from pathlib import Path
from typing import Any, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from accelerate.utils import broadcast_object_list
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console
from instanovo.common import AccelerateDeNovoPredictor, DataProcessor
from instanovo.constants import (
    DIFFUSION_START_STEP,
    REFINEMENT_COLUMN,
    REFINEMENT_PROBABILITY_COLUMN,
    SpecialTokens,
)
from instanovo.diffusion.data import DiffusionDataProcessor
from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
from instanovo.inference.diffusion import DiffusionDecoder
from instanovo.inference.interfaces import Decoder
from instanovo.transformer.predict import TransformerPredictor
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs"


class DiffusionPredictor(AccelerateDeNovoPredictor):
    """Predictor for the InstaNovo+ model."""

    def __init__(
        self,
        config: DictConfig,
    ):
        self.refine = config.get("refine", False)
        self.refine_all = config.get("refine_all", True)
        self.refine_threshold = np.log(config.get("refine_threshold", 0.9))
        self.precursor_tolerance = config.get("filter_precursor_ppm", 50)
        super().__init__(config)

    # Possibly merge with transformer load_model
    def load_model(self) -> Tuple[nn.Module, DictConfig]:
        """Setup the model."""
        default_model = InstaNovoPlus.get_pretrained()[0]
        model_path = self.config.get("instanovo_plus_model", default_model)

        logger.info(f"Loading InstaNovo+ model {model_path}")
        if model_path in InstaNovoPlus.get_pretrained():
            # Using a pretrained model from models.json
            model, model_config = InstaNovoPlus.from_pretrained(
                model_path, override_config={"peak_embedding_dtype": "float32"} if self.config.get("mps", False) else None
            )
        else:
            model_path = self.s3.get_local_path(model_path)
            assert model_path is not None
            model, model_config = InstaNovoPlus.load(
                model_path, override_config={"peak_embedding_dtype": "float32"} if self.config.get("mps", False) else None
            )

        return model, model_config

    def postprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Load previous predictions for refinement."""
        if not self.refine:
            return dataset

        data_path = self.config.get("data_path", None)
        prediction_paths = []
        if self.accelerator.is_main_process:
            if OmegaConf.is_list(data_path):
                # Grouped refinement
                for group in data_path:
                    path = group.get("refinement_path")
                    if path is None:
                        raise ValueError("refinement_path must be specified per group when `refine` is True in pipeline mode.")
                    path = self.s3.get_local_path(path)
                    prediction_paths.append(path)
            else:
                path = self.config.get("refinement_path", None)
                if path is None:
                    raise ValueError("refinement_path must be specified when `refine` is True.")
                path = self.s3.get_local_path(path)
                prediction_paths.append(path)

            if self.accelerator.num_processes > 1:
                logger.info(f"Broadcasting {len(prediction_paths)} refinement paths")

        prediction_paths = broadcast_object_list([prediction_paths], from_process=0)[0]
        if self.accelerator.num_processes > 1:
            logger.info(f"Received {len(prediction_paths)} paths for refinement")

        predictions_id_col = self.config.get("refinement_id_col", "spectrum_id")
        dataset_id_col = self.config.get("dataset_id_col", "spectrum_id")
        predictions_refine_col = self.config.get("prediction_refine_col", "predictions_tokenised")
        prediction_confidence_col = self.config.get("prediction_confidence_col", None)

        for path in prediction_paths:
            columns = set(pl.scan_csv(path).collect_schema().keys())
            if predictions_id_col not in columns:
                raise ValueError(f"Column '{predictions_id_col}' does not exist in {path}.")
            if predictions_refine_col not in columns:
                raise ValueError(f"Column '{predictions_refine_col}' does not exist in {path}.")
            if prediction_confidence_col not in columns and prediction_confidence_col is not None:
                raise ValueError(f"Column '{prediction_confidence_col}' does not exist in {path}.")

        if dataset_id_col not in dataset.column_names:
            raise ValueError(f"Column '{dataset_id_col}' does not exist in dataset.")

        target_schema = {predictions_id_col: pl.String, predictions_refine_col: pl.String}

        if prediction_confidence_col is not None:
            target_schema.update({prediction_confidence_col: pl.Float64})

        logger.info(f"Reading {len(prediction_paths)} refinement file(s)")
        predictions_df = pl.concat(
            [pl.read_csv(path, columns=list(target_schema.keys()), schema_overrides=target_schema) for path in prediction_paths],
            how="vertical",
        )

        id_to_predictions = dict(
            zip(
                predictions_df[predictions_id_col],
                predictions_df[predictions_refine_col],
                strict=False,
            )
        )
        if prediction_confidence_col is not None:
            id_to_confidence = dict(
                zip(
                    predictions_df[predictions_id_col],
                    predictions_df[prediction_confidence_col],
                    strict=False,
                )
            )
        else:
            logger.warning("'prediction_confidence_col' not set. Setting all input confidence scores to 0.")

        def add_predictions_column(row: dict[str, Any]) -> dict[str, Any]:
            prediction = id_to_predictions.get(row[dataset_id_col], None)
            row[REFINEMENT_COLUMN] = self._clean_predictions(prediction)
            if prediction_confidence_col is not None:
                row[REFINEMENT_PROBABILITY_COLUMN] = id_to_confidence.get(row[dataset_id_col], None)
            else:
                row[REFINEMENT_PROBABILITY_COLUMN] = 0
            return row

        logger.info("Adding refinement columns to dataset")
        dataset = dataset.map(add_predictions_column)

        num_none_refine = sum(1 for x in dataset[REFINEMENT_COLUMN] if x is None)
        if num_none_refine > 0:
            logger.info(f"Refinement is missing for {num_none_refine} / {len(dataset)} spectra ({((num_none_refine / len(dataset)) * 100):.2f}%)")

        return dataset

    def _clean_predictions(self, predictions: str | None) -> str:
        if predictions is None:
            return ""
        # Replace invalid tokens with PAD token
        tokens = self.model.residue_set.tokenize(predictions)
        tokens = [token if token in self.model.residue_set.vocab else SpecialTokens.PAD_TOKEN.value for token in tokens]
        return ", ".join(tokens)

    def setup_data_processor(self) -> DataProcessor:
        """Setup the data processor."""
        processor = DiffusionDataProcessor(
            self.residue_set,
            n_peaks=self.model_config.get("n_peaks", 200),
            min_mz=self.model_config.get("min_mz", 50.0),
            max_mz=self.model_config.get("max_mz", 2500.0),
            min_intensity=self.model_config.get("min_intensity", 0.01),
            remove_precursor_tol=self.model_config.get("remove_precursor_tol", 2.0),
            return_str=False,
            reverse_peptide=False,
            add_eos=False,
            peptide_pad_length=self.model_config.get("max_length", 40),
            peptide_pad_value=self.residue_set.PAD_INDEX,
            use_spectrum_utils=False,
            annotated=not self.denovo,
            metadata_columns=["group"],
        )

        if self.refine:
            processor.add_metadata_columns([REFINEMENT_COLUMN, REFINEMENT_PROBABILITY_COLUMN])

        return processor

    def setup_decoder(self) -> Decoder:
        """Setup the decoder."""
        return DiffusionDecoder(model=self.model)  # type: ignore

    def get_predictions(self, batch: Any) -> dict[str, Any]:
        """Get the predictions for a batch."""
        num_beams = self.config.get("num_beams", 1)
        batch_size = batch["spectra"].size(0)

        batch_results: dict[str, Any] = self.decoder.decode(
            initial_sequence=batch[REFINEMENT_COLUMN] if self.refine else None,
            spectra=batch["spectra"],
            precursors=batch["precursors"],
            spectra_padding_mask=batch["spectra_mask"],
            start_step=DIFFUSION_START_STEP if self.refine else None,  # type: ignore
            beam_size=num_beams,
            return_encoder_output=self.save_encoder_outputs,
            encoder_output_reduction=self.encoder_output_reduction,
        )  # type: ignore

        if "peptides" in batch:
            targets = [self.residue_set.decode(seq, reverse=False) for seq in batch["peptides"]]
        else:
            targets = [None] * batch_size

        batch_results["targets"] = targets

        if self.refine and not self.refine_all:
            unrefined_preds = [self.residue_set.decode(seq, reverse=False) for seq in batch[REFINEMENT_COLUMN]]
            batch_results["unrefined_predictions"] = unrefined_preds
            unrefined_matches = []
            for i in range(batch_size):
                matches, _ = self.metrics.matches_precursor(
                    unrefined_preds[i],
                    batch["precursors"][i][2],
                    batch["precursors"][i][1],
                    prec_tol=self.precursor_tolerance,
                )
                unrefined_matches.append(matches)

            for i in range(batch_size):
                refine_prob = batch[REFINEMENT_PROBABILITY_COLUMN][i].item()
                if self.refine_threshold is not None:
                    if (
                        refine_prob < self.refine_threshold
                        and unrefined_matches[i] > batch_results["meets_precursor"][i]
                        and refine_prob > batch_results["prediction_log_probability"][i]
                    ):
                        batch_results["meets_precursor"][i] = unrefined_matches[i]
                        batch_results["predictions"][i] = unrefined_preds[i]
                        batch_results["prediction_log_probability"][i] = refine_prob
                else:
                    # Ensemble based on ppm match
                    if unrefined_matches[i] > batch_results["meets_precursor"][i]:
                        batch_results["meets_precursor"][i] = unrefined_matches[i]
                        batch_results["predictions"][i] = unrefined_preds[i]
                        batch_results["prediction_log_probability"][i] = refine_prob

        batch_results.pop("meets_precursor", None)

        return batch_results


class CombinedPredictor(TransformerPredictor):
    """Predictor for the combined InstaNovo+ model."""

    diffusion_load_model = DiffusionPredictor.load_model
    diffusion_get_predictions = DiffusionPredictor.get_predictions

    def __init__(self, config: DictConfig):
        self.refine = config.get("refine", False)
        self.refine_all = config.get("refine_all", True)
        self.refine_threshold = np.log(config.get("refine_threshold", 0.9))
        self.precursor_tolerance = config.get("filter_precursor_ppm", 50)
        super().__init__(config)

        # Manually prepare the diffusion model since it is not prepared in the parent class
        if self.refine:
            logger.info("Running in refinement mode.")
            self.diffusion_model: nn.Module = self.accelerator.prepare(self.diffusion_model)

    def load_model(self) -> Tuple[nn.Module, DictConfig]:
        """Setup the model."""
        self.transformer_model, transformer_model_config = super().load_model()

        if not self.refine:
            return self.transformer_model, transformer_model_config

        self.diffusion_model, diffusion_model_config = self.diffusion_load_model()  # type: ignore

        self.diffusion_residue_set = self.diffusion_model.residue_set

        # Compare residue sets
        transformer_residue_set = self.transformer_model.residue_set.index_to_residue
        diffusion_residue_set = self.diffusion_residue_set.index_to_residue
        if transformer_residue_set != diffusion_residue_set:
            raise ValueError("Transformer and diffusion residue sets do not match")

        # Compare max length
        self.transformer_max_length = transformer_model_config.get("max_length", 40)
        self.diffusion_max_length = diffusion_model_config.get("max_length", 40)
        if self.transformer_max_length != self.diffusion_max_length:
            logger.warning(
                f"Transformer and diffusion max length do not match. "
                f"Transformer: {self.transformer_max_length}, "
                f"Diffusion: {self.diffusion_max_length}"
            )

        return self.transformer_model, transformer_model_config

    def setup_decoder(self) -> Decoder:
        """Setup the decoder."""
        # Diffusion decoder
        if self.refine:
            self.diffusion_decoder = DiffusionDecoder(model=self.diffusion_model)
        else:
            self.diffusion_decoder = None  # type: ignore

        # Transformer decoder
        self.transformer_decoder = super().setup_decoder()
        return self.transformer_decoder  # type: ignore

    def _tokenize_and_pad(self, refinement: list[str]) -> torch.Tensor:
        """Tokenize and pad the transformer predictions."""
        encodings = []
        for refine in refinement:
            refine_tokenized = self.diffusion_residue_set.tokenize(refine)

            refine_encoding = self.diffusion_residue_set.encode(refine_tokenized, add_eos=False, return_tensor="pt")

            refine_encoding = refine_encoding[: self.diffusion_max_length]

            # Diffusion always padded to fixed length
            refine_padded = torch.full(
                (max(self.diffusion_max_length, refine_encoding.shape[0]),),
                fill_value=self.diffusion_residue_set.PAD_INDEX,
                dtype=refine_encoding.dtype,
                device=refine_encoding.device,
            )
            refine_padded[: refine_encoding.shape[0]] = refine_encoding

            encodings.append(refine_padded)

        encodings, _ = DiffusionDataProcessor._pad_and_mask(encodings)
        return encodings

    def get_predictions(self, batch: Any) -> dict[str, Any]:
        """Get the predictions for a batch."""
        # set self.model to use the correct model
        self.decoder = self.transformer_decoder
        transformer_predictions = super().get_predictions(batch)

        if not self.refine:
            return transformer_predictions  # type: ignore

        batch[REFINEMENT_COLUMN] = self._tokenize_and_pad(transformer_predictions["predictions"]).to(self.accelerator.device)
        batch[REFINEMENT_PROBABILITY_COLUMN] = torch.tensor(transformer_predictions["prediction_log_probability"]).to(self.accelerator.device)

        self.decoder = self.diffusion_decoder  # type: ignore
        diffusion_predictions = self.diffusion_get_predictions(batch)  # type: ignore

        predictions = {
            "predictions": diffusion_predictions["predictions"],
            "prediction_log_probability": diffusion_predictions["prediction_log_probability"],
            "prediction_token_log_probabilities": diffusion_predictions["prediction_token_log_probabilities"],
            "targets": transformer_predictions["targets"],
        }

        # Don't need to keep these
        transformer_predictions.pop("targets")
        diffusion_predictions.pop("targets")

        predictions.update({f"instanovo_{k}": v for k, v in transformer_predictions.items()})
        predictions.update({f"instanovoplus_{k}": v for k, v in diffusion_predictions.items()})

        return predictions
