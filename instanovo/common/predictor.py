from __future__ import annotations

import os
import sys
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, InitProcessGroupKwargs
from datasets import Dataset, Value
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console, set_rank
from instanovo.common.dataset import DataProcessor
from instanovo.common.utils import Timer
from instanovo.constants import ANNOTATED_COLUMN, ANNOTATION_ERROR, PREDICTION_COLUMNS, MSColumns
from instanovo.inference import Decoder
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.device_handler import validate_and_configure_device
from instanovo.utils.metrics import Metrics
from instanovo.utils.s3 import S3FileHandler

load_dotenv()

# Automatic rank logger
logger = ColorLog(console, __name__).logger


class AccelerateDeNovoPredictor(metaclass=ABCMeta):
    """Predictor class that uses the Accelerate library."""

    @property
    def s3(self) -> S3FileHandler:
        """Get the S3 file handler.

        Returns:
            S3FileHandler: The S3 file handler
        """
        return self._s3

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

        # Hide progress bar from HF datasets
        disable_progress_bar()

        self.targets: list | None = None
        self.output_path = self.config.get("output_path", None)
        self.pred_df: pd.DataFrame | None = None
        self.results_dict: dict | None = None

        self.prediction_tokenised_col = self.config.get("prediction_tokenised_col", "predictions_tokenised")
        self.prediction_col = self.config.get("prediction_col", "predictions")
        self.log_probs_col = self.config.get("log_probs_col", "log_probs")
        self.token_log_probs_col = self.config.get("token_log_probs_col", "token_log_probs")

        # Encoder output config
        self.save_encoder_outputs = config.get("save_encoder_outputs", False)
        self.encoder_output_path = config.get("encoder_output_path", None)
        self.encoder_output_reduction = config.get("encoder_output_reduction", "mean")
        if self.save_encoder_outputs and self.encoder_output_path is None:
            raise ValueError(
                "Expected 'encoder_output_path' but found None. "
                "Please specify it in the config file or with the cli flag encoder-output-path=path/to/encoder_outputs.parquet",
            )
        if self.save_encoder_outputs and self.encoder_output_reduction not in ["mean", "max", "sum", "full"]:
            raise ValueError(
                f"Invalid encoder output reduction: {self.encoder_output_reduction}. Please choose from 'mean', 'max', 'sum', or 'full'."
            )
        if self.encoder_output_reduction == "full":
            raise NotImplementedError("Full encoder output reduction is not yet implemented.")

        self.accelerator = self.setup_accelerator()

        if self.accelerator.is_main_process:
            logger.info(f"Config:\n{OmegaConf.to_yaml(self.config)}")

        # Whether to check metrics
        self.denovo = self.config.get("denovo", False)
        self._group_output: dict[str, str] | None = None
        self._group_mapping: dict[str, str] | None = None

        self._s3: S3FileHandler = S3FileHandler()

        logger.info("Loading model...")
        self.model, self.model_config = self.load_model()

        self.model = self.model.to(self.accelerator.device)
        self.model = self.model.eval()
        logger.info("Model loaded.")

        self.residue_set = self.model.residue_set
        self.residue_set.update_remapping(self.config.get("residue_remapping", {}))
        logger.info(f"Vocab: {self.residue_set.index_to_residue}")

        logger.info("Loading dataset...")
        self.test_dataset = self.load_dataset()

        logger.info(f"Data loaded: {len(self.test_dataset):,} test samples.")

        self.test_dataloader = self.build_dataloader(self.test_dataset)
        logger.info("Data loader built.")

        # Print sample batch
        self.print_sample_batch()

        logger.info("Initializing decoder.")
        self.decoder = self.setup_decoder()

        logger.info("Initializing metrics.")
        self.metrics = self.setup_metrics()

        # Prepare accelerator
        self.model, self.test_dataloader = self.accelerator.prepare(self.model, self.test_dataloader)

        self.running_loss = None
        self.steps_per_inference = len(self.test_dataloader)

        logger.info(f"Total batches: {self.steps_per_inference:,d}")

        # Final sync after setup
        self.accelerator.wait_for_everyone()

    @abstractmethod
    def load_model(self) -> Tuple[nn.Module, DictConfig]:
        """Load the model."""
        ...

    @abstractmethod
    def setup_decoder(self) -> Decoder:
        """Setup the decoder."""
        ...

    @abstractmethod
    def setup_data_processor(self) -> DataProcessor:
        """Setup the data processor."""
        ...

    @abstractmethod
    def get_predictions(self, batch: Any) -> dict[str, Any]:
        """Get the predictions for a batch."""
        ...

    def postprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Postprocess the dataset."""
        return dataset

    def load_dataset(self) -> Dataset:
        """Load the test dataset.

        Returns:
            Dataset:
                The test dataset
        """
        data_path = self.config.get("data_path", None)
        if OmegaConf.is_list(data_path):
            # If validation data is a list, we assume the data is grouped
            # Each list item should include a result_name, input_path, and output_path

            _new_data_paths = []
            self._group_mapping = {}  # map file paths to group name
            self._group_output = {}  # map group name to output path (for saving predictions)

            for group in data_path:
                path = group.get("input_path")
                name = group.get("result_name")
                for fp in SpectrumDataFrame._convert_file_paths(path):  # e.g. expands list of globs
                    self._group_mapping[fp] = name
                _new_data_paths.append(path)
                self._group_output[name] = group.get("output_path")
            self.group_data_paths = data_path
            data_path = _new_data_paths

        logger.info(f"Loading data from {data_path}")
        try:
            sdf = SpectrumDataFrame.load(
                data_path,
                lazy=False,
                is_annotated=not self.denovo,
                column_mapping=self.config.get("column_map", None),
                shuffle=False,
                add_spectrum_id=True,
                add_source_file_column=True,
            )
        except ValueError as e:
            # More descriptive error message in predict mode.
            if str(e) == ANNOTATION_ERROR:
                raise ValueError(
                    "The sequence column is missing annotations, are you trying to run de novo prediction? Add the `denovo=True` flag"
                ) from e
            else:
                raise

        dataset = sdf.to_dataset(in_memory=True)

        subset = self.config.get("subset", 1.0)
        if not 0 < subset <= 1:
            raise ValueError(
                f"Invalid subset value: {subset}. Must be a float greater than 0 and less than or equal to 1."  # noqa: E501
            )

        original_size = len(dataset)
        max_charge = self.config.get("max_charge", 10)
        model_max_charge = self.model_config.get("max_charge", 10)
        if max_charge > model_max_charge:
            logger.warning(f"Inference has been configured with max_charge={max_charge}, but model has max_charge={model_max_charge}.")
            logger.warning(f"Overwriting max_charge config to model value: {model_max_charge}.")
            max_charge = model_max_charge

        precursor_charge_col = MSColumns.PRECURSOR_CHARGE.value
        dataset = dataset.filter(lambda row: (row[precursor_charge_col] <= max_charge) and (row[precursor_charge_col] > 0))

        # Filter invalid sequences
        if not self.denovo:
            supported_residues = set(self.residue_set.vocab)
            supported_residues.update(set(self.residue_set.residue_remapping.keys()))
            data_residues = sdf.get_vocabulary(self.residue_set.tokenize)
            if len(data_residues - supported_residues) > 0:
                logger.warning(
                    f"Found {len(data_residues - supported_residues):,d} unsupported residues! "
                    "These rows will be dropped in evaluation mode. Please adjust the metrics "
                    "calculations accordingly."
                )
                logger.warning(f"New residues found: \n{data_residues - supported_residues}")
                logger.warning(f"Residues supported: \n{supported_residues}")
                logger.warning("Please check residue remapping if a different convention has been used.")
                dataset = dataset.filter(
                    lambda row: all(residue in supported_residues for residue in set(self.residue_set.tokenize(row[ANNOTATED_COLUMN])))
                )

        if len(dataset) < original_size:
            logger.warning(
                f"Found {original_size - len(dataset):,d} rows with charge > {max_charge} or <= 0. "
                "This could mean the charge column is missing or contains invalid values. "
                "These rows will be skipped."
            )

        if subset < 1.0:
            dataset = dataset.train_test_split(test_size=subset, seed=42)["test"]

        if len(dataset) == 0:
            logger.warning("No data found, exiting.")
            sys.exit()

        # Optional dataset postprocessing
        dataset = self.postprocess_dataset(dataset)

        # Used to group validation outputs
        if self._group_mapping is not None:
            logger.info("Computing groups.")
            groups = [self._group_mapping.get(row.get("source_file"), "no_group") for row in dataset]
            dataset = dataset.add_column("group", groups, feature=Value("string"))

            if self.accelerator.is_main_process:
                logger.info("Sequences per group:")
                group_counts = Counter(groups)
                for group, count in group_counts.items():
                    logger.info(f" - {group}: {count:,d}")

                self.using_groups = True
        else:
            dataset = dataset.add_column("group", ["no_group"] * len(dataset), feature=Value("string"))
            self.using_groups = False

        # Force add a unique prediction_id column
        # This will be used to order predictions
        dataset = dataset.add_column("prediction_id", np.arange(len(dataset)), feature=Value("int32"))

        return dataset

    def print_sample_batch(self) -> None:
        """Print a sample batch of the training data."""
        if self.accelerator.is_main_process:
            # sample_batch = next(iter(self.train_dataloader))
            sample_batch = next(iter(self.test_dataloader))
            logger.info("Sample batch:")
            for key, value in sample_batch.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.shape
                    value_type = value.dtype
                else:
                    value_shape = len(value)
                    value_type = type(value)

                logger.info(f" - {key}: {value_type}, {value_shape}")

    def setup_metrics(self) -> Metrics:
        """Setup the metrics."""
        return Metrics(self.residue_set, self.config.get("max_isotope_error", 1))

    def setup_accelerator(self) -> Accelerator:
        """Setup the accelerator."""
        # TODO: How do we specify device without accelerator?
        timeout = timedelta(seconds=self.config.get("timeout", 3600))
        validate_and_configure_device(self.config)
        accelerator = Accelerator(
            cpu=self.config.get("force_cpu", False),
            mixed_precision="fp16" if torch.cuda.is_available() and not self.config.get("force_cpu", False) else "no",
            dataloader_config=DataLoaderConfiguration(split_batches=False),
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timeout)],
        )

        device = accelerator.device  # Important, this forces ranks to choose a device.

        if accelerator.num_processes > 1:
            set_rank(accelerator.local_process_index)

        if accelerator.is_main_process:
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Torch version: {torch.__version__}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Predicting with {accelerator.num_processes} devices")
            logger.info(f"Per-device batch size: {self.config['batch_size']}")

        logger.info(f"Using device: {device}")

        return accelerator

    def build_dataloader(self, test_dataset: Dataset) -> torch.utils.data.DataLoader:
        """Setup the dataloaders."""
        test_processor = self.setup_data_processor()
        test_processor.add_metadata_columns(["prediction_id", "group"])

        test_dataset = test_processor.process_dataset(test_dataset)

        pin_memory = self.config.get("pin_memory", False)
        if self.accelerator.device == torch.device("cpu") or self.config.get("mps", False):
            pin_memory = False

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            collate_fn=test_processor.collate_fn,
            num_workers=self.config.get("num_workers", 8),
            pin_memory=pin_memory,
            prefetch_factor=self.config.get("prefetch_factor", 2),
            drop_last=False,
        )
        return test_dataloader

    def predict(self) -> pd.DataFrame:
        """Predict the test dataset."""
        all_predictions: dict[str, list] = defaultdict(list)
        all_encoder_outputs: list[np.ndarray] = []
        test_step = 0

        logger.info("Predicting...")
        inference_timer = Timer(self.steps_per_inference)
        print_batch_size = True
        for i, batch in enumerate(self.test_dataloader):
            if print_batch_size:
                logger.info(f"Batch {i} shape: {batch['spectra'].shape[0]}")
                print_batch_size = False

            # Implementation specific
            with torch.no_grad(), self.accelerator.autocast():
                batch_predictions = self.get_predictions(batch)

            # Pass through prediction_id and group columns
            # prediction_id is automatically cast to tensor
            batch_predictions["prediction_id"] = [x.item() for x in batch["prediction_id"]]
            batch_predictions["group"] = batch["group"]

            # Some outputs are required from get_predictions
            for k in PREDICTION_COLUMNS:
                if k not in batch_predictions:
                    raise ValueError(f"Prediction column {k} not found in batch predictions.")
                all_predictions[k].extend(batch_predictions[k])

            if "encoder_output" in batch_predictions:
                # Always ensure it is removed with pop even if we do not save it.
                encoder_output = batch_predictions.pop("encoder_output")
                if self.save_encoder_outputs:
                    all_encoder_outputs.extend(encoder_output)

            # Additional prediction info
            if self.config.get("save_all_predictions", False):
                for k, v in batch_predictions.items():
                    if k in PREDICTION_COLUMNS:
                        continue
                    all_predictions[k].extend(v)

            test_step += 1
            inference_timer.step()

            if (i + 1) % self.config.get("log_interval", 50) == 0 or (i + 1) == self.steps_per_inference:
                logger.info(
                    f"[Batch {i + 1:05d}/{self.steps_per_inference:05d}] "
                    f"[{inference_timer.get_time_str()}/{inference_timer.get_eta_str()}] "  # noqa: E501
                    f"{inference_timer.get_step_time_rate_str()}: "
                )

        logger.info(f"Time taken for {self.config.get('data_path', None)} is {inference_timer.get_delta():.1f} seconds")

        logger.info("Prediction complete.")
        self.accelerator.wait_for_everyone()

        if self.accelerator.num_processes > 1:
            logger.info("Gathering predictions from all processes...")

        # Broadcast all predictions to all processes
        for key, value in all_predictions.items():
            all_predictions[key] = self.accelerator.gather_for_metrics(value, use_gather_object=True)

        if self.save_encoder_outputs:
            all_encoder_outputs = self.accelerator.gather_for_metrics(all_encoder_outputs, use_gather_object=True)

        if self.accelerator.is_main_process:
            pred_df = self.predictions_to_df(all_predictions)
            pred_df = self.postprocess_predictions(pred_df)

            results_dict = None
            if not self.denovo:
                logger.info("Calculating metrics...")

                results_dict = self.calculate_metrics(pred_df)

            self.save_predictions(pred_df, results_dict)

            if self.save_encoder_outputs:
                self.save_encoder_outputs_to_parquet(pred_df["spectrum_id"].tolist(), all_encoder_outputs)
        else:
            pred_df = None

        return pred_df

    def _tokens_to_string(self, tokens: list[str] | None) -> str:
        """Convert a list of tokens to a ProForma compliant string."""
        if tokens is None:
            return ""
        peptide = ""
        if len(tokens) > 1 and not tokens[0][0].isalpha():
            # Assume n-terminal
            peptide = tokens[0] + "-"
            tokens = tokens[1:]
        return peptide + "".join(tokens)

    def predictions_to_df(self, predictions: dict[str, list]) -> pd.DataFrame:
        """Convert the predictions to a pandas DataFrame.

        Args:
            predictions: The predictions dictionary

        Returns:
            pd.DataFrame: The predictions dataframe
        """
        index_cols = self.config.get("index_columns", ["precursor_mz", "precursor_charge"])
        index_cols = [x for x in index_cols if x in self.test_dataset.column_names]
        index_cols.append("prediction_id")
        index_df = self.test_dataset.to_pandas()[index_cols]

        pred_df = pd.DataFrame(predictions)
        # Drop duplicates caused by padding multiple processes
        pred_df = pred_df.drop_duplicates(subset=["prediction_id"], keep="first")

        pred_df = index_df.merge(pred_df, on="prediction_id", how="left")

        # Some column processing
        pred_df["predictions_tokenised"] = pred_df["predictions"].map(lambda x: ", ".join(x))
        pred_df["predictions"] = pred_df["predictions"].map(self._tokens_to_string)
        pred_df["targets"] = pred_df["targets"].map(self._tokens_to_string)
        pred_df["delta_mass_ppm"] = pred_df.apply(
            lambda row: np.min(np.abs(self.metrics.matches_precursor(row["predictions_tokenised"], row["precursor_mz"], row["precursor_charge"])[1])),
            axis=1,
        )

        pred_df = pred_df.rename(
            columns={
                "predictions": self.prediction_col,
                "predictions_tokenised": self.prediction_tokenised_col,
                "prediction_log_probability": self.log_probs_col,
                "prediction_token_log_probabilities": self.token_log_probs_col,
            }
        )

        if self.denovo:
            pred_df.drop(columns=["targets"], inplace=True)

        return pred_df

    def postprocess_predictions(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the predictions.

        Optionally, this can be used to modify the predictions, eg. ensembling.
        By default, this does nothing.

        Args:
            pred_df: The predictions dataframe

        Returns:
            pd.DataFrame: The postprocessed predictions dataframe
        """
        return pred_df

    def calculate_metrics(
        self,
        pred_df: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Calculate the metrics.

        Args:
            pred_df: The predictions dataframe

        Returns:
            dict[str, Any] | None: The results dictionary containing the metrics
        """
        predictions = pred_df[self.prediction_tokenised_col].copy()
        targets = pred_df["targets"]
        log_probs = pred_df[self.log_probs_col]
        delta_mass_ppm = pred_df["delta_mass_ppm"]

        aa_prec, aa_recall, pep_recall, pep_prec = self.metrics.compute_precision_recall(targets, predictions)
        aa_er = self.metrics.compute_aa_er(targets, predictions)
        auc = self.metrics.calc_auc(
            targets,
            predictions,
            np.exp(log_probs),
        )

        logger.info("Performance:")
        logger.info(f"  aa_er       {aa_er:.5f}")
        logger.info(f"  aa_prec     {aa_prec:.5f}")
        logger.info(f"  aa_recall   {aa_recall:.5f}")
        logger.info(f"  pep_prec    {pep_prec:.5f}")
        logger.info(f"  pep_recall  {pep_recall:.5f}")
        logger.info(f"  auc         {auc:.5f}")

        fdr = self.config.get("filter_fdr_threshold", None)
        if fdr:
            _, threshold = self.metrics.find_recall_at_fdr(
                targets,
                predictions,
                np.exp(log_probs),
                fdr=fdr,
            )
            aa_prec, aa_recall, pep_recall, pep_prec = self.metrics.compute_precision_recall(
                targets,
                predictions,
                np.exp(log_probs),
                threshold=threshold,
            )
            logger.info(f"Performance at {fdr * 100:.1f}% FDR:")
            logger.info(f"  aa_prec     {aa_prec:.5f}")
            logger.info(f"  aa_recall   {aa_recall:.5f}")
            logger.info(f"  pep_prec    {pep_prec:.5f}")
            logger.info(f"  pep_recall  {pep_recall:.5f}")
            logger.info(f"  confidence  {threshold:.5f}")

        filter_precursor_ppm = self.config.get("filter_precursor_ppm", None)
        if filter_precursor_ppm and delta_mass_ppm is not None:
            idx = delta_mass_ppm < filter_precursor_ppm  # type: ignore
            logger.info(f"Performance with filtering at {filter_precursor_ppm} ppm delta mass:")
            if np.sum(idx) > 0:
                filtered_preds = pd.Series(predictions)
                filtered_preds[~idx] = ""
                aa_prec, aa_recall, pep_recall, pep_prec = self.metrics.compute_precision_recall(targets, filtered_preds)
                logger.info(f"  aa_prec     {aa_prec:.5f}")
                logger.info(f"  aa_recall   {aa_recall:.5f}")
                logger.info(f"  pep_prec    {pep_prec:.5f}")
                logger.info(f"  pep_recall  {pep_recall:.5f}")
                logger.info(f"Rows filtered: {len(predictions) - np.sum(idx)} ({(len(predictions) - np.sum(idx)) / len(predictions) * 100:.2f}%)")
                if np.sum(idx) < 1000:
                    logger.info(f"Metrics calculated on a small number of samples ({np.sum(idx)}), interpret with care!")
            else:
                logger.info("No predictions met criteria, skipping metrics.")

        model_confidence_no_pred = self.config.get("filter_confidence", None)
        if model_confidence_no_pred:
            idx = np.exp(log_probs) > model_confidence_no_pred
            logger.info(f"Performance with filtering confidence < {model_confidence_no_pred}")
            if np.sum(idx) > 0:
                filtered_preds = pd.Series(predictions)
                filtered_preds[~idx] = ""
                aa_prec, aa_recall, pep_recall, pep_prec = self.metrics.compute_precision_recall(targets, filtered_preds)
                logger.info(f"  aa_prec     {aa_prec:.5f}")
                logger.info(f"  aa_recall   {aa_recall:.5f}")
                logger.info(f"  pep_prec    {pep_prec:.5f}")
                logger.info(f"  pep_recall  {pep_recall:.5f}")
                logger.info(f"Rows filtered: {len(predictions) - np.sum(idx)} ({(len(predictions) - np.sum(idx)) / len(predictions) * 100:.2f}%)")
                if np.sum(idx) < 1000:
                    logger.info(f"Metrics calculated on a small number of samples ({np.sum(idx)}), interpret with care!")
            else:
                logger.info("No predictions met criteria, skipping metrics.")

        # Evaluate individual result files
        if self.using_groups and not self.denovo:
            logger.info("Evaluating individual result files.")
            # TODO Handle better with pred_df
            _preds = pd.Series(predictions)
            _targs = pd.Series(targets)
            _probs = pd.Series(log_probs)

            # TODO Make this more generic
            results_dict: dict[str, Any] = {
                "run_name": self.config.get("run_name"),
                "instanovo_model": self.config.get("instanovo_model"),
                "num_beams": self.config.get("num_beams", 1),
                "use_knapsack": self.config.get("use_knapsack", False),
            }
            for group in pred_df["group"].unique():
                if group == "no_group":
                    continue
                idx = pred_df["group"] == group
                _group_preds = _preds[idx].reset_index(drop=True)
                _group_targs = _targs[idx].reset_index(drop=True)
                _group_probs = _probs[idx].reset_index(drop=True)
                aa_prec, aa_recall, pep_recall, pep_prec = self.metrics.compute_precision_recall(_group_targs, _group_preds)
                aa_er = self.metrics.compute_aa_er(_group_targs, _group_preds)
                auc = self.metrics.calc_auc(_group_targs, _group_preds, _group_probs)

                results_dict.update(
                    {
                        f"{group}_aa_prec": [aa_prec],
                        f"{group}_aa_recall": [aa_recall],
                        f"{group}_pep_recall": [pep_recall],
                        f"{group}_pep_prec": [pep_prec],
                        f"{group}_aa_er": [aa_er],
                        f"{group}_auc": [auc],
                    }
                )

                fdr = self.config.get("filter_fdr_threshold", None)
                if fdr:
                    _, threshold = self.metrics.find_recall_at_fdr(_group_targs, _group_preds, np.exp(_group_probs), fdr=fdr)
                    _, _, pep_recall_at_fdr, _ = self.metrics.compute_precision_recall(
                        _group_targs,
                        _group_preds,
                        np.exp(_group_probs),
                        threshold=threshold,
                    )

                    results_dict.update(
                        {
                            f"{group}_pep_recall_at_{fdr:.3f}_fdr": [pep_recall_at_fdr],
                        }
                    )
            return results_dict
        return None

    def save_predictions(self, pred_df: pd.DataFrame, results_dict: dict[str, list] | None = None) -> None:
        """Save the predictions to a file.

        Args:
            pred_df: The predictions dataframe
            results_dict: The results dictionary containing the metrics
        """
        # Save metrics to a file
        if self.using_groups and not self.denovo and results_dict is not None:
            result_path = self.config.get("result_file_path")
            logger.info(f"Saving metrics to {result_path}.")
            local_path = self.s3.get_local_path(result_path, missing_ok=True)
            if local_path is not None and os.path.exists(local_path):
                results_df = pd.read_csv(local_path)
                results_df = pd.concat([results_df, pd.DataFrame(results_dict)], ignore_index=True, join="outer")
            else:
                results_df = pd.DataFrame(results_dict)

            self.s3.upload_to_s3_wrapper(results_df.to_csv, result_path, index=False)

        # Save individual result files per group
        if self.using_groups and self._group_output is not None and pred_df is not None:
            logger.info("Saving individual result files per group.")
            for group in pred_df["group"].unique():
                idx = pred_df["group"] == group
                if self._group_output.get(group) is not None:
                    self.s3.upload_to_s3_wrapper(pred_df[idx].to_csv, self._group_output[group], index=False)

        # Save output
        if self.output_path is not None and pred_df is not None:
            logger.info(f"Saving predictions to {self.output_path}.")
            self.s3.upload_to_s3_wrapper(pred_df.to_csv, self.output_path, index=False)
            logger.info(f"Predictions saved to {self.output_path}")

            # Upload to Aichor
            if S3FileHandler._aichor_enabled() and not self.output_path.startswith("s3://"):
                self.s3.upload(self.output_path, S3FileHandler.convert_to_s3_output(self.output_path))

    def save_encoder_outputs_to_parquet(self, spectrum_ids: list[str], encoder_outputs: list[np.ndarray]) -> None:
        """Save the encoder outputs to a file.

        Args:
            encoder_outputs: The encoder outputs
            spectrum_ids: The spectrum ids
        """
        if len(encoder_outputs) == 0:
            logger.warning(f"No encoder outputs were returned by the decoder {type(self.decoder)}.")
            logger.warning("Skipping encoder output saving.")
            return

        if self.encoder_output_path is not None:
            encoder_outputs_fp32 = np.stack(encoder_outputs).astype(np.float32)
            encoder_output_df = pl.DataFrame(
                {"spectrum_id": spectrum_ids, **{f"spectrum_encoding_{i}": encoder_outputs_fp32[:, i] for i in range(encoder_outputs_fp32.shape[1])}}
            )

            logger.info(f"Saving encoder outputs to {self.encoder_output_path}.")
            self.s3.upload_to_s3_wrapper(encoder_output_df.write_parquet, self.encoder_output_path)
