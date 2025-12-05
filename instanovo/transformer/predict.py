import os
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.common import AccelerateDeNovoPredictor, DataProcessor
from instanovo.constants import MASS_SCALE, MAX_MASS
from instanovo.inference import (
    BeamSearchDecoder,
    Decoder,
    GreedyDecoder,
    Knapsack,
    KnapsackBeamSearchDecoder,
)
from instanovo.transformer.data import TransformerDataProcessor
from instanovo.transformer.model import InstaNovo
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs"


class TransformerPredictor(AccelerateDeNovoPredictor):
    """Predictor for the InstaNovo model."""

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.num_beams = config.get("num_beams", 1)
        # Change this to a generic save all outputs function
        self.save_beams = config.get("save_beams", False)
        super().__init__(config)

    def load_model(self) -> Tuple[nn.Module, DictConfig]:
        """Setup the model."""
        default_model = InstaNovo.get_pretrained()[0]
        model_path = self.config.get("instanovo_model", default_model)

        logger.info(f"Loading InstaNovo model {model_path}")
        if model_path in InstaNovo.get_pretrained():
            # Using a pretrained model from models.json
            model, model_config = InstaNovo.from_pretrained(
                model_path, override_config={"peak_embedding_dtype": "float32"} if self.config.get("mps", False) else None
            )
        else:
            model_path = self.s3.get_local_path(model_path)
            assert model_path is not None
            model, model_config = InstaNovo.load(
                model_path, override_config={"peak_embedding_dtype": "float32"} if self.config.get("mps", False) else None
            )

        return model, model_config

    def setup_data_processor(self) -> DataProcessor:
        """Setup the data processor."""
        processor = TransformerDataProcessor(
            self.residue_set,
            n_peaks=self.model_config.get("n_peaks", 200),
            min_mz=self.model_config.get("min_mz", 50.0),
            max_mz=self.model_config.get("max_mz", 2500.0),
            min_intensity=self.model_config.get("min_intensity", 0.01),
            remove_precursor_tol=self.model_config.get("remove_precursor_tol", 2.0),
            return_str=False,
            use_spectrum_utils=False,
            annotated=not self.denovo,
            metadata_columns=["group"],
        )

        return processor

    def setup_decoder(self) -> Decoder:
        """Setup the decoder."""
        float_dtype = torch.float32 if self.config.get("force_fp32", False) else torch.float64
        if self.config.get("use_knapsack", False):
            logger.info(f"Using Knapsack Beam Search with {self.num_beams} beam(s)")
            knapsack_path = self.config.get("knapsack_path", None)
            if knapsack_path is None or not os.path.exists(knapsack_path):
                logger.info("Knapsack path missing or not specified, generating...")
                knapsack = _setup_knapsack(self.model, self.config.get("max_isotope_error", 1))
                decoder: Decoder = KnapsackBeamSearchDecoder(self.model, knapsack, float_dtype=float_dtype)
                if knapsack_path is not None:
                    logger.info(f"Saving knapsack to {knapsack_path}")
                    knapsack.save(knapsack_path)
            else:
                logger.info("Knapsack path found. Loading...")
                decoder = KnapsackBeamSearchDecoder.from_file(self.model, knapsack_path, float_dtype=float_dtype)
        elif self.num_beams > 1:
            logger.info(f"Using Beam Search with {self.num_beams} beam(s)")
            decoder = BeamSearchDecoder(self.model, float_dtype=float_dtype)
        else:
            logger.info(f"Using Greedy Search with {self.num_beams} beam(s)")
            decoder = GreedyDecoder(
                model=self.model,
                suppressed_residues=self.config.get("suppressed_residues", None),
                disable_terminal_residues_anywhere=self.config.get("disable_terminal_residues_anywhere", True),
                float_dtype=float_dtype,
            )
        return decoder

    def get_predictions(self, batch: Any) -> dict[str, Any]:
        """Get the predictions for a batch."""
        batch_size = batch["spectra"].size(0)

        batch_predictions: dict[str, Any] = self.decoder.decode(
            spectra=batch["spectra"],
            precursors=batch["precursors"],
            beam_size=self.num_beams,
            max_length=self.config.get("max_length", 40),
            return_beam=self.num_beams > 1,
            return_encoder_output=self.save_encoder_outputs,
            encoder_output_reduction=self.encoder_output_reduction,
        )

        if "peptides" in batch:
            targets = [self.residue_set.decode(seq, reverse=True) for seq in batch["peptides"]]
        else:
            targets = [None] * batch_size

        batch_predictions["targets"] = targets

        return batch_predictions


def _setup_knapsack(model: InstaNovo, max_isotope: int = 2) -> Knapsack:
    residue_masses = dict(model.residue_set.residue_masses.copy())
    for special_residue in list(model.residue_set.residue_to_index.keys())[:3]:
        residue_masses[special_residue] = 0
    residue_indices = model.residue_set.residue_to_index
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=MAX_MASS,
        mass_scale=MASS_SCALE,
        max_isotope=max_isotope,
    )
