from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console
from instanovo.common import AccelerateDeNovoTrainer, DataProcessor
from instanovo.inference import Decoder, GreedyDecoder
from instanovo.transformer.data import TransformerDataProcessor
from instanovo.transformer.model import InstaNovo
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs"


class TransformerTrainer(AccelerateDeNovoTrainer):
    """Trainer for the InstaNovo model."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def setup_model(self) -> nn.Module:
        """Setup the model."""
        config = self.config.get("model", {})
        model = InstaNovo(
            residue_set=self.residue_set,
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            encoder_layers=config.get("encoder_layers", config.get("n_layers", 9)),
            decoder_layers=config.get("decoder_layers", config.get("n_layers", 9)),
            dropout=config["dropout"],
            max_charge=config["max_charge"],
            use_flash_attention=config.get("use_flash_attention", False),
            conv_peak_encoder=config.get("conv_peak_encoder", False),
            peak_embedding_dtype="float32" if self.config.get("mps", False) else "float64",
        )
        return model

    def update_vocab(self, model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update the vocabulary of the model."""
        return self._update_vocab(  # type: ignore
            model_state,
            target_layers=["head.weight", "head.bias", "aa_embed.weight"],
            resolution=self.config.get("residue_conflict_resolution", "delete"),
        )

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def setup_decoder(self) -> Decoder:
        """Setup the decoder."""
        return GreedyDecoder(model=self.model, float_dtype=torch.float32 if self.config.get("mps", False) else torch.float64)

    def setup_data_processors(self) -> tuple[DataProcessor, DataProcessor]:
        """Setup the datasets."""
        train_processor = TransformerDataProcessor(
            self.residue_set,
            n_peaks=self.config.model.get("n_peaks", 200),
            min_mz=self.config.model.get("min_mz", 50.0),
            max_mz=self.config.model.get("max_mz", 2500.0),
            min_intensity=self.config.model.get("min_intensity", 0.01),
            remove_precursor_tol=self.config.model.get("remove_precursor_tol", 2.0),
            return_str=False,
            use_spectrum_utils=False,
        )

        valid_processor = TransformerDataProcessor(
            self.residue_set,
            n_peaks=self.config.model.get("n_peaks", 200),
            min_mz=self.config.model.get("min_mz", 50.0),
            max_mz=self.config.model.get("max_mz", 2500.0),
            min_intensity=self.config.model.get("min_intensity", 0.01),
            remove_precursor_tol=self.config.model.get("remove_precursor_tol", 2.0),
            return_str=False,
            use_spectrum_utils=False,
        )

        return train_processor, valid_processor

    def add_checkpoint_state(self) -> dict[str, Any]:
        """Add checkpoint state."""
        return {}

    def save_model(self, is_best_checkpoint: bool = False) -> None:
        """Save the model."""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = self.config.get("model_save_folder_path", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        if self.config.get("keep_model_every_interval", False):
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{self.epoch:02d}_step_{self.global_step + 1}.ckpt")
        else:
            model_path = os.path.join(checkpoint_dir, "model_latest.ckpt")
            if Path(model_path).exists() and Path(model_path).is_file():
                Path(model_path).unlink()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint_state = {
            "state_dict": unwrapped_model.state_dict(),
            "config": OmegaConf.to_container(self.config.model),
            "residues": self.residue_set.residue_masses,
            "epoch": self.epoch,
            "global_step": self.global_step + 1,
        }
        checkpoint_state.update(self.add_checkpoint_state())

        torch.save(checkpoint_state, model_path)
        logger.info(f"Saved model to {model_path}")

        if S3FileHandler._aichor_enabled():
            self.s3.upload(model_path, S3FileHandler.convert_to_s3_output(model_path))

        if is_best_checkpoint and self.accelerator.is_main_process:
            best_model_path = os.path.join(checkpoint_dir, "model_best.ckpt")
            if Path(best_model_path).exists() and Path(best_model_path).is_file():
                Path(best_model_path).unlink()

            shutil.copy(model_path, best_model_path)

            if S3FileHandler._aichor_enabled():
                self.s3.upload(best_model_path, S3FileHandler.convert_to_s3_output(best_model_path))

        logger.info(f"Saved checkpoint to {model_path}")

    def forward(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for the model to calculate loss."""
        preds = self.model(
            x=batch["spectra"],
            p=batch["precursors"],
            y=batch["peptides"],
            x_mask=batch["spectra_mask"],
            y_mask=batch["peptides_mask"],
        )

        preds = preds[:, :-1].reshape(-1, preds.shape[-1])

        loss = self.loss_fn(preds, batch["peptides"].flatten())

        return loss, {"loss": loss}

    def get_predictions(self, batch: Any) -> tuple[list[str] | list[list[str]], list[str] | list[list[str]]]:
        """Get the predictions for a batch."""
        # Greedy decoding
        batch_predictions = self.decoder.decode(
            spectra=batch["spectra"],
            precursors=batch["precursors"],
            beam_size=self.config.get("n_beams", 1),
            max_length=self.config.get("max_length", 40),
        )

        targets = [self.residue_set.decode(i, reverse=True) for i in batch["peptides"]]

        return batch_predictions["predictions"], targets


@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="instanovo")
def main(config: DictConfig) -> None:
    """Train the model."""
    logger.info("Initializing training.")
    trainer = TransformerTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
