from __future__ import annotations

import datetime
import logging
import os
import sys
import time
import warnings
from typing import Any
from typing import cast
from typing import Tuple

import hydra
import neptune
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from pathlib import Path

import instanovo.utils.s3 as s3

from instanovo.inference import GreedyDecoder
from instanovo.inference import Decoder
from instanovo.inference import ScoredSequence
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import remove_modifications
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo
from instanovo.types import Peptide
from instanovo.types import PeptideMask
from instanovo.types import PrecursorFeatures
from instanovo.types import ResidueLogits
from instanovo.types import Spectrum
from instanovo.types import SpectrumMask
from instanovo.utils import Metrics
from instanovo.utils import ResidueSet
from instanovo.utils import SpectrumDataFrame
from instanovo.constants import ANNOTATION_ERROR, ANNOTATED_COLUMN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_PATH = Path(__file__).parent.parent / "configs"

warnings.filterwarnings("ignore", message=".*does not have many workers*")


class PTModule(ptl.LightningModule):
    """PTL wrapper for model."""

    def __init__(
        self,
        config: DictConfig | dict[str, Any],
        model: InstaNovo,
        decoder: Decoder,
        metrics: Metrics,
        sw: SummaryWriter,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        compile: bool = True,
        fp16: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.decoder = decoder
        self.metrics = metrics
        self.sw = sw
        self.optim = optim
        self.scheduler = scheduler

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        self.running_loss = None
        self._reset_valid_metrics()
        self.steps = 0
        self.train_epoch_start_time: float | None = None
        self.train_start_time: float | None = None
        self.valid_epoch_start_time: float | None = None
        self.valid_epoch_step = 0

        # Update rates based on bs=32
        self.step_scale = 32 / config["train_batch_size"]

        @torch.compile(dynamic=False, mode="reduce-overhead", disable=~compile)
        @torch.autocast("cuda", dtype=torch.float16, enabled=fp16)
        def compiled_forward(
            spectra: Tensor,
            precursors: Tensor,
            peptides: Tensor,
            spectra_mask: Tensor,
            peptides_mask: Tensor,
        ) -> Tensor:
            """Compiled forward pass."""
            return self.forward(
                spectra, precursors, peptides, spectra_mask, peptides_mask
            )

        self.compiled_forward = compiled_forward

    def forward(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        peptides: list[str] | Integer[Peptide, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"],
        peptides_mask: Bool[PeptideMask, " batch"],
    ) -> Float[ResidueLogits, " batch token+1"]:
        """Model forward pass."""
        return self.model(spectra, precursors, peptides, spectra_mask, peptides_mask)  # type: ignore

    def training_step(  # need to update this
        self,
        batch: tuple[
            Float[Spectrum, " batch"],
            Float[PrecursorFeatures, " batch"],
            Bool[SpectrumMask, " batch"],
            Integer[Peptide, " batch"],
            Bool[PeptideMask, " batch"],
        ],
    ) -> Float[Tensor, " batch"]:
        """A single training step.

        Args:
            batch (tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]) :
                A batch of MS/MS spectra, precursor information, and peptide
                sequences as torch Tensors.

        Returns:
            torch.FloatTensor: training loss
        """
        if self.train_epoch_start_time is None:
            self.train_epoch_start_time = time.time()
            self.valid_epoch_start_time = None
        if self.train_start_time is None:
            self.train_start_time = time.time()

        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        peptides_mask = peptides_mask.to(self.device)

        peptides = peptides.to(self.device)

        # preds = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        preds = self.compiled_forward(
            spectra, precursors, peptides, spectra_mask, peptides_mask
        )

        # Cut off EOS's prediction, ignore_index should take care of masking
        # EOS at positions < sequence_length will have a label of ignore_index
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        loss = self.loss_fn(preds, peptides.flatten())

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        if (
            (self.steps + 1)
            % int(self.config.get("console_logging_steps", 2000) * self.step_scale)
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            delta = time.time() - self.train_epoch_start_time
            epoch_step = self.steps % len(self.trainer.train_dataloader)
            est_total = (
                delta
                / (epoch_step + 1)
                * (len(self.trainer.train_dataloader) - epoch_step - 1)
            )

            logger.info(
                f"[TRAIN] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs-1:02d} Step {self.steps+1:06d}] "
                + f"[Batch {epoch_step + 1:05d}/{len(self.trainer.train_dataloader):05d}] [{_format_time(delta)}/{_format_time(est_total)}, {(delta / (epoch_step + 1)):.3f}s/it]: "
                + f"train_loss_raw={loss.item():.4f}, running_loss={self.running_loss:.4f}, LR={lr:.6f}"
            )

        if (self.steps + 1) % int(
            self.config.get("tensorboard_logging_steps", 500) * self.step_scale
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.sw.add_scalar("train/loss_raw", loss.item(), self.steps + 1)
            self.sw.add_scalar("train/loss_smooth", self.running_loss, self.steps + 1)
            self.sw.add_scalar("optim/lr", lr, self.steps + 1)
            self.sw.add_scalar(
                "optim/epoch", self.trainer.current_epoch, self.steps + 1
            )

        self.steps += 1

        return loss

    def validation_step(
        self,
        batch: Tuple[
            Float[Spectrum, " batch"],
            Float[PrecursorFeatures, " batch"],
            Bool[SpectrumMask, " batch"],
            Integer[Peptide, " batch"],
            Bool[PeptideMask, " batch"],
        ],
        *args: Any,
    ) -> float:
        """Single validation step."""
        if self.valid_epoch_start_time is None:
            logger.info(
                f"[VALIDATION] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs-1:02d}] Starting validation."
            )
            self.valid_epoch_start_time = time.time()

        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        # peptides = peptides.to(self.device)
        # peptides_mask = peptides_mask.to(self.device)

        # Loss
        peptides = peptides.to(self.device)

        with torch.no_grad():
            preds = self.forward(
                spectra, precursors, peptides, spectra_mask, peptides_mask
            )
        # Cut off EOS's prediction, ignore_index should take care of masking
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        loss = self.loss_fn(preds, peptides.flatten())

        # Greedy decoding
        with torch.no_grad():
            p = self.decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=self.config["n_beams"],
                max_length=self.config["max_length"],
            )
            p = cast(list[ScoredSequence], p)

        y = [x.sequence if isinstance(x, ScoredSequence) else [] for x in p]
        targets = [s for s in self.model.batch_idx_to_aa(peptides, reverse=True)]

        aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(
            targets, y
        )
        aa_er = self.metrics.compute_aa_er(targets, y)

        self.valid_metrics["valid_loss"].append(loss.item())
        self.valid_metrics["aa_er"].append(aa_er)
        self.valid_metrics["aa_prec"].append(aa_prec)
        self.valid_metrics["aa_recall"].append(aa_recall)
        self.valid_metrics["pep_recall"].append(pep_recall)

        if (
            (self.valid_epoch_step + 1)
            % int(self.config.get("console_logging_steps", 2000) * self.step_scale)
        ) == 0:
            delta = time.time() - self.valid_epoch_start_time
            epoch_step = self.valid_epoch_step % len(self.trainer.val_dataloaders)
            est_total = (
                delta
                / (epoch_step + 1)
                * (len(self.trainer.val_dataloaders) - epoch_step - 1)
            )

            logger.info(
                f"[VALIDATION] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs-1:02d} Step {self.steps+1:06d}] "
                + f"[Batch {epoch_step:05d}/{len(self.trainer.val_dataloaders):05d}] [{_format_time(delta)}/{_format_time(est_total)}, {(delta / (epoch_step + 1)):.3f}s/it]"
            )

        self.valid_epoch_step += 1

        return float(loss.item())

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        epoch = self.trainer.current_epoch
        self.sw.add_scalar("eval/train_loss", self.running_loss, epoch)

        delta = time.time() - cast(float, self.train_start_time)
        epoch = self.trainer.current_epoch
        est_total = delta / (epoch + 1) * (self.trainer.max_epochs - epoch - 1)
        logger.info(
            f"[TRAIN] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs-1:02d}] Epoch complete, total time {_format_time(delta)}, remaining time {_format_time(est_total)}, {_format_time(delta / (epoch + 1))} per epoch"
        )

        self.running_loss = None
        self.train_epoch_start_time = None
        self.train_epoch_step = 0

    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics at the end of each epoch."""
        epoch = self.trainer.current_epoch
        if self.steps == 0:
            # Don't record sanity check validation
            self._reset_valid_metrics()
            return
        for k, v in self.valid_metrics.items():
            self.sw.add_scalar(f"eval/{k}", np.mean(v), epoch)

        valid_loss = np.mean(self.valid_metrics["valid_loss"])
        logger.info(
            f"[VALIDATION] [Epoch {epoch:02d}/{self.trainer.max_epochs-1:02d}] train_loss={self.running_loss if self.running_loss else 0:.5f}, valid_loss={valid_loss:.5f}"
        )
        logger.info(
            f"[VALIDATION] [Epoch {epoch:02d}/{self.trainer.max_epochs-1:02d}] Metrics:"
        )
        for metric in ["aa_er", "aa_prec", "aa_recall", "pep_recall"]:
            val = np.mean(self.valid_metrics[metric])
            logger.info(
                f"[VALIDATION] [Epoch {epoch:02d}/{self.trainer.max_epochs-1:02d}] - {metric:11s}{val:.3f}"
            )

        self.valid_epoch_start_time = None
        self.valid_epoch_step = 0
        self._reset_valid_metrics()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save config with checkpoint."""
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attempt to load config with checkpoint."""
        self.config = checkpoint["config"]

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        return [self.optim], {"scheduler": self.scheduler, "interval": "step"}

    def _reset_valid_metrics(self) -> None:
        valid_metrics = ["valid_loss", "aa_er", "aa_prec", "aa_recall", "pep_recall"]
        self.valid_metrics: dict[str, list[float]] = {x: [] for x in valid_metrics}


# flake8: noqa: CR001
def train(
    config: DictConfig,
) -> None:
    """Training function."""
    torch.manual_seed(config.get("seed", 101))
    torch.set_float32_matmul_precision("high")
    # os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

    time_now = datetime.datetime.now().strftime("_%y_%m_%d_%H_%M")
    if s3.register_tb():
        config["tb_summarywriter"] = os.environ["AICHOR_LOGS_PATH"]
    else:
        config["tb_summarywriter"] = config["tb_summarywriter"] + time_now

    if config.get("report_to", "") == "neptune":
        os.environ["NEPTUNE_PROJECT"] = "InstaDeep/denovo-sequencing"
        run = neptune.init_run(
            with_id=None,
            description=config.get("run_name", "instanovo_acpt_base") + time_now,
        )
        run.assign({"config": OmegaConf.to_yaml(config)})
        sw = NeptuneSummaryWriter(config["tb_summarywriter"], run)
    else:
        sw = SummaryWriter(config["tb_summarywriter"])

    # Transformer vocabulary
    residue_set = ResidueSet(
        residue_masses=config["residues"],
        residue_remapping=config["residue_remapping"],
    )
    logger.info(f"Vocab: {residue_set.index_to_residue}")

    logger.info("Loading data")

    try:
        train_sdf = SpectrumDataFrame.load(
            config.get("train_path"),
            lazy=config.get("lazy_loading", True),
            is_annotated=True,
            shuffle=True,
            partition=config.get("train_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
        )
        valid_sdf = SpectrumDataFrame.load(
            config.get("valid_path", None) or config.get("train_path"),
            lazy=config.get("lazy_loading", True),
            is_annotated=True,
            shuffle=False,
            partition=config.get("valid_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
        )
    except ValueError as e:
        # More descriptive error message in predict mode.
        if str(e) == ANNOTATION_ERROR:
            raise ValueError(
                "The sequence column is missing annotations, are you trying to run de novo prediction? Add the --denovo flag"
            )
        else:
            raise

    if config.get("valid_path", None) is None:
        logger.info("Validation path not specified, generating from training set.")
        sequences = list(train_sdf.get_unique_sequences())
        sequences = sorted(list(set([remove_modifications(x) for x in sequences])))
        train_unique, valid_unique = train_test_split(
            sequences,
            test_size=config.get("valid_subset_of_train"),
            random_state=42,
        )
        train_unique = set(train_unique)
        valid_unique = set(valid_unique)

        train_sdf.filter_rows(
            lambda row: remove_modifications(row["sequence"]) in train_unique
        )
        valid_sdf.filter_rows(
            lambda row: remove_modifications(row["sequence"]) in valid_unique
        )
        # Save splits
        # TODO: Optionally load the data splits
        # TODO: Allow loading of data splits in `predict.py`
        # TODO: Upload to Aichor
        split_path = os.path.join(
            config.get("model_save_folder_path", "./checkpoints"), "splits.csv"
        )
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        pd.DataFrame(
            {
                "modified_sequence": list(train_unique) + list(valid_unique),
                "split": ["train"] * len(train_unique) + ["valid"] * len(valid_unique),
            }
        ).to_csv(str(split_path), index=False)
        logger.info(f"Data splits saved to {split_path}")

    # Check residues
    if config.get("perform_data_checks", True):
        logger.info(
            f"Checking for unknown residues in {len(train_sdf) + len(valid_sdf):,d} rows."
        )
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        data_residues = set()
        data_residues.update(train_sdf.get_vocabulary(residue_set.tokenize))
        data_residues.update(valid_sdf.get_vocabulary(residue_set.tokenize))
        if len(data_residues - supported_residues) > 0:
            logger.warning(
                "Unsupported residues found in evaluation set! These rows will be dropped."
            )
            logger.info(f"New residues found: \n{data_residues-supported_residues}")
            logger.info(f"Residues supported: \n{supported_residues}")
            original_size = (len(train_sdf), len(valid_sdf))
            train_sdf.filter_rows(
                lambda row: all(
                    [
                        residue in supported_residues
                        for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                    ]
                )
            )
            valid_sdf.filter_rows(
                lambda row: all(
                    [
                        residue in supported_residues
                        for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                    ]
                )
            )
            new_size = (len(train_sdf), len(valid_sdf))
            logger.warning(
                f"{original_size[0]-new_size[0]:,d} ({(original_size[0]-new_size[0])/original_size[0]*100:.2f}%) training rows dropped."
            )
            logger.warning(
                f"{original_size[1]-new_size[1]:,d} ({(original_size[1]-new_size[1])/original_size[1]*100:.2f}%) validation rows dropped."
            )

        # Check charge values:
        original_size = (len(train_sdf), len(valid_sdf))
        train_sdf.filter_rows(
            lambda row: (row["precursor_charge"] <= config.get("max_charge", 10))
            and (row["precursor_charge"] > 0)
        )
        if len(train_sdf) < original_size[0]:
            logger.warning(
                f"Found {original_size[0] - len(train_sdf)} rows in training set with charge > {config.get('max_charge', 10)} or <= 0. These rows will be skipped."
            )

        valid_sdf.filter_rows(
            lambda row: (row["precursor_charge"] <= config.get("max_charge", 10))
            and (row["precursor_charge"] > 0)
        )
        if len(valid_sdf) < original_size[1]:
            logger.warning(
                f"Found {original_size[1] - len(valid_sdf)} rows in training set with charge > {config.get('max_charge', 10)}. These rows will be skipped."
            )

    train_sdf.sample_subset(fraction=config.get("train_subset", 1.0), seed=42)
    valid_sdf.sample_subset(fraction=config.get("valid_subset", 1.0), seed=42)

    train_ds = SpectrumDataset(
        train_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        peptide_pad_length=config.get("max_length", 40)
        if config.get("compile_model", False)
        else 0,
        pad_spectrum_max_length=config.get("compile_model", False)
        or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
    )
    valid_ds = SpectrumDataset(
        valid_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        pad_spectrum_max_length=config.get("compile_model", False)
        or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
    )

    logger.info(
        f"Data loaded: {len(train_ds):,} training samples; {len(valid_ds):,} validation samples"
    )
    # logger.info(f"Train columns: {train_df.columns}")
    # logger.info(f"Valid columns: {valid_df.columns}")

    train_sequences = pl.Series(list(train_sdf.get_unique_sequences()))
    valid_sequences = pl.Series(list(valid_sdf.get_unique_sequences()))
    if config.get("blacklist", None):
        logger.info(
            "Checking if any training set overlaps with blacklisted sequences..."
        )
        blacklist_df = pd.read_csv(config["blacklist"])
        leakage = any(
            train_sequences.map_elements(
                remove_modifications, return_dtype=pl.String
            ).is_in(blacklist_df["sequence"])
        )
        if leakage:
            raise ValueError(
                "Portion of training set sequences overlaps with blacklisted sequences."
            )
        else:
            logger.info("No blacklisted sequences!")

    if config.get("perform_data_checks", True):
        logger.info("Checking if any validation set overlaps with training set...")
        leakage = any(valid_sequences.is_in(train_sequences))
        if leakage:
            raise ValueError(
                "Portion of validation set sequences overlaps with training set."
            )
        else:
            logger.info("No data leakage!")

    # Check how many times model will save
    if config.get("save_model", True):
        total_epochs = config.get("epochs", 30)
        epochs_per_save = 1 / (
            len(train_ds)
            / config.get("train_batch_size", 256)
            / config.get("ckpt_interval")
        )
        if epochs_per_save > total_epochs:
            logger.warning(
                f"Model checkpoint will never save. Attempting to save every {epochs_per_save:.2f} epochs but only training for {total_epochs:d} epochs. Check ckpt_interval in config."
            )
        else:
            logger.info(f"Model checkpointing every {epochs_per_save:.2f} epochs.")

    # Check warmup
    if config.get("warmup_iters", 100_000) > len(train_ds) / config.get(
        "train_batch_size", 256
    ):
        logger.warning(
            "Model warmup is greater than one epoch of the training set. Check warmup_iters in config"
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        num_workers=0,  # SDF requirement is 0
        shuffle=False,  # SDF requirement
        collate_fn=collate_batch,
        # multiprocessing_context="fork",
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=config["predict_batch_size"],
        num_workers=0,  # SDF requirement is 0
        shuffle=False,
        collate_fn=collate_batch,
        # multiprocessing_context="fork",
    )

    # Update rates based on bs=32
    step_scale = 32 / config["train_batch_size"]
    logger.info(f"Updates per epoch: {len(train_dl):,}, step_scale={step_scale}")

    batch = next(iter(train_dl))
    spectra, precursors, spectra_mask, peptides, peptides_mask = batch
    logger.info("Sample batch:")
    logger.info(f" - spectra.shape={spectra.shape}")
    logger.info(f" - precursors.shape={precursors.shape}")
    logger.info(f" - spectra_mask.shape={spectra_mask.shape}")
    logger.info(f" - peptides.shape={peptides.shape}")
    logger.info(f" - peptides_mask.shape={peptides_mask.shape}")

    # init model
    model = InstaNovo(
        residue_set=residue_set,
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_charge=config["max_charge"],
        use_flash_attention=config["use_flash_attention"],
        conv_peak_encoder=config["conv_peak_encoder"],
    )

    if not config["train_from_scratch"]:
        model_path = config["resume_checkpoint"]
    else:
        model_path = None

    if model_path is not None:
        logger.info(f"Loading model checkpoint from '{model_path}'")
        model_state = torch.load(model_path, map_location="cpu")
        # check if PTL checkpoint
        if "state_dict" in model_state:
            model_state = {
                k.replace("model.", ""): v for k, v in model_state["state_dict"].items()
            }

        aa_embed_size = model_state["head.weight"].shape[0]
        if aa_embed_size != len(residue_set):
            state_keys = ["head.weight", "head.bias", "aa_embed.weight"]
            logger.warning(
                f"Model expects vocab size of {len(residue_set)}, checkpoint has {aa_embed_size}."
            )
            logger.warning(
                "Assuming a change was made to the residues in the configuration file."
            )
            logger.warning(f"Automatically converting {state_keys} to match expected.")

            new_model_state = model.state_dict()

            resolution = config.get("residue_conflict_resolution", "delete")
            for k in state_keys:
                # initialise weights to normal distribution with weight 1/sqrt(dim)
                tmp = torch.normal(
                    mean=0,
                    std=1.0 / np.sqrt(config["dim_model"]),
                    size=new_model_state[k].shape,
                    dtype=new_model_state[k].dtype,
                )
                if "bias" in k:
                    # initialise bias to zeros
                    tmp = torch.zeros_like(tmp)

                if resolution == "delete":
                    del model_state[k]
                elif resolution == "random":
                    model_state[k] = tmp
                elif resolution == "partial":
                    tmp[:aa_embed_size] = model_state[k][
                        : min(tmp.shape[0], aa_embed_size)
                    ]
                    model_state[k] = tmp
                else:
                    raise ValueError(
                        f"Unknown residue_conflict_resolution type '{resolution}'"
                    )

            logger.warning(
                f"Model checkpoint has {len(state_keys)} weights updated with '{resolution}' conflict resolution"
            )

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(model.state_dict().keys())]
        )
        if k_missing > 0:
            logger.warning(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            logger.warning(f"Model state is missing {k_missing} keys!")
        model.load_state_dict(model_state, strict=False)

    logger.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    if not config["conv_peak_encoder"]:
        logger.info("Test forward pass:")
        with torch.no_grad():
            y = model(spectra, precursors, peptides, spectra_mask, peptides_mask)
            logger.info(f" - y.shape={y.shape}")

    # Train on GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    decoder = GreedyDecoder(model=model)
    metrics = Metrics(residue_set, config["isotope_error_range"])

    # Use as an additional data sanity check
    if config.get("perform_data_checks", True):
        logger.info("Sanity checking precursor masses for training set...")
        train_sdf.validate_precursor_mass(metrics)
        logger.info("Sanity checking precursor masses for validation set...")
        valid_sdf.validate_precursor_mass(metrics)

    # init optim
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = WarmupScheduler(optim, config["warmup_iters"])
    strategy = _get_strategy()

    ptmodel = PTModule(
        config,
        model,
        decoder,
        metrics,
        sw,
        optim,
        scheduler,
        config["compile_model"],
        config["fp16"],
    )

    if config["save_model"]:
        logger.info("Model saving enabled")

        # returns input if s3 disabled
        ckpt_path = s3.convert_to_s3_output(config["model_save_folder_path"])

        if s3._s3_enabled():
            callbacks = [
                s3.PLCheckpointWrapper(
                    dirpath=config["model_save_folder_path"],
                    save_top_k=-1,
                    save_weights_only=config["save_weights_only"],
                    every_n_train_steps=config["ckpt_interval"],
                    s3_ckpt_path=ckpt_path,
                    strategy=strategy,
                )
            ]
        else:
            callbacks = [
                ptl.callbacks.ModelCheckpoint(
                    dirpath=config["model_save_folder_path"],
                    save_top_k=-1,
                    save_weights_only=config["save_weights_only"],
                    every_n_train_steps=config["ckpt_interval"],
                )
            ]

        logger.info(
            f"Saving every {config['ckpt_interval']} training steps to {ckpt_path}"
        )
    else:
        logger.info("Model saving disabled")
        callbacks = None

    logger.info("Initializing PL trainer.")
    trainer = ptl.Trainer(
        accelerator="auto",
        precision="16-mixed" if config["fp16"] else None,
        callbacks=callbacks,
        devices="auto",
        logger=config["logger"],
        max_epochs=config["epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        accumulate_grad_batches=config["grad_accumulation"],
        gradient_clip_val=config["gradient_clip_val"],
        enable_progress_bar=False,
        strategy=strategy,
        val_check_interval=config["val_check_interval"],
    )

    # Train the model.
    logger.info("Starting PL trainer.")
    trainer.fit(ptmodel, train_dl, valid_dl)

    logger.info("Training complete.")


def _get_strategy() -> DDPStrategy | str:
    """Get the strategy for the Trainer.

    The DDP strategy works best when multiple GPUs are used. It can work for
    CPU-only, but definitely fails using MPS (the Apple Silicon chip) due to
    Gloo.

    Returns
    -------
    Optional[DDPStrategy]
        The strategy parameter for the Trainer.
    """
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)

    return "auto"


def _set_author_neptune_api_token() -> None:
    """Set the variable NEPTUNE_API_TOKEN based on the email of commit author.

    It is useful on AIchor to have proper owner of each run.
    """
    try:
        author_email = os.environ["VCS_AUTHOR_EMAIL"]
    # we are not on AIchor
    except KeyError:
        return

    author_email, _ = author_email.split("@")
    author_email = author_email.replace("-", "_").replace(".", "_").upper()

    logger.info(
        f"Checking for Neptune API token under {author_email}__NEPTUNE_API_TOKEN."
    )
    try:
        author_api_token = os.environ[f"{author_email}__NEPTUNE_API_TOKEN"]
        os.environ["NEPTUNE_API_TOKEN"] = author_api_token
        logger.info(f"Set token for {author_email}.")
    except KeyError:
        logger.info(f"Neptune credentials for user {author_email} not found.")


class NeptuneSummaryWriter(SummaryWriter):
    """Combine SummaryWriter with NeptuneWriter."""

    def __init__(self, log_dir: str, run: neptune.Run) -> None:
        super().__init__(log_dir=log_dir)
        self.run = run

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int | None = None
    ) -> None:
        """Record scalar to tensorboard and Neptune."""
        super().add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
        )
        self.run[tag].append(scalar_value, step=global_step)


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int) -> None:
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="instanovo")
def main(config: DictConfig) -> None:
    """Train the model."""
    logger.info("Initializing training.")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    _set_author_neptune_api_token()

    # Unnest hydra configs
    # TODO Use the nested configs by default
    sub_configs_list = ["model", "dataset", "residues"]
    for sub_name in sub_configs_list:
        if sub_name in config:
            with open_dict(config):
                temp = config[sub_name]
                del config[sub_name]
                config.update(temp)

    logger.info(f"Imported hydra config:\n{OmegaConf.to_yaml(config)}")

    if config["n_gpu"] > 1:
        raise ValueError("n_gpu > 1 currently not supported.")

    train(config)


if __name__ == "__main__":
    main()
