from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any
from typing import Tuple

import hydra
import neptune
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
import yaml
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import train_test_split
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import instanovo.utils.s3 as s3
from instanovo.inference.beam_search import BeamSearchDecoder
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import load_ipc_shards
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo
from instanovo.types import Peptide
from instanovo.types import PeptideMask
from instanovo.types import PrecursorFeatures
from instanovo.types import ResidueLogits
from instanovo.types import Spectrum
from instanovo.types import SpectrumMask
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PTModule(ptl.LightningModule):
    """PTL wrapper for model."""

    def __init__(
        self,
        config: dict[str, Any],
        model: InstaNovo,
        decoder: BeamSearchDecoder,
        metrics: Metrics,
        sw: SummaryWriter,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        # device: str = 'cpu',
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

        # Update rates based on bs=32
        self.step_scale = 32 / config["train_batch_size"]

    def forward(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        peptides: list[str] | Integer[Peptide, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"],
        peptides_mask: Bool[PeptideMask, " batch"],
    ) -> Tuple[Float[ResidueLogits, " batch"], Integer[Peptide, " batch"]]:
        """Model forward pass."""
        return self.model(spectra, precursors, peptides, spectra_mask, peptides_mask)  # type: ignore

    def training_step(  # need to update this
        self,
        batch: tuple[
            Float[Spectrum, " batch"],
            Float[PrecursorFeatures, " batch"],
            Bool[SpectrumMask, " batch"],
            list[str] | Integer[Peptide, " batch"],
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
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        peptides_mask = peptides_mask.to(self.device)

        peptides = peptides.to(self.device)

        preds = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        # Cut off EOS's prediction, ignore_index should take care of masking
        # EOS at positions < sequence_length will have a label of ignore_index
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        loss = self.loss_fn(preds, peptides.flatten())

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        if (
            (self.steps + 1) % int(self.config.get("console_logging_steps", 2000) * self.step_scale)
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            logger.info(
                f"[Step {self.steps+1:06d}]: train_loss_raw={loss.item():.4f}, running_loss={self.running_loss:.4f}, LR={lr}"
            )

        if (self.steps + 1) % int(
            self.config.get("tensorboard_logging_steps", 500) * self.step_scale
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.sw.add_scalar("train/loss_raw", loss.item(), self.steps + 1)
            self.sw.add_scalar("train/loss_smooth", self.running_loss, self.steps + 1)
            self.sw.add_scalar("optim/lr", lr, self.steps + 1)
            self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps + 1)

        self.steps += 1

        return loss

    def validation_step(
        self,
        batch: Tuple[
            Float[Spectrum, " batch"],
            Float[PrecursorFeatures, " batch"],
            Bool[SpectrumMask, " batch"],
            list[str] | Integer[Peptide, " batch"],
            Bool[PeptideMask, " batch"],
        ],
        *args: Any,
    ) -> float:
        """Single validation step."""
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        # peptides = peptides.to(self.device)
        # peptides_mask = peptides_mask.to(self.device)

        # Loss
        peptides = peptides.to(self.device)

        with torch.no_grad():
            preds = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
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

        y = [x.sequence if type(x) != list else "" for x in p]
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

        return float(loss.item())

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        epoch = self.trainer.current_epoch
        self.sw.add_scalar("eval/train_loss", self.running_loss, epoch)

        self.running_loss = None

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
            f"[Epoch {epoch:02d}] train_loss={self.running_loss if self.running_loss else 0:.5f}, valid_loss={valid_loss:.5f}"
        )
        logger.info(f"[Epoch {epoch:02d}] Metrics:")
        for metric in ["aa_er", "aa_prec", "aa_recall", "pep_recall"]:
            val = np.mean(self.valid_metrics[metric])
            logger.info(f"[Epoch {epoch:02d}] - {metric:11s}{val:.3f}")

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
    train_path = config.get("train_path")
    valid_path = config.get("valid_path")
    if config.get("use_shards", False):
        train_df = load_ipc_shards(train_path, split="train")
        if valid_path is None:
            # Split training set if no valid_path is specified, used for nine-species training
            logger.info("Validation path not specified, generating from training set")
            train_unique = train_df["modified_sequence"].unique().sort()
            train_seq, valid_seq = train_test_split(
                train_unique, test_size=config.get("valid_subset_of_train"), random_state=42
            )
            valid_df = train_df.filter(train_df["modified_sequence"].is_in(valid_seq))
            train_df = train_df.filter(train_df["modified_sequence"].is_in(train_seq))
            # Save splits
            # TODO: Optionally load the data splits
            # TODO: Allow loading of data splits in `predict.py`
            # TODO: Upload to Aichor
            split_path = Path(config.get("model_save_folder_path", "./checkpoints")).joinpath(
                "splits.csv"
            )
            pd.DataFrame(
                {
                    "modified_sequence": list(train_seq) + list(valid_seq),
                    "split": ["train"] * train_seq.shape[0] + ["valid"] * valid_seq.shape[0],
                }
            ).to_csv(str(split_path), index=False)
            logger.info(f"Data splits saved to {split_path}")
        else:
            valid_df = load_ipc_shards(valid_path, split="valid")
        train_df = train_df.sample(fraction=config["train_subset"], seed=42)
        valid_df = valid_df.sample(fraction=config["valid_subset"], seed=42)

        # Check residues
        logger.info(
            f"Checking for unknown residues in {train_df.shape[0] + valid_df.shape[0]:,d} rows."
        )
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        train_df = train_df.with_columns(
            pl.col("modified_sequence")
            .map_elements(
                lambda x: all([y in supported_residues for y in residue_set.tokenize(x)]),
                return_dtype=pl.Boolean,
            )
            .alias("supported")
        )
        valid_df = valid_df.with_columns(
            pl.col("modified_sequence")
            .map_elements(
                lambda x: all([y in supported_residues for y in residue_set.tokenize(x)]),
                return_dtype=pl.Boolean,
            )
            .alias("supported")
        )

        if (~train_df["supported"]).sum() > 0 or (~valid_df["supported"]).sum() > 0:
            logger.warning("Unsupported residues found! These rows will be dropped.")
            df_residues = set()
            for x in train_df["modified_sequence"]:
                df_residues.update(set(residue_set.tokenize(x)))
            for x in valid_df["modified_sequence"]:
                df_residues.update(set(residue_set.tokenize(x)))
            logger.info(f"New residues found: \n{df_residues-supported_residues}")
            logger.info(f"Residues supported: \n{supported_residues}")
            original_size = (train_df.shape[0], valid_df.shape[0])
            train_df = train_df.filter(pl.col("supported"))
            valid_df = valid_df.filter(pl.col("supported"))
            new_size = (train_df.shape[0], valid_df.shape[0])
            logger.warning(
                f"{original_size[0]-new_size[0]:,d} ({(original_size[0]-new_size[0])/original_size[0]*100:.2f}%) training rows dropped."
            )
            logger.warning(
                f"{original_size[1]-new_size[1]:,d} ({(original_size[1]-new_size[1])/original_size[1]*100:.2f}%) validation rows dropped."
            )

    elif train_path.endswith(".ipc"):
        train_df = pl.read_ipc(train_path)
        train_df = train_df.sample(fraction=config["train_subset"], seed=42)
        valid_df = pl.read_ipc(valid_path)
        valid_df = valid_df.sample(fraction=config["valid_subset"], seed=42)
    elif train_path.endswith(".csv"):
        train_df = pd.read_csv(train_path)
        train_df = train_df.sample(frac=config["train_subset"], random_state=42)
        valid_df = pd.read_csv(valid_path)
        valid_df = valid_df.sample(frac=config["valid_subset"], random_state=42)

    train_ds = SpectrumDataset(train_df, residue_set, config["n_peaks"], return_str=False)
    valid_ds = SpectrumDataset(valid_df, residue_set, config["n_peaks"], return_str=False)

    logger.info(
        f"Data loaded: {len(train_ds):,} training samples; {len(valid_ds):,} validation samples"
    )
    logger.info(f"Train columns: {train_df.columns}")
    logger.info(f"Valid columns: {valid_df.columns}")

    logger.info("Checking if any validation set overlaps with training set...")
    leakage = any(valid_df["modified_sequence"].is_in(train_df["modified_sequence"]))
    if leakage:
        raise ValueError("Portion of validation set sequences overlaps with training set.")
    else:
        logger.info("No data leakage!")

    train_dl = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        num_workers=config["n_workers"],
        shuffle=True,
        collate_fn=collate_batch,
        multiprocessing_context="fork",
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=config["predict_batch_size"],
        num_workers=config["n_workers"],
        shuffle=False,
        collate_fn=collate_batch,
        multiprocessing_context="fork",
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
            logger.warning("Assuming a change was made to the residues in the configuration file.")
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
                    tmp[:aa_embed_size] = model_state[k][: min(tmp.shape[0], aa_embed_size)]
                    model_state[k] = tmp
                else:
                    raise ValueError(f"Unknown residue_conflict_resolution type '{resolution}'")

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

    logger.info("Test forward pass:")
    with torch.no_grad():
        y = model(spectra, precursors, peptides, spectra_mask, peptides_mask)
        logger.info(f" - y.shape={y.shape}")

    # Train on GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # decoder = GreedyDecoder(model, i2s, max_length=config["max_length"])
    decoder = BeamSearchDecoder(model=model)
    metrics = Metrics(residue_set, config["isotope_error_range"])

    # init optim
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = WarmupScheduler(optim, config["warmup_iters"])
    strategy = _get_strategy()

    ptmodel = PTModule(config, model, decoder, metrics, sw, optim, scheduler)

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

        logger.info(f"Saving every {config['ckpt_interval']} training steps to {ckpt_path}")
    else:
        logger.info("Model saving disabled")
        callbacks = None

    logger.info("Initializing PL trainer.")
    trainer = ptl.Trainer(
        accelerator="auto",
        callbacks=callbacks,
        devices="auto",
        logger=config["logger"],
        max_epochs=config["epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        accumulate_grad_batches=config["grad_accumulation"],
        gradient_clip_val=config["gradient_clip_val"],
        strategy=strategy,
    )

    # Train the model.
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


class NeptuneSummaryWriter(SummaryWriter):
    """Combine SummaryWriter with NeptuneWriter."""

    def __init__(self, log_dir: str, run: neptune.Run) -> None:
        super().__init__(log_dir=log_dir)
        self.run = run

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None) -> None:
        """Record scalar to tensorboard and Neptune."""
        super().add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
        )
        self.run[tag].append(scalar_value, step=global_step)


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


@hydra.main(config_path="../../configs", version_base=None, config_name="instanovo")
def main(config: DictConfig) -> None:
    """Train the model."""
    logger.info("Initializing training.")

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
