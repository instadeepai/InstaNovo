from __future__ import annotations

import argparse
import datetime
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
import yaml
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from instanovo.inference.beam_search import BeamSearchDecoder
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo
from instanovo.utils.metrics import Metrics

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
        spectra: Tensor,
        precursors: Tensor,
        peptides: list[str] | Tensor,
        spectra_mask: Tensor,
        peptides_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Model forward pass."""
        return self.model(spectra, precursors, peptides, spectra_mask, peptides_mask)  # type: ignore

    def training_step(  # need to update this
        self,
        batch: tuple[Tensor, Tensor, Tensor, list[str] | Tensor, Tensor],
    ) -> torch.Tensor:
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
        # peptides = peptides.to(self.device)
        # peptides_mask = peptides_mask.to(self.device)

        preds, truth = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        # cut off EOS's prediction, ignore_index should take care of masking
        # preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        preds = preds[:, :-1, :].reshape(-1, self.model.decoder.vocab_size + 1)

        loss = self.loss_fn(preds, truth.flatten())

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        if ((self.steps + 1) % int(2000 * self.step_scale)) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            logging.info(
                f"[Step {self.steps-1:06d}]: train_loss_raw={loss.item():.4f}, running_loss={self.running_loss:.4f}, LR={lr}"
            )

        if (self.steps + 1) % int(500 * self.step_scale) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.sw.add_scalar("train/loss_raw", loss.item(), self.steps - 1)
            self.sw.add_scalar("train/loss_smooth", self.running_loss, self.steps - 1)
            self.sw.add_scalar("optim/lr", lr, self.steps - 1)
            self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps - 1)

        self.steps += 1

        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, list[str] | Tensor, Tensor], *args: Any
    ) -> torch.Tensor:
        """Single validation step."""
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        # peptides = peptides.to(self.device)
        # peptides_mask = peptides_mask.to(self.device)

        # Loss
        with torch.no_grad():
            preds, truth = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        preds = preds[:, :-1, :].reshape(-1, self.model.decoder.vocab_size + 1)
        loss = self.loss_fn(preds, truth.flatten())

        # Greedy decoding
        with torch.no_grad():
            # y, _ = decoder(spectra, precursors, spectra_mask)
            p = self.decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=self.config["n_beams"],
                max_length=self.config["max_length"],
            )

        # targets = self.model.batch_idx_to_aa(peptides)
        y = ["".join(x.sequence) if not isinstance(x, list) else "" for x in p]
        targets = peptides

        aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(targets, y)
        aa_er = self.metrics.compute_aa_er(targets, y)

        self.valid_metrics["valid_loss"].append(loss.item())
        self.valid_metrics["aa_er"].append(aa_er)
        self.valid_metrics["aa_prec"].append(aa_prec)
        self.valid_metrics["aa_recall"].append(aa_recall)
        self.valid_metrics["pep_recall"].append(pep_recall)

        return loss.item()

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        epoch = self.trainer.current_epoch
        self.sw.add_scalar(f"eval/train_loss", self.running_loss, epoch)

        self.running_loss = None

    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics at the end of each epoch."""
        epoch = self.trainer.current_epoch
        for k, v in self.valid_metrics.items():
            self.sw.add_scalar(f"eval/{k}", np.mean(v), epoch)

        valid_loss = np.mean(self.valid_metrics["valid_loss"])
        logging.info(
            f"[Epoch {epoch:02d}] train_loss={self.running_loss:.5f}, valid_loss={valid_loss:.5f}"
        )
        logging.info(f"[Epoch {epoch:02d}] Metrics:")
        for metric in ["aa_er", "aa_prec", "aa_recall", "pep_recall"]:
            val = np.mean(self.valid_metrics[metric])
            logging.info(f"[Epoch {epoch:02d}] - {metric:11s}{val:.3f}")

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
    train_path: str,
    valid_path: str,
    config: dict,
    model_path: str | None = None,
) -> None:
    """Training function."""
    config["tb_summarywriter"] = config["tb_summarywriter"] + datetime.datetime.now().strftime(
        "_%y_%m_%d_%H_%M"
    )

    sw = SummaryWriter(config["tb_summarywriter"])

    # Transformer vocabulary, should we include an UNK token?
    if config["dec_type"] != "depthcharge":
        vocab = ["PAD", "<s>", "</s>"] + list(config["residues"].keys())
    else:
        vocab = list(config["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    logging.info(f"Vocab: {i2s}")

    logging.info("Loading data")
    if train_path.endswith(".ipc"):
        train_df = pl.read_ipc(train_path)
        train_df = train_df.sample(fraction=config["train_subset"], seed=0)
        valid_df = pl.read_ipc(valid_path)
        valid_df = valid_df.sample(fraction=config["valid_subset"], seed=0)
    elif train_path.endswith(".csv"):
        train_df = pd.read_csv(train_path)
        train_df = train_df.sample(frac=config["train_subset"], random_state=0)
        valid_df = pd.read_csv(valid_path)
        valid_df = valid_df.sample(frac=config["valid_subset"], random_state=0)

    train_ds = SpectrumDataset(train_df, s2i, config["n_peaks"], return_str=True)
    valid_ds = SpectrumDataset(valid_df, s2i, config["n_peaks"], return_str=True)

    logging.info(
        f"Data loaded: {len(train_ds):,} training samples; {len(valid_ds):,} validation samples"
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        num_workers=config["n_workers"],
        shuffle=True,
        collate_fn=collate_batch,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=config["predict_batch_size"],
        num_workers=config["n_workers"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Update rates based on bs=32
    step_scale = 32 / config["train_batch_size"]
    logging.info(f"Updates per epoch: {len(train_dl):,}, step_scale={step_scale}")

    batch = next(iter(train_dl))
    spectra, precursors, spectra_mask, peptides, peptides_mask = batch

    logging.info("Sample batch:")
    logging.info(f" - spectra.shape={spectra.shape}")
    logging.info(f" - precursors.shape={precursors.shape}")
    logging.info(f" - spectra_mask.shape={spectra_mask.shape}")
    logging.info(f" - len(peptides)={len(peptides)}")
    logging.info(f" - peptides_mask={peptides_mask}")

    # init model
    model = InstaNovo(
        i2s=i2s,
        residues=config["residues"],
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        max_charge=config["max_charge"],
        use_depthcharge=config["use_depthcharge"],
        enc_type=config["enc_type"],
        dec_type=config["dec_type"],
        dec_precursor_sos=config["dec_precursor_sos"],
    )

    if model_path is not None:
        logging.info(f"Loading model checkpoint from '{model_path}'")
        model_state = torch.load(model_path, map_location="cpu")
        # check if PTL checkpoint
        if "state_dict" in model_state:
            model_state = {k.replace("model.", ""): v for k, v in model_state["state_dict"].items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(model.state_dict().keys())]
        )
        if k_missing > 0:
            logging.warning(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            logging.warning(f"Model state is missing {k_missing} keys!")
        model.load_state_dict(model_state, strict=False)

    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    logging.info("Test forward pass:")
    with torch.no_grad():
        y, _ = model(spectra, precursors, peptides, spectra_mask, peptides_mask)
        logging.info(f" - y.shape={y.shape}")

    # Train on GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # decoder = GreedyDecoder(model, i2s, max_length=config["max_length"])
    decoder = BeamSearchDecoder(model=model)
    metrics = Metrics(config["residues"], config["isotope_error_range"])

    # init optim
    # assert s2i["PAD"] == 0  # require PAD token to be index 0, all padding should use zeros
    # loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = WarmupScheduler(optim, config["warmup_iters"])
    strategy = _get_strategy()

    ptmodel = PTModule(config, model, decoder, metrics, sw, optim, scheduler)

    if config["save_model"]:
        callbacks = [
            ptl.callbacks.ModelCheckpoint(
                dirpath=config["model_save_folder_path"],
                save_top_k=-1,
                save_weights_only=config["save_weights_only"],
                every_n_train_steps=config["ckpt_interval"],
            )
        ]
    else:
        callbacks = None

    logging.info("Initializing PL trainer.")
    trainer = ptl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
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

    logging.info("Training complete.")


def _get_strategy() -> DDPStrategy | None:
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

    return None


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


def main() -> None:
    """Train the model."""
    logging.info("Initializing training.")

    parser = argparse.ArgumentParser()

    parser.add_argument("train_path")
    parser.add_argument("valid_path")
    parser.add_argument("--config", default="base.yaml")
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--n_workers", default=8)

    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"../../configs/instanovo/{args.config}"
    )

    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)

    # config["residues"] = {str(aa): float(mass) for aa, mass in config["residues"].items()}
    config["n_gpu"] = int(args.n_gpu)
    config["n_workers"] = int(args.n_workers)

    if config["n_gpu"] > 1:
        raise Exception("n_gpu > 1 currently not supported.")

    if not config["train_from_scratch"]:
        model_path = config["resume_checkpoint"]
    else:
        model_path = None

    train(args.train_path, args.valid_path, config, model_path)


if __name__ == "__main__":
    main()
