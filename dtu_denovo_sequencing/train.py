import argparse
import logging
import random
from dataclasses import dataclass
from typing import Optional

import jiwer
import numpy as np
import torch
import torch.nn as nn
from omegaconf import MISSING
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dtu_denovo_sequencing.config import i2s
from dtu_denovo_sequencing.config import ModelConfig
from dtu_denovo_sequencing.dataset import collate_batch
from dtu_denovo_sequencing.dataset import load_all
from dtu_denovo_sequencing.dataset import SpecDataset
from dtu_denovo_sequencing.model import TransNovo


@dataclass
class TrainConfig:
    # Model settings
    model_cfg: ModelConfig = ModelConfig()

    device: str = "cuda"
    seed: int = 1775

    # Note: while a higher batch size may work, some samples are very long
    # This causes significant GPU bottlenecking at max VRAM as it tries to process the sequence
    batch_size: int = 16
    num_workers: int = 1
    grad_accumulation: int = 4
    fp16: bool = True

    summary_interval: int = 32
    checkpoint_interval: int = 40000
    stdout_interval: int = 40000
    validation_interval: int = 40000

    # Data settings
    checkpoint_path: str = MISSING
    train_data_path: str = MISSING
    logs_path: str = MISSING
    load_weights: Optional[str] = None

    # Keep these first two consistent for all implementations! Always use sklearn.model_selection.train_test_split
    test_split_seed: int = 100  # seed for splitting test set
    test_split: float = 0.1  # test split proportion of full dataset
    valid_split: float = 0.1  # valid split proportion of full dataset
    valid_proportion: float = (
        0.1  # Validate only on a portion of the full validation set (saves time!)
    )

    # Training settings
    epochs: int = 100
    lr: float = 1e-4
    warmup_lr_start: float = 1e-7
    warmup_steps: int = 5000


# flake8: noqa: CR001
def train(cfg: TrainConfig) -> None:
    device = cfg.device

    logging.info(f"Creating model")
    model = TransNovo(cfg.model_cfg).to(device)
    if cfg.fp16:
        logging.info(f"Running in mixed precision")

    logging.info(
        f"Model: {np.sum([p.numel() for p in model.parameters()]):,} parameters, {device}"
    )

    if cfg.load_weights:
        model.load_state_dict(torch.load(cfg.load_weights, map_location="cpu"))
        logging.info(f"Model: loaded weights from {cfg.load_weights}")

    # Vocabulary ('#' used for 'M(ox)')
    s2i = {k: v for v, k in enumerate(i2s)}

    # Load dataset
    logging.info(f"Loading train dataset from {cfg.train_data_path}")
    data_df = load_all(cfg.train_data_path)

    # Split dataset
    seq_unique = data_df["Sequence"].unique()

    # test unused during training, seed and proportion must remain constant!
    train, test = train_test_split(
        seq_unique,
        random_state=cfg.test_split_seed,
        test_size=int(len(seq_unique) * cfg.test_split),
    )

    # this seed can change
    train, valid = train_test_split(
        train,
        random_state=cfg.test_split_seed,
        test_size=int(len(seq_unique) * cfg.valid_split),
    )

    logging.info(
        f"train: {len(train)} unique sequences, valid: {len(valid)} unique sequences, test: {len(test)} unique sequences"
    )

    logging.info(f"Shrinking validation set by a factor of {cfg.valid_proportion}")

    valid = valid[: int(len(valid) * cfg.valid_proportion)]

    logging.info("Splitting dataset")
    train_df = data_df[data_df["Sequence"].isin(train)]
    valid_df = data_df[data_df["Sequence"].isin(valid)]

    logging.info(
        f"train: {len(train_df)} total samples, valid: {len(valid_df)} total samples"
    )

    train_set = SpecDataset(train_df, s2i, i2s)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        # num_workers=cfg.num_workers,
    )

    valid_set = SpecDataset(valid_df, s2i, i2s)
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        # num_workers=cfg.num_workers,
    )

    # PyTorch setup
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg.warmup_lr_start / cfg.lr,
        total_iters=cfg.warmup_steps,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_set.PAD)
    logging.info(f"nominal batch_size: {cfg.batch_size * cfg.grad_accumulation}")

    # fp16 grad scaler
    if cfg.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # Tensorboard
    sw = SummaryWriter(cfg.logs_path)

    steps = 0
    smooth_loss = None

    model.train()

    for epoch in tqdm(range(cfg.epochs)):

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            (x, y, x_pad, y_pad) = batch
            x = x.to(device).float()
            y = y.to(device).to(torch.long)
            x_pad = x_pad.to(device).to(torch.bool)
            y_pad = y_pad.to(device).to(torch.bool)

            # add SOS to y
            y_sos = torch.cat(
                [torch.ones((y.shape[0], 1)).to(y.device).long(), y[:, :-1]], dim=-1
            )

            preds = model(x, x_pad, y_sos)

            loss = loss_fn(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))

            if cfg.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((i + 1) % cfg.grad_accumulation == 0) or (i + 1 == len(train_loader)):
                if cfg.fp16:
                    scaler.unscale_(optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                scheduler.step()

            if smooth_loss is None:
                smooth_loss = float(loss.item())
            else:
                smooth_loss = smooth_loss + 0.1 * (float(loss.item()) - smooth_loss)

            if ((steps + 1) % cfg.summary_interval) == 0:
                sw.add_scalar("training/loss_smooth", smooth_loss, steps)
                sw.add_scalar("training/loss_raw", loss.item(), steps)
                sw.add_scalar("opt/lr", float(optimizer.param_groups[0]["lr"]), steps)
                if cfg.fp16:
                    sw.add_scalar("opt/grad_norm", gnorm, steps)

            if ((steps + 1) % cfg.validation_interval) == 0:
                logging.info("running validation")
                model.eval()
                valid_loss = 0
                cer = 0

                text_preds = []
                text_targets = []
                for _, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    (x, y, x_pad, y_pad) = batch
                    x = x.to(device).float()
                    y = y.to(device).to(torch.long)
                    x_pad = x_pad.to(device).to(torch.bool)
                    y_pad = y_pad.to(device).to(torch.bool)

                    y_sos = torch.cat(
                        [torch.ones((y.shape[0], 1)).to(y.device).long(), y[:, :-1]],
                        dim=-1,
                    )

                    with torch.no_grad():
                        preds = model(x, x_pad, y_sos)

                    loss = loss_fn(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))

                    seq = (
                        (torch.ones((x.shape[0], 1)) * train_set.SOS).long().to(device)
                    )

                    with torch.no_grad():
                        x = model.input_embed(x.float())
                        x = model.transformer.encoder(x, src_key_padding_mask=x_pad)

                    for _ in range(cfg.model_cfg.max_len):
                        with torch.no_grad():
                            # y_hat = model(x.float(), x_pad, seq)
                            yy = model.pos_enc(
                                model.seq_embed(seq).transpose(0, 1)
                            ).transpose(0, 1)
                            yy = model.transformer.decoder(
                                yy, memory=x, memory_key_padding_mask=x_pad
                            )
                            y_hat = model.head(yy)

                        pred = y_hat.argmax(dim=-1)[:, -1]
                        seq = torch.cat([seq, pred[:, None]], dim=-1)

                    text_preds += [train_set.seq_to_aa(s[1:]) for s in seq]
                    text_targets += [train_set.seq_to_aa(s) for s in y]

                    valid_loss += loss.item() / len(valid_loader)
                cer = jiwer.cer(text_preds, text_targets)

                logging.info(
                    f"Epoch {epoch+1:4d}/{cfg.epochs}: train_loss={smooth_loss:.4f}, \
                            valid_loss={valid_loss:.4f}, valid_cer={cer:.3f}"
                )

                sw.add_scalar("validation/loss", valid_loss, steps)
                sw.add_scalar("validation/cer", cer, steps)

                model.train()

            if ((steps + 1) % cfg.checkpoint_interval) == 0:
                torch.save(
                    model.state_dict(), f"{cfg.checkpoint_path}/model_{steps}.ckpt"
                )

            steps += 1


def main() -> None:
    print("Initializing Training Process..")
    logging.getLogger().setLevel(logging.INFO)

    # Setup CLI args
    parser = argparse.ArgumentParser(
        usage="\n"
        + "-" * 10
        + " Default config "
        + "-" * 10
        + "\n"
        + str(OmegaConf.to_yaml(OmegaConf.structured(TrainConfig)))
    )

    # Parse args
    a, _ = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli()

    base_cfg = OmegaConf.structured(TrainConfig)
    cfg: TrainConfig = OmegaConf.merge(base_cfg, override_cfg)
    logging.info(f"Running with config:\n {OmegaConf.to_yaml(cfg)}")

    # Set seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # Launch training
    train(cfg)


if __name__ == "__main__":
    main()
