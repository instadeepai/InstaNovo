import argparse
import logging
import random
from dataclasses import dataclass

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

    batch_size: int = 12
    num_workers: int = 16
    grad_accumulation: int = 8

    summary_interval: int = 25
    checkpoint_interval: int = 1
    stdout_interval: int = 100
    validation_interval: int = 1

    # Data settings
    checkpoint_path: str = MISSING
    train_data_path: str = MISSING
    train_valid_split: float = MISSING

    # Training settings
    epochs: int = 100
    lr: float = 5e-4


# flake8: noqa: CR001
def train(cfg: TrainConfig) -> None:
    device = cfg.device

    model = TransNovo(cfg.model_cfg).to(device)

    # Vocabulary ('#' used for '(ox)')
    s2i = {k: v for v, k in enumerate(i2s)}

    # Load dataset
    logging.info(f"Loading train dataset from {cfg.train_data_path}")
    data_df = load_all(cfg.train_data_path)

    # Split dataset
    seq_unique = data_df["Sequence"].unique()

    train, valid = train_test_split(
        seq_unique, random_state=cfg.seed, test_size=cfg.train_valid_split
    )

    logging.info(
        f"train: {len(train)} unique sequences, valid: {len(valid)} unique sequences"
    )

    logging.info("Splitting dataset")
    # train_df = data_df.iloc[[seq in train for seq in data_df['Sequence']]]
    # valid_df = data_df.iloc[[seq in valid for seq in data_df['Sequence']]]
    train_df = data_df[data_df["Sequence"].isin(train)]
    valid_df = data_df[data_df["Sequence"].isin(valid)]
    train_df.shape, valid_df.shape
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
        shuffle=True,
        collate_fn=collate_batch,
        # num_workers=cfg.num_workers,
    )

    # PyTorch setup
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_set.PAD)

    # Tensorboard
    sw = SummaryWriter()

    steps = 0

    for epoch in tqdm(range(cfg.epochs)):
        model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            (x, y, x_pad, y_pad) = batch
            x = x.to(device).to(torch.float)
            y = y.to(device).to(torch.long)
            x_pad = x_pad.to(device).to(torch.bool)
            y_pad = y_pad.to(device).to(torch.bool)

            # add SOS to y
            y_sos = torch.cat(
                [torch.ones((y.shape[0], 1)).to(y.device).long(), y[:, :-1]], dim=-1
            )

            preds = model(x, x_pad, y_sos)

            loss = loss_fn(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))

            loss.backward()

            # Gradient accumulation
            if ((i + 1) % cfg.grad_accumulation == 0) or (i + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                steps += 1

            train_loss += loss.item() / len(train_loader)

            if steps % cfg.summary_interval == 0:
                # sw.add_scalar("training/loss_smooth", train_loss, steps)
                sw.add_scalar("training/loss_raw", loss.item(), steps)

        sw.add_scalar("training/epoch_loss", train_loss, steps)

        if ((epoch + 1) % cfg.validation_interval) == 0:
            model.eval()
            valid_loss = 0
            cer = 0

            text_preds = []
            text_targets = []
            for _, batch in enumerate(valid_loader):
                (x, y, x_pad, y_pad) = batch
                x = x.to(device).to(torch.float)
                y = y.to(device).to(torch.long)
                x_pad = x_pad.to(device).to(torch.bool)
                y_pad = y_pad.to(device).to(torch.bool)

                y_sos = torch.cat(
                    [torch.ones((y.shape[0], 1)).to(y.device).long(), y[:, :-1]], dim=-1
                )

                with torch.no_grad():
                    preds = model(x, x_pad, y_sos)

                loss = loss_fn(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))

                seq = (torch.ones((x.shape[0], 1)) * train_set.SOS).long().to(device)
                for _ in range(cfg.model_cfg.max_len):
                    with torch.no_grad():
                        y_hat = model(x.float(), x_pad, seq)
                    pred = y_hat.argmax(dim=-1)[:, -1]
                    seq = torch.cat([seq, pred[:, None]], dim=-1)

                text_preds += [train_set.seq_to_aa(s[1:]) for s in seq]
                text_targets += [train_set.seq_to_aa(s) for s in y]

                valid_loss += loss.item() / len(valid_loader)
            cer = jiwer.cer(text_preds, text_targets)

            logging.info(
                f"Epoch {epoch+1:4d}/{cfg.epochs}: train_loss={train_loss:.4f}, \
                        valid_loss={valid_loss:.4f}, valid_cer={cer:.3f}"
            )

            sw.add_scalar("eval/loss", valid_loss, steps)
            sw.add_scalar("eval/cer", cer, steps)

        if ((epoch + 1) % cfg.checkpoint_interval) == 0:
            torch.save(model.state_dict(), f"{cfg.checkpoint_path}/model_{epoch}.ckpt")


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
