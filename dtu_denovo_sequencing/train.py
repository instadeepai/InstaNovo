import argparse
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import deepspeed
import jiwer
import numpy as np
import torch
import torch.nn as nn
from fastprogress.fastprogress import master_bar
from fastprogress.fastprogress import progress_bar
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

# DS2 train command:
# deepspeed --num_nodes 1 ./dtu_denovo_sequencing/train.py checkpoint_path=runs/trans-debug/ train_data_path=./data/denovo_dataset_v1/ batch_size=12 --deepspeed --deepspeed_config=deepspeed_cfg.json


@dataclass
class DistributedConfig:
    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:54321"
    # n_nodes: int = 1 # Handled by deepspeed
    n_gpus_per_node: int = 1


@dataclass
class TrainConfig:
    # Distributed settings
    distributed: DistributedConfig = DistributedConfig()
    # Model settings
    model_cfg: ModelConfig = ModelConfig()

    device: str = "cuda"
    seed: int = 1775

    # Note: while a higher batch size may work, some samples are very long
    # This causes significant GPU bottlenecking at max VRAM as it tries to process the sequence
    batch_size: int = 8
    num_workers: int = 16

    summary_interval: int = 25
    checkpoint_interval: int = 160_000
    stdout_interval: int = 100
    validation_interval: int = 40_000

    # Learning settings -- managed by deepspeed cfg
    max_steps: int = 100_000_000  # 500_000 #1_000_000

    # Data settings
    checkpoint_path: str = MISSING
    train_data_path: str = MISSING
    resume_checkpoint: str = ""

    # Keep these first two consistent for all implementations! Always use sklearn.model_selection.train_test_split
    test_split_seed: int = 100  # seed for splitting test set
    test_split: float = 0.1  # test split proportion of full dataset
    valid_split: float = 0.1  # valid split proportion of full dataset
    valid_proportion: float = (
        0.1  # Validate only on a portion of the full validation set
    )


# flake8: noqa: CR001
def train(rank: int, cfg: TrainConfig, deepspeed_cfg: argparse.Namespace) -> None:
    print(f"[RANK {rank}] Deepspeed cfg: {deepspeed_cfg}")

    # -------------------
    # Setup distributed
    if cfg.distributed.n_gpus_per_node > 1:
        deepspeed.init_distributed(
            backend=cfg.distributed.dist_backend, init_method=cfg.distributed.dist_url
        )

    device = torch.device(f"cuda:{rank:d}")

    # --------------------
    # Define model and loss
    logging.info(f"Creating model")
    model = TransNovo(cfg.model_cfg).to(device)

    if rank == 0:
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Model initialized as:\n {model}")
        logging.info(f"checkpoints directory : {cfg.checkpoint_path}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
    print(
        f"[RANK {rank}] Model has {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )

    # ------------------------
    # Get train and validation data
    # Vocabulary ('#' used for 'M(ox)')
    s2i = {k: v for v, k in enumerate(i2s)}

    # Load dataset
    # logging.info(f"Loading train dataset from {cfg.train_data_path}")
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

    valid = valid[: int(len(valid) * cfg.valid_proportion)]

    train_df = data_df[data_df["Sequence"].isin(train)]
    valid_df = data_df[data_df["Sequence"].isin(valid)]

    train_ds = SpecDataset(train_df, s2i, i2s)
    valid_ds = SpecDataset(valid_df, s2i, i2s)

    # ------------------------
    # Initialize deepspeed wrapper
    model_engine, optim, train_dl, scheduler = deepspeed.initialize(
        args=deepspeed_cfg,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_ds,
        collate_fn=collate_batch,
    )
    # fix broken deepspeed gradient accumulated dl sizes.
    real_train_len = len(train_dl.data_sampler) // train_dl.batch_size
    train_dl.len = real_train_len

    try:
        ds_log_path = (
            Path(model_engine.tensorboard_output_path())
            / model_engine.tensorboard_job_name()
        )
    except Exception as e:
        ds_log_path = (
            Path(model_engine.monitor.tb_monitor.output_path)
            / model_engine.monitor.tb_monitor.job_name
        )

    if cfg.resume_checkpoint != "":
        _, client_sd = model_engine.load_checkpoint(
            cfg.checkpoint_path, cfg.resume_checkpoint
        )
        steps = client_sd["steps"] + 1
        last_epoch = client_sd["last_epoch"]
    else:
        steps = 0
        last_epoch = 0
        client_sd = {}

    fp16 = model_engine.fp16_enabled()

    loss_fn = nn.CrossEntropyLoss(ignore_index=train_ds.PAD)

    # --------------------------
    # Logging init
    max_epochs = math.ceil(cfg.max_steps / len(train_dl))
    print(f"[RANK {rank}] deepspeed fp16={fp16} | max epochs: {max_epochs}")

    if rank == 0:
        print(f"[RANK {rank}] deepspeed logging to {ds_log_path}")
        try:
            sw = model_engine.get_summary_writer()
        except Exception as e:
            sw = model_engine.monitor.tb_monitor.summary_writer

        mb = master_bar(range(max(0, last_epoch), max_epochs))
        sw.add_text(
            "config", "```\n" + OmegaConf.to_yaml(cfg) + "\n```", global_step=steps
        )
        smooth_loss = None
        valid_dl = DataLoader(
            valid_ds,
            cfg.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=cfg.num_workers,
        )
    else:
        mb = range(max(0, last_epoch), max_epochs)

    # --------------------------
    # Training loop
    model_engine.train()

    for epoch in mb:
        if rank == 0:
            start = time.time()
            mb.write(f"Epoch: {epoch + 1}")
            pb = progress_bar(enumerate(train_dl), total=len(train_dl), parent=mb)
        else:
            pb = enumerate(train_dl)

        if steps > cfg.max_steps:
            break

        for i, batch in pb:
            # -----------------------
            #  Read batch
            if rank == 0:
                start_b = time.time()
            x, y, x_pad, y_pad = batch
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).to(torch.long)
            x_pad = x_pad.to(device, non_blocking=True).to(torch.bool)
            y_pad = y_pad.to(device, non_blocking=True).to(torch.bool)

            if fp16:
                x = x.to(torch.float16)

            # add SOS to y
            y_sos = torch.cat(
                [torch.ones((y.shape[0], 1)).to(y.device).long(), y[:, :-1]], dim=-1
            )

            preds = model_engine(x, x_pad, y_sos)

            loss = loss_fn(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))

            # Backwards
            model_engine.backward(loss)
            if steps % cfg.summary_interval == 0:
                gnorm = model_engine.get_global_grad_norm()
            model_engine.step()

            # checkpointing
            if steps % cfg.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = f"{cfg.checkpoint_path}/ckpt_{steps:08d}.pt"

                client_sd["steps"] = steps
                client_sd["last_epoch"] = epoch
                client_sd["cfg_yaml"] = OmegaConf.to_yaml(cfg)

                model_engine.save_checkpoint(
                    cfg.checkpoint_path,
                    Path(checkpoint_path).stem,
                    client_state=client_sd,
                )

                print(f"[RANK {rank}] Saved checkpoint to {checkpoint_path}")

            # ----------------------
            # Validation & logging
            if rank == 0:
                if smooth_loss is None:
                    smooth_loss = float(loss.item())
                else:
                    smooth_loss = smooth_loss + 0.1 * (float(loss.item()) - smooth_loss)

                # STDOUT logging
                if steps % cfg.stdout_interval == 0:
                    mb.write(
                        "steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB".format(
                            steps,
                            loss.item(),
                            time.time() - start_b,
                            torch.cuda.max_memory_allocated() / 1e9,
                        )
                    )
                if steps % (cfg.stdout_interval // 5) == 0:
                    mb.child.comment = (
                        "steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}".format(
                            steps, loss.item(), time.time() - start_b
                        )
                    )

                # Tensorboard summary logging
                if steps % cfg.summary_interval == 0:
                    sw.add_scalar("training/loss_smooth", smooth_loss, steps)
                    sw.add_scalar("training/loss_raw", loss.item(), steps)
                    sw.add_scalar("opt/lr", float(optim.param_groups[0]["lr"]), steps)
                    if gnorm is not None:
                        sw.add_scalar("opt/grad_norm", float(gnorm), steps)

                # Validation
                if steps % cfg.validation_interval == 0 and steps != 0:
                    model_engine.eval()
                    # loss_fn.eval()

                    valid_loss = 0
                    cer = 0
                    text_preds = []
                    text_targets = []
                    with torch.no_grad():
                        for _, batch in progress_bar(
                            enumerate(valid_dl), total=len(valid_dl), parent=mb
                        ):
                            (x, y, x_pad, y_pad) = batch
                            x = x.to(device).float()
                            y = y.to(device).to(torch.long)
                            x_pad = x_pad.to(device).to(torch.bool)
                            y_pad = y_pad.to(device).to(torch.bool)

                            if fp16:
                                x = x.to(torch.float16)

                            y_sos = torch.cat(
                                [
                                    torch.ones((y.shape[0], 1)).to(y.device).long(),
                                    y[:, :-1],
                                ],
                                dim=-1,
                            )

                            preds = model(x, x_pad, y_sos)

                            loss = loss_fn(
                                preds.reshape(-1, preds.shape[-1]), y.reshape(-1)
                            )

                            seq = (
                                (torch.ones((x.shape[0], 1)) * valid_ds.SOS)
                                .long()
                                .to(device)
                            )

                            x = model.input_embed(x)
                            x = model.transformer.encoder(x, src_key_padding_mask=x_pad)

                            for _ in range(cfg.model_cfg.max_len):
                                # y_hat = model(x.float(), x_pad, seq)
                                yy = model.pos_enc(
                                    model.seq_embed(seq).transpose(0, 1)
                                ).transpose(0, 1)
                                yy = model.transformer.decoder(
                                    yy,
                                    memory=x,
                                    tgt_mask=model.transformer.generate_square_subsequent_mask(
                                        seq.shape[1]
                                    ).to(
                                        x.device
                                    ),
                                    memory_key_padding_mask=x_pad,
                                )
                                y_hat = model.head(yy)

                                pred = y_hat.argmax(dim=-1)[:, -1]
                                seq = torch.cat([seq, pred[:, None]], dim=-1)

                            text_preds += [valid_ds.seq_to_aa(s[1:]) for s in seq]
                            text_targets += [valid_ds.seq_to_aa(s) for s in y]

                            valid_loss += loss.item() / len(valid_dl)
                        cer = jiwer.cer(text_preds, text_targets)

                        logging.info(
                            f"step {steps+1:4d}/{cfg.max_steps}: train_loss={smooth_loss:.4f}, \
                                    valid_loss={valid_loss:.4f}, valid_cer={cer:.3f}"
                        )

                        sw.add_scalar("validation/loss", valid_loss, steps)
                        sw.add_scalar("validation/cer", cer, steps)

                    model_engine.train()
                    # loss_fn.train()
                    sw.add_scalar(
                        "memory/max_allocated_gb",
                        torch.cuda.max_memory_allocated() / 1e9,
                        steps,
                    )
                    sw.add_scalar(
                        "memory/max_reserved_gb",
                        torch.cuda.max_memory_reserved() / 1e9,
                        steps,
                    )
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

            steps += 1
            if steps > cfg.max_steps:
                print(f"[RANK {rank}] FINISHED TRAINING")
                break

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)
                )
            )
    print("Training completed!")


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

    deepspeed.add_config_arguments(parser)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    # Parse args
    a, _ = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli()

    # We must remove any config arguments deepspeed injected,
    # otherwise we will have duplicate deepspeed keys in `override_cfg`
    # and cli args `a`.
    keys_to_drop = []
    for key in override_cfg:
        if key.startswith("--"):
            keys_to_drop.append(key)
    for key in keys_to_drop:
        delattr(override_cfg, key)

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
    train(a.local_rank, cfg, a)


if __name__ == "__main__":
    main()
