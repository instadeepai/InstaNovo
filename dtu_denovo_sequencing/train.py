from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import deepspeed
import jiwer
import numpy as np
import s3fs
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastprogress.fastprogress import master_bar
from fastprogress.fastprogress import progress_bar
from omegaconf import MISSING
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tensorboard.compat.tensorflow_stub.io.gfile import _REGISTERED_FILESYSTEMS
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dtu_denovo_sequencing.config import i2s
from dtu_denovo_sequencing.config import ModelConfig
from dtu_denovo_sequencing.dataset import collate_batch
from dtu_denovo_sequencing.dataset import load_all
from dtu_denovo_sequencing.dataset import SpecDataset
from dtu_denovo_sequencing.model import TransNovo
from dtu_denovo_sequencing.utils import evaluation


load_dotenv()

# DS2 train command:
# deepspeed --num_nodes 1 ./dtu_denovo_sequencing/train.py checkpoint_path=runs/trans-debug/ train_data_path=./data/denovo_dataset_v1/ batch_size=12 --deepspeed --deepspeed_config=deepspeed_cfg.json

# For debugging NCCL failures, print an explicit warning message as well as basic NCCL initialization information.
os.environ["NCCL_DEBUG"] = "INFO"


@dataclass
class DistributedConfig:
    """Configuration settings for distributes training."""

    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:54321"
    # n_nodes: int = 1 # Handled by deepspeed
    n_gpus_per_node: int = 1


@dataclass
class TrainConfig:
    """Configuration settings for training."""

    # Distributed settings
    distributed: DistributedConfig = DistributedConfig()
    # Model settings
    model_cfg: ModelConfig = ModelConfig()

    device: str = "cuda"
    seed: int = 1775

    # Validation set batch_size
    batch_size: int = 4
    # num_workers > 0 causes occasional dataloader freeze
    num_workers: int = 0

    summary_interval: int = 25
    checkpoint_interval: int = 40_000  # 20_000
    stdout_interval: int = 100
    validation_interval: int = 20_000  # 10_000

    # Learning settings -- managed by deepspeed cfg
    max_steps: int = 100_000_000  # 500_000 #1_000_000

    # Data settings
    checkpoint_path: str = "/runs"
    train_data_path: str = MISSING
    resume_checkpoint: str = ""

    # Keep these first two consistent for all implementations! Always use sklearn.model_selection.train_test_split
    test_split_seed: int = 100  # seed for splitting test set
    test_split: float = 0.1  # test split proportion of full dataset
    valid_split: float = 0.1  # valid split proportion of full dataset
    valid_proportion: float = 0.1  # validate only on a portion of the full validation set


# flake8: noqa: CR001
def train(rank: int, cfg: TrainConfig, deepspeed_cfg: argparse.Namespace) -> None:
    """Launch the training."""
    # If running on AIchor, set monitors to use output paths created by AIchor
    if "AICHOR_LOGS_PATH" in os.environ:
        assert "S3_ENDPOINT" in os.environ

        fs = _REGISTERED_FILESYSTEMS["s3"]
        if fs._s3_endpoint is None:
            # Set custom S3 endpoint explicitly. Not sure why this isn't picked up here:
            # https://github.com/tensorflow/tensorboard/blob/153cc747fdbeca3545c81947d4880d139a185c52/tensorboard/compat/tensorflow_stub/io/gfile.py#L227
            fs._s3_endpoint = os.environ["S3_ENDPOINT"]
        register_filesystem("s3", fs)

        deepspeed_cfg_dict = vars(deepspeed_cfg)
        deepspeed_cfg_path = Path(deepspeed_cfg_dict["deepspeed_config"])
        with open(deepspeed_cfg_path) as f:
            ds_config = json.load(f)

        ds_config["tensorboard"]["output_path"] = os.environ["AICHOR_LOGS_PATH"]

        logging.info(f"Updated ds_config: {ds_config}")
        new_deepspeed_cfg_path = deepspeed_cfg_path.with_name(
            f"{deepspeed_cfg_path.stem}_aichor{deepspeed_cfg_path.suffix}"
        )
        with open(new_deepspeed_cfg_path, "w") as f:
            json.dump(ds_config, f, indent=4)
            logging.info(f"Written updated config to {new_deepspeed_cfg_path}")

        deepspeed_cfg_dict["deepspeed_config"] = str(new_deepspeed_cfg_path)
        deepspeed_cfg = Namespace(**deepspeed_cfg_dict)

    logging.info(f"[RANK {rank}] Deepspeed cfg: {deepspeed_cfg}")

    # -------------------
    # Setup distributed
    if cfg.distributed.n_gpus_per_node > 1:
        deepspeed.init_distributed(dist_backend=cfg.distributed.dist_backend)

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
    logging.info(
        f"[RANK {rank}] Model has {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )

    # ------------------------
    # Get train and validation data
    # Vocabulary ('#' used for 'M(ox)')
    s2i = {k: v for v, k in enumerate(i2s)}

    # Load dataset
    # logging.info(f"Loading train dataset from {cfg.train_data_path}")
    data_df = load_all(cfg.train_data_path, verbose=True)

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

    train_ds = SpecDataset(train_df.copy(), s2i, i2s)
    valid_ds = SpecDataset(valid_df.copy(), s2i, i2s)

    # remove all references to original dataframes to save memory
    del train_df, valid_df, data_df

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

    if cfg.resume_checkpoint != "":
        _, client_sd = model_engine.load_checkpoint(cfg.checkpoint_path, cfg.resume_checkpoint)
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
    logging.info(f"[RANK {rank}] deepspeed fp16={fp16} | max epochs: {max_epochs}")

    if rank == 0:
        try:
            ds_log_path = f"{model_engine.monitor.tb_monitor.output_path}{model_engine.monitor.tb_monitor.job_name}"
        except AttributeError as err:
            logging.warning(f"AttributeError: {err}")
            ds_log_path = "<NOT SET>"

        logging.info(
            f"[RANK {rank}] cfg.checkpoint_path: '{cfg.checkpoint_path}', cfg.resume_checkpoint: '{cfg.resume_checkpoint}'"
        )

        logging.info(f"[RANK {rank}] deepspeed logging to {ds_log_path}")
        try:
            sw = model_engine.get_summary_writer()
        except Exception as e:
            sw = model_engine.monitor.tb_monitor.summary_writer

        mb = master_bar(range(max(0, last_epoch), max_epochs))
        sw.add_text("config", "```\n" + OmegaConf.to_yaml(cfg) + "\n```", global_step=steps)
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
            y_sos = torch.cat([torch.ones((y.shape[0], 1)).to(y.device).long(), y[:, :-1]], dim=-1)

            preds = model_engine(x, x_pad, y_sos)

            loss = loss_fn(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))

            # Backwards
            model_engine.backward(loss)
            if steps % cfg.summary_interval == 0:
                gnorm = model_engine.get_global_grad_norm()
            model_engine.step()

            # ----------------------
            # checkpointing
            if steps % cfg.checkpoint_interval == 0 and steps != 0:
                checkpoint_tag = f"ckpt_{steps:08d}"

                client_sd["steps"] = steps
                client_sd["last_epoch"] = epoch
                client_sd["cfg_yaml"] = OmegaConf.to_yaml(cfg)

                local_dir = os.path.join(cfg.checkpoint_path, checkpoint_tag)
                logging.info(
                    f"[RANK {rank}] Saving checkpoint locally to {local_dir} with client_sd: {client_sd}"
                )
                if not os.path.exists(local_dir):
                    logging.info(f"[RANK {rank}] Creating {local_dir}")
                    os.makedirs(local_dir, exist_ok=True)

                # First save checkpoint locally, must be done on all ranks
                # Hangs to synchronise all threads.
                # Also calls .barrier() at the end to ensure all threads are done writing.
                logging.info(f"[RANK {rank}] Saving checkpoint locally to {local_dir}")
                model_engine.save_checkpoint(
                    save_dir=cfg.checkpoint_path,
                    tag=checkpoint_tag,
                    client_state=client_sd,
                )
                logging.info(f"[RANK {rank}] Saved checkpoint locally to {local_dir}")

                if rank == 0:
                    # now upload to S3
                    logging.info(f"[RANK {rank}] Creating s3fs.core.S3FileSystem")
                    s3 = s3fs.core.S3FileSystem(
                        client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")}
                    )
                    logging.info(f" [RANK {rank}] Created s3fs.core.S3FileSystem: {s3}")

                    # Prepare for checkpoint load by ensuring all parameters are partitioned
                    # https://github.com/microsoft/DeepSpeed/blob/6273dffc2f192275a08268b683c309a328b52191/deepspeed/runtime/engine.py#L2752
                    if model_engine.zero_optimization_partition_weights():
                        logging.info("model_engine.optimizer.checkpoint_event_prologue()")
                        model_engine.optimizer.checkpoint_event_prologue()
                        logging.info("model_engine.optimizer.checkpoint_event_prologue() done")

                    # https://github.com/microsoft/DeepSpeed/blob/6273dffc2f192275a08268b683c309a328b52191/deepspeed/runtime/engine.py#L2789
                    ckpt_list = model_engine._get_all_ckpt_names(
                        cfg.checkpoint_path, checkpoint_tag
                    )
                    logging.info(f"ckpt_list: {ckpt_list}")
                    for local_chkpt_path in ckpt_list:
                        logging.info(f"local_chkpt_path: {local_chkpt_path}")
                        relative_path = Path(local_chkpt_path).relative_to(cfg.checkpoint_path)
                        logging.info(f"relative_path: {relative_path}")
                        s3_chkpt_path = f"{os.environ['AICHOR_OUTPUT_PATH']}{relative_path}"
                        logging.info(f"s3_chkpt_path: {s3_chkpt_path}")

                        with open(local_chkpt_path, "rb") as local_fp, s3.open(
                            s3_chkpt_path, "wb"
                        ) as remote_fp:
                            remote_fp.write(local_fp.read())
                            logging.info(f"Wrote {local_chkpt_path} to {s3_chkpt_path}")

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
                    mb.child.comment = "steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}".format(
                        steps, loss.item(), time.time() - start_b
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
                    logging.info(f"[RANK {rank}] Starting validation at steps : {steps:d}")
                    model_engine.eval()
                    # loss_fn.eval()

                    valid_loss = 0
                    cer = 0
                    text_preds = []
                    text_targets = []
                    with torch.no_grad():
                        logging.info(f"[RANK {rank}] Total validation steps : {len(valid_dl):d}")
                        for i, batch in progress_bar(
                            enumerate(valid_dl), total=len(valid_dl), parent=mb
                        ):
                            # for i , batch in enumerate(valid_dl):
                            # logging.info(f"[RANK {rank}] Validaton step : {i:d}")
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

                            loss = loss_fn(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))

                            seq = (torch.ones((x.shape[0], 1)) * valid_ds.SOS).long().to(device)

                            bias = x[:, :, 0]
                            x = model.input_embed(x)
                            x = model.transformer.encoder(x, bias=bias, src_key_padding_mask=x_pad)
                            # x = model.transformer.encoder(x, src_key_padding_mask=x_pad)

                            for _ in range(cfg.model_cfg.max_len):
                                # y_hat = model(x.float(), x_pad, seq)
                                yy = model.pos_enc(model.seq_embed(seq))
                                yy = model.transformer.decoder(
                                    yy,
                                    memory=x,
                                    tgt_mask=model.transformer.generate_square_subsequent_mask(
                                        seq.shape[1]
                                    ).to(x.device),
                                    memory_key_padding_mask=x_pad,
                                )
                                y_hat = model.head(yy)

                                pred = y_hat.argmax(dim=-1)[:, -1]
                                seq = torch.cat([seq, pred[:, None]], dim=-1)

                            text_preds += [valid_ds.seq_to_aa(s[1:]) for s in seq]
                            text_targets += [valid_ds.seq_to_aa(s) for s in y]

                            valid_loss += loss.item() / len(valid_dl)
                            # logging.info(f"[RANK {rank}] valid_loss : {valid_loss:.3f}")
                        cer = jiwer.cer(text_preds, text_targets)
                        aa_precision, aa_recall, pep_recall = evaluation.aa_match_metrics(
                            *evaluation.aa_match_batch(text_preds, text_targets, s2i)
                        )

                        logging.info(
                            f"step {steps+1:4d}/{cfg.max_steps}: train_loss={smooth_loss:.4f}, \
                                    valid_loss={valid_loss:.4f}, valid_cer={cer:.3f}, \
                                    valid_aa_precision={aa_precision:.3f}, \
                                    valid_aa_recall={aa_recall:.3f}, \
                                    valid_pep_recall={pep_recall:.3f}"
                        )

                        sw.add_scalar("validation/loss", valid_loss, steps)
                        sw.add_scalar("validation/cer", cer, steps)
                        sw.add_scalar("validation/aa_precision", aa_precision, steps)
                        sw.add_scalar("validation/aa_recall", aa_recall, steps)
                        sw.add_scalar("validation/pep_recall", pep_recall, steps)

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

                    logging.info(f"[RANK {rank}] Validation complete")

            steps += 1
            if steps > cfg.max_steps:
                logging.info(f"[RANK {rank}] FINISHED TRAINING")
                break

        if rank == 0:
            logging.info(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")
    logging.info("Training completed!")


def main() -> None:
    """Train the model."""
    logging.info("Initializing Training Process..")
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
    deepspeed_cfg, _ = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli()

    # We must remove any config arguments deepspeed injected,
    # otherwise we will have duplicate deepspeed keys in `override_cfg`
    # and cli args `deepspeed_cfg`.
    keys_to_drop = []
    for key in override_cfg:
        if key.startswith("--"):
            keys_to_drop.append(key)
    for key in keys_to_drop:
        delattr(override_cfg, key)

    base_cfg = OmegaConf.structured(TrainConfig)
    train_cfg: TrainConfig = OmegaConf.merge(base_cfg, override_cfg)

    logging.info(f"Running with config:\n {OmegaConf.to_yaml(train_cfg)}")
    # Set seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_cfg.seed)

    # Launch training
    train(deepspeed_cfg.local_rank, train_cfg, deepspeed_cfg)


if __name__ == "__main__":
    main()
