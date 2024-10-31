from __future__ import annotations

import logging
import os
from pathlib import Path
import re

import pytorch_lightning as pl
import s3fs
from pytorch_lightning.strategies import DDPStrategy
from tensorboard.compat.tensorflow_stub.io.gfile import _REGISTERED_FILESYSTEMS
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def upload(source: str, target: str) -> None:
    """Upload local file to s3."""
    if not _s3_enabled():
        return
    s3 = _create_s3fs()

    logger.info(f"Uploading {source} to {target}")
    with open(source, "rb") as local_fp, s3.open(target, "wb") as remote_fp:
        remote_fp.write(local_fp.read())
        logger.info(f"Wrote {source} to {target}")


def download(source: str, target: str) -> None:
    """Download file from s3 and save locally."""
    if not _s3_enabled():
        return
    s3 = _create_s3fs()

    logger.info(f"Downloading {source} to {target}")
    with open(target, "wb") as local_fp, s3.open(source, "rb") as remote_fp:
        local_fp.write(remote_fp.read())
        logger.info(f"Wrote {source} to {target}")


def get_checkpoint_path(model_filename: str, local_path: str = "checkpoints") -> str:
    """Download checkpoint from s3 and return local checkpoint path."""
    if not _s3_enabled():
        return model_filename

    # Environment variable specific to Aichor compute platform
    model_filename = f"{os.environ['AICHOR_INPUT_PATH']}{model_filename}"

    os.makedirs(local_path, exist_ok=True)
    local_path = f"{local_path}/model.ckpt"

    logger.info(f"Downloading {model_filename} from S3 to {local_path}")
    download(model_filename, target=local_path)

    return local_path


def register_tb() -> bool:
    """Register s3 filesystem to tensorboard."""
    if not _s3_enabled():
        return False
    fs = _REGISTERED_FILESYSTEMS["s3"]
    if fs._s3_endpoint is None:
        # Set custom S3 endpoint explicitly. Not sure why this isn't picked up here:
        # https://github.com/tensorflow/tensorboard/blob/153cc747fdbeca3545c81947d4880d139a185c52/tensorboard/compat/tensorflow_stub/io/gfile.py#L227
        fs._s3_endpoint = os.environ["S3_ENDPOINT"]
    register_filesystem("s3", fs)
    return True


def convert_to_s3_output(path: str) -> str:
    """Convert local directory to s3 output path if possible."""
    if _s3_enabled():
        # Environment variable specific to Aichor compute platform
        output_path = os.environ["AICHOR_OUTPUT_PATH"]
        if "s3://" not in output_path:
            output_path = f"s3://{output_path}/output/"
        return output_path + str(Path(path))
    return path


def _clean_filepath(filepath: str) -> str:
    """Convert file path to AIchor compatible path, without disallowed characters."""
    pattern = r"[^a-zA-Z0-9\-_\.]"
    clean_filepath = re.sub(pattern, "", filepath)
    return clean_filepath


class PLCheckpointWrapper(pl.callbacks.ModelCheckpoint):
    """Wrapper for PL ModelCheckpoint callback to upload checkpoints to s3."""

    def __init__(
        self,
        strategy: DDPStrategy,
        dirpath: str | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: str | None = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        s3_ckpt_path: str | None = None,
    ) -> None:
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            None,
            every_n_epochs,
            save_on_train_epoch_end,
        )
        self.s3_ckpt_path = s3_ckpt_path
        self.strategy = strategy

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if not self.s3_ckpt_path:
            return

        # upload checkpoint to s3
        if self.strategy and not isinstance(self.strategy, str):
            self.strategy.barrier()

        # Special characters are not allowed in AIchor bucket names. See: https://docs.aichor.ai/docs/user-manual/buckets/
        suffix = _clean_filepath(filepath.split("/")[-1])
        target = f"{self.s3_ckpt_path}/{suffix}"

        upload(filepath, target)


def _s3_enabled() -> bool:
    if "AICHOR_LOGS_PATH" in os.environ:
        assert "S3_ENDPOINT" in os.environ
        return True
    return False


def _create_s3fs() -> s3fs.core.S3FileSystem:
    url = os.environ.get("S3_ENDPOINT")
    logging.info(f"Creating s3fs.core.S3FileSystem, Endpoint: {url}")
    s3 = s3fs.core.S3FileSystem(client_kwargs={"endpoint_url": url})
    logging.info(f"Created s3fs.core.S3FileSystem: {s3}")

    return s3
