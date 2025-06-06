from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import lightning as L
import s3fs
from lightning.pytorch.strategies import DDPStrategy
from tensorboard.compat.tensorflow_stub.io.gfile import _REGISTERED_FILESYSTEMS, register_filesystem

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class S3FileHandler:
    """A utility class for handling files stored locally or on S3.

    Attributes:
        temp_dir (tempfile.TemporaryDirectory): A temporary directory
            for storing downloaded S3 files.
    """

    def __init__(self) -> None:
        """Initializes the S3FileHandler."""
        self.s3 = S3FileHandler._create_s3fs()
        self.temp_dir = tempfile.TemporaryDirectory()

    @staticmethod
    def s3_enabled() -> bool:
        """Check if s3 is environment variable is present."""
        return "S3_ENDPOINT" in os.environ

    @staticmethod
    def _create_s3fs() -> s3fs.core.S3FileSystem | None:
        if not S3FileHandler.s3_enabled():
            return None

        url = os.environ.get("S3_ENDPOINT")
        logger.info(f"Creating s3fs.core.S3FileSystem, Endpoint: {url}")
        s3 = s3fs.core.S3FileSystem(client_kwargs={"endpoint_url": url})
        logger.info(f"Created s3fs.core.S3FileSystem: {s3}")

        return s3

    def _download_from_s3(self, s3_path: str) -> str:
        """Downloads a file from S3 to a temporary directory.

        Args:
            s3_path (str): The S3 path of the file.

        Returns:
            str: The local file path where the file is saved.
        """
        if self.s3 is None:
            return s3_path
        parsed = urlparse(s3_path)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        local_path = os.path.join(self.temp_dir.name, os.path.basename(s3_path))

        logger.info(f"Downloading {bucket}/{key} to {local_path}")
        self.s3.get(f"{bucket}/{key}", local_path)
        return local_path

    def download(self, s3_path: str, local_path: str) -> None:
        """Downloads a local from S3.

        Args:
            s3_path (str): The source S3 path (e.g., s3://bucket/key).
            local_path (str): The path to the local file to be written.
        """
        if not s3_path.startswith("s3://") or self.s3 is None:
            return
        parsed = urlparse(s3_path)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        dir_path = os.path.dirname(local_path)
        if dir_path:  # Only create directories if there's a directory component
            os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Downloading {bucket}/{key} to {local_path}")
        self.s3.get(f"{bucket}/{key}", local_path)

    def get_local_path(self, path: str, missing_ok: bool = False) -> str | None:
        """Returns a local file path. If the input path is an S3 path, the file is downloaded first.

        Args:
            path (str): The local or S3 path.

        Returns:
            str: The local file path.
        """
        if path.startswith("s3://") and self.s3 is not None:
            if not self.s3.exists(path):
                if missing_ok:
                    return None
                else:
                    raise FileNotFoundError(f"Could not find {path}.")

            local_path = os.path.join(self.temp_dir.name, os.path.basename(path))
            self.download(path, local_path)
            return local_path
        return path  # Already a local path

    def upload(self, local_path: str, s3_path: str) -> None:
        """Uploads a local file to S3.

        Args:
            local_path (str): The path to the local file.
            s3_path (str): The destination S3 path (e.g., s3://bucket/key).
        """
        if not s3_path.startswith("s3://") or self.s3 is None:
            return
        parsed = urlparse(s3_path)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")

        logger.info(f"Uploading {local_path} to {bucket}/{key}")
        self.s3.put(local_path, f"{bucket}/{key}")

    def upload_to_s3_wrapper(
        self, save_func: Callable[..., Any], s3_path: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Calls a save function and uploads the resulting file to S3.

        Args:
            save_func (Callable[..., Any]): The function to save a file (e.g., torch.save).
            s3_path (str): The destination S3 path.
            *args: Additional positional arguments for the save function.
            **kwargs: Additional keyword arguments for the save function.
        """
        if not s3_path.startswith("s3://") or self.s3 is None:
            if os.path.dirname(s3_path):
                os.makedirs(os.path.dirname(s3_path), exist_ok=True)
            return save_func(s3_path, *args, **kwargs)

        local_path = os.path.join(self.temp_dir.name, os.path.basename(urlparse(s3_path).path))
        result = save_func(local_path, *args, **kwargs)
        self.upload(local_path, s3_path)
        return result

    @staticmethod
    def _aichor_enabled() -> bool:
        if "AICHOR_LOGS_PATH" in os.environ:
            assert "S3_ENDPOINT" in os.environ
            return True
        return False

    @staticmethod
    def register_tb() -> bool:
        """Register s3 filesystem to tensorboard."""
        if not S3FileHandler._aichor_enabled():
            return False
        fs = _REGISTERED_FILESYSTEMS["s3"]
        if fs._s3_endpoint is None:
            # Set custom S3 endpoint explicitly. Not sure why this isn't picked up here:
            # https://github.com/tensorflow/tensorboard/blob/153cc747fdbeca3545c81947d4880d139a185c52/tensorboard/compat/tensorflow_stub/io/gfile.py#L227
            fs._s3_endpoint = os.environ["S3_ENDPOINT"]
        register_filesystem("s3", fs)
        return True

    @staticmethod
    def convert_to_s3_output(path: str) -> str:
        """Convert local directory to s3 output path if possible."""
        if S3FileHandler._aichor_enabled():
            # Environment variable specific to Aichor compute platform
            output_path = os.environ["AICHOR_OUTPUT_PATH"]
            if "s3://" not in output_path:
                output_path = f"s3://{output_path}/output"

            if path.startswith("/"):
                path = str(Path(path).relative_to("/"))

            return os.path.join(output_path, path)
        return path

    def cleanup(self) -> None:
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def __del__(self) -> None:
        self.cleanup()


def _clean_filepath(filepath: str) -> str:
    """Convert file path to AIchor compatible path, without disallowed characters."""
    pattern = r"[^a-zA-Z0-9\-_\.]"
    clean_filepath = re.sub(pattern, "", filepath)
    return clean_filepath


class PLCheckpointWrapper(L.pytorch.callbacks.ModelCheckpoint):
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
        s3: S3FileHandler | None = None,
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
        self.s3 = s3
        self.strategy = strategy

    def _save_checkpoint(self, trainer: L.pytorch.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if self.s3_ckpt_path is None or self.s3 is None:
            return

        # upload checkpoint to s3
        if self.strategy and not isinstance(self.strategy, str):
            self.strategy.barrier()

        # Special characters are not allowed in AIchor bucket names. See: https://docs.aichor.ai/docs/user-manual/buckets/
        suffix = _clean_filepath(filepath.split("/")[-1])
        target = f"{self.s3_ckpt_path}/{suffix}"

        self.s3.upload(filepath, target)


def _create_s3fs() -> s3fs.core.S3FileSystem:
    url = os.environ.get("S3_ENDPOINT")
    logging.info(f"Creating s3fs.core.S3FileSystem, Endpoint: {url}")
    s3 = s3fs.core.S3FileSystem(client_kwargs={"endpoint_url": url})
    logging.info(f"Created s3fs.core.S3FileSystem: {s3}")
    return s3
