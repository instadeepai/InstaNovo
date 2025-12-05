from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import s3fs
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

    def __init__(
        self, s3_endpoint: str | None = None, aws_access_key_id: str | None = None, aws_secret_access_key: str | None = None, verbose: bool = True
    ) -> None:
        """Initializes the S3FileHandler.

        Args:
            s3_endpoint: Optional S3 endpoint to use. If not provided,
                the S3_ENDPOINT environment variable will be used.
            aws_access_key_id: Optional AWS access key ID to use. If not provided,
                the AWS_ACCESS_KEY_ID environment variable will be used.
            aws_secret_access_key: Optional AWS secret access key to use. If not provided,
                the AWS_SECRET_ACCESS_KEY environment variable will be used.
            verbose: Whether to log verbose messages.
        """
        self.s3 = S3FileHandler._create_s3fs(
            verbose,
            s3_endpoint,
            aws_access_key_id,
            aws_secret_access_key,
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verbose = verbose

    @staticmethod
    def s3_enabled() -> bool:
        """Check if s3 is environment variable is present."""
        return "S3_ENDPOINT" in os.environ

    @staticmethod
    def _create_s3fs(
        verbose: bool = True,
        s3_endpoint: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> s3fs.core.S3FileSystem | None:
        if not S3FileHandler.s3_enabled() and s3_endpoint is None:
            return None

        if s3_endpoint is None:
            assert "S3_ENDPOINT" in os.environ
        if aws_access_key_id is None:
            assert "AWS_ACCESS_KEY_ID" in os.environ
        if aws_secret_access_key is None:
            assert "AWS_SECRET_ACCESS_KEY" in os.environ

        url = s3_endpoint or os.environ.get("S3_ENDPOINT")
        if verbose:
            logger.info(f"Creating s3fs.core.S3FileSystem, Endpoint: {url}")
        s3 = s3fs.core.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key, client_kwargs={"endpoint_url": url})
        if verbose:
            logger.info(f"Created s3fs.core.S3FileSystem: {s3}")

        return s3

    def _log_if_verbose(self, message: str) -> None:
        """Log a message if verbose logging is enabled.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            logger.info(message)

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

        self._log_if_verbose(f"Downloading {bucket}/{key} to {local_path}")
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
        self._log_if_verbose(f"Downloading {bucket}/{key} to {local_path}")
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

        self._log_if_verbose(f"Uploading {local_path} to {bucket}/{key}")
        self.s3.put(local_path, f"{bucket}/{key}")

    def upload_to_s3_wrapper(self, save_func: Callable[..., Any], s3_path: str, *args: Any, **kwargs: Any) -> Any:
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

    def listdir(self, path: str) -> list[str]:
        """List the contents of a directory on S3.

        Args:
            path (str): The path to the directory.
        """
        if not path.startswith("s3://") or self.s3 is None:
            return []

        parsed = urlparse(path)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        return self.s3.listdir(f"{bucket}/{key}", detail=False)  # type: ignore[no-any-return]

    @staticmethod
    def _aichor_enabled() -> bool:
        """Check if Aichor is enabled."""
        if "AICHOR_LOGS_PATH" in os.environ:
            assert "S3_ENDPOINT" in os.environ
            return True
        return False

    @staticmethod
    def register_tb() -> bool:
        """Register s3 filesystem to tensorboard.

        Returns:
            bool: Whether the registration was successful.
        """
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
        """Convert local directory to s3 output path if possible.

        Args:
            path (str): The local path to convert.

        Returns:
            str: The s3 output path.
        """
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
