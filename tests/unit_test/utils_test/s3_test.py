import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from instanovo.utils.s3 import (
    PLCheckpointWrapper,
    S3FileHandler,
    _clean_filepath,
)


@patch("instanovo.utils.s3.S3FileHandler.s3_enabled", return_value=True)
@patch("instanovo.utils.s3.S3FileHandler._create_s3fs")
def test_s3_file_handler_upload(mock_create_s3fs: Any, mock_s3_enabled: Any) -> None:
    """Test uploading local file to s3 using S3FileHandler."""
    mock_s3 = MagicMock()
    mock_create_s3fs.return_value = mock_s3

    handler = S3FileHandler()
    handler.upload("source.txt", "s3://bucket/target.txt")

    mock_s3.put.assert_called_once_with("source.txt", "bucket/target.txt")


@patch("instanovo.utils.s3.S3FileHandler.s3_enabled", return_value=True)
@patch("instanovo.utils.s3.S3FileHandler._create_s3fs")
def test_s3_file_handler_download(mock_create_s3fs: Any, mock_s3_enabled: Any) -> None:
    """Test downloading file from s3 using S3FileHandler."""
    mock_s3 = MagicMock()
    mock_create_s3fs.return_value = mock_s3

    handler = S3FileHandler()
    handler.download("s3://bucket/source.txt", "target.txt")

    mock_s3.get.assert_called_once_with("bucket/source.txt", "target.txt")


@patch("instanovo.utils.s3.S3FileHandler.s3_enabled", return_value=True)
@patch("instanovo.utils.s3.S3FileHandler._create_s3fs")
def test_s3_file_handler_get_local_path(mock_create_s3fs: Any, mock_s3_enabled: Any) -> None:
    """Test getting local path for s3 file using S3FileHandler."""
    mock_s3 = MagicMock()
    mock_create_s3fs.return_value = mock_s3
    mock_s3.exists.return_value = True

    handler = S3FileHandler()
    local_path = handler.get_local_path("s3://bucket/source.txt")

    assert local_path is not None
    assert local_path.startswith(tempfile.gettempdir())
    mock_s3.get.assert_called_once()


@patch("instanovo.utils.s3.S3FileHandler.s3_enabled", return_value=True)
@patch("instanovo.utils.s3.S3FileHandler._create_s3fs")
def test_s3_file_handler_get_local_path_missing(
    mock_create_s3fs: Any, mock_s3_enabled: Any
) -> None:
    """Test getting local path for missing s3 file using S3FileHandler."""
    mock_s3 = MagicMock()
    mock_create_s3fs.return_value = mock_s3
    mock_s3.exists.return_value = False

    handler = S3FileHandler()
    with pytest.raises(FileNotFoundError):
        handler.get_local_path("s3://bucket/source.txt")


@patch("instanovo.utils.s3.S3FileHandler.s3_enabled", return_value=True)
@patch("instanovo.utils.s3.S3FileHandler._create_s3fs")
def test_s3_file_handler_get_local_path_missing_ok(
    mock_create_s3fs: Any, mock_s3_enabled: Any
) -> None:
    """Test getting local path for missing s3 file with missing_ok=True using S3FileHandler."""
    mock_s3 = MagicMock()
    mock_create_s3fs.return_value = mock_s3
    mock_s3.exists.return_value = False

    handler = S3FileHandler()
    local_path = handler.get_local_path("s3://bucket/source.txt", missing_ok=True)
    assert local_path is None


@patch("instanovo.utils.s3.S3FileHandler._aichor_enabled", return_value=True)
@patch.dict(os.environ, {"AICHOR_OUTPUT_PATH": "s3://output/"})
def test_s3_file_handler_convert_to_s3_output(mock_aichor_enabled: Any) -> None:
    """Tests conversion of local directory to s3 output path using S3FileHandler."""
    assert S3FileHandler.convert_to_s3_output("/local/path") == "s3://output/local/path"


def test_clean_filepath() -> None:
    """Tests conversion of file path to AIchor compatible path."""
    filepath = "some/file@path#withchars-epoch=2000.txt"
    assert _clean_filepath(filepath) == "somefilepathwithchars-epoch2000.txt"


def test_s3_enabled() -> None:
    """Tests environment variable checks."""
    with patch.dict(os.environ, {}, clear=True):
        assert not S3FileHandler.s3_enabled()

    with patch.dict(os.environ, {"S3_ENDPOINT": "example/end_point"}, clear=True):
        assert S3FileHandler.s3_enabled()


@patch.dict(os.environ, {"S3_ENDPOINT": "example/end_point"})
@patch("s3fs.core.S3FileSystem")
def test_create_s3fs(mock_s3_file_system: Any) -> None:
    """Tests s3 creation."""
    mock_s3_instance = MagicMock()
    mock_s3_file_system.return_value = mock_s3_instance

    s3_instance = S3FileHandler._create_s3fs()

    mock_s3_file_system.assert_called_once_with(client_kwargs={"endpoint_url": "example/end_point"})
    assert s3_instance == mock_s3_instance


@patch("instanovo.utils.s3.S3FileHandler")
def test_pl_wrapper(mock_s3_handler: Any) -> None:
    """Tests PL ModelCheckpoint calls."""
    mock_s3_instance = MagicMock()
    mock_s3_handler.return_value = mock_s3_instance

    pl_wrapper = PLCheckpointWrapper(
        strategy="strategy", s3_ckpt_path="model/path", s3=mock_s3_instance
    )

    assert pl_wrapper.strategy == "strategy"
    assert pl_wrapper.s3_ckpt_path == "model/path"
    assert pl_wrapper.s3 == mock_s3_instance

    mock_trainer = MagicMock()
    filepath = "path/to/checkpoint=0000$7.ckpt"
    pl_wrapper._save_checkpoint(mock_trainer, filepath)

    mock_s3_instance.upload.assert_called_once_with(filepath, "model/path/checkpoint00007.ckpt")
