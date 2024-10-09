import os
from typing import Any
from unittest.mock import patch, mock_open, MagicMock

import pytest

from instanovo.utils.s3 import (
    upload,
    download,
    get_checkpoint_path,
    convert_to_s3_output,
    _clean_filepath,
    _s3_enabled,
    _create_s3fs,
    PLCheckpointWrapper,
)


@patch("instanovo.utils.s3._s3_enabled", return_value=True)
@patch("instanovo.utils.s3._create_s3fs")
@patch("builtins.open", new_callable=mock_open, read_data=b"data")
def test_upload(
    mock_open_local: Any, mock_create_s3fs: Any, mock_s3_enabled: Any
) -> None:
    """Test function that uploads local file and saves to s3."""
    mock_s3 = MagicMock()
    mock_create_s3fs.return_value = mock_s3

    mock_s3.open = mock_open()

    upload("source.txt", "s3://bucket/target.txt")

    mock_open_local.assert_called_with("source.txt", "rb")
    mock_open_local().read.assert_called_once()

    mock_s3.open.assert_called_with("s3://bucket/target.txt", "wb")
    mock_s3.open().write.assert_called_once_with(b"data")


@patch("instanovo.utils.s3._s3_enabled", return_value=True)
@patch("instanovo.utils.s3._create_s3fs")
@patch("builtins.open", new_callable=mock_open)
def test_download(
    mock_open_local: Any, mock_create_s3fs: Any, mock_s3_enabled: Any
) -> None:
    """Test function that downloads file from s3 and saves locally."""
    mock_s3 = MagicMock()
    mock_create_s3fs.return_value = mock_s3

    mock_s3.open = mock_open(read_data=b"data")

    download("s3://bucket/source.txt", "target.txt")

    mock_s3.open.assert_called_with("s3://bucket/source.txt", "rb")
    mock_s3.open().read.assert_called_once()

    mock_open_local.assert_called_with("target.txt", "wb")
    mock_open_local().write.assert_called_once_with(b"data")


@patch("instanovo.utils.s3._s3_enabled", return_value=True)
@patch("instanovo.utils.s3.os.makedirs")
@patch("instanovo.utils.s3.download")
@patch.dict(os.environ, {"AICHOR_INPUT_PATH": "/input/"})
def test_get_checkpoint_path(
    mock_download: Any, mock_makedirs: Any, mock_s3_enabled: Any
) -> None:
    """Test downloading checkpoint path from s3 and returning local path."""
    local_path = get_checkpoint_path("model.ckpt")
    assert local_path == "checkpoints/model.ckpt"

    mock_makedirs.assert_called_once_with("checkpoints", exist_ok=True)
    mock_download.assert_called_once_with(
        "/input/model.ckpt", target="checkpoints/model.ckpt"
    )


@patch("instanovo.utils.s3._s3_enabled", return_value=True)
@patch.dict(os.environ, {"AICHOR_OUTPUT_PATH": "s3://output/"})
def test_convert_to_s3_output(mock_s3_enabled: Any) -> None:
    """Tests conversion of local directory to s3 output path."""
    assert convert_to_s3_output("/local/path") == "s3://output//local/path"


def test_clean_filepath() -> None:
    """Tests conversion of file path to AIchor compatible path."""
    filepath = "some/file@path#withchars-epoch=2000.txt"
    assert _clean_filepath(filepath) == "somefilepathwithchars-epoch2000.txt"


def test_s3_enabled() -> None:
    """Tests environment variable checks."""
    with patch.dict(os.environ, {}, clear=True):
        assert not _s3_enabled()

    with patch.dict(
        os.environ, {"AICHOR_LOGS_PATH": "example/aichor/path"}, clear=True
    ):
        with pytest.raises(AssertionError):
            _s3_enabled()

    with patch.dict(
        os.environ,
        {"AICHOR_LOGS_PATH": "example/aichor/path", "S3_ENDPOINT": "example/end_point"},
        clear=True,
    ):
        assert _s3_enabled()


@patch.dict(os.environ, {"S3_ENDPOINT": "example/end_point"})
@patch("s3fs.core.S3FileSystem")
def test_create_s3fs(mock_s3_file_system: Any) -> None:
    """Tests s3 creation."""
    mock_s3_instance = MagicMock()
    mock_s3_file_system.return_value = mock_s3_instance

    s3_instance = _create_s3fs()

    mock_s3_file_system.assert_called_once_with(
        client_kwargs={"endpoint_url": "example/end_point"}
    )

    assert s3_instance == mock_s3_instance


@patch("instanovo.utils.s3.upload")
def test_pl_wrapper(mock_upload: Any) -> None:
    """Tests PL ModelCheckpoint calls."""
    pl_wrapper = PLCheckpointWrapper(strategy="strategy", s3_ckpt_path="model/path")

    assert pl_wrapper.strategy == "strategy"
    assert pl_wrapper.s3_ckpt_path == "model/path"

    mock_trainer = MagicMock()
    filepath = "path/to/checkpoint=0000$7.ckpt"
    pl_wrapper._save_checkpoint(mock_trainer, filepath)

    mock_upload.assert_called_once_with(
        "path/to/checkpoint=0000$7.ckpt", "model/path/checkpoint00007.ckpt"
    )
