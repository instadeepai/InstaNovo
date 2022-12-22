from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Generator

import pytest
from cloudpathlib.local import LocalS3Client
from cloudpathlib.local import LocalS3Path

from dtu_denovo_sequencing.bucket.s3 import S3BucketManager


@pytest.fixture(scope="module")
def tmp_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create tmp directory to store all the files linked to the tests."""
    # mypy does not understand the output type
    return tmp_path_factory.mktemp("s3_test")  # type: ignore


@pytest.fixture(scope="module")
def local_s3_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tmp directory to mock local S3."""
    return tmp_path_factory.mktemp("local_s3")  # type: ignore


@pytest.fixture(scope="module", autouse=True)
def monkey_module() -> Generator:
    """Fixture to use monkeypatch at module scope.

    Inspiration from https://stackoverflow.com/a/53963978/11194702.
    """
    from _pytest.monkeypatch import MonkeyPatch

    monkey_patch = MonkeyPatch()

    yield monkey_patch

    monkey_patch.undo()


@pytest.fixture(scope="module", autouse=True)
def _bucket_client_mock(monkey_module: pytest.MonkeyPatch, local_s3_directory: Path) -> None:
    """Fixture to mock the bucket clients for all tests.

    We mock the cloudpath advised in:
    https://cloudpathlib.drivendata.org/stable/testing_mocked_cloudpathlib/
    """
    LocalS3Client._default_storage_temp_dir = local_s3_directory
    monkey_module.setattr("dtu_denovo_sequencing.bucket.s3.S3Client", LocalS3Client)
    monkey_module.setattr("dtu_denovo_sequencing.bucket.s3.S3Path", LocalS3Path)


@contextlib.contextmanager
def s3_ctx() -> Generator:
    """Context manager to temporarily set required env variables with dummy values."""
    old_environ = dict(os.environ)
    os.environ.update(
        {
            "AWS_ACCESS_KEY_ID": "dummy_key_id",
            "AWS_SECRET_ACCESS_KEY": "dummy_access_key",
            "S3_ENDPOINT": "dummy_endpoint",
        }
    )

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def test_upload_local_file_bucket_file(tmp_directory: Path) -> None:
    """Ensure the upload is working as expected when bucket_path + local_path are files."""
    local_path = tmp_directory / "my_file_to_upload.txt"
    local_path.touch()
    bucket_path = "s3://my_bucket/directory/my_file_uploaded.txt"

    expected_s3_path = bucket_path

    with s3_ctx():
        s3 = S3BucketManager()
        s3_path = s3.upload(local_path=local_path, bucket_path=bucket_path)

    assert s3_path == expected_s3_path
    assert LocalS3Path(s3_path).exists()


def test_upload_local_file_bucket_directory(tmp_directory: Path) -> None:
    """Ensure the upload is working as expected when bucket_path is a directory and local_path a file.

    The file uploaded must have the same file name as local_path.
    """
    local_path = tmp_directory / "my_file_to_upload.txt"
    local_path.touch()
    bucket_path = "s3://my_bucket/directory"

    expected_s3_path = f"{bucket_path}/{local_path.name}"

    with s3_ctx():
        s3 = S3BucketManager()
        s3_path = s3.upload(local_path=local_path, bucket_path=bucket_path)

    assert s3_path == expected_s3_path
    assert LocalS3Path(s3_path).exists()


def test_upload_local_directory_bucket_directory(tmp_directory: Path) -> None:
    """Ensure the upload is working as expected when bucket_path and local_path are directories.

    All the files and sub directories of local_path must be uploaded to bucket_path.
    """
    local_path = tmp_directory / "directory_to_upload"
    local_path.mkdir()

    local_file1 = local_path / "file1.txt"
    local_file2 = local_path / "file2.txt"

    local_sub_directory = local_path / "sub_directory"
    local_sub_directory.mkdir()
    local_sub_directory_file = local_sub_directory / "file.txt"

    local_file1.touch()
    local_file2.touch()
    local_sub_directory_file.touch()

    bucket_path = "s3://my_bucket/directory_uploaded"

    expected_s3_path = bucket_path

    with s3_ctx():
        s3 = S3BucketManager()
        s3_path = s3.upload(local_path=local_path, bucket_path=bucket_path)

    assert s3_path == expected_s3_path
    assert (LocalS3Path(s3_path) / local_file1.name).exists()
    assert (LocalS3Path(s3_path) / local_file2.name).exists()
    assert (
        LocalS3Path(s3_path) / local_sub_directory_file.parent.name / local_sub_directory_file.name
    ).exists()


def test_download_local_file_bucket_file(tmp_directory: Path) -> None:
    """Ensure the download is working as expected when bucket_path + local_path are files."""
    local_path = tmp_directory / "my_file_downloaded.txt"
    bucket_path = LocalS3Path("s3://my_bucket/my_file_to_download.txt")
    bucket_path.touch()
    expected_local_path = str(local_path)

    with s3_ctx():
        s3 = S3BucketManager()
        output_local_path = s3.download(local_path=local_path, bucket_path=str(bucket_path))

    assert expected_local_path == output_local_path
    assert Path(expected_local_path).exists()


def test_download_local_directory_bucket_file(tmp_directory: Path) -> None:
    """Ensure the download is working as expected when bucket_path is a directory and local_path a file.

    The file download must have the same file name as bucket_path.
    """
    local_path = tmp_directory / "my_target_directory_to_download_file"
    bucket_path = LocalS3Path("s3://my_bucket/my_file_to_download.txt")
    bucket_path.touch()
    expected_local_path = str(local_path / bucket_path.name)

    with s3_ctx():
        s3 = S3BucketManager()
        output_local_path = s3.download(local_path=local_path, bucket_path=str(bucket_path))

    assert expected_local_path == output_local_path
    assert Path(expected_local_path).exists()


def test_download_local_directory_bucket_directory(tmp_directory: Path) -> None:
    """Ensure the download is working as expected when bucket_path and local_path are directories.

    All the files and sub directories of bucket_path must be uploaded to local_path.
    """
    local_path = tmp_directory / "my_target_directory_to_download_directory"
    bucket_path = LocalS3Path("s3://my_bucket/directory_to_download")

    bucket_path.mkdir()
    bucket_file1 = bucket_path / "file1.txt"
    bucket_file2 = bucket_path / "file2.txt"

    bucket_sub_directory = bucket_path / "sub_directory_to_download"
    bucket_sub_directory.mkdir()
    bucket_sub_directory_file = bucket_sub_directory / "file.txt"

    bucket_file1.touch()
    bucket_file2.touch()
    bucket_sub_directory_file.touch()

    expected_local_path = str(local_path)

    with s3_ctx():
        s3 = S3BucketManager()
        output_local_path = s3.download(local_path=local_path, bucket_path=str(bucket_path))

    assert expected_local_path == output_local_path
    assert (Path(output_local_path) / bucket_file1.name).exists()
    assert (Path(output_local_path) / bucket_file2.name).exists()
    assert (
        Path(output_local_path)
        / bucket_sub_directory_file.parent.name
        / bucket_sub_directory_file.name
    ).exists()
