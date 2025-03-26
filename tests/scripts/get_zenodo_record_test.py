import os
import zipfile
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from rich.progress import Progress
from typer.testing import CliRunner

from instanovo.scripts.get_zenodo_record import RECORD_ID, app, get_zenodo, main, unzip_zenodo


@pytest.fixture
def mock_response() -> Mock:
    """Create a mock response for requests.get."""
    mock = Mock()
    mock.headers.get.return_value = "1024"  # Mock content length
    # Create sample data to simulate chunks
    mock.iter_content.return_value = [b"test data" * 100]
    return mock


@pytest.fixture
def mock_zipfile() -> MagicMock:
    """Create a mock zipfile."""
    mock = MagicMock(spec=zipfile.ZipFile)
    return mock


@pytest.fixture
def runner() -> CliRunner:
    """Create a Typer CLI test runner."""
    return CliRunner()


class TestZenodoDownload:
    """Test cases for zenodo download functions."""

    def test_get_zenodo_success(self, mock_response: Mock, tmp_path: Any) -> None:
        """Test successful download of a Zenodo record."""
        # Setup
        zenodo_url: str = f"https://zenodo.org/records/{RECORD_ID}/files/test.zip"
        zip_path: str = str(tmp_path / "test.zip")
        progress: MagicMock = MagicMock(spec=Progress)
        task_id: int = 1

        # Mock requests.get
        with patch("requests.get", return_value=mock_response):
            # Execute
            get_zenodo(zenodo_url, zip_path, progress, task_id)

            # Assert
            assert os.path.exists(zip_path)
            # Check progress was updated with total size
            progress.update.assert_any_call(task_id, total=1024)
            # Check progress was updated for each chunk
            progress.update.assert_any_call(task_id, advance=len(b"test data" * 100))

    def test_get_zenodo_request_exception(self) -> None:
        """Test handling of request exception during download."""
        # Setup
        zenodo_url: str = f"https://zenodo.org/records/{RECORD_ID}/files/test.zip"
        zip_path: str = "test.zip"
        progress: MagicMock = MagicMock(spec=Progress)
        task_id: int = 1

        # Mock requests.get to raise an exception
        with patch("requests.get", side_effect=requests.RequestException("Network error")):
            # Execute and assert
            with pytest.raises(requests.RequestException):
                get_zenodo(zenodo_url, zip_path, progress, task_id)

    def test_unzip_zenodo_success(self, tmp_path: Any) -> None:
        """Test successful extraction of a zip file."""
        # Setup
        zip_path: str = str(tmp_path / "test.zip")
        extract_path: str = str(tmp_path / "extracted")

        # Create a real test zip file
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.writestr("test.txt", "This is test content")

        # Execute
        unzip_zenodo(zip_path, extract_path)

        # Assert
        assert os.path.exists(extract_path)
        assert os.path.exists(os.path.join(extract_path, "test.txt"))

    def test_unzip_zenodo_bad_zipfile(self, tmp_path: Any) -> None:
        """Test handling of a bad zip file."""
        # Setup
        zip_path: str = str(tmp_path / "bad.zip")
        extract_path: str = str(tmp_path / "extracted")

        # Create an invalid zip file
        with open(zip_path, "w") as f:
            f.write("This is not a valid zip file")

        # Execute and assert
        with pytest.raises(zipfile.BadZipFile):
            unzip_zenodo(zip_path, extract_path)


class TestMainCommand:
    """Test cases for the main Typer command."""

    def test_cli_successful_execution(self, runner: CliRunner, tmp_path: Any) -> None:
        """Test successful execution of the CLI command."""
        # Setup
        zip_path: str = str(tmp_path / "test.zip")
        extract_path: str = str(tmp_path / "extracted")

        # Mock the functions to avoid actual network/file operations
        with patch("instanovo.scripts.get_zenodo_record.get_zenodo") as mock_get:
            with patch("instanovo.scripts.get_zenodo_record.unzip_zenodo") as mock_unzip:
                with patch("os.path.exists", return_value=False):
                    with patch("os.listdir", return_value=[]):
                        # Execute CLI command
                        result = runner.invoke(
                            app,
                            [
                                "--zenodo-url",
                                f"https://zenodo.org/records/{RECORD_ID}/files/test.zip",
                                "--zip-path",
                                zip_path,
                                "--extract-path",
                                extract_path,
                            ],
                        )

        # Assert
        assert result.exit_code == 0
        mock_get.assert_called_once()
        mock_unzip.assert_called_once()

    def test_cli_skips_when_directory_exists(self, runner: CliRunner, tmp_path: Any) -> None:
        """Test CLI skips download when target directory exists."""
        # Setup - create directory and a file
        extract_path: str = str(tmp_path)
        resource_dir: str = os.path.join(extract_path, "instanovo_test_resources")
        os.makedirs(resource_dir, exist_ok=True)
        with open(os.path.join(resource_dir, "dummy.txt"), "w") as f:
            f.write("test")

        # Execute CLI command
        result = runner.invoke(
            app,
            [
                "--zenodo-url",
                f"https://zenodo.org/records/{RECORD_ID}/files/test.zip",
                "--zip-path",
                "test.zip",
                "--extract-path",
                extract_path,
            ],
        )

        # Assert
        assert result.exit_code == 0
        assert "Skipping download" in result.stdout

    def test_main_with_parameters(self) -> None:
        """Test main function with different parameters."""
        with patch("instanovo.scripts.get_zenodo_record.get_zenodo") as mock_get:
            with patch("instanovo.scripts.get_zenodo_record.unzip_zenodo") as mock_unzip:
                with patch("os.path.exists", return_value=False):
                    with patch("os.listdir", return_value=[]):
                        # We need to patch the Progress creation and usage
                        with patch("rich.progress.Progress"):
                            # Test parameters
                            zenodo_url: str = "https://zenodo.org/records/12345/files/test.zip"
                            zip_path: str = "custom.zip"
                            extract_path: str = "./custom"

                            main(
                                zenodo_url=zenodo_url, zip_path=zip_path, extract_path=extract_path
                            )

        # Verify that the functions were called with correct parameters
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == zenodo_url
        assert mock_get.call_args[0][1] == zip_path

        mock_unzip.assert_called_once()
        assert mock_unzip.call_args[0][0] == zip_path
        assert mock_unzip.call_args[0][1] == extract_path


@pytest.mark.parametrize(
    "zenodo_url,zip_path,extract_path",
    [
        (f"https://zenodo.org/records/{RECORD_ID}/files/custom.zip", "custom.zip", "./custom"),
        ("https://zenodo.org/records/12345/files/test.zip", "other.zip", "./other"),
    ],
)
def test_cli_with_different_parameters(
    runner: CliRunner, zenodo_url: str, zip_path: str, extract_path: str
) -> None:
    """Test CLI with different parameters."""
    with patch("instanovo.scripts.get_zenodo_record.get_zenodo") as mock_get:
        with patch("instanovo.scripts.get_zenodo_record.unzip_zenodo") as mock_unzip:
            with patch("os.path.exists", return_value=False):
                with patch("os.listdir", return_value=[]):
                    # Execute CLI command with different parameters
                    result = runner.invoke(
                        app,
                        [
                            "--zenodo-url",
                            zenodo_url,
                            "--zip-path",
                            zip_path,
                            "--extract-path",
                            extract_path,
                        ],
                    )

    # Assert
    assert result.exit_code == 0
    mock_get.assert_called_once()
    mock_unzip.assert_called_once()
