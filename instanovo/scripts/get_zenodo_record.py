# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "typer",
#     "rich",
# ]
# ///

from __future__ import annotations

import logging
import os
import zipfile

import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


RECORD_ID = "17816199"


def get_zenodo(zenodo_url: str, zip_path: str, progress: Progress, task_id: TaskID) -> None:
    """Fetches specified zenodo record."""
    try:
        response = requests.get(zenodo_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        progress.update(task_id, total=total_size)

        with open(zip_path, "wb") as file:
            for data in response.iter_content(chunk_size=8192):
                file.write(data)
                progress.update(task_id, advance=len(data))

        logger.info(f"Zip file at url {zenodo_url} downloaded to {zip_path}.")

    except requests.RequestException as e:
        logger.error(f"Failed to download the Zenodo record: {e}")
        raise


def unzip_zenodo(zip_path: str, extract_path: str) -> None:
    """Extracts zip file to specified location."""
    try:
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        logger.info(f"Zip files extracted to {extract_path}")

    except zipfile.BadZipFile as e:
        logger.error(f"Failed to unzip the file: {e}")
        raise


app = typer.Typer()


@app.command()
def main(
    zenodo_url: str = typer.Option(
        f"https://zenodo.org/records/{RECORD_ID}/files/instanovo_test_resources.zip",
        help="URL of the Zenodo record to download",
    ),
    zip_path: str = typer.Option(
        "./tests/instanovo_test_resources.zip",
        help="Path where the downloaded zip file will be saved",
    ),
    extract_path: str = typer.Option("./tests/instanovo_test_resources", help="Path where the zip file contents will be extracted"),
) -> None:
    """Downloads and extracts the zenodo record used for unit and integration tests."""
    if os.path.exists(f"{extract_path}") and os.listdir(f"{extract_path}"):
        if os.path.exists(f"{extract_path}/record_id.txt"):
            with open(f"{extract_path}/record_id.txt", "r") as f:
                record_id = f.read().strip()
            if record_id == RECORD_ID:
                typer.echo(f"Record is up to date, skipping download and extraction. Path '{extract_path}' already exists and is non-empty.")
                raise typer.Exit()
            else:
                typer.echo("Record is outdated, downloading new record.")
        else:
            typer.echo("Record ID is not documented, downloading new record.")

    console = Console()  # Create a Console instance for Rich
    progress = Progress(
        DownloadColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,  # Pass the console instance to Progress
    )

    task_id = progress.add_task("download", filename=zip_path, start=False)

    with progress:
        progress.start_task(task_id)
        get_zenodo(zenodo_url, zip_path, progress, task_id)

    unzip_zenodo(zip_path, extract_path)

    with open(f"{extract_path}/record_id.txt", "w") as f:
        f.write(RECORD_ID)


if __name__ == "__main__":
    app()
