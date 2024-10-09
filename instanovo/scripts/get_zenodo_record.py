from __future__ import annotations

import logging
import os
import requests
import zipfile
from tqdm import tqdm

import click


logger = logging.getLogger()
logger.setLevel(logging.INFO)


RECORD_ID = "13898491"


def get_zenodo(zenodo_url: str, zip_path: str) -> None:
    """Fetches specified zenodo record."""
    try:
        response = requests.get(zenodo_url, stream=True)

        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
            with open(zip_path, "wb") as file:
                for data in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(data))
                    file.write(data)

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


@click.command()
@click.option(
    "--zenodo-url",
    default=f"https://zenodo.org/records/{RECORD_ID}/files/instanovo_test_resources.zip",
)
@click.option("--zip-path", default="./tests/instanovo_test_resources.zip")
@click.option("--extract-path", default="./tests")
def main(
    zenodo_url: str,
    zip_path: str,
    extract_path: str,
) -> None:
    """Downloads and extracts the zenodo record used for unit and integration tests."""
    if os.path.exists(extract_path + "/instanovo_test_resources") and os.listdir(
        extract_path + "/instanovo_test_resources"
    ):
        print(
            f"Skipping download and extraction. Path '{extract_path}'/instanovo_test_resources already exists and is non-empty."
        )
        return

    get_zenodo(zenodo_url, zip_path)
    unzip_zenodo(zip_path, extract_path)

    os.makedirs("./tests/instanovo_test_resources/train_test", exist_ok=True)


if __name__ == "__main__":
    main()
