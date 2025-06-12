import os
from pathlib import Path

import requests
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


def download_file(url: str, path: Path, model_id: str, file_name: str) -> None:
    """Download a file with a progress bar.

    Args:
        url (str):
            The URL to download the file from.
        path (Path):
            The path to save the file to.
        model_id (str):
            The model ID.
        file_name (str):
            The name of the file to download.
    """
    # If not cached, download the file with a progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    logger.info(f"Downloading model {model_id} from {url}")

    with (
        open(path, "wb") as file,
        tqdm(
            desc=file_name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    if not os.path.getsize(path) == total_size:
        raise ValueError(
            f"Downloaded file is incomplete. Expected size of {total_size} "
            "bytes does not match downloaded size of "
            f"{os.path.getsize(path)} bytes."
        )

    logger.info(f"Cached model {model_id} at {path}")
