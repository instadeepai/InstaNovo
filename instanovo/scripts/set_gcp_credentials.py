# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "python-dotenv",
# ]
# ///
"""Set Google Storage credentials."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def to_base64(json_path: Path | str) -> str:
    """Convert a JSON file with GS credentials to a base64 encoded string."""
    with open(json_path) as f:
        credentials = f.read()
    return base64.b64encode(credentials.encode("ascii")).decode()


def set_credentials() -> None:
    """Set the GS credentials.

    - To access GCP buckets, the credentials are stored in a json file.
    - For runs on AIchor we only have access to the encoded string which should be decoded and saved

    Raises:
        OSError: if 'GOOGLE_APPLICATION_CREDENTIALS' is not set OR
                 if 'GOOGLE_APPLICATION_CREDENTIALS' file does not exist and
                 'GS_CREDENTIALS_ENCODED' is not set

    """
    try:
        gcp_crendentials_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    except KeyError:
        msg = (
            "To use GCP buckets you should set 'GOOGLE_APPLICATION_CREDENTIALS' env variable. "
            "It corresponds to the path to the json file with the credentials."
        )
        raise OSError(msg) from None

    if gcp_crendentials_path.exists():
        return

    try:
        gcp_credentials_encoded = os.environ["GS_CREDENTIALS_ENCODED"]
    except KeyError:
        msg = (
            "If the json file 'GOOGLE_APPLICATION_CREDENTIALS' does not exist, you must set 'GS_CREDENTIALS_ENCODED' as the base64 encoded json file."
        )
        raise OSError(msg) from None

    credentials = json.loads(base64.b64decode(gcp_credentials_encoded).decode())
    with open(gcp_crendentials_path, "w") as f:
        json.dump(credentials, f)
    print(f"Created {gcp_crendentials_path}")  # noqa: T201


if __name__ == "__main__":
    # print(to_base64('ext-dtu-denovo-sequencing-gcp-6cfd4324b948.json'))
    set_credentials()
