# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
# ]
# ///
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


def update_ipc(source: Path, target: Path) -> pl.DataFrame:
    """Update .ipc file to new schema format."""
    if not source.suffix.lower().endswith("ipc"):
        raise ValueError("Incorrect file type - .ipc file required.")

    logger.info(f"Processing {source}.")

    df = pl.read_ipc(source=source)
    df = df.rename(
        {
            "Sequence": "sequence",
            "Modified sequence": "modified_sequence",
            "MS/MS m/z": "precursor_mz",
            "Charge": "precursor_charge",
            "Intensity": "intensity_array",
            "Mass values": "mz_array",
        }
    )

    df = df.with_columns(df["modified_sequence"].str.strip_chars("_"))

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(target)


def main() -> None:
    """Update ipc file."""
    parser = argparse.ArgumentParser()

    parser.add_argument("source", help="source file or folder")
    parser.add_argument("target", help="target ipc file to be saved")

    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)

    update_ipc(source, target)


if __name__ == "__main__":
    main()
