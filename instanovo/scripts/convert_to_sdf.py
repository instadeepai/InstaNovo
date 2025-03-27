# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "instanovo",
#     "typer",
# ]
# ///
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from instanovo.__init__ import console
from instanovo.utils import SpectrumDataFrame
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

app = typer.Typer()


class Partition(str, Enum):
    """Partition of saved dataset."""

    train = "train"
    valid = "valid"
    test = "test"


@app.command()
def convert(
    source: Annotated[str, typer.Argument(help="Source file(s)")],
    target: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, dir_okay=True, help="Target folder to save data shards"
        ),
    ],
    name: Annotated[Optional[str], typer.Option(help="Name of saved dataset")],
    partition: Annotated[Partition, typer.Option(help="Partition of saved dataset")],
    max_charge: Annotated[int, typer.Option(help="Maximum charge to filter out")] = 10,
    shard_size: Annotated[int, typer.Option(help="Length of saved data shards")] = 1_000_000,
    is_annotated: Annotated[
        bool, typer.Option("--is-annotated", help="whether dataset is annotated")
    ] = False,
) -> None:
    """Convert data to SpectrumDataFrame and save as *.parquet file(s)."""
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading {source}")
    sdf = SpectrumDataFrame.load(
        source,
        is_annotated=is_annotated,
        name=name,
        partition=partition.value,
        max_shard_size=shard_size,
        lazy=True,
    )
    logger.info(f"Loaded {len(sdf):,d} rows")

    logger.info(f"Filtering max_charge <= {max_charge}")
    sdf.filter_rows(lambda row: row["precursor_charge"] <= max_charge)

    logger.info(f"Saving {len(sdf):,d} rows to {target}")
    sdf.save(
        target,
        name=name,
        partition=partition.value,
        max_shard_size=shard_size,
    )

    logger.info("Saving complete.")
    del sdf


if __name__ == "__main__":
    app()
