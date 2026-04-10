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
        typer.Argument(exists=True, file_okay=False, dir_okay=True, help="Target folder to save data shards"),
    ],
    name: Annotated[Optional[str], typer.Option(help="Name of saved dataset")],
    partition: Annotated[Partition, typer.Option(help="Partition of saved dataset")],
    max_charge: Annotated[int, typer.Option(help="Maximum charge to filter out")] = 10,
    shard_size: Annotated[int, typer.Option(help="Length of saved data shards")] = 1_000_000,
    is_annotated: Annotated[bool, typer.Option("--is-annotated", help="whether dataset is annotated")] = False,
    add_spectrum_id: Annotated[bool, typer.Option("--add-spectrum-id", help="Add spectrum id column")] = False,
    include_ms1: Annotated[
        bool,
        typer.Option(
            "--include-ms1", help="Include MS1 scans alongside MS2. Adds ms_level column (1=MS1, 2=MS2). Precursor columns are empty for MS1 rows."
        ),
    ] = False,
) -> None:
    """Convert data to SpectrumDataFrame and save as *.parquet file(s)."""
    from instanovo.utils.data_handler import SpectrumDataFrame

    logging.basicConfig(level=logging.INFO)

    ms_label = " (MS1+MS2)" if include_ms1 else " (MS2 only)"

    logger.info(f"Loading {source}{ms_label}")

    if include_ms1:
        ms_levels: list[int] = [1, 2]
        # Direct load with MS1+MS2 support — only works for mzML/mzXML files
        source_lower = source.lower()
        if source_lower.endswith(".mzml"):
            sdf = SpectrumDataFrame.load_mzml(source, ms_levels=ms_levels)
        elif source_lower.endswith(".mzxml"):
            from instanovo.utils.msreader import read_mzxml

            sdf = SpectrumDataFrame.from_polars(SpectrumDataFrame._df_from_dict(read_mzxml(source, ms_levels=ms_levels)))
        else:
            logger.warning("--include-ms1 only applies to mzML/mzXML files. Ignoring flag.")
            sdf = SpectrumDataFrame.load(
                source,
                is_annotated=is_annotated,
                name=name,
                partition=partition.value,
                max_shard_size=shard_size,
                lazy=True,
                add_spectrum_id=add_spectrum_id,
            )
    else:
        sdf = SpectrumDataFrame.load(
            source,
            is_annotated=is_annotated,
            name=name,
            partition=partition.value,
            max_shard_size=shard_size,
            lazy=True,
            add_spectrum_id=add_spectrum_id,
        )

    logger.info(f"Loaded {len(sdf):,d} rows")

    if include_ms1:
        # Filter MS2 rows by charge but keep all MS1 rows
        logger.info(f"Filtering MS2 rows with max_charge <= {max_charge} (keeping all MS1 rows)")
    else:
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
