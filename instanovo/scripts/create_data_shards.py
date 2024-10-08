# DO NOT MERGE TO GITHUB

from __future__ import annotations

import os
import pickle
import zipfile
from typing import Iterator

import click
import pandas as pd
import polars as pl
import tqdm

from instanovo.transformer.dataset import remove_modifications


def get_shards(
    path: str, seq_filter: list[str] | set[str], max_shard_size: int
) -> Iterator[pd.DataFrame]:
    """Load all zipped pickle files."""
    current_shard = None
    for filename in tqdm.tqdm(os.listdir(path)):
        dst = os.path.join(path, filename)
        with zipfile.ZipFile(dst, "r") as f:
            df = pickle.loads(f.read(f.namelist()[0]))
            df = df[df["Sequence"].map(remove_modifications).isin(seq_filter)].copy()
            df["Exp"] = "_".join(filename.split("_")[:-3])
            if current_shard is None:
                current_shard = df
            elif len(current_shard) + len(df) < max_shard_size:
                current_shard = pd.concat([current_shard, df])
            else:
                yield pd.concat(
                    [current_shard, df[: (max_shard_size - len(current_shard))]]
                )
                current_shard = df[(max_shard_size - len(current_shard)) :]
    yield current_shard


@click.command()
@click.option("--data-path", "-d")
@click.option("--output-path", "-o")
@click.option("--splits-file", "-f")
@click.option("--split", "-s", default="train")
@click.option("--max-size", "-m", default=1_000_000)
@click.option("--remap-columns", "-r", is_flag=True, default=False)
def main(
    data_path: str,
    output_path: str,
    splits_file: str,
    split: str,
    max_size: int,
    remap_columns: bool,
) -> None:
    """Split pickle files into .ipc data shards."""
    splits = pd.read_csv(splits_file)

    split_options = splits["split"].unique()
    if split not in split_options:
        raise ValueError(
            f"Unknown split {split}. Please select one of {list(split_options)}."
        )

    # TODO: find a better way to configure this.
    col_map = {
        "Modified sequence": "modified_sequence",
        "MS/MS m/z": "precursor_mz",
        "Sequence": "sequence",
        "Precursor m/z": "precursor_mz",
        "Theoretical m/z": "theoretical_mz",
        "Mass": "precursor_mass",
        "Charge": "precursor_charge",
        "Mass values": "mz_array",
        "Mass spectrum": "mz_array",
        "Intensity": "intensity_array",
        "Raw intensity spectrum": "intensity_array",
    }

    drop_cols = [
        "Normalized intensity spectrum",
        "Mass matches",
        "Mass intensities",
        "Mass errors",
        "Matches",
    ]

    filter_sequences = set(splits[splits["Split"] == split]["Sequence"].to_list())
    shards = get_shards(
        path=data_path, seq_filter=filter_sequences, max_shard_size=max_size
    )
    os.makedirs(output_path, exist_ok=True)
    for index, shard in enumerate(shards):
        polars_df = pl.DataFrame(shard)
        if remap_columns:
            polars_df = polars_df.rename(
                {k: v for k, v in col_map.items() if k in polars_df.columns}
            )
            polars_df = polars_df.drop(
                [col for col in drop_cols if col in polars_df.columns]
            )
            polars_df = polars_df.with_columns(
                pl.col("modified_sequence").apply(lambda x: x[1:-1])
            )
        polars_df.write_ipc(os.path.join(output_path, f"{split}_shard_{index}.ipc"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
