from __future__ import annotations

import logging
import os
import random
import re

import click
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def remove_modifications(x: str) -> str:
    """Remove modifications in peptide sequence."""
    x = re.findall(r"[A-Z]", x)
    return "".join(x)


def create_split_csv(directory: str) -> None:
    """Creates a csv file with the split tag for each sequence in the nine species dataset."""
    unique_sequences = set()
    for filename in os.listdir(directory):
        if filename.endswith(".ipc"):
            df = pl.read_ipc(os.path.join(directory, filename))
            df = df.with_columns(
                df["modified_sequence"]
                .map_elements(remove_modifications, return_dtype=str)
                .alias("sequence")
            )  # overwrite sequence column

            unique_sequences.update(df["sequence"].unique())
            del df

    # for reproducibility
    unique_sequences = sorted(unique_sequences)
    random.seed(42)
    random.shuffle(unique_sequences)

    train_sequences, temp_sequences = train_test_split(
        unique_sequences, test_size=0.2, random_state=42
    )
    val_sequences, test_sequences = train_test_split(
        temp_sequences, test_size=0.5, random_state=42
    )

    train_df = pd.DataFrame({"sequence": train_sequences, "split": "train"})
    val_df = pd.DataFrame({"sequence": val_sequences, "split": "valid"})
    test_df = pd.DataFrame({"sequence": test_sequences, "split": "test"})

    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv(os.path.join(directory, "species_split.csv"), index=False)
    logger.info(f"Total number of unique sequences processed: {len(unique_sequences)}")


def create_shards(
    directory: str, split_csv_path: str, holdout_file_path: str = "None"
) -> None:
    """Read in the species ipcs and create train, validate and test shards accordingly."""
    splits_df = pd.read_csv(split_csv_path)
    logger.info("Species split file read in.")

    train_dfs = []
    valid_dfs = []
    test_dfs = []

    for filename in os.listdir(directory):
        if filename.endswith(".ipc") and filename not in holdout_file_path:
            logger.info(f"Processing {filename}.")
            train_df, valid_df, test_df = process_splits(directory, filename, splits_df)

            check_split_duplicates(train_df, valid_df, test_df)

            train_dfs.append(train_df)
            valid_dfs.append(valid_df)
            test_dfs.append(test_df)

    if train_dfs:
        train_df = pl.concat(train_dfs)
        train_df.write_ipc(os.path.join(directory, "train.ipc"))

    if valid_dfs:
        valid_df = pl.concat(valid_dfs)
        valid_df.write_ipc(os.path.join(directory, "valid.ipc"))

    if holdout_file_path != "None":
        test_df = pl.read_ipc(os.path.join(holdout_file_path))
        test_dfs.append(test_df)

    if test_dfs:
        test_df = pl.concat(test_dfs)
        test_df.write_ipc(os.path.join(directory, "test.ipc"))

    check_split_duplicates(train_df, valid_df, test_df)


def process_splits(
    directory: str, filename: str, splits_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create the train, valid and test portions of a specified species."""
    df_species = pl.read_ipc(os.path.join(directory, filename))
    df_species = df_species.with_columns(
        df_species["modified_sequence"]
        .map_elements(remove_modifications, return_dtype=str)
        .alias("sequence")
    )

    df_species = df_species.with_columns(
        (
            pl.col("sequence").is_in(
                splits_df.loc[splits_df["split"] == "train", "sequence"]
            )
        ).alias("is_train"),
        (
            pl.col("sequence").is_in(
                splits_df.loc[splits_df["split"] == "valid", "sequence"]
            )
        ).alias("is_valid"),
        (
            pl.col("sequence").is_in(
                splits_df.loc[splits_df["split"] == "test", "sequence"]
            )
        ).alias("is_test"),
    )

    train_df = (
        df_species.filter(pl.col("is_train"))
        .drop("is_train")
        .drop("is_valid")
        .drop("is_test")
    )
    valid_df = (
        df_species.filter(pl.col("is_valid"))
        .drop("is_train")
        .drop("is_valid")
        .drop("is_test")
    )
    test_df = (
        df_species.filter(pl.col("is_test"))
        .drop("is_train")
        .drop("is_valid")
        .drop("is_test")
    )

    return train_df, valid_df, test_df


def check_split_duplicates(
    train_df: pl.DataFrame, valid_df: pl.DataFrame, test_df: pl.DataFrame
) -> None:
    """Check for duplicate sequences between train, valid, and test datasets."""
    train_sequences = set(train_df["sequence"].to_list())
    valid_sequences = set(valid_df["sequence"].to_list())
    test_sequences = set(test_df["sequence"].to_list())

    train_valid_overlap = train_sequences.intersection(valid_sequences)
    train_test_overlap = train_sequences.intersection(test_sequences)
    valid_test_overlap = valid_sequences.intersection(test_sequences)

    if train_valid_overlap:
        logger.warning(
            f"Duplicates found between train and valid sets: {len(train_valid_overlap)} sequences"
        )

    if train_test_overlap:
        logger.warning(
            f"Duplicates found between train and test sets: {len(train_test_overlap)} sequences"
        )

    if valid_test_overlap:
        logger.warning(
            f"Duplicates found between valid and test sets: {len(valid_test_overlap)} sequences"
        )


def check_csv_duplicates(df: pd.DataFrame) -> None:
    """Check for duplicates in the 'sequence' column and print the result."""
    duplicated_sequences = df[df.duplicated(subset=["sequence"], keep=False)]
    duplicated_count = duplicated_sequences.shape[0]
    total_sequences = df.shape[0]
    duplicated_percentage = (duplicated_count / total_sequences) * 100

    if duplicated_count > 0:
        logger.info("There are duplicates in the 'sequence' column:")
        logger.info(duplicated_sequences)
        logger.info(f"Percentage of duplicated sequences: {duplicated_percentage:.2f}%")
    else:
        logger.info("No duplicates found in the 'sequence' column.")


# TODO functionality to output if check passes or not
def dataset_split_check(df: pl.DataFrame, dataset_name: str) -> None:
    """Calculates the percentage of each species in the dataframe."""
    if df.is_empty():
        logger.info("Empty dataframe.")
    counts = df["experiment_name"].value_counts()
    count_col = counts["count"]
    percentages = (count_col / len(df) * 100).alias("percentage")
    counts = counts.with_columns(percentages)
    logger.info(f"Species proportions in {dataset_name}: \n{counts}")


@click.command()
@click.argument("directory")
@click.option("--holdout_file_path", default="None")
@click.option("--split_csv_path", default="")
@click.option("--check_split", default=False)
def main(
    directory: str, holdout_file_path: str, split_csv_path: str, check_split: bool
) -> None:
    """Run the dataset split code and proportion checks."""
    if not os.path.exists(split_csv_path):
        create_split_csv(directory)

    create_shards(
        directory, os.path.join(directory, "species_split.csv"), holdout_file_path
    )

    if check_split:
        dataset_split_check(pl.read_ipc(os.path.join(directory, "train.ipc")), "train")
        dataset_split_check(pl.read_ipc(os.path.join(directory, "valid.ipc")), "valid")
        dataset_split_check(pl.read_ipc(os.path.join(directory, "test.ipc")), "test")

    check_csv_duplicates(df=pd.read_csv(os.path.join(directory, "species_split.csv")))


# TODO remove
# EXAMPLE USAGE:
# python scripts/splits_and_shards.py "../9_species_v2/massive.ucsd.edu/v05/MSV000090982/formatted_ipc/" --split_csv_path "../9_species_v2/massive.ucsd.edu/v05/MSV000090982/formatted_ipc/split.csv"
if __name__ == "__main__":
    main()
