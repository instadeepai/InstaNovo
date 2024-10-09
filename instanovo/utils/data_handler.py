from __future__ import annotations

import logging

from matchms.exporting import save_as_mgf
from matchms.importing import load_from_mgf
from matchms import Spectrum
import pandas as pd
import polars as pl
from pathlib import Path
from enum import Enum
import glob
import os
import random
import tempfile
from typing import Iterator, Callable, Any, cast
import uuid
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil
from datasets import Dataset, load_dataset

from instanovo.constants import (
    PROTON_MASS_AMU,
    MSColumns,
    MS_TYPES,
    ANNOTATED_COLUMN,
    ANNOTATION_ERROR,
)
from instanovo.utils import Metrics
from instanovo.utils.msreader import read_mzml, read_mzxml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SpectrumDataFrame:
    """Spectra data class.

    A data class for interacting with mass spectra data, including loading, processing,
    and saving mass spectrometry data in various formats such as Parquet, MGF, and others.

    Supports lazy loading, shuffling, and handling of large datasets by processing them
    in chunks.

    Attributes:
        is_annotated (bool): Whether the dataset is annotated with peptide sequences.
        has_predictions (bool): Whether the dataset contains predictions.
        is_lazy (bool): Whether lazy loading mode is used.
    """

    # flake8: noqa: CCR001
    def __init__(
        self,
        df: pl.DataFrame | None = None,
        file_paths: str | list[str] | None = None,
        is_annotated: bool = False,
        has_predictions: bool = False,
        is_lazy: bool = False,
        shuffle: bool = False,
        max_shard_size: int = 100_000,
        custom_load_fn: Callable | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> None:
        """Initialize SpectrumDataFrame.

        Args:
            df (pl.DataFrame | None): In-memory polars DataFrame with mass spectra data.
            file_paths (str | list[str] | None): Path(s) to the input data files.
            custom_load_fn (Callable | None): Custom function for loading data files.
            is_annotated (bool): Whether the dataset is annotated.
            has_predictions (bool): Whether predictions are present in the dataset.
            is_lazy (bool): Whether to use lazy loading mode.
            shuffle (bool): Whether to shuffle the dataset.
            max_shard_size (int): Maximum size of data shards for chunking large datasets.

        Raises:
            ValueError: If neither `df` nor `file_paths` is specified, or both are given.
            FileNotFoundError: If no files are found matching the given `file_paths`.
        """
        self._is_annotated: bool = is_annotated
        self._has_predictions: bool = has_predictions  # if we have outputs/predictions
        self._is_lazy: bool = is_lazy  # or streaming
        self._shuffle: bool = shuffle
        self._max_shard_size = max_shard_size
        self._custom_load_fn = custom_load_fn
        self.executor = None
        self._temp_directory = None

        if df is None and file_paths is None:
            raise ValueError("Must specify either df or file_paths, both are None.")

        # native refers to data being stored as a list of parquet files.
        self._is_native = file_paths is not None
        if df is not None and self._is_native:
            raise ValueError("Must specify either df or file_paths, not both.")

        if self._is_native:
            # Get all file paths
            self._file_paths = self._convert_file_paths(
                cast(str | list[str], file_paths)
            )

            if len(self._file_paths) == 0:
                raise FileNotFoundError(f"No files matching '{file_paths}' were found.")

            # If any of the files are not .parquet, create a tempdir with the converted files.
            if not all([fp.endswith(".parquet") for fp in self._file_paths]):
                # If lazy make tempdir, if not convert to non-native and load all contents into df
                # Only iterate over non-parquet files
                df_iterator = SpectrumDataFrame.get_data_shards(
                    self._file_paths,
                    max_shard_size=self._max_shard_size,
                    custom_load_fn=custom_load_fn,
                    column_mapping=column_mapping,
                )

                if self._is_lazy:
                    self._temp_directory = tempfile.mkdtemp()

                    new_file_paths = [
                        fp for fp in self._file_paths if fp.endswith(".parquet")
                    ]
                    for temp_df in df_iterator:
                        # TODO: better way to generate id than hash?
                        temp_parquet_path = os.path.join(
                            self._temp_directory, f"temp_{uuid.uuid4().hex}.parquet"
                        )
                        temp_df.write_parquet(temp_parquet_path)
                        new_file_paths.append(temp_parquet_path)
                    self._file_paths = new_file_paths
                else:
                    self.df = pl.concat(
                        [
                            SpectrumDataFrame._cast_columns(temp_df)
                            for temp_df in df_iterator
                        ]
                    )
                    # Native is disabled if not lazy
                    self._is_native = False
                    self._file_paths = []

            if self._file_paths is not None:
                self._filter_series_per_file: dict[str, pl.Series] = {
                    fp: pl.Series(
                        np.full(pl.scan_parquet(fp).collect().height, True, dtype=bool)
                    )
                    for fp in self._file_paths
                }
        else:
            self.df = df

        # Check all columns
        self._check_type_spec()

        # Shuffled file handling, uses a two-step shuffle to optimise efficiency
        self._current_index_in_file = (
            0  # index in the current file, used in shuffle mode
        )
        self._next_file_index = 0  # index of the next file to be loaded in _file_paths
        self._current_file: str | None = None  # filename of the current file
        self._current_file_len = 0  # length of the current file
        self._current_file_data: pl.DataFrame | None = (
            None  # loaded data of the current file
        )
        self._current_file_position = 0  # starting index of the current file, used to

        if self._shuffle:
            if self._is_native:
                self._shuffle_file_order()
            else:
                self.df = SpectrumDataFrame._shuffle_df(self.df)
        elif self._is_native:
            # Sort files alphabetically
            # TODO: Do we want this? Keeps consistent when loading files across devices
            self._file_paths.sort()
            self._update_file_indices()

        if self._is_lazy:
            # When lazy loading, use async loaders to keep next file ready at all times.
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.loop = asyncio.get_event_loop()
            self._next_file: str | None = None  # Future file name
            self._next_file_future: pl.DataFrame | None = None  # Future file data
            self._preload_task: asyncio.Task | None = None

    @staticmethod
    def _shuffle_df(df: pl.DataFrame) -> pl.DataFrame:
        """Shuffle the rows of the given DataFrame."""
        shuffled_indices = np.random.permutation(len(df))
        return df[shuffled_indices]
        # return df.with_row_count("row_nr").select(pl.all().shuffle())

    @staticmethod
    def _sanitise_peptide(peptide: str) -> str:
        """Sanitise peptide sequence."""
        # Some datasets save sequence wrapped with _ or .
        if peptide[0] == "_":
            peptide = peptide[1:-1]
        if peptide[0] == ".":
            peptide = peptide[1:-1]
        return peptide

    @staticmethod
    def _cast_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Cast the columns of the DataFrame to the appropriate data types based on MS_TYPES."""
        return df.with_columns(
            [
                pl.col(column.value).cast(dtype)
                for column, dtype in MS_TYPES.items()
                if column.value in df.columns
            ]
        )

    @staticmethod
    def _convert_file_paths(file_paths: str | list[str]) -> list[str]:
        """Convert a string or list of file paths to a list of file paths.

        Args:
            file_paths (str | list[str]): File path or list of file paths.

        Returns:
            list[str]: A list of resolved file paths.

        Raises:
            ValueError: If input is a directory or not a valid file path.
        """
        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                raise ValueError(
                    "Input must be a string (filepath or glob) or a list of file paths. Found directory."
                )
            if "*" in file_paths or "?" in file_paths or "[" in file_paths:
                # Glob notation: path/to/data/*.parquet
                return glob.glob(file_paths)
            else:
                # Single file
                return [file_paths]
        elif not isinstance(file_paths, list):
            ValueError(
                "Input must be a string (filepath or glob) or a list of file paths."
            )
        return file_paths

    # flake8: noqa: CR001
    @staticmethod
    def get_data_shards(
        file_paths: list[str],
        custom_load_fn: Callable | None = None,
        column_mapping: dict[str, str] | None = None,
        max_shard_size: int = 100_000,
        add_index_cols: bool = True,
    ) -> Iterator[pl.DataFrame]:
        """Load data files into DataFrames one at a time to save memory.

        Args:
            file_paths (list[str]): List of file paths to be loaded.
            custom_load_fn (Callable | None): Custom function to load the files.
            max_shard_size (int): Maximum size of data shards.
            add_index_cols (bool): Whether to add special indexing columns.

        Yields:
            Iterator[pl.DataFrame]: DataFrames containing mass spectra data.
        """
        column_mapping = column_mapping or {}
        current_shard = None
        for fp in file_paths:
            if fp.endswith(".parquet"):
                continue

            if custom_load_fn is not None:
                df = custom_load_fn(fp)
            else:
                df = SpectrumDataFrame._df_from_any(fp)

            if df is None:
                logger.info(f"Skipping {fp}")
                continue

            # Add special columns for indexing
            if add_index_cols:
                exp_name = Path(fp).stem
                df = df.with_columns(
                    pl.lit(exp_name).alias("experiment_name").cast(pl.Utf8)
                )
                if "scan_number" in df.columns:
                    df = df.with_columns(
                        (
                            pl.col("experiment_name")
                            + ":"
                            + pl.col("scan_number").cast(pl.Utf8)
                        ).alias("spectrum_id")
                    )

            df = df.rename({k: v for k, v in column_mapping.items() if k in df.columns})

            # If df > shard_size, split it up first
            while len(df) > max_shard_size:
                yield df[:max_shard_size]
                df = df[max_shard_size:]

            # Assumes df < shard_size
            if current_shard is None:
                current_shard = df
            elif len(current_shard) + len(df) < max_shard_size:
                current_shard = pl.concat([current_shard, df])
            else:
                yield pl.concat(
                    [current_shard, df[: (max_shard_size - len(current_shard))]]
                )
                current_shard = df[(max_shard_size - len(current_shard)) :]
        yield current_shard

    def filter_rows(self, filter_fn: Callable) -> None:
        """Apply a filter function to rows of the DataFrame.

        Args:
            filter_fn (Callable): Function used to filter rows.
        """
        if self._is_native:
            for fp in self._file_paths:
                df = pl.scan_parquet(fp).collect()
                new_filter = df.select(
                    [
                        pl.struct(df.columns)
                        .map_elements(filter_fn, return_dtype=bool)
                        .alias("result"),
                    ]
                )["result"]
                self._filter_series_per_file[fp] &= new_filter

            if self._current_file is not None:
                self._current_file_data = self._load_parquet_data(self._current_file)
            if not self._shuffle:
                self._update_file_indices()
        else:
            new_filter = self.df.select(
                [
                    pl.struct(self.df.columns)
                    .map_elements(filter_fn, return_dtype=bool)
                    .alias("result"),
                ]
            )["result"]
            self.df = self.df.filter(new_filter)

    def reset_filter(self) -> None:
        """Reset the filters applied to the DataFrame."""
        if not self._is_native:
            raise NotImplementedError(
                "Filter reset is not supported in non-native mode."
            )
        self._filter_series_per_file = {
            fp: pl.Series(
                np.full(pl.scan_parquet(fp).collect().height, True, dtype=bool)
            )
            for fp in self._file_paths
        }
        if self._current_file is not None:
            self._current_file_data = self._load_parquet_data(self._current_file)
        if not self._shuffle:
            self._update_file_indices()

    def _update_file_indices(self) -> None:
        """Update mapping from index to file in native non-shuffle mode."""
        if self._shuffle:
            raise ValueError("Cannot use file indexing in shuffle mode.")

        self._index_to_file_index = pl.concat(
            [
                pl.Series(np.full(self._filter_series_per_file[fp].sum(), i, dtype=int))
                for i, fp in enumerate(self._file_paths)
            ]
        )

        cumulative_position = 0
        self._file_begin_index: dict[str, int] = {}
        for fp in self._file_paths:
            self._file_begin_index[fp] = cumulative_position
            height = self._filter_series_per_file[fp].sum()
            cumulative_position += height

    def _shuffle_file_order(self) -> None:
        """Shuffle the order of files in native mode."""
        random.shuffle(self._file_paths)

    def _load_parquet_data(self, file_path: str) -> pl.DataFrame:
        """Load data from a parquet file and apply the filters."""
        return (
            pl.scan_parquet(file_path)
            .filter(self._filter_series_per_file[file_path])
            .collect()
        )

    def _load_next_file(self) -> None:
        """Load the next file in sequence for lazy loading."""
        # This function is exclusive to native mode i.e. always lazy
        self._current_file = self._file_paths[self._next_file_index]
        # Scan file, filter, and collect
        if self._current_file == self._next_file and self._next_file_future is not None:
            self._current_file_data = self._next_file_future
        else:
            self._current_file_data = self._load_parquet_data(self._current_file)

        # Update next file loading
        if self._shuffle:
            self._current_file_data = SpectrumDataFrame._shuffle_df(
                self._current_file_data
            )  # Shuffle rows
        self._current_file_len = self._current_file_data.shape[0]

        # Update future
        if len(self._file_paths) > 0:
            future_file_index = self._next_file_index + 1
            if future_file_index >= len(self._file_paths):
                if self._shuffle:
                    self._shuffle_file_order()
                future_file_index = 0

            self._next_file = self._file_paths[future_file_index]

            self._next_file_future = None
            self._start_preload_next(self._next_file)

    def _start_preload_next(self, file_path: str) -> None:
        """Start preloading the next file asynchronously."""
        if self._preload_task is None or self._preload_task.done():
            self._preload_task = self.loop.create_task(
                self._preload_next_file(file_path)
            )

    async def _preload_next_file(self, file_path: str) -> None:
        """Asynchronously preload the next file."""
        try:
            self._next_file_future = await self.loop.run_in_executor(
                self.executor, self._load_parquet_data, file_path
            )
        except Exception as e:
            print(f"Error preloading file {file_path}: {e}")
            self._next_file_future = None

    def __len__(self) -> int:  # noqa: CCE001
        """Returns the total number of rows in the SpectrumDataFrame.

        Returns:
            int: Number of rows in the DataFrame.
        """
        if self._is_native:
            return sum([v.sum() for v in self._filter_series_per_file.values()])
        return int(self.df.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict[str, Any]: Dictionary containing the data from the specified row.

        Raises:
            IndexError: If the DataFrame is empty or the index is out of range.
        """
        if len(self) == 0:
            raise IndexError("Attempt to index empty SpectrumDataFrame")

        # In shuffle, idx is ignored.
        if self._is_native:
            if self._shuffle:
                # If no file is loaded or we have finished the current file
                if (
                    self._current_file_data is None
                    or self._current_index_in_file >= self._current_file_len
                ):
                    self._current_index_in_file = 0

                    self._load_next_file()
                    self._next_file_index += 1
                    if self._next_file_index >= len(self._file_paths):
                        self._next_file_index = 0

                # for mypy
                assert self._current_file_data is not None

                row = self._current_file_data[self._current_index_in_file]

                self._current_index_in_file += 1
            else:
                # In native mode without shuffle, idx is used.
                selected_file_index = self._index_to_file_index[idx]

                # If the index is outside the currently loaded file, load the new file
                if (
                    self._current_file_data is None
                    or self._file_paths[selected_file_index] != self._current_file
                ):
                    self._next_file_index = selected_file_index
                    self._load_next_file()

                # for mypy
                assert self._current_file is not None
                assert self._current_file_data is not None

                # Find the relative index within the current file
                file_begin_index = self._file_begin_index[self._current_file]
                index_in_file = idx
                if file_begin_index > 0:
                    index_in_file = idx % self._file_begin_index[self._current_file]

                row = self._current_file_data[index_in_file]
        else:
            # We're in non-native non-lazy mode
            if self._shuffle:
                # Shuffle if we have passed through all entries
                if self._current_index_in_file >= self.df.height:
                    self.df = SpectrumDataFrame._shuffle_df(self.df)
                    self._current_index_in_file = 0

                row = self.df[self._current_index_in_file]

                self._current_index_in_file += 1
            else:
                row = self.df[idx]

        row = SpectrumDataFrame._cast_columns(row)

        # Squeeze all entries
        row_dict: dict[str, Any] = {
            k: v[0] for k, v in row.to_dict(as_series=False).items()
        }

        if self.is_annotated:
            row_dict[ANNOTATED_COLUMN] = SpectrumDataFrame._sanitise_peptide(
                row_dict[ANNOTATED_COLUMN]
            )

        return row_dict

    # def iterable(self, df: pl.DataFrame) -> None:
    #     """Iterates dataset. Supports streaming?"""
    #     pass

    @property
    def is_annotated(self) -> bool:
        """Check if the dataset is annotated.

        Returns:
            bool: True if annotated, False otherwise.
        """
        return self._is_annotated

    @property
    def has_predictions(self) -> bool:
        """Check if the dataset contains predictions.

        Returns:
            bool: True if predictions are present, False otherwise.
        """
        return self._has_predictions

    @property
    def is_lazy(self) -> bool:
        """Check if lazy loading mode is enabled.

        Returns:
            bool: True if lazy loading is enabled, False otherwise.
        """
        return self._is_lazy

    def save(
        self,
        target: str,
        partition: str | None = None,
        name: str | None = None,
        max_shard_size: int | None = None,
    ) -> None:
        """Save the dataset in parquet format with the option to partition and shard the data.

        Args:
            target (str): Directory to save the dataset.
            partition (str | None): Partition name to be included in the file names.
            name (str | None): Dataset name to be included in the file names.
            max_shard_size (int | None): Maximum size of the data shards.
        """
        max_shard_size = max_shard_size or self._max_shard_size
        partition = partition or "default"
        name = name or "ms"

        total_num_files = (len(self) // max_shard_size) + 1

        shards = self._to_parquet_chunks(target, max_shard_size)

        os.makedirs(target, exist_ok=True)

        for i, shard in enumerate(shards):
            filename = (
                f"dataset-{name}-{partition}-{i:04d}-{total_num_files:04d}.parquet"
            )
            shard.write_parquet(os.path.join(target, filename))

    def _to_parquet_chunks(
        self, target: str, max_shard_size: int = 1_000_000
    ) -> Iterator[pl.DataFrame]:
        """Generate DataFrame chunks to be saved as parquet files.

        Args:
            target (str): Directory to save the parquet files.
            max_shard_size (int): Maximum size of the data shards.

        Yields:
            Iterator[pl.DataFrame]: Chunks of DataFrames to be saved.
        """
        if self._is_native:
            current_shard = None
            for fp in self._file_paths:
                # Load each file with filtering
                df = self._load_parquet_data(fp)

                while len(df) > max_shard_size:
                    yield df[:max_shard_size]
                    df = df[max_shard_size:]

                # Assumes df < shard_size
                if current_shard is None:
                    current_shard = df
                elif len(current_shard) + len(df) < max_shard_size:
                    current_shard = pl.concat([current_shard, df])
                else:
                    yield pl.concat(
                        [current_shard, df[: (max_shard_size - len(current_shard))]]
                    )
                    current_shard = df[(max_shard_size - len(current_shard)) :]
            yield current_shard
        else:
            df = self.df
            while len(df) > max_shard_size:
                yield df[:max_shard_size]
                df = df[max_shard_size:]
            yield df

    def write_csv(self, target: str) -> None:
        """Write the dataset to a CSV file.

        Args:
            target (str): Path to the output CSV file.
        """
        self.to_pandas().to_csv(target, index=False)

    def write_ipc(self, target: str) -> None:
        """Write the dataset to a Polars ipc file.

        Args:
            target (str): Path to the output ipc file.
        """
        df = self.to_polars()
        if self._is_native:
            df = df.collect().rechunk()
        df.write_ipc(target)

    def write_mgf(self, target: str, export_style: str | None = None) -> None:
        """Write the dataset to an MGF file using Matchms format.

        Args:
            target (str): Path to the output MGF file.
            export_style (str | None): Style of export to be used (optional).
        """
        export_style = export_style or "matchms"
        spectra = self.to_matchms()

        # Check if the file exists and delete it if it does
        if os.path.exists(target):
            try:
                os.remove(target)
            except OSError as e:
                logger.warn(f"Error deleting existing file '{target}': {e}")
                return  # Exit the method if we can't delete the file

        save_as_mgf(spectra, target, export_style=export_style)

    def write_pointnovo(self, spectrum_source: str, feature_target: str) -> None:
        """Write the dataset in PointNovo format.

        Args:
            spectrum_source (str): Source of the spectrum data.
            feature_target (str): Target for the features.
        """
        raise NotImplementedError()

    def write_mzxml(self, target: str) -> None:
        """Write the dataset in mzXML format.

        Args:
            target (str): Path to the output mzXML file.
        """
        raise NotImplementedError()

    def write_mzml(self, target: str) -> None:
        """Write the dataset in mzML format.

        Args:
            target (str): Path to the output mzML file.
        """
        raise NotImplementedError()

    def to_pandas(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame.

        Warning:
            This function loads the entire dataset into memory. For large datasets,
            this may consume a significant amount of RAM.

        Returns:
            pd.DataFrame: The dataset in pandas DataFrame format.
        """
        return self.to_polars(return_lazy=False).to_pandas()

    def to_polars(self, return_lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
        """Convert the dataset to a polars DataFrame.

        Args:
            return_lazy (bool): Return LazyFrame when in lazy mode. Defaults to True.

        Returns:
            pl.DataFrame | pl.LazyFrame: The dataset in polars DataFrame format
        """
        if self._is_native:
            dfs = []
            for fp in self._file_paths:
                dfs.append(pl.scan_parquet(fp).filter(self._filter_series_per_file[fp]))
            df = pl.concat(dfs)
            if return_lazy:
                return df
            return df.collect().rechunk()
        return self.df

    def to_matchms(self) -> list[Spectrum]:
        """Convert the dataset to a list of Matchms spectrum objects.

        Warning:
            This function loads the entire dataset into memory. For large datasets,
            this may consume a significant amount of RAM.

        Returns:
            list[Spectrum]: List of Matchms spectrum objects.
        """
        df = self.to_polars(return_lazy=False)
        return SpectrumDataFrame._df_to_matchms(df)

    def export_predictions(self, target: str, export_type: str | Enum) -> None:
        """Export the predictions from the dataset.

        Args:
            target (str): Target path to save the predictions.
            export_type (str | Enum): Type of export format.
        """
        if isinstance(export_type, str):
            pass
        raise NotImplementedError()

    @classmethod
    def load(
        cls,
        source: str | Path,
        source_type: str = "default",
        is_annotated: bool = False,
        shuffle: bool = False,
        name: str | None = None,
        partition: str | None = None,
        custom_load: Callable | None = None,
        column_mapping: dict[str, str] | None = None,
        lazy: bool = True,
        max_shard_size: int = 1_000_000,
    ) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from a source.

        Args:
            source (str | Path): Path to the source file or directory.
            source_type (str): Type of the source (default is "default").
            is_annotated (bool): Whether the dataset is annotated.
            shuffle (bool): Whether to shuffle the dataset.
            name (str | None): Name of the dataset.
            partition (str | None): Partition name of the dataset.
            lazy (bool): Whether to use lazy loading mode.
            max_shard_size (int): Maximum size of data shards.

        Returns:
            SpectrumDataFrame: The loaded SpectrumDataFrame.
        """
        partition = partition or "default"
        name = name or "ms"

        # Native mode
        if (
            isinstance(source, str)
            and os.path.isdir(source)
            and source_type == "default"
        ):
            # /path/to/folder/dataset-name-train-0000-of-0001.parquet
            source = os.path.join(source, f"dataset-{name}-{partition}-*-*.parquet")

        return cls(
            file_paths=cast(str, source),  # We don't support Path directly
            is_lazy=lazy,
            custom_load_fn=custom_load,
            column_mapping=column_mapping,
            max_shard_size=max_shard_size,
            shuffle=shuffle,
            is_annotated=is_annotated,
        )

    @staticmethod
    def _df_from_any(source: str, source_type: str | None = None) -> pl.DataFrame:
        """Load a DataFrame from various source formats (MGF, IPC, etc.).

        Args:
            source (str): Path to the source file.
            source_type (str | None): Type of the source file.

        Returns:
            pl.DataFrame: The loaded DataFrame.
        """
        if source_type is None:
            # Try to infer
            source_type = source.split(".")[-1].lower()

        match source_type:
            case "ipc":
                return SpectrumDataFrame._df_from_ipc(source)
            case "mgf":
                return SpectrumDataFrame._df_from_mgf(source)
            case "mzml":
                return SpectrumDataFrame._df_from_mzml(source)
            case "mzxml":
                return SpectrumDataFrame._df_from_mzxml(source)
            case "_":
                logger.info(f"Unknown filetype {source_type} of {source}")
                return None

    @classmethod
    def load_mgf(cls, source: str) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from an MGF file.

        Args:
            source (str): Path to the MGF file.

        Returns:
            SpectrumDataFrame: The loaded SpectrumDataFrame.
        """
        spectra = list(load_from_mgf(source))
        return cls.from_matchms(spectra)

    @staticmethod
    def _df_from_mgf(source: str) -> pl.DataFrame:
        """Load a polars DataFrame from an MGF file.

        Args:
            source (str): Path to the MGF file.

        Returns:
            pl.DataFrame: The loaded polars DataFrame.
        """
        spectra = list(load_from_mgf(source))
        return SpectrumDataFrame._df_from_matchms(spectra)

    @classmethod
    def load_pointnovo(
        cls, spectrum_source: str, feature_source: str
    ) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from PointNovo format.

        Args:
            spectrum_source (str): Source of spectrum data.
            feature_source (str): Source of feature data.
        """
        raise NotImplementedError()

    @classmethod
    def load_csv(
        cls,
        source: str,
        column_mapping: dict[str, str] | None = None,
        lazy: bool = False,
        annotated: bool = False,
    ) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from a CSV file.

        Args:
            source (str): Path to the CSV file.
            column_mapping (dict[str, str] | None): Mapping of columns to rename.
            lazy (bool): Whether to use lazy loading mode.
            annotated (bool): Whether the dataset is annotated.

        Returns:
            SpectrumDataFrame: The loaded SpectrumDataFrame.
        """
        df = pl.read_csv(source)
        if column_mapping is not None:
            df = df.rename({k: v for k, v in column_mapping.items() if k in df.columns})
        return cls(df, is_annotated=annotated, is_lazy=lazy)

    @classmethod
    def load_mzxml(cls, source: str) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from an mzXML file.

        Args:
            source (str): Path to the mzXML file.

        Returns:
            SpectrumDataFrame: The loaded SpectrumDataFrame.
        """
        # spectra = list(load_from_mzxml(source))
        # return cls.from_matchms(spectra)
        df = SpectrumDataFrame._df_from_dict(read_mzxml(source))
        return cls.from_polars(df)

    @staticmethod
    def _df_from_mzxml(source: str) -> pl.DataFrame:
        """Load a polars DataFrame from an MGF file.

        Args:
            source (str): Path to the MGF file.

        Returns:
            pl.DataFrame: The loaded polars DataFrame.
        """
        # spectra = list(load_from_mzxml(source))
        # return SpectrumDataFrame._df_from_matchms(spectra)
        return SpectrumDataFrame._df_from_dict(read_mzxml(source))

    @classmethod
    def load_mzml(cls, source: str) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from an mzML file.

        Args:
            source (str): Path to the mzML file.

        Returns:
            SpectrumDataFrame: The loaded SpectrumDataFrame.
        """
        # spectra = list(load_from_mzml(source))
        # return cls.from_matchms(spectra)
        df = SpectrumDataFrame._df_from_dict(read_mzml(source))
        return cls.from_polars(df)

    @staticmethod
    def _df_from_mzml(source: str) -> pl.DataFrame:
        """Load a polars DataFrame from an MGF file.

        Args:
            source (str): Path to the MGF file.

        Returns:
            pl.DataFrame: The loaded polars DataFrame.
        """
        # spectra = list(load_from_mzml(source))
        # return SpectrumDataFrame._df_from_matchms(spectra)
        return SpectrumDataFrame._df_from_dict(read_mzml(source))

    @classmethod
    def from_huggingface(
        cls,
        dataset: str | Dataset,
        shuffle: bool = False,
        is_annotated: bool = False,
        **kwargs: Any,
    ) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from HuggingFace directory or Dataset instance.

        Warning:
            This function loads the entire dataset into memory. For large datasets,
            this may consume a significant amount of RAM.

        Args:
            dataset (str | Dataset): Path to HuggingFace or Dataset instance.

        Returns:
            SpectrumDataFrame: The loaded SpectrumDataFrame.
        """
        if isinstance(dataset, str):
            dataset = load_dataset(dataset, **kwargs)
        # TODO: Explore dataset.to_pandas(batched=True)
        return cls.from_pandas(
            dataset.to_pandas(), shuffle=shuffle, is_annotated=is_annotated
        )

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, shuffle: bool = False, is_annotated: bool = False
    ) -> "SpectrumDataFrame":
        """Create a SpectrumDataFrame from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The pandas DataFrame.

        Returns:
            SpectrumDataFrame: The resulting SpectrumDataFrame.
        """
        df = pl.from_pandas(df)
        return cls.from_polars(df, shuffle=shuffle, is_annotated=is_annotated)

    @classmethod
    def from_polars(
        cls, df: pl.DataFrame, shuffle: bool = False, is_annotated: bool = False
    ) -> "SpectrumDataFrame":
        """Create a SpectrumDataFrame from a polars DataFrame.

        Args:
            df (pl.DataFrame): The polars DataFrame.

        Returns:
            SpectrumDataFrame: The resulting SpectrumDataFrame.
        """
        return cls(
            df=df,
            shuffle=shuffle,
            is_annotated=is_annotated,
        )

    @classmethod
    def load_ipc(
        cls, source: str, shuffle: bool = False, is_annotated: bool = False
    ) -> "SpectrumDataFrame":
        """Load a SpectrumDataFrame from IPC format.

        Args:
            source (str): Path to the IPC file.

        Returns:
            SpectrumDataFrame: The loaded SpectrumDataFrame.
        """
        df = cls._df_from_ipc(source)
        return cls(
            df,
            is_lazy=False,
            shuffle=shuffle,
            is_annotated=is_annotated,
        )

    @staticmethod
    def _df_from_ipc(source: str) -> pl.DataFrame:
        """Load a polars DataFrame from an IPC file.

        Args:
            source (str): Path to the IPC file.

        Returns:
            pl.DataFrame: The loaded polars DataFrame.
        """
        df = pl.read_ipc(source)
        if "modified_sequence" in df.columns:
            df = df.with_columns(pl.col("modified_sequence").alias(ANNOTATED_COLUMN))
        return df

    @classmethod
    def from_matchms(
        cls, spectra: list[Spectrum], shuffle: bool = False, is_annotated: bool = False
    ) -> "SpectrumDataFrame":
        """Create a SpectrumDataFrame from Matchms spectrum objects.

        Args:
            spectra (list): List of Matchms spectrum objects.
            shuffle (bool, optional): If True, shuffle the data. Defaults to False.
            is_annotated (bool, optional): If True, treat the spectra as annotated. Defaults to False.

        Returns:
            SpectrumDataFrame: The resulting SpectrumDataFrame.

        Raises:
            ValueError: If the input parameters are invalid or incompatible.
        """
        df = SpectrumDataFrame._df_from_matchms(spectra)
        return cls(
            df=df,
            shuffle=shuffle,
            is_annotated=is_annotated,
        )

    @staticmethod
    def _df_from_dict(data: dict[str, Any]) -> pl.DataFrame:
        df = pl.DataFrame(
            {
                "scan_number": pl.Series(
                    np.arange(len(data["sequence"])), dtype=pl.Int64
                ),
                ANNOTATED_COLUMN: pl.Series(data["sequence"], dtype=pl.Utf8),
                # Calculate precursor mass
                MSColumns.PRECURSOR_MASS.value: pl.Series(
                    np.array(data["precursor_mz"]) * np.array(data["precursor_charge"])
                    - np.array(data["precursor_charge"]) * PROTON_MASS_AMU,
                    dtype=MS_TYPES[MSColumns.PRECURSOR_MASS],
                ),
                MSColumns.PRECURSOR_MZ.value: pl.Series(
                    data["precursor_mz"], dtype=MS_TYPES[MSColumns.PRECURSOR_MZ]
                ),
                MSColumns.PRECURSOR_CHARGE.value: pl.Series(
                    data["precursor_charge"], dtype=MS_TYPES[MSColumns.PRECURSOR_CHARGE]
                ),
                MSColumns.RETENTION_TIME.value: pl.Series(
                    data["retention_time"], dtype=MS_TYPES[MSColumns.RETENTION_TIME]
                ),
                MSColumns.MZ_ARRAY.value: pl.Series(
                    data["mz_array"], dtype=MS_TYPES[MSColumns.MZ_ARRAY]
                ),
                MSColumns.INTENSITY_ARRAY.value: pl.Series(
                    data["intensity_array"], dtype=MS_TYPES[MSColumns.INTENSITY_ARRAY]
                ),
            }
        )
        return df

    @staticmethod
    def _df_from_matchms(spectra: list[Spectrum]) -> pl.DataFrame:
        """Load a polars DataFrame from a list of Matchms spectra.

        Args:
            spectra (list[Spectrum]): List of Matchms spectrum objects.

        Returns:
            pl.DataFrame: The loaded polars DataFrame.
        """
        data: dict[str, list[Any]] = {
            "scan_number": [],
            "sequence": [],
            "precursor_mass": [],
            "precursor_mz": [],
            "precursor_charge": [],
            "retention_time": [],
            "mz_array": [],
            "intensity_array": [],
        }

        for i, spectrum in enumerate(spectra):
            data["scan_number"].append(i)
            data["sequence"].append(spectrum.metadata.get("peptide_sequence", ""))
            data["precursor_mass"].append(spectrum.metadata.get("pepmass", 0.0))
            data["precursor_mz"].append(spectrum.metadata.get("precursor_mz", 0.0))
            data["precursor_charge"].append(spectrum.metadata.get("charge", 0))
            data["retention_time"].append(spectrum.metadata.get("retention_time", 0.0))
            data["mz_array"].append(spectrum.peaks.mz)
            data["intensity_array"].append(spectrum.peaks.intensities)

        df = SpectrumDataFrame._df_from_dict(data)

        return df

    @staticmethod
    def _df_to_matchms(df: pl.DataFrame) -> list[Spectrum]:
        """Convert a polars DataFrame to a list of Matchms spectra.

        Args:
            df (pl.DataFrame): The input polars DataFrame.

        Returns:
            list[Spectrum]: List of Matchms spectrum objects.
        """
        spectra = []

        for row in df.iter_rows(named=True):
            metadata = {
                "peptide_sequence": row[ANNOTATED_COLUMN],
                # "pepmass": row[MSColumns.PRECURSOR_MASS.value],
                "precursor_mz": row[MSColumns.PRECURSOR_MZ.value],
                "charge": row[MSColumns.PRECURSOR_CHARGE.value],
                "retention_time": row[MSColumns.RETENTION_TIME.value],
            }

            mz_array = np.array(row[MSColumns.MZ_ARRAY.value])
            intensity_array = np.array(row[MSColumns.INTENSITY_ARRAY.value])

            spectrum = Spectrum(mz_array, intensity_array, metadata=metadata)
            spectra.append(spectrum)

        return spectra

    def _check_type_spec(self) -> None:
        """Check the data type specifications for the DataFrame columns.

        This method validates that important columns have the correct data types.
        """
        # Check expected columns, use ENUM constant for this.
        expected_cols = [
            c.value
            for c in [
                MSColumns.MZ_ARRAY,
                MSColumns.INTENSITY_ARRAY,
                MSColumns.PRECURSOR_MZ,
                MSColumns.PRECURSOR_CHARGE,
            ]
        ]
        if self.is_annotated:
            expected_cols.append(ANNOTATED_COLUMN)

        if self._is_native:
            # Check all parquet files
            for fp in self._file_paths:
                columns = pl.scan_parquet(fp).collect_schema().keys()
                missing_cols = [col for col in expected_cols if col not in columns]
                if missing_cols:
                    break
        else:
            # Check only self.df
            missing_cols = [col for col in expected_cols if col not in self.df.columns]

        if missing_cols:
            plural_s = "s" if len(missing_cols) > 1 else ""
            missing_col_names = ", ".join(missing_cols)
            raise ValueError(
                f"Column{plural_s} missing! Missing column{plural_s}: {missing_col_names}"
            )

        # Check annotated column is actually annotated
        if self.is_annotated:
            if self._is_native:
                for fp in self._file_paths:
                    has_annotations = (
                        pl.scan_parquet(fp)
                        .select(
                            (
                                (pl.col(ANNOTATED_COLUMN).is_not_null())
                                & (pl.col(ANNOTATED_COLUMN) != "")
                            ).all()
                        )
                        .collect()
                        .to_numpy()[0]
                    )
                    if not has_annotations:
                        raise ValueError(ANNOTATION_ERROR)
            else:
                has_annotations = self.df.select(
                    (
                        (pl.col(ANNOTATED_COLUMN).is_not_null())
                        & (pl.col(ANNOTATED_COLUMN) != "")
                    ).all()
                ).to_numpy()[0]
                if not has_annotations:
                    raise ValueError(ANNOTATION_ERROR)

    @classmethod
    def concatenate(
        cls, sdf: list["SpectrumDataFrame"], strict: bool = True
    ) -> "SpectrumDataFrame":
        """Concatenate a list of SpectrumDataFrames.

        Warning:
            This function loads the entire dataset into memory. For large datasets,
            this may consume a significant amount of RAM.

        Args:
            df (list[SpectrumDataFrame]): List of SpectrumDataFrames to concatenate.
            strict (bool): Whether to perform strict concatenation.

        Returns:
            SpectrumDataFrame: The resulting concatenated SpectrumDataFrame.
        """
        return cls(pl.concat([x.to_polars(return_lazy=False) for x in sdf]))

    def get_unique_sequences(self) -> set[str]:
        """Retrieve unique peptide sequences from the dataset.

        Returns:
            set[str]: A set of unique peptide sequences.

        Raises:
            ValueError: If the dataset is not annotated.
        """
        if not self.is_annotated:
            raise ValueError("Only annotated datasets have sequences.")

        if self._is_native:
            sequences = set()
            for fp in self._file_paths:
                df_unique = (
                    pl.scan_parquet(fp)
                    .filter(self._filter_series_per_file[fp])
                    .select(pl.col(ANNOTATED_COLUMN).unique())
                    .collect()
                )
                sequences.update(set(df_unique[ANNOTATED_COLUMN].to_list()))
            return sequences
        else:
            return set(self.df[ANNOTATED_COLUMN].unique())

    def get_vocabulary(self, tokenize_fn: Callable) -> set[str]:
        """Get the vocabulary of unique residues from peptide sequences.

        Args:
            tokenize_fn (Callable): Function to tokenize peptide sequences.

        Returns:
            set[str]: A set of unique residues.

        Raises:
            ValueError: If the dataset is not annotated.
        """
        if not self.is_annotated:
            raise ValueError("Only annotated datasets have residue vocabularies.")

        sequences = self.get_unique_sequences()
        residues = set()
        for x in sequences:
            residues.update(set(tokenize_fn(x)))
        return residues

    def validate_precursor_mass(self, metrics: Metrics, tolerance: float = 50) -> int:
        """Validate precursor mz matching the annotations.

        Args:
            metrics (Metrics): InstaNovo metrics class for calculating sequence mass.
            tolerance (float): Tolerance to match precursor mass in ppm.

        Returns:
            int: Number of precursor matches

        Raises:
            ValueError: If none of the sequences match the precursor mz.
            ValueError: If SpectrumDataFrame is not annotated.
        """
        if not self.is_annotated:
            raise ValueError("Cannot verify precursor mass without annotations.")

        if self._is_native:
            num_matches_precursor = 0
            for fp in self._file_paths:
                result = (
                    pl.scan_parquet(fp)
                    .filter(self._filter_series_per_file[fp])
                    .select(
                        [
                            pl.col(ANNOTATED_COLUMN),
                            pl.col(MSColumns.PRECURSOR_MZ.value),
                            pl.col(MSColumns.PRECURSOR_CHARGE.value),
                        ]
                    )
                    .with_columns(
                        [
                            pl.struct(
                                [
                                    ANNOTATED_COLUMN,
                                    pl.col(MSColumns.PRECURSOR_MZ.value),
                                    pl.col(MSColumns.PRECURSOR_CHARGE.value),
                                ]
                            )
                            .map_elements(
                                lambda x: metrics.matches_precursor(
                                    x[ANNOTATED_COLUMN],
                                    x[MSColumns.PRECURSOR_MZ.value],
                                    x[MSColumns.PRECURSOR_CHARGE.value],
                                    prec_tol=tolerance,
                                )[0],
                                return_dtype=bool,
                            )
                            .alias("precursor_match")
                        ]
                    )
                    .select(
                        pl.col("precursor_match").sum().alias("num_matches_precursor")
                    )
                )
                num_matches_precursor += result.collect()["num_matches_precursor"][0]
        else:
            result = (
                self.df.select(
                    [
                        pl.col(ANNOTATED_COLUMN),
                        pl.col(MSColumns.PRECURSOR_MZ.value),
                        pl.col(MSColumns.PRECURSOR_CHARGE.value),
                    ]
                )
                .with_columns(
                    [
                        pl.struct(
                            pl.col(ANNOTATED_COLUMN),
                            pl.col(MSColumns.PRECURSOR_MZ.value),
                            pl.col(MSColumns.PRECURSOR_CHARGE.value),
                        )
                        .map_elements(
                            lambda x: metrics.matches_precursor(
                                x[ANNOTATED_COLUMN],
                                x[MSColumns.PRECURSOR_MZ.value],
                                x[MSColumns.PRECURSOR_CHARGE.value],
                                prec_tol=tolerance,
                            )[0],
                            return_dtype=bool,
                        )
                        .alias("precursor_match")
                    ]
                )
                .select(pl.col("precursor_match").sum().alias("num_matches_precursor"))
            )
            num_matches_precursor = result["num_matches_precursor"][0]

        if num_matches_precursor == 0:
            raise ValueError(
                "None of the sequence labels in the dataset match the precursor mz. Check sequences and residue set for errors."
            )
        elif num_matches_precursor < len(self):
            logger.warn(
                f"{len(self) - num_matches_precursor:,d} ({(1 - num_matches_precursor/len(self))*100:.2f}%) of the sequence labels do not match the precursor mz to {tolerance}ppm."
            )

        return num_matches_precursor

    # def filter(self, *args, **kwargs) -> "SpectrumDataFrame":
    #     return self.df.filter(*args, **kwargs)

    def sample_subset(self, fraction: float, seed: int | None = None) -> None:
        """Sample a subset of the dataset.

        Args:
            fraction (float): Fraction of the dataset to sample.
            seed (int): Random seed for reproducibility.
        """
        if fraction >= 1:
            return
        if self._is_native:
            for fp in self._file_paths:
                if seed:
                    np.random.seed(seed)
                filter = self._filter_series_per_file[fp].to_numpy()
                filter[filter] = np.random.choice(
                    [True, False], size=filter.sum(), p=[fraction, 1 - fraction]
                )
                self._filter_series_per_file[fp] &= pl.Series(filter)
        else:
            self.df = self.df.sample(fraction=fraction, seed=seed)

    def validate_data(self) -> bool:
        """Validate the integrity of the dataset.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        raise NotImplementedError()

    def __del__(self) -> None:
        """Clean up the resources when the object is destroyed.

        This includes shutting down the thread pool executor and removing temporary files.
        """
        if self.executor:
            self.executor.shutdown(wait=True)
        if self._temp_directory is not None and os.path.exists(self._temp_directory):
            shutil.rmtree(self._temp_directory)
