from __future__ import annotations

import re
from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from datasets import Dataset
from torch import nn


class DataProcessor(metaclass=ABCMeta):
    """Data processor abstract class.

    This class is used to process the data before it is used in the model.
    It is designed to be used with the `Dataset` class from the HuggingFace `datasets` library.

    It includes two main methods:
    - `process_row`: Processes a row of data.
    - `collate_fn`: Collates a batch of data. To be passed to the `DataLoader` class.

    Additionally, it includes a way to pass metadata columns that will be kept after processing a dataset.
    These metadata columns will also bypass the `collate_fn`.
    """

    @property
    def metadata_columns(self) -> set[str]:
        """Get the metadata columns.

        These columns are kept after processing a dataset.

        Returns:
            list[str]: The metadata columns.
        """
        return self._metadata_columns

    def __init__(self, metadata_columns: list[str] | set[str] | None = None):
        """Initialize the data processor.

        Args:
            metadata_columns: The metadata columns to add to the expected columns.
        """
        self._metadata_columns: set[str] = set(metadata_columns or [])

    @abstractmethod
    def _get_expected_columns(self) -> list[str]:
        """Get the expected columns.

        These are the columns that will be returned by the `process_row` method.

        Returns:
            list[str]: The expected columns.
        """
        ...

    @abstractmethod
    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of data.

        Args:
            row (dict[str, Any]): The row of data to process in dict format.

        Returns:
            dict[str, Any]: The processed row with resulting columns.
        """
        ...

    @abstractmethod
    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Logic for collating a batch.

        Args:
            batch (list[dict[str, Any]]): The batch to collate.

        Returns:
            dict[str, Any]: The collated batch.
        """
        ...

    def process_dataset(self, dataset: Dataset, return_format: str | None = "torch") -> Dataset:
        """Process a dataset by mapping the `process_row` method.

        The resulting dataset has the columns expected by the `collate_fn` method.

        Args:
            dataset (Dataset): The dataset to process.
            return_format (str | None): The format to return the dataset in.
                Default is "torch".

        Returns:
            Dataset: The processed dataset.
        """
        dataset = dataset.map(self.process_row)
        dataset.set_format(type=return_format, columns=self.get_expected_columns())
        return dataset

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch.

        Metadata columns are added after collation.

        Args:
            batch (list[dict[str, Any]]): The batch to collate.

        Returns:
            dict[str, Any]: The collated batch with metadata.
        """
        return_batch: dict[str, Any] = self._collate_batch(batch)

        # Add metadata
        metadata = {}
        for col in self.metadata_columns:
            if col in return_batch:
                continue
            metadata[col] = [row[col] if col in row else None for row in batch]

        return_batch.update(metadata)

        return return_batch

    def get_expected_columns(self) -> list[str]:
        """Get the expected columns to be kept in the dataset after processing.

        These columns are expected by the `collate_fn` method and include
        both data and metadata columns.

        Returns:
            list[str]: The expected columns.
        """
        return self._get_expected_columns() + list(self.metadata_columns)

    def add_metadata_columns(self, columns: list[str] | set[str]) -> None:
        """Add expected metadata columns.

        Args:
            columns (list[str] | set[str]): The columns to add.
        """
        self._metadata_columns.update(set(columns))

    @staticmethod
    def _pad_and_mask(
        tensor_list: list[torch.tensor] | tuple[torch.tensor],
    ) -> tuple[torch.tensor, torch.tensor]:
        """Pad and mask a list of tensors.

        Args:
            tensor_list (list[torch.tensor] | tuple[torch.tensor]): The list of tensors to pad and mask.

        Returns:
            tuple[torch.tensor, torch.tensor]: The padded and masked tensors.
        """
        ll = torch.tensor([y.shape[0] for y in tensor_list], dtype=torch.long)
        padded_tensor = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
        attention_mask = torch.arange(padded_tensor.shape[1], dtype=torch.long)[None, :] >= ll[:, None]
        return padded_tensor, attention_mask

    @staticmethod
    def remove_modifications(peptide: str, replace_isoleucine_with_leucine: bool = True) -> str:
        """Remove modifications and optionally replace Isoleucine with Leucine.

        Args:
            peptide (str): The peptide to remove modifications from.
            replace_isoleucine_with_leucine (bool): Whether to replace Isoleucine with Leucine.

        Returns:
            str: The peptide with modifications removed.
        """
        # remove [UNIMOD as it will be picked up by the regex
        peptide = peptide.replace("UNIMOD", "")
        # use regex to extract only A-Z
        peptide = re.findall(r"[A-Z]", peptide)
        # replace I with L
        if replace_isoleucine_with_leucine:
            peptide = ["L" if aa == "I" else aa for aa in peptide]
        return "".join(peptide)
