from __future__ import annotations

import os
import sys

import pytest
import requests
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset

# Add the root directory to the PYTHONPATH
# This allows pytest to find the modules for testing

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)


@pytest.fixture(scope="session")
def checkpoints_dir() -> str:
    """A pytest fixture to create and provide the absolute path of a 'checkpoints' directory.

    Ensures the directory exists for storing checkpoint files during the test session.
    """
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    return os.path.abspath(checkpoints_dir)


@pytest.fixture(scope="session")
def instanovo_checkpoint(checkpoints_dir: str) -> str:
    """A pytest fixture to download and provide the path of the InstaNovo model checkpoint.

    Downloads from a predefined URL if the checkpoint file doesn't exist locally.
    """
    url = "https://github.com/instadeepai/InstaNovo/releases/download/0.1.4/instanovo_yeast.pt"
    checkpoint_path = os.path.join(checkpoints_dir, "instanovo_yeast.pt")

    if not os.path.isfile(checkpoint_path):
        response = requests.get(url)
        with open(checkpoint_path, "wb") as file:
            file.write(response.content)

    return os.path.abspath(checkpoint_path)


@pytest.fixture(scope="session")
def dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """A pytest fixture to load and provide a dataset for testing.

    Loads a specific subset (1% of test data) from the 'instanovo_ninespecies_exclude_yeast' dataset.
    """
    return load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test[:1%]")


@pytest.fixture(scope="session")
def knapsack_dir(checkpoints_dir: str) -> str:
    """A pytest fixture to create and provide the absolute path of a 'knapsack' directory within the checkpoints directory for storing test artifacts."""
    knapsack_dir = os.path.join(checkpoints_dir, "knapsack")
    return os.path.abspath(knapsack_dir)
