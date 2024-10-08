from __future__ import annotations

import os
import random
import sys
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as ptl
import torch
from google.cloud import storage
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import open_dict

from instanovo.transformer.model import InstaNovo
from instanovo.scripts.set_gcp_credentials import set_credentials


# Add the root directory to the PYTHONPATH
# This allows pytest to find the modules for testing

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)


def reset_seed(seed: int = 42) -> None:
    """Function to reset seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    ptl.seed_everything(seed)


@pytest.fixture()
def _reset_seed() -> None:
    """A pytest fixture to reset the seeds at the start of relevant tests."""
    reset_seed()


@pytest.fixture(scope="session")
def checkpoints_dir() -> str:
    """A pytest fixture to create and provide the absolute path of a 'checkpoints' directory.

    Ensures the directory exists for storing checkpoint files during the test session.
    """
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    return os.path.abspath(checkpoints_dir)


@pytest.fixture(scope="session")
def instanovo_config() -> DictConfig:
    """A pytest fixture to read in a Hydra config for the Instanovo model unit and integration test."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="instanovo_unit_test")

    sub_configs_list = ["model", "dataset", "residues"]
    for sub_name in sub_configs_list:
        if sub_name in cfg:
            with open_dict(cfg):
                temp = cfg[sub_name]
                del cfg[sub_name]
                cfg.update(temp)

    return cfg


@pytest.fixture(scope="session")
def dir_paths() -> tuple[str, str]:
    """A pytest fixture that returns the root and data directories for the unit and integration tests."""
    root_dir = "./data/denovo_code_tests"
    data_dir = os.path.join(root_dir, "example_data")
    return root_dir, data_dir


@pytest.fixture(scope="session")
def _get_gcp_test_bucket(dir_paths: tuple[str, str]) -> None:
    """A pytest fixture to download the GCP data files and model checkpoint for the Instanovo model unit and integration test. A train_test folder is created for the saving of additional model checkpoints created in the model training test."""
    set_credentials()
    storage_client = storage.Client()

    root_dir, data_dir = dir_paths
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, "train_test"), exist_ok=True)

    bucket = storage_client.get_bucket("denovo_code_tests")

    blobs = bucket.list_blobs()
    for blob in blobs:
        blob.download_to_filename(os.path.join(root_dir, f"{blob.name}"))


@pytest.fixture(scope="session")
def instanovo_checkpoint(dir_paths: tuple[str, str]) -> str:
    """A pytest fixture that returns the InstaNovo model checkpoint used for unit and integration tests."""
    root_dir, _ = dir_paths
    return os.path.join(root_dir, "model.ckpt")


@pytest.fixture(scope="session")
def instanovo_model(
    instanovo_checkpoint: str, _get_gcp_test_bucket: None
) -> tuple[Any, Any]:
    """A pytest fixture that returns the InstaNovo model and config used for unit and integration tests."""
    model, config = InstaNovo.load(path=instanovo_checkpoint)
    return model, config


@pytest.fixture(scope="session")
def residue_set(instanovo_model: tuple[Any, Any]) -> Any:
    """A pytest fixture to return the model's residue set used for unit and integration tests."""
    model, config = instanovo_model
    return model.residue_set
