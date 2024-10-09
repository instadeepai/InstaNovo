from __future__ import annotations

import os
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as ptl
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict

from instanovo.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
from instanovo.transformer.model import InstaNovo
from instanovo.transformer.predict import _setup_knapsack


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
    """A pytest fixture to read in a Hydra config for the Instanovo model unit and integration tests."""
    with initialize(version_base=None, config_path="../instanovo/configs"):
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
def instanovo_inference_config() -> DictConfig:
    """A pytest fixture to read in a Hydra config for inference of the Instanovo model unit and integration tests."""
    with initialize(version_base=None, config_path="../instanovo/configs/inference"):
        cfg = compose(config_name="unit_test")

    return cfg


@pytest.fixture(scope="session")
def dir_paths() -> tuple[str, str]:
    """A pytest fixture that returns the root and data directories for the unit and integration tests."""
    root_dir = "./tests/instanovo_test_resources"
    data_dir = os.path.join(root_dir, "example_data")
    return root_dir, data_dir


@pytest.fixture(scope="session")
def instanovo_checkpoint(dir_paths: tuple[str, str]) -> str:
    """A pytest fixture that returns the InstaNovo model checkpoint used for unit and integration tests."""
    root_dir, _ = dir_paths
    return os.path.join(root_dir, "model.ckpt")


@pytest.fixture(scope="session")
def instanovo_model(
    instanovo_checkpoint: str,
) -> tuple[Any, Any]:
    """A pytest fixture that returns the InstaNovo model and config used for unit and integration tests."""
    model, config = InstaNovo.load(path=instanovo_checkpoint)
    return model, config


@pytest.fixture(scope="session")
def residue_set(instanovo_model: tuple[Any, Any]) -> Any:
    """A pytest fixture to return the model's residue set used for unit and integration tests."""
    model, _ = instanovo_model
    return model.residue_set


@pytest.fixture(scope="session")
def instanovo_output(dir_paths: tuple[str, str]) -> pd.DataFrame:
    """A pytest fixture to load the pre-computed InstaNovo model predictions for unit and integration tests."""
    root_dir, _ = dir_paths
    return pd.read_csv(os.path.join(root_dir, "predictions.csv"))


@pytest.fixture(scope="session")
def knapsack_dir(dir_paths: tuple[str, str]) -> str:
    """A pytest fixture to create and provide the absolute path of a 'knapsack' directory within the checkpoints directory for storing test artifacts."""
    root_dir, _ = dir_paths
    knapsack_dir = os.path.join(root_dir, "example_knapsack")
    return os.path.abspath(knapsack_dir)


@pytest.fixture(scope="session")
def setup_knapsack_decoder(
    instanovo_model: tuple[Any, Any], knapsack_dir: str
) -> KnapsackBeamSearchDecoder:
    """A pytest fixture to create a Knapsack object."""
    model, _ = instanovo_model

    if os.path.exists(knapsack_dir):
        decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_dir)
        print("Loaded knapsack decoder.")

    else:
        knapsack = _setup_knapsack(model)

        knapsack.save(path=knapsack_dir)
        print("Created and saved knapsack.")

        decoder = KnapsackBeamSearchDecoder(model, knapsack)
        print("Loaded knapsack decoder.")

    return decoder
