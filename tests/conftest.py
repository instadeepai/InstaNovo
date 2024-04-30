from __future__ import annotations

import json
import os
import random
import sys
import zipfile
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as ptl
import requests
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from instanovo.constants import MASS_SCALE
from instanovo.diffusion.multinomial_diffusion import MultinomialDiffusion
from instanovo.inference.diffusion import DiffusionDecoder
from instanovo.inference.knapsack import Knapsack
from instanovo.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo

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
def instanovoplus_checkpoint(checkpoints_dir: str) -> str:
    """A pytest fixture to download and provide the path of the InstaNovo+ model checkpoint.

    Downloads from a predefined URL if the checkpoint file doesn't exist locally.
    """
    url = "https://github.com/instadeepai/InstaNovo/releases/download/0.1.5/instanovoplus_yeast.zip"
    zip_file_path = os.path.join(checkpoints_dir, "instanovoplus_yeast.zip")
    checkpoint_path = os.path.join(checkpoints_dir, "diffusion_checkpoint")

    if not os.path.isdir(checkpoint_path):
        response = requests.get(url)
        with open(zip_file_path, "wb") as file:
            file.write(response.content)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(checkpoints_dir)
    return os.path.abspath(checkpoint_path)


@pytest.fixture(scope="session")
def dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """A pytest fixture to load and provide a dataset for testing.

    Loads a specific subset (1% of test data) from the 'ms_ninespecies_benchmark' dataset.
    """
    return load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test[:1%]")


@pytest.fixture(scope="session")
def instanovo_model(instanovo_checkpoint: str) -> tuple[Any, Any]:
    """A pytest fixture to load an InstaNovo model from a specified checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = InstaNovo.load(instanovo_checkpoint)
    model = model.to(device).eval()

    return model, config


@pytest.fixture(scope="session")
def instanovoplus_model(
    instanovoplus_checkpoint: str,
) -> tuple[MultinomialDiffusion, DiffusionDecoder]:
    """A pytest fixture to load an InstaNovo+ model from a specified checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_model = MultinomialDiffusion.load(instanovoplus_checkpoint)
    diffusion_model = diffusion_model.to(device).eval()
    diffusion_decoder = DiffusionDecoder(model=diffusion_model)

    return diffusion_model, diffusion_decoder


@pytest.fixture(scope="session")
def knapsack_dir(checkpoints_dir: str) -> str:
    """A pytest fixture to create and provide the absolute path of a 'knapsack' directory within the checkpoints directory for storing test artifacts."""
    knapsack_dir = os.path.join(checkpoints_dir, "knapsack")
    return os.path.abspath(knapsack_dir)


@pytest.fixture(scope="session")
def setup_knapsack_decoder(
    instanovo_model: tuple[Any, Any], knapsack_dir: str
) -> KnapsackBeamSearchDecoder:
    """A pytest fixture to create a Knapsack object."""
    model, config = instanovo_model

    if os.path.exists(knapsack_dir):
        decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_dir)
        print("Loaded knapsack decoder.")

    else:
        residue_masses = model.peptide_mass_calculator.masses
        residue_masses["$"] = 0
        residue_indices = model.decoder._aa2idx

        knapsack = Knapsack.construct_knapsack(
            residue_masses=residue_masses,
            residue_indices=residue_indices,
            max_mass=4000.00,
            mass_scale=MASS_SCALE,
        )

        knapsack.save(path=knapsack_dir)
        print("Created and saved knapsack.")

        decoder = KnapsackBeamSearchDecoder(model, knapsack)
        print("Loaded knapsack decoder.")

    return decoder


@pytest.fixture(scope="session")
def load_preds(
    instanovo_model: InstaNovo,
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    setup_knapsack_decoder: KnapsackBeamSearchDecoder,
) -> list[str]:
    """A pytest fixture to fetch/load predictions from the InstaNovo model to be used as input to the InstaNovo+ model."""
    file_path = "tests/instanovo_predictions.json"
    preds = []

    if os.path.exists(file_path):
        print("Loading InstaNovo predictions from JSON file.")
        with open(file_path) as f:
            preds = json.load(f)

    else:
        print("Computing InstaNovo predictions and saving to JSON file.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        preds = []

        model, config = instanovo_model
        s2i = {v: k for k, v in model.i2s.items()}

        ds = SpectrumDataset(df=dataset, s2i=s2i, n_peaks=config["n_peaks"], return_str=True)
        dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_batch)

        output_file = "tests/instanovo_predictions.json"

        with open(output_file, "w") as json_file:
            for _, batch in tqdm(enumerate(dl), total=len(dl)):
                spectra, precursors, _, peptides, _ = batch
                spectra = spectra.to(device)
                precursors = precursors.to(device)

                with torch.no_grad():
                    p = setup_knapsack_decoder.decode(
                        spectra=spectra,
                        precursors=precursors,
                        beam_size=config["n_beams"],
                        max_length=config["max_length"],
                    )

                preds += ["".join(x.sequence) if not isinstance(x, list) else "" for x in p]

            json.dump(preds, json_file)
    return preds
