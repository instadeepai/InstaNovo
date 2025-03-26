from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import polars as pl
import pytest
import torch
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.transformer.model import InstaNovo
from instanovo.transformer.predict import get_preds
from instanovo.transformer.train import train
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="To reduce computational runtime, this test is skipped unless a GPU is available.",
)
@pytest.mark.usefixtures("_reset_seed")
def test_train_model(
    tmp_path: Path,
    instanovo_config: DictConfig,
    instanovo_inference_config: DictConfig,
) -> None:
    """Test training an InstaNovo model and doing inference end-to-end."""
    assert instanovo_config.residues == {
        "A": 10.5,
        "B": 20.75,
        "C": 15.68,
        "D": 18.25,
        "E": 12.33,
    }
    assert instanovo_config.learning_rate == 1e-3
    assert instanovo_config.epochs == 1

    temp_train_config = copy.deepcopy(instanovo_config)
    temp_inference_config = copy.deepcopy(instanovo_inference_config)

    temp_train_config["model_save_folder_path"] = str(
        tmp_path
    )  # save the model in a temporary directory

    temp_train_config = copy.deepcopy(instanovo_config)
    temp_inference_config = copy.deepcopy(instanovo_inference_config)

    temp_train_config["model_save_folder_path"] = str(
        tmp_path
    )  # save the model in a temporary directory

    logger.info("Training model.")
    train(temp_train_config)

    logger.info("Loading model.")
    checkpoint_path = os.path.join(
        temp_train_config["model_save_folder_path"], "epoch=0-step=480.ckpt"
    )
    assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
    model, config = InstaNovo.load(checkpoint_path)

    temp_inference_config["output_path"] = os.path.join(
        temp_train_config["model_save_folder_path"], "test_sample_preds.csv"
    )
    logger.info("Computing predictions and saving to specified file.")
    get_preds(
        config=temp_inference_config,
        model=model,
        model_config=config,
    )

    assert os.path.exists(temp_inference_config["output_path"]), (
        f"Output file {temp_inference_config['output_path']} does not exist."
    )

    pred_df = pl.read_csv(temp_inference_config["output_path"])
    assert pred_df["targets"][0] == "DDCA"

    if tmp_path is not None and os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
