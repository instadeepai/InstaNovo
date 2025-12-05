from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import torch
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.diffusion.predict import DiffusionPredictor
from instanovo.diffusion.train import DiffusionTrainer
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.device_handler import check_device
from instanovo.utils.s3 import S3FileHandler
from tests.conftest import reset_seed

logger = ColorLog(console, __name__).logger


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.mps.is_available(),
    reason="To reduce computational runtime, this test is skipped unless a GPU is available.",
)
@pytest.mark.usefixtures("_reset_seed")
def test_train_model(
    tmp_path: Path,
    instanovoplus_config: DictConfig,
    instanovoplus_inference_config: DictConfig,
) -> None:
    """Test training an InstaNovo model and doing inference end-to-end."""
    # Monkey patch _aichor_enabled to always return False in tests
    # to prevent PermissionErrors when trying to write to s3 bucket
    S3FileHandler._aichor_enabled = lambda: False  # type: ignore[method-assign]

    assert instanovoplus_config.residues.residues == {
        "A": 10.5,
        "B": 20.75,
        "C": 15.68,
        "D": 18.25,
        "E": 12.33,
    }
    assert instanovoplus_config.learning_rate == 1e-3
    # assert instanovoplus_config.training_steps == 481

    temp_train_config = copy.deepcopy(instanovoplus_config)
    temp_inference_config = copy.deepcopy(instanovoplus_inference_config)

    temp_train_config["model_save_folder_path"] = str(tmp_path)  # save the model in a temporary directory

    with patch.object(S3FileHandler, "_aichor_enabled", return_value=False):
        logger.info("Training model.")
        check_device(config=temp_train_config)
        trainer = DiffusionTrainer(temp_train_config)
        reset_seed()
        trainer.train()

        logger.info("Loading model.")
        checkpoint_path = os.path.join(temp_train_config["model_save_folder_path"], "model_latest.ckpt")
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."

        temp_inference_config["output_path"] = os.path.join(temp_train_config["model_save_folder_path"], "test_sample_preds.csv")
        logger.info("Computing predictions and saving to specified file.")

        check_device(config=temp_inference_config)
        predictor = DiffusionPredictor(temp_inference_config)
        reset_seed()
        predictor.predict()

        assert os.path.exists(temp_inference_config["output_path"]), f"Output file {temp_inference_config['output_path']} does not exist."

        pred_df = pl.read_csv(temp_inference_config["output_path"])
        assert pred_df["targets"][0] == "DDCA"

        if tmp_path is not None and os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
