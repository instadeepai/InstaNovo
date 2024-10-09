from __future__ import annotations

import logging
import os

import polars as pl
import pytest
import torch
from omegaconf import DictConfig

from instanovo.transformer.model import InstaNovo
from instanovo.transformer.predict import get_preds
from instanovo.transformer.train import train

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="To reduce computational runtime, this test is skipped unless a GPU is available.",
)
@pytest.mark.usefixtures("_reset_seed")
def test_train_model(
    instanovo_config: DictConfig,
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test training an InstaNovo model and doing inference end-to-end."""
    root_dir, _ = dir_paths
    assert instanovo_config.residues == {
        "A": 10.5,
        "B": 20.75,
        "C": 15.68,
        "D": 18.25,
        "E": 12.33,
    }
    assert instanovo_config.learning_rate == 1e-3
    assert instanovo_config.model_save_folder_path == os.path.join(
        root_dir, "train_test"
    )
    assert instanovo_config.epochs == 5

    logger.info("Training model.")
    train(instanovo_config)

    logger.info("Loading model.")
    model, config = InstaNovo.load(
        os.path.join(root_dir, "train_test", "epoch=4-step=2420.ckpt")
    )

    logger.info("Computing predictions and saving to specified file.")
    get_preds(
        config=instanovo_inference_config,
        model=model,
        model_config=config,
    )

    pred_df = pl.read_csv(instanovo_inference_config["output_path"])
    print(pred_df)
    assert instanovo_inference_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    # Even with fixed seeds, model training can introduce randomness, and so we perform a general check on the predicted peptide lengths.
    assert len(pred_df["targets"][0]) == len(pred_df["preds"][0])
