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
    os.getenv("AICHOR_ORGANIZATION_NAME") is None,
    reason="We only run this test on Aichor.",
)
@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
@pytest.mark.usefixtures("_get_gcp_test_bucket")
@pytest.mark.usefixtures("_reset_seed")
def test_model(
    instanovo_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Training an InstaNovo model and doing inference end-to-end."""
    root_dir, data_dir = dir_paths
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_path = os.path.join(root_dir, "train_test", "test_preds.csv")

    logger.info("Computing predictions and saving to specified file.")
    get_preds(
        data_path=os.path.join(data_dir, "test.ipc"),
        model=model,
        config=config,
        denovo=False,
        output_path=pred_path,
        device=device,
    )

    pred_df = pl.read_csv(pred_path)

    assert config["subset"] == 0.01
    assert pred_df["targets"][0] == "BCDEBD"
    assert (
        pred_df["preds"][0] == "DBDECB"
        if os.getenv("AICHOR_ORGANIZATION_NAME")
        else "BEBDDC"
    )

    if os.getenv("AICHOR_ORGANIZATION_NAME"):
        assert pred_df["log_probs"][0] == pytest.approx(-5.2449049949646, rel=1e-2)
    else:
        assert pred_df["log_probs"][0] == pytest.approx(-3.870894432067871, rel=1e-2)
