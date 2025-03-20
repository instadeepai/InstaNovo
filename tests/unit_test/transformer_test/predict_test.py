import copy
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch
from omegaconf import DictConfig

from instanovo.transformer.predict import _format_time, get_preds
from instanovo.utils.data_handler import SpectrumDataFrame


def test_preds_with_beam(
    instanovo_model: tuple[Any, Any],
    instanovo_inference_config: DictConfig,
) -> None:
    """Test transformer inference with beam search."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instanovo_inference_config["device"] = device

    model, config = instanovo_model
    temp_config = copy.deepcopy(instanovo_inference_config)

    # Test beam search
    temp_config["save_beams"] = True
    get_preds(
        config=temp_config,
        model=model,
        model_config=config,
    )
    pred_df = pl.read_csv(temp_config["output_path"])

    assert temp_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["preds"][0] == "CADD"
    assert pred_df["log_probs"][0] == pytest.approx(-2.01, rel=1e-1)
    assert pred_df[f"preds_beam_{0}"][0] == "CADD"
    assert pred_df[f"log_probs_beam_{0}"][0] == pytest.approx(-2.01, rel=1e-1)


def test_preds_with_knapsack(
    instanovo_model: tuple[Any, Any],
    instanovo_inference_config: DictConfig,
) -> None:
    """Test transformer inference with knapsack beam search."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instanovo_inference_config["device"] = device

    model, config = instanovo_model
    temp_config = copy.deepcopy(instanovo_inference_config)

    # Test knapsack beam search
    temp_config["save_beams"] = False
    temp_config["use_knapsack"] = True

    get_preds(
        config=temp_config,
        model=model,
        model_config=config,
    )
    pred_df = pl.read_csv(temp_config["output_path"])

    assert temp_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["preds"][0] == "CADD"
    assert pred_df["log_probs"][0] == pytest.approx(-2.01, rel=1e-1)


def test_preds_with_greedy(
    instanovo_model: tuple[Any, Any],
    instanovo_inference_config: DictConfig,
) -> None:
    """Test transformer inference with greedy search."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instanovo_inference_config["device"] = device

    model, config = instanovo_model
    temp_config = copy.deepcopy(instanovo_inference_config)

    # Test greedy search
    temp_config["use_knapsack"] = False
    temp_config["num_beams"] = 1

    get_preds(
        config=temp_config,
        model=model,
        model_config=config,
    )
    pred_df = pl.read_csv(temp_config["output_path"])

    assert temp_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["preds"][0] == "CADD"
    assert pred_df["log_probs"][0] == pytest.approx(-2.01, rel=1e-1)


def test_row_drop(
    instanovo_model: tuple[Any, Any],
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference when dropping a row."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instanovo_inference_config["device"] = device
    _, data_dir = dir_paths

    model, config = instanovo_model
    temp_config = copy.deepcopy(instanovo_inference_config)

    # Test dropping rows
    temp_config["data_path"] = (
        data_dir + "/test_sample_2.ipc"  # contains an invalid sequence
    )
    get_preds(
        config=temp_config,
        model=model,
        model_config=config,
    )
    pred_df = pl.read_csv(temp_config["output_path"])
    assert len(pred_df) == 1


def test_parquet_preds(
    instanovo_model: tuple[Any, Any],
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference for parquet input."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instanovo_inference_config["device"] = device
    _, data_dir = dir_paths

    model, config = instanovo_model
    temp_config = copy.deepcopy(instanovo_inference_config)

    sdf = SpectrumDataFrame(file_paths=data_dir + "/test_sample.mgf")
    sdf.save(Path(data_dir), partition="example_parquet")

    # Test prediction on parquet file
    temp_config["data_path"] = (
        data_dir + "/dataset-ms-example_parquet-0000-0001.parquet"
    )
    get_preds(
        config=temp_config,
        model=model,
        model_config=config,
    )
    pred_df = pl.read_csv(temp_config["output_path"])
    assert pred_df["targets"][0] == "DDCA"


def test_mzml_preds(
    instanovo_model: tuple[Any, Any],
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference for mzML input."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instanovo_inference_config["device"] = device
    _, data_dir = dir_paths

    model, config = instanovo_model
    temp_config = copy.deepcopy(instanovo_inference_config)

    # Test prediction on mzml file
    temp_config["data_path"] = data_dir + "/example.mzML"
    temp_config["denovo"] = True

    get_preds(
        config=temp_config,
        model=model,
        model_config=config,
    )
    pred_df = pl.read_csv(temp_config["output_path"])
    assert len(pred_df) == 1


def test_mzxml_preds(
    instanovo_model: tuple[Any, Any],
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference for mzxml input."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instanovo_inference_config["device"] = device
    _, data_dir = dir_paths

    model, config = instanovo_model
    temp_config = copy.deepcopy(instanovo_inference_config)

    # Test prediction on mzxml file
    temp_config["data_path"] = data_dir + "/example.mzxml"
    temp_config["denovo"] = True

    get_preds(
        config=temp_config,
        model=model,
        model_config=config,
    )
    pred_df = pl.read_csv(temp_config["output_path"])
    assert len(pred_df) == 1


def test_format_time() -> None:
    """Test time format function."""
    assert _format_time(seconds=4567.7) == "01:16:07"
