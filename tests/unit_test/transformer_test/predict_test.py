import copy
from typing import Any
from unittest.mock import patch, MagicMock

import polars as pl
import pytest
import torch
from omegaconf import DictConfig

from instanovo.transformer.predict import get_preds
from instanovo.transformer.predict import main
from instanovo.transformer.predict import _format_time
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
    sdf.save(data_dir, partition="example_parquet")

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


@patch("instanovo.transformer.model.InstaNovo.load")
@patch("instanovo.transformer.predict.get_preds")
def test_main(mock_get_preds: Any, mock_load: Any) -> None:
    """Test the main function call of predict."""
    mock_config = DictConfig({})
    mock_model = MagicMock()
    mock_model_config = MagicMock()
    mock_load.return_value = (mock_model, mock_model_config)

    with pytest.raises(ValueError, match="Expected data_path but found None"):
        main(mock_config)

    mock_config["data_path"] = "fake_path/test_data.ipc"

    with pytest.raises(ValueError, match="Expected model_path but found None"):
        main(mock_config)

    mock_config["model_path"] = "fake_path/model.ckpt"
    with pytest.raises(
        FileNotFoundError, match="No file found at path: fake_path/model.ckpt"
    ):
        main(mock_config)

    mock_param1 = MagicMock()
    mock_param1.numel.return_value = 100
    mock_param2 = MagicMock()
    mock_param2.numel.return_value = 100
    mock_model.parameters.return_value = [mock_param1, mock_param2]

    mock_config["save_beams"] = True
    mock_config["num_beams"] = 1

    with patch("os.path.isfile", return_value=True):
        main(mock_config)

    assert not mock_config["save_beams"]

    mock_load.assert_called_once_with(
        mock_config["model_path"], update_residues_to_unimod=True
    )
    mock_get_preds.assert_called_once_with(mock_config, mock_model, mock_model_config)


def test_format_time() -> None:
    """Test time format function."""
    assert _format_time(seconds=4567.7) == "01:16:07"
