import copy
from pathlib import Path

import polars as pl
import pytest
from omegaconf import DictConfig

from instanovo.transformer.predict import TransformerPredictor
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.device_handler import check_device


def test_preds_with_beam(
    instanovo_inference_config: DictConfig,
) -> None:
    """Test transformer inference with beam search."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)

    # Test beam search
    temp_config["save_all_predictions"] = True

    TransformerPredictor(temp_config).predict()
    pred_df = pl.read_csv(temp_config["output_path"])

    assert temp_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["predictions"][0] == "CADD"
    assert pred_df["log_probs"][0] == pytest.approx(-2.01, rel=1e-1)
    assert pred_df[f"predictions_beam_{0}"][0] == "CADD"
    assert pred_df[f"predictions_log_probability_beam_{0}"][0] == pytest.approx(-2.01, rel=1e-1)


def test_preds_with_knapsack(
    instanovo_inference_config: DictConfig,
) -> None:
    """Test transformer inference with knapsack beam search."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)

    # Test knapsack beam search
    temp_config["save_all_predictions"] = False
    temp_config["use_knapsack"] = True

    TransformerPredictor(temp_config).predict()
    pred_df = pl.read_csv(temp_config["output_path"])

    assert temp_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["predictions"][0] == "CADD"
    assert pred_df["log_probs"][0] == pytest.approx(-2.01, rel=1e-1)


def test_preds_with_greedy(
    instanovo_inference_config: DictConfig,
) -> None:
    """Test transformer inference with greedy search."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)

    # Test greedy search
    temp_config["use_knapsack"] = False
    temp_config["num_beams"] = 1

    TransformerPredictor(temp_config).predict()
    pred_df = pl.read_csv(temp_config["output_path"])

    assert temp_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["predictions"][0] == "CADD"
    assert pred_df["log_probs"][0] == pytest.approx(-2.01, rel=1e-1)


def test_row_drop(
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference when dropping a row."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)

    _, data_dir = dir_paths

    # Test dropping rows
    temp_config["data_path"] = (
        data_dir + "/test_sample_2.ipc"  # contains an invalid sequence
    )
    TransformerPredictor(temp_config).predict()
    pred_df = pl.read_csv(temp_config["output_path"])
    assert len(pred_df) == 1


def test_parquet_preds(
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference for parquet input."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)

    _, data_dir = dir_paths

    sdf = SpectrumDataFrame(file_paths=data_dir + "/test_sample.mgf")
    sdf.save(Path(data_dir), partition="example_parquet")

    # Test prediction on parquet file
    temp_config["data_path"] = data_dir + "/dataset-ms-example_parquet-0000-0001.parquet"
    TransformerPredictor(temp_config).predict()
    pred_df = pl.read_csv(temp_config["output_path"])
    assert pred_df["targets"][0] == "DDCA"


def test_mzml_preds(
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference for mzML input."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)

    _, data_dir = dir_paths

    # Test prediction on mzml file
    temp_config["data_path"] = data_dir + "/example.mzML"
    temp_config["denovo"] = True

    TransformerPredictor(temp_config).predict()
    pred_df = pl.read_csv(temp_config["output_path"])
    assert len(pred_df) == 1


def test_mzxml_preds(
    instanovo_inference_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test transformer inference for mzxml input."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)

    _, data_dir = dir_paths

    # Test prediction on mzxml file
    temp_config["data_path"] = data_dir + "/example.mzxml"
    temp_config["denovo"] = True

    TransformerPredictor(temp_config).predict()
    pred_df = pl.read_csv(temp_config["output_path"])
    assert len(pred_df) == 1
