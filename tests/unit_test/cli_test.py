import copy
import csv
import os
from pathlib import Path
from typing import List

import hydra
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from typer import Typer
from typer.testing import CliRunner, Result

from instanovo.cli import combined_cli, diffusion_cli, transformer_cli
from instanovo.utils.device_handler import check_device

runner = CliRunner()


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory: pytest.FixtureRequest) -> pytest.TempPathFactory:
    """Fixture for a temporary directory to store test files."""
    return tmp_path_factory.mktemp("test_data")


def test_transformer_predict(tmp_dir: Path, dir_paths: tuple[str, str], instanovo_inference_config: DictConfig) -> None:
    """Test the 'predict' command of the instanovo_cli with different input files."""
    _, data_dir = dir_paths
    input_files = [data_dir + "/example.mgf", data_dir + "/example.mzML"]
    output_files = [str(tmp_dir / "output_mgf.csv"), str(tmp_dir / "output_mzML.csv")]

    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(config=temp_config)
    config_path = data_dir + "/temp_config.yaml"
    with open(Path(config_path), "w") as f:
        OmegaConf.save(temp_config, f)

    for input_file, output_file in zip(input_files, output_files, strict=False):
        result: Result = runner.invoke(
            transformer_cli,
            [
                "predict",
                "--data-path",
                input_file,
                "--output-path",
                output_file,
                "--config-path",
                "../../" + data_dir,
                "--config-name",
                "temp_config",
                "num_beams=1",
                "--denovo",
            ],
        )
        assert result.exit_code == 0

    for output_file in output_files:
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert reader.fieldnames == [
                "experiment_name",
                "scan_number",
                "spectrum_id",
                "precursor_mz",
                "precursor_charge",
                "prediction_id",
                "predictions",
                "log_probs",
                "token_log_probs",
                "group",
                "predictions_tokenised",
                "delta_mass_ppm",
            ]
            data = rows[0]
            assert data["precursor_charge"] == "2"
            assert data["precursor_mz"] == "506.257934570313"
            assert data["experiment_name"] == "example"
            if output_file.endswith("mgf"):
                assert len(rows) == 3
                assert data["scan_number"] == "0"
                assert data["spectrum_id"] == "example:0"
            elif output_file.endswith("mzML"):
                assert len(rows) == 1
                assert data["scan_number"] == "1"
                assert data["spectrum_id"] == "example:1"


@pytest.mark.parametrize("cli", [combined_cli, transformer_cli, diffusion_cli])
def test_predict_nonexisting_config_name(cli: Typer) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting config name."""
    with pytest.raises(
        hydra.errors.MissingConfigException,
        match="Cannot find primary config 'nonexisting'. Check that it's in your config search path.",
    ):
        runner.invoke(
            cli,
            ["predict", "--config-name", "nonexisting", "--output-path", "output.csv"],
            catch_exceptions=False,
        )


@pytest.mark.parametrize("cli", [combined_cli, transformer_cli, diffusion_cli])
def test_predict_nonexisting_config_path(cli: Typer) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting config path."""
    with pytest.raises(hydra.errors.MissingConfigException, match="Primary config directory not found."):
        runner.invoke(
            cli,
            ["predict", "--config-path", "nonexisting", "--output-path", "output.csv"],
            catch_exceptions=False,
        )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], "Run predictions with InstaNovo and optionally with InstaNovo+."),
        (["--help"], "Run predictions with InstaNovo and optionally with InstaNovo+."),
        (
            ["transformer", "--help"],
            "Run predictions or train with only the transformer-based InstaNovo model.",
        ),
        (
            ["diffusion", "--help"],
            "Run predictions or train with only the diffusion-based InstaNovo+ model.",
        ),
    ],
)
def test_main_help(args: List[str], expected: str) -> None:
    """Test the main command help text is displayed."""
    result: Result = runner.invoke(combined_cli, args)
    assert result.exit_code == 0
    assert expected in result.stdout


def test_predict_transformer(caplog: pytest.FixtureRequest) -> None:
    """Test the 'predict' command of the instanovo_cli."""
    with caplog.at_level("INFO"):
        cmd_args = ["transformer", "predict", "--config-name", "unit_test"]
        if torch.backends.mps.is_available():
            cmd_args.extend(["mps=true"])

        result: Result = runner.invoke(combined_cli, cmd_args)
        assert result.exit_code == 0
        assert "data_path: tests/instanovo_test_resources/example_data/test_sample.mgf" in caplog.text
        assert "max_charge: 3" in caplog.text
        assert "denovo: false" in caplog.text

        assert "instanovo_model: tests/instanovo_test_resources/model.ckpt" in caplog.text
        assert "output_path: tests/instanovo_test_resources/test_sample_preds.csv" in caplog.text
        assert "knapsack_path: tests/instanovo_test_resources/example_knapsack" in caplog.text
        assert "use_knapsack: false" in caplog.text
        assert "num_beams: 5" in caplog.text
        assert "max_length: 6" in caplog.text


def test_predict_transformer_pipeline(caplog: pytest.FixtureRequest) -> None:
    """Test the 'predict' command of the instanovo_cli."""
    with caplog.at_level("INFO"):
        cmd_args = ["transformer", "predict", "--config-name", "pipeline_unit_test"]
        if torch.backends.mps.is_available():
            cmd_args.extend(["mps=true"])

        result: Result = runner.invoke(combined_cli, cmd_args)
        assert result.exit_code == 0
        assert "max_charge: 3" in caplog.text
        assert "denovo: false" in caplog.text

        assert "instanovo_model: tests/instanovo_test_resources/model.ckpt" in caplog.text
        assert "result_name: test" in caplog.text
        assert "input_path: tests/instanovo_test_resources/example_data/test_sample.mgf" in caplog.text
        assert "output_path: tests/instanovo_test_resources/test_sample_preds.csv" in caplog.text
        assert "result_file_path: tests/instanovo_test_resources/instanovo_results.csv" in caplog.text
        assert "knapsack_path: tests/instanovo_test_resources/example_knapsack" in caplog.text
        assert "use_knapsack: false" in caplog.text
        assert "num_beams: 1" in caplog.text
        assert "max_length: 6" in caplog.text


def test_predict_diffusion(caplog: pytest.FixtureRequest) -> None:
    """Test the 'diffusion predict' command of the instanovo_cli."""
    with caplog.at_level("INFO"):
        cmd_args = ["diffusion", "predict", "--config-name", "instanovoplus_unit_test"]
        if torch.backends.mps.is_available():
            cmd_args.extend(["mps=true"])

        result: Result = runner.invoke(combined_cli, cmd_args)
        assert result.exit_code == 0
        assert "data_path: tests/instanovo_test_resources/example_data/test_sample.mgf" in caplog.text
        assert "max_charge: 3" in caplog.text
        assert "denovo: false" in caplog.text
        assert "instanovo_plus_model: tests/instanovo_test_resources/instanovoplus" in caplog.text
        assert "output_path: tests/instanovo_test_resources/instanovoplus/test_sample_preds.csv" in caplog.text
        assert "knapsack_path: null" in caplog.text
        assert "max_length: 6" in caplog.text


def test_predict(caplog: pytest.FixtureRequest) -> None:
    """Test the 'predict' command of the instanovo_cli."""
    with caplog.at_level("INFO"):
        cmd_args = ["predict", "--config-name", "unit_test"]
        if torch.backends.mps.is_available():
            cmd_args.extend(["mps=true"])

        result: Result = runner.invoke(combined_cli, cmd_args)
        assert result.exit_code == 0
        assert "data_path: tests/instanovo_test_resources/example_data/test_sample.mgf" in caplog.text
        assert "max_charge: 3" in caplog.text
        assert "denovo: false" in caplog.text

        assert "instanovo_model: tests/instanovo_test_resources/model.ckpt" in caplog.text
        assert "output_path: tests/instanovo_test_resources/test_sample_preds.csv" in caplog.text
        assert "knapsack_path: tests/instanovo_test_resources/example_knapsack" in caplog.text
        assert "use_knapsack: false" in caplog.text
        assert "num_beams: 5" in caplog.text
        assert "max_length: 6" in caplog.text


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            [
                "predict",
                "--instanovo-model",
                "nonexisting",
                "--output-path",
                "predictions.csv",
            ],
            "InstaNovo",
        ),
        # (["predict", "--instanovo-plus-model", "nonexisting"], "InstaNovo+"), #TODO
        (
            [
                "predict",
                "--instanovo-model",
                "nonexisting",
                "--instanovo-plus-model",
                "nonexisting",
                "--output-path",
                "predictions.csv",
            ],
            "InstaNovo",
        ),
        (
            [
                "transformer",
                "predict",
                "--instanovo-model",
                "nonexisting",
                "--output-path",
                "predictions.csv",
            ],
            "InstaNovo",
        ),
        #    (["diffusion", "predict", "--instanovo-plus-model", "nonexisting"], "InstaNovo+") #TODO
    ],
)
def test_model(
    args: List[str],
    expected: str,
    dir_paths: tuple[str, str],
) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting instanovo model."""
    _, data_dir = dir_paths
    with pytest.raises(ValueError, match=f"{expected} model ID 'nonexisting' is not supported."):
        runner.invoke(
            combined_cli,
            args + ["--data-path", data_dir + "/example.mgf"],
            catch_exceptions=False,
        )


@pytest.mark.parametrize(
    "args",
    [
        [
            "predict",
            "--instanovo-model",
            "checkpoint.invalid",
            "--output-path",
            "predictions.csv",
        ],
        [
            "transformer",
            "predict",
            "--instanovo-model",
            "checkpoint.invalid",
            "--output-path",
            "predictions.csv",
        ],
    ],
)
def test_instanovo_model_suffix(args: List[str], dir_paths: tuple[str, str]) -> None:
    """Test the 'predict' command of the instanovo cli with a file with a non-supported suffix."""
    _, data_dir = dir_paths
    with runner.isolated_filesystem():
        with open("checkpoint.invalid", "w") as f:
            f.write("dummy checkpoint file")
        with pytest.raises(
            ValueError,
            match="Checkpoint file 'checkpoint.invalid' should end with extension '.ckpt'.",
        ):
            runner.invoke(
                combined_cli,
                args + ["--data-path", data_dir + "/example.mgf"],
                catch_exceptions=False,
            )


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        (".ckpt", r"\*.yaml, \*.pt"),
        (".yaml", r"\*.ckpt, \*.pt"),
        (".pt", r"\*.ckpt, \*.yaml"),
        (".txt", r"\*.ckpt, \*.yaml, \*.pt"),
    ],
)
@pytest.mark.parametrize(
    "args",
    [
        # ["predict", "--instanovo-plus-model", "checkpoint_dir"], #TODO
        [
            "diffusion",
            "predict",
            "--instanovo-plus-model",
            "checkpoint_dir",
            "--output-path",
            "predictions.csv",
        ]
    ],
)
def test_instanovoplus_model(
    args: List[str],
    extension: str,
    expected: str,
    dir_paths: tuple[str, str],
) -> None:
    """Test the 'predict' command of the instanovo plus cli with a non existing file."""
    _, data_dir = dir_paths
    with runner.isolated_filesystem():
        os.mkdir("checkpoint_dir")
        with open(f"checkpoint_dir/file.{extension}", "w") as f:
            f.write("dummy checkpoint file")

        with pytest.raises(
            ValueError,
            match=(
                r"The directory 'checkpoint_dir' is missing the following required file\(s\): "
                f"{expected}."
            ),
        ):
            runner.invoke(
                combined_cli,
                args + ["--data-path", data_dir + "/example.mgf"],
                catch_exceptions=False,
            )


@pytest.mark.parametrize(
    "args",
    [
        ["predict", "--data-path", "*.mgf", "--output-path", "output.csv"],
        ["transformer", "predict", "--data-path", "*.mgf", "--output-path", "output.csv"],
        ["diffusion", "predict", "--data-path", "*.mgf", "--output-path", "output.csv"],
    ],
)
def test_nonexisting_data_path(args: List[str]) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting data path."""
    with runner.isolated_filesystem():
        with pytest.raises(
            ValueError,
            match=r"The data_path '\*.mgf' doesn't correspond to any file\(s\).",
        ):
            runner.invoke(combined_cli, args, catch_exceptions=False)


@pytest.mark.parametrize(
    "args",
    [
        ["predict", "--config-name", "default", "--output-path", "output.csv"],
        ["transformer", "predict", "--config-name", "default", "--output-path", "output.csv"],
        ["diffusion", "predict", "--config-name", "default", "--output-path", "output.csv"],
    ],
)
def test_no_data_path(args: List[str]) -> None:
    """Test the 'predict' command of the instanovo_cli with no data path."""
    with runner.isolated_filesystem():
        with pytest.raises(
            ValueError,
            match=(
                r"Expected 'data_path' but found None. Please specify it in the "
                r"`config/inference/<your_config>.yaml` configuration file or with the cli flag "
                r"`--data-path='path/to/data'`. Allows `.mgf`, `.mzml`, `.mzxml`, a directory, or a"
                r" `.parquet` file. Glob notation is supported:  eg.: "
                r"`--data-path='./experiment/\*.mgf'`."
            ),
        ):
            runner.invoke(combined_cli, args, catch_exceptions=False)


@pytest.mark.parametrize(
    "args",
    [
        ["predict", "--config-name", "default"],
        ["transformer", "predict", "--config-name", "default"],
        ["diffusion", "predict", "--config-name", "default"],
    ],
)
def test_no_output_path(args: List[str], dir_paths: tuple[str, str]) -> None:
    """Test the 'predict' command of the instanovo_cli with no output path."""
    _, data_dir = dir_paths

    with runner.isolated_filesystem():
        with pytest.raises(
            ValueError,
            match=(
                r"Expected 'output_path' but found None (when refining|in denovo mode|"
                r"in denovo or refine mode). Please specify it in the "
                r"`config/inference/<your_config>.yaml` configuration "
                r"file or with the cli flag `--output-path=path/to/output_file`."
            ),
        ):
            runner.invoke(
                combined_cli,
                args + ["--data-path", data_dir + "/example.mgf"],
                catch_exceptions=False,
            )


def test_version_command() -> None:
    """Test the version command."""
    result: Result = runner.invoke(combined_cli, ["version"])
    assert result.exit_code == 0
    assert "InstaNovo" in result.stdout
    assert "InstaNovo+" in result.stdout
    assert "PyTorch" in result.stdout
