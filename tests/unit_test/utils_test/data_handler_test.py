import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import polars as pl
import pytest
from numpy import array, nan

from instanovo.utils.data_handler import SpectrumDataFrame
from tests.conftest import reset_seed


def test_init() -> None:
    """Test spectrum data frame basic initialisation."""
    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)
    sdf = SpectrumDataFrame(df)

    assert not sdf._is_annotated
    assert not sdf._has_predictions
    assert not sdf._is_lazy
    assert not sdf._shuffle
    assert sdf._max_shard_size == 100000
    assert sdf._custom_load_fn is None
    assert sdf.executor is None
    assert sdf._temp_directory is None

    assert not sdf._is_native

    assert df.equals(sdf.df)

    assert sdf._current_index_in_file == (0)
    assert sdf._next_file_index == 0
    assert sdf._current_file is None
    assert sdf._current_file_len == 0
    assert sdf._current_file_position == 0


def test_properties() -> None:
    """Test spectrum data frame properties."""
    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)
    sdf = SpectrumDataFrame(df)

    assert not sdf.is_annotated
    assert not sdf.has_predictions
    assert not sdf.is_lazy


def test_errors(tmp_path: Any, dir_paths: tuple[str, str]) -> None:
    """Test data handler error catching."""
    with pytest.raises(ValueError, match="Must specify either df or file_paths, both are None."):
        _ = SpectrumDataFrame()

    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    with pytest.raises(ValueError, match="Must specify either df or file_paths, not both."):
        _ = SpectrumDataFrame(df=df, file_paths=tmp_path)

    _, data_dir = dir_paths
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input must be a string (filepath or glob) or a list of file paths. Found directory."
        ),
    ):
        _ = SpectrumDataFrame(file_paths=data_dir)


def test_spec_init(dir_paths: tuple[str, str]) -> None:
    """Test spectrum data frame specified initialisation."""
    _, data_dir = dir_paths

    sdf = SpectrumDataFrame(file_paths=data_dir + "/valid.ipc", is_annotated=True)

    assert not sdf._is_native
    assert sdf._file_paths == []


def test_lazy(dir_paths: tuple[str, str]) -> None:
    """Test spectrum data frame lazy loading."""
    _, data_dir = dir_paths

    sdf = SpectrumDataFrame(file_paths=data_dir + "/valid.ipc", is_annotated=True, is_lazy=True)

    assert sdf._is_native
    assert sdf._temp_directory is not None
    assert sdf._file_paths is not None

    sdf = SpectrumDataFrame(file_paths=data_dir + "/val*.ipc", is_annotated=True, is_lazy=True)

    assert sdf._is_native
    assert sdf._temp_directory is not None
    assert sdf._file_paths is not None

    assert isinstance(sdf.executor, ThreadPoolExecutor)
    assert sdf.loop is not None


def test_sanitise_peptides() -> None:
    """Test sanitise peptide sequence."""
    assert SpectrumDataFrame._sanitise_peptide("_ABC_") == "ABC"
    assert SpectrumDataFrame._sanitise_peptide(".ABC.") == "ABC"


def test_length() -> None:
    """Test get length."""
    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)
    sdf = SpectrumDataFrame(df)

    assert len(sdf) == 1


def test_get_item() -> None:
    """Test fetching item at specified index."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25],
            [8.84, 19.215, 21.8, 29.64, 67.6, 39.55, 30.81, 50.965, 52.25, 28.25],
            [9.84, 20.215, 22.8, 30.64, 68.6, 40.55, 31.81, 51.965, 53.25, 29.25],
        ],
        "intensity_array": [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ],
        "precursor_mz": [35.83, 36.83, 37.83],
        "precursor_charge": [2, 3, 4],
        "sequence": ["ABCDEAAABC", "XYZVWXYZVW", "_MNOPQRSMNO_"],
    }
    df = pl.DataFrame(data)
    sdf = SpectrumDataFrame(df, is_annotated=True)

    assert sdf[2] == {
        "mz_array": [
            9.84,
            20.215,
            22.8,
            30.64,
            68.6,
            40.55,
            31.81,
            51.965,
            53.25,
            29.25,
        ],
        "intensity_array": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        "precursor_mz": 37.83,
        "precursor_charge": 4,
        "sequence": "MNOPQRSMNO",
    }

    reset_seed()
    sdf = SpectrumDataFrame(df, is_annotated=True, shuffle=True)
    assert sdf._current_index_in_file == 0
    assert sdf[1] == {
        "mz_array": [
            7.84,
            18.215,
            20.8,
            28.64,
            66.6,
            38.55,
            29.81,
            49.965,
            51.25,
            27.25,
        ],
        "intensity_array": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "precursor_mz": 35.83,
        "precursor_charge": 2,
        "sequence": "ABCDEAAABC",
    }


def test_check_type_spec() -> None:
    """Test data checker."""
    data: dict = {}
    df = pl.DataFrame(data)

    with pytest.raises(
        ValueError,
        match=(
            "Columns missing! Missing columns: "
            "mz_array, intensity_array, precursor_mz, precursor_charge"
        ),
    ):
        _ = SpectrumDataFrame(df)

    with pytest.raises(
        ValueError,
        match=(
            "Columns missing! Missing columns: "
            "mz_array, intensity_array, precursor_mz, precursor_charge, sequence"
        ),
    ):
        _ = SpectrumDataFrame(df, is_annotated=True)

    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "",
    }
    df = pl.DataFrame(data)

    with pytest.raises(
        ValueError,
        match=(
            "Attempting to load annotated dataset, but some or all sequence "
            "annotations are missing."
        ),
    ):
        _ = SpectrumDataFrame(df=df, is_annotated=True)


def test_parquet_init(dir_paths: tuple[str, str], tmp_path: Any) -> None:
    """Test spectrum data frame parquet creation and initialisation."""
    _, data_dir = dir_paths
    sdf = SpectrumDataFrame(
        file_paths=[data_dir + "/example.mgf"],
        is_annotated=True,
        is_lazy=False,
        shuffle=False,
    )

    assert not sdf._is_native
    sdf.save(tmp_path, partition="example_mgf")

    expected_df = pd.DataFrame(
        {
            "scan_number": [0, 1, 2],
            "sequence": ["FHHTIGGSR", "GPAGPQGPR", "TTVINM[15.99]PR"],
            "precursor_mass": [1010.501317140626, 1253.140739138673, 946.491246339844],
            "precursor_mz": [506.257934570313, 418.720855712891, 474.252899169922],
            "precursor_charge": [2, 3, 2],
            "retention_time": [100.0, 200.0, 300.0],
            "mz_array": [
                array([10.0, 20.0, 30.0, 40.0]),
                array([10.0, 20.0, 30.0, 40.0]),
                array([10.0, 20.0, 30.0, 40.0]),
            ],
            "intensity_array": [
                array([1.0, 1.5, 1.0, 1.5]),
                array([1.0, 1.5, 1.0, 1.5]),
                array([1.0, 1.5, 1.0, 1.5]),
            ],
            "experiment_name": ["example", "example", "example"],
            "spectrum_id": ["example:0", "example:1", "example:2"],
        }
    )

    path = tmp_path / "dataset-ms-example_mgf-0000-0001.parquet"
    df = pl.read_parquet(path).to_pandas()

    assert df.equals(expected_df)

    sdf = SpectrumDataFrame(
        file_paths=[str(path)],
        is_annotated=True,
        shuffle=True,
        is_lazy=True,
    )

    assert sdf._is_native

    sdf = SpectrumDataFrame(
        file_paths=[str(path)],
        is_annotated=True,
        shuffle=True,
        is_lazy=False,
    )

    assert not sdf._is_native


def test_mgf_to_parquet(dir_paths: tuple[str, str], tmp_path: Any) -> None:
    """Test mgf to parquet conversion with lazy loading."""
    _, data_dir = dir_paths
    sdf = SpectrumDataFrame(
        file_paths=[data_dir + "/example.mgf"],
        is_annotated=True,
        is_lazy=True,
        shuffle=False,
    )

    sdf.save(tmp_path, partition="example_lazy_mgf")

    expected_df = pd.DataFrame(
        {
            "scan_number": [0, 1, 2],
            "sequence": ["FHHTIGGSR", "GPAGPQGPR", "TTVINM[15.99]PR"],
            "precursor_mass": [1010.501317140626, 1253.140739138673, 946.491246339844],
            "precursor_mz": [506.257934570313, 418.720855712891, 474.252899169922],
            "precursor_charge": [2, 3, 2],
            "retention_time": [100.0, 200.0, 300.0],
            "mz_array": [
                array([10.0, 20.0, 30.0, 40.0]),
                array([10.0, 20.0, 30.0, 40.0]),
                array([10.0, 20.0, 30.0, 40.0]),
            ],
            "intensity_array": [
                array([1.0, 1.5, 1.0, 1.5]),
                array([1.0, 1.5, 1.0, 1.5]),
                array([1.0, 1.5, 1.0, 1.5]),
            ],
            "experiment_name": ["example", "example", "example"],
            "spectrum_id": ["example:0", "example:1", "example:2"],
        }
    )

    path = tmp_path / "dataset-ms-example_lazy_mgf-0000-0001.parquet"
    df = pl.read_parquet(path).to_pandas()

    assert df.equals(expected_df)


def test_mzml_to_parquet(dir_paths: tuple[str, str], tmp_path: Any) -> None:
    """Test mzml to parquet conversion."""
    _, data_dir = dir_paths
    sdf = SpectrumDataFrame(
        file_paths=[data_dir + "/example.mzML"],
        is_annotated=False,
        is_lazy=False,
        shuffle=False,
    )
    sdf.save(tmp_path, partition="example_mzml")

    expected_df = pd.DataFrame(
        {
            "scan_number": [1],
            "sequence": [""],
            "precursor_mass": [1010.501317140626],
            "precursor_mz": [506.257934570313],
            "precursor_charge": [2],
            "retention_time": [nan],
            "mz_array": [array([10.0, 20.0, 30.0, 40.0])],
            "intensity_array": [array([1.0, 1.5, 1.0, 1.5])],
            "experiment_name": ["example"],
            "spectrum_id": ["example:1"],
        }
    )

    df = pl.read_parquet(tmp_path / "dataset-ms-example_mzml-0000-0001.parquet").to_pandas()
    assert df.equals(expected_df)

    sdf = SpectrumDataFrame(
        file_paths=[data_dir + "/example.mzML"],
        is_annotated=False,
        is_lazy=True,
        shuffle=False,
    )
    sdf.save(tmp_path, partition="example_lazy_mzml")

    df = pl.read_parquet(tmp_path / "dataset-ms-example_lazy_mzml-0000-0001.parquet").to_pandas()
    assert df.equals(expected_df)


def test_mzxml_to_parquet(dir_paths: tuple[str, str], tmp_path: Any) -> None:
    """Test mzxml to parquet conversion."""
    _, data_dir = dir_paths
    sdf = SpectrumDataFrame(
        file_paths=[data_dir + "/example.mzxml"],
        is_annotated=False,
        is_lazy=False,
        shuffle=False,
    )
    sdf.save(tmp_path, partition="example_mzxml")

    expected_df = pd.DataFrame(
        {
            "scan_number": [1],
            "sequence": [""],
            "precursor_mass": [1010.501317140626],
            "precursor_mz": [506.257934570313],
            "precursor_charge": [2],
            "retention_time": [nan],
            "mz_array": [array([10.0, 30.0])],
            "intensity_array": [array([20.0, 40.0])],
            "experiment_name": ["example"],
            "spectrum_id": ["example:1"],
        }
    )

    df = pl.read_parquet(tmp_path / "dataset-ms-example_mzxml-0000-0001.parquet").to_pandas()
    assert df.equals(expected_df)

    sdf = SpectrumDataFrame(
        file_paths=[data_dir + "/example.mzxml"],
        is_annotated=False,
        is_lazy=True,
        shuffle=False,
    )
    sdf.save(tmp_path, partition="example_lazy_mzxml")

    df = pl.read_parquet(tmp_path / "dataset-ms-example_lazy_mzxml-0000-0001.parquet").to_pandas()
    assert df.equals(expected_df)
