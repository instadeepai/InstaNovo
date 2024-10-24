from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
from numpy import array

from instanovo.scripts.convert_to_ipc import convert_mgf_ipc
from instanovo.scripts.convert_to_ipc import convert_mzml_mzxml_ipc
from instanovo.scripts.convert_to_ipc import main


def test_mgf_to_ipc(dir_paths: tuple[str, str], tmp_path: Path) -> None:
    """Tests mgf to ipc conversion."""
    _, data_dir = dir_paths

    convert_mgf_ipc(Path(data_dir + "/example.mgf"), tmp_path / "from_mgf_current.ipc")

    df_current = pl.read_ipc(tmp_path / "from_mgf_current.ipc").to_pandas()
    df_example = pl.read_ipc(os.path.join(data_dir, "from_mgf_example.ipc")).to_pandas()

    pd.testing.assert_frame_equal(df_current, df_example, check_like=True)


def test_mzml_to_ipc(dir_paths: tuple[str, str], tmp_path: Path) -> None:
    """Tests mzml to ipc conversion."""
    _, data_dir = dir_paths

    convert_mzml_mzxml_ipc(
        Path(data_dir + "/example.mzML"), tmp_path / "from_mzml_current.ipc"
    )

    df_current = pl.read_ipc(tmp_path / "from_mzml_current.ipc").to_pandas()
    df_example = pl.read_ipc(
        os.path.join(data_dir, "from_mzml_example.ipc")
    ).to_pandas()

    pd.testing.assert_frame_equal(df_current, df_example, check_like=True)


def test_mzxml_to_ipc(dir_paths: tuple[str, str], tmp_path: Path) -> None:
    """Tests mzxml to ipc conversion."""
    _, data_dir = dir_paths

    convert_mzml_mzxml_ipc(
        Path(data_dir + "/example.mzxml"), tmp_path / "from_mzxml_current.ipc"
    )

    df_current = pl.read_ipc(tmp_path / "from_mzxml_current.ipc").to_pandas()
    df_example = pl.read_ipc(
        os.path.join(data_dir, "from_mzxml_example.ipc")
    ).to_pandas()

    pd.testing.assert_frame_equal(df_current, df_example, check_like=True)


def test_main_mgf_conversion(tmp_path: Path) -> None:
    """Tests the mgf conversion call in main."""
    source_file = tmp_path / "source.mgf"
    target_file = tmp_path / "target.ipc"

    test_args = [
        "instanovo/scripts/convert_to_ipc.py",
        str(source_file),
        str(target_file),
        "--source_type",
        "mgf",
        "--max_charge",
        "5",
        "--verbose",
    ]

    with patch("sys.argv", test_args):
        with patch.object(Path, "is_file", return_value=True):
            with patch(
                "instanovo.scripts.convert_to_ipc.convert_mgf_ipc"
            ) as mock_convert_mgf_ipc:
                mock_convert_mgf_ipc.return_value = None
                main()

                mock_convert_mgf_ipc.assert_called_once_with(
                    source_file, target_file, 5, use_old_schema=False, verbose=True
                )


def test_main_file_conversion(tmp_path: Path, dir_paths: tuple[str, str]) -> None:
    """Tests the mgf conversion call in main for an example file."""
    _, data_dir = dir_paths
    source_file = data_dir + "/example.mgf"
    target_file = tmp_path / "target.ipc"

    test_args = [
        "instanovo/scripts/convert_to_ipc.py",
        str(source_file),
        str(target_file),
        "--source_type",
        "mgf",
        "--max_charge",
        "2",
        "--verbose",
    ]

    with patch("sys.argv", test_args):
        main()

    expected_df = pd.DataFrame(
        {
            "experiment_name": ["example", "example"],
            "evidence_index": [1, 2],
            "scan_number": [1, 3],
            "sequence": ["FHHTIGGSR", "TTVINMPR"],
            "modified_sequence": ["FHHTIGGSR", "TTVINM[15.99]PR"],
            "precursor_mass": [1012.515869140626, 948.505798339844],
            "precursor_mz": [506.257934570313, 474.252899169922],
            "precursor_charge": [2, 2],
            "retention_time": [100.0, 300.0],
            "mz_array": [
                array([10.0, 20.0, 30.0, 40.0]),
                array([10.0, 20.0, 30.0, 40.0]),
            ],
            "intensity_array": [
                array([1.0, 1.5, 1.0, 1.5]),
                array([1.0, 1.5, 1.0, 1.5]),
            ],
            "scans": ["1", "3"],
        }
    )

    df = pl.read_ipc(tmp_path / "target.ipc").to_pandas()

    assert df.equals(expected_df)


def test_main_mzml_conversion(tmp_path: Path) -> None:
    """Tests the mzml conversion call in main."""
    source_file = tmp_path / "source.mzML"
    target_file = tmp_path / "target.ipc"

    test_args = [
        "instanovo/scripts/convert_to_ipc.py",
        str(source_file),
        str(target_file),
        "--source_type",
        "mzml",
        "--max_charge",
        "5",
        "--verbose",
    ]

    with patch("sys.argv", test_args):
        with patch.object(Path, "is_file", return_value=True):
            with patch(
                "instanovo.scripts.convert_to_ipc.convert_mzml_mzxml_ipc"
            ) as mock_convert_mzml_mzxml_ipc:
                mock_convert_mzml_mzxml_ipc.return_value = None
                main()

                mock_convert_mzml_mzxml_ipc.assert_called_once_with(
                    source_file, target_file, 5, use_old_schema=False, verbose=True
                )


def test_main_mxzml_conversion(tmp_path: Path) -> None:
    """Tests the mzxml conversion call in main."""
    source_file = tmp_path / "source.mzxml"
    target_file = tmp_path / "target.ipc"

    test_args = [
        "instanovo/scripts/convert_to_ipc.py",
        str(source_file),
        str(target_file),
        "--source_type",
        "mzxml",
        "--max_charge",
        "5",
        "--verbose",
    ]

    with patch("sys.argv", test_args):
        with patch.object(Path, "is_file", return_value=True):
            with patch(
                "instanovo.scripts.convert_to_ipc.convert_mzml_mzxml_ipc"
            ) as mock_convert_mzml_mzxml_ipc:
                mock_convert_mzml_mzxml_ipc.return_value = None
                main()

                mock_convert_mzml_mzxml_ipc.assert_called_once_with(
                    source_file, target_file, 5, use_old_schema=False, verbose=True
                )
