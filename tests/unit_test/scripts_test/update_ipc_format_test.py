from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from instanovo.scripts.update_ipc_format import main, update_ipc


def test_update_ipc_format(tmp_path: Path) -> None:
    """Test code to update an ipc file from the old format to the new format."""
    source_file = tmp_path / "source.ipc"
    target_file = tmp_path / "target.ipc"

    source_df = pl.DataFrame(
        {
            "Sequence": ["PEPA", "PEPB"],
            "Modified sequence": ["_PEPA_", "_PEPB_"],
            "MS/MS m/z": [100.0, 200.0],
            "Charge": [2, 3],
            "Intensity": [[10, 20], [30, 40]],
            "Mass values": [[100, 200], [300, 400]],
        }
    )
    source_df.write_ipc(source_file)

    update_ipc(source_file, target_file)

    assert target_file.exists()

    target_df = pl.read_ipc(target_file)

    expected_df = pl.DataFrame(
        {
            "sequence": ["PEPA", "PEPB"],
            "modified_sequence": ["PEPA", "PEPB"],
            "precursor_mz": [100.0, 200.0],
            "precursor_charge": [2, 3],
            "intensity_array": [[10, 20], [30, 40]],
            "mz_array": [[100, 200], [300, 400]],
        }
    )

    pd.testing.assert_frame_equal(target_df.to_pandas(), expected_df.to_pandas())


def test_incorrect_file_error() -> None:
    """Test incorrect file type error catching."""
    with pytest.raises(ValueError, match="Incorrect file type - .ipc file required."):
        update_ipc(Path("incorrect_file_type.csv"), Path("target.ipc"))


@patch("instanovo.scripts.update_ipc_format.update_ipc", autospec=True)
def test_main(mock_update_ipc: Any) -> None:
    """Test main method call."""
    with patch(
        "sys.argv",
        ["instanovo.utils.update_ipc_format.py", "source.ipc", "target.ipc"],
    ):
        main()

    mock_update_ipc.assert_called_once_with(Path("source.ipc"), Path("target.ipc"))
