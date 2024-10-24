from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
from numpy import array

from instanovo.scripts.convert_to_sdf import main


def test_main(tmp_path: Path, dir_paths: tuple[str, str]) -> None:
    """Tests converting and saving an input file to a parquet file."""
    _, data_dir = dir_paths
    source_file = data_dir + "/example.mgf"
    target_file = tmp_path / "target.parquet"

    test_args = [
        "instanovo/scripts/convert_to_sdf.py",
        source_file,
        str(target_file),
        "--is_annotated",
        "True",
        "--max_charge",
        "2",
    ]

    with patch("sys.argv", test_args):
        main()

    expected_df = pd.DataFrame(
        {
            "scan_number": [0, 2],
            "sequence": ["FHHTIGGSR", "TTVINM[15.99]PR"],
            "precursor_mass": [1010.501317140626, 946.491246339844],
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
            "experiment_name": ["example", "example"],
            "spectrum_id": ["example:0", "example:2"],
        }
    )

    df = pl.read_parquet(tmp_path / "target.parquet").to_pandas()

    assert df.equals(expected_df)
