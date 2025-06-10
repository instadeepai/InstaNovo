import os

import pandas as pd
import polars as pl
from numpy import array
from typer.testing import CliRunner

from instanovo.scripts.convert_to_sdf import app

runner = CliRunner()


def test_convert_command(dir_paths: tuple[str, str]) -> None:
    """Tests converting and saving an input file to a parquet file using the CLI."""
    _, data_dir = dir_paths
    # Convert to absolute path to ensure it works even when current directory changes
    source_file = os.path.abspath(os.path.join(data_dir, "example.mgf"))

    with runner.isolated_filesystem() as temp_dir:
        # Create a target directory
        target_dir = os.path.join(temp_dir, "output")
        os.makedirs(target_dir)

        # Use Typer's CliRunner to invoke the command
        result = runner.invoke(
            app,
            [
                source_file,
                target_dir,
                "--name",
                "example",
                "--partition",
                "train",
                "--max-charge",
                "2",
                "--is-annotated",
            ],
        )

        # Check if the command executed successfully
        assert result.exit_code == 0, f"Command failed with error: {result.stdout}"

        # In the new version, the output file is named based on name and partition
        expected_output_file = os.path.join(target_dir, "dataset-example-train-0000-0001.parquet")
        assert os.path.exists(expected_output_file), f"Output file {expected_output_file} not found"

        # Read the output file and verify its contents
        df = pl.read_parquet(expected_output_file).to_pandas()

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

        # Check if the dataframe contents match the expected values
        pd.testing.assert_frame_equal(df, expected_df)
