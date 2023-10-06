from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyopenms
from depthcharge.data.hdf5 import AnnotatedSpectrumIndex
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_mgf_ipc(source: str, target: str, max_charge: int = 10) -> pl.DataFrame:
    """Convert .mgf file to Polars .ipc."""
    if not Path(source).name.endswith(".mgf"):
        raise ValueError(
            f"Attempted to use MGF mode with a non-mgf file: '{source}'. Please specify with --source_type"
        )

    tmp_dir = tempfile.TemporaryDirectory()
    idx_fn = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    valid_charge = np.arange(1, max_charge + 1)
    index = AnnotatedSpectrumIndex(idx_fn, source, valid_charge=valid_charge)

    logger.info(f"Loaded {index.n_spectra} spectra, converting...")

    data_dict: dict[str, Any] = {
        "Sequence": [],
        "Modified sequence": [],
        "MS/MS m/z": [],
        "Charge": [],
        "Mass values": [],
        "Intensity": [],
    }

    for i in tqdm(range(index.n_spectra), total=index.n_spectra):
        (
            mz_array,
            int_array,
            precursor_mz,
            precursor_charge,
            peptide,
        ) = index[i]

        data_dict["Sequence"].append(peptide)
        data_dict["Modified sequence"].append(f"_{peptide}_")
        data_dict["MS/MS m/z"].append(precursor_mz)
        data_dict["Charge"].append(precursor_charge)
        data_dict["Mass values"].append(mz_array)
        data_dict["Intensity"].append(int_array)

    df = pl.DataFrame(data_dict)

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(target)

    tmp_dir.cleanup()

    return df


# flake8: noqa: CR001
def convert_mzml_ipc(
    source: str,
    target: str,
    max_charge: int = 10,
    use_old_schema: bool = True,
    verbose: bool = True,
) -> None:
    """Convert mzml to polars ipc."""
    schema = {
        "experiment_name": str,
        "evidence_index": int,
        "scan_number": int,
        "sequence": str,
        "modified_sequence": str,
        "precursor_mass": float,
        "precursor_mz": pl.Float64,
        "precursor_charge": int,
        "precursor_intensity": pl.Float64,
        "retention_time": pl.Float64,
        "mz_array": pl.List(pl.Float64),
        "intensity_array": pl.List(pl.Float64),
    }

    if use_old_schema:
        schema = {
            "experiment_name": str,
            "evidence_index": int,
            "scan_number": int,
            "Sequence": str,
            "Modified sequence": str,
            "Mass": float,
            "MS/MS m/z": pl.Float64,
            "Charge": int,
            "precursor_intensity": pl.Float64,
            "retention_time": pl.Float64,
            "Mass values": pl.List(pl.Float64),
            "Intensity": pl.List(pl.Float64),
        }

    df = pl.DataFrame(schema=schema)

    source = Path(source)

    if source.is_file():
        filenames = [source]
    else:
        filenames = source.iterdir()

    for filepath in filenames:
        if verbose:
            logger.info(f"Processing {filepath}...")

        if not filepath.is_file():
            if verbose:
                logger.warning("File not found, skipping...")
            continue

        exp = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(str(filepath), exp)

        evidence_index = 0

        exp_iter = iter(exp)
        if verbose:
            exp_iter = tqdm(exp_iter, total=len(exp.getSpectra()))

        data = []
        for spectrum in exp_iter:
            if spectrum.getMSLevel() != 2:
                continue

            mz_array, int_array = spectrum.get_peaks()
            precursor = spectrum.getPrecursors()[0]

            if precursor.getCharge() > max_charge:
                continue

            scan_id = int(re.findall(r"=(\d+)", spectrum.getNativeID())[-1])

            data.append(
                [
                    filepath.stem,
                    evidence_index,
                    scan_id,
                    "",
                    "" if not use_old_schema else "__",
                    precursor.getUnchargedMass(),
                    precursor.getMZ(),
                    precursor.getCharge(),
                    precursor.getIntensity(),
                    spectrum.getRT(),
                    list(mz_array),
                    list(int_array),
                ]
            )

            evidence_index += 1
        df = pl.concat([df, pl.DataFrame(data, schema=schema)])

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(target)


def main() -> None:
    """Convert data to ipc."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("source", help="source file or folder")
    parser.add_argument("target", help="target ipc file to be saved")
    parser.add_argument(
        "--source_type", default="mgf", choices=["mgf", "mzml", "csv"], help="type of input data"
    )
    parser.add_argument("--max_charge", default=10, help="maximum charge to filter out")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_old_schema", action="store_true")

    args = parser.parse_args()

    source = args.source
    target = args.target

    if args.source_type == "mgf":
        convert_mgf_ipc(source, target, args.max_charge)
    elif args.source_type == "csv":
        df = pd.read_csv(source)
        df = pl.from_pandas(df)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        df.write_ipc(target)
    elif args.source_type == "mzml":
        convert_mzml_ipc(
            source,
            target,
            args.max_charge,
            use_old_schema=args.use_old_schema,
            verbose=args.verbose,
        )
    else:
        assert Exception(f"Source type {args.source_type} not supported.")


if __name__ == "__main__":
    main()
