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
from matchms.importing import load_from_mgf
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_mgf_ipc(
    source: Path,
    target: Path,
    max_charge: int = 10,
    use_old_schema: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Convert .mgf file to Polars .ipc."""
    schema = {
        "experiment_name": str,
        "evidence_index": int,
        "scan_number": int,
        "sequence": str,
        "modified_sequence": str,
        "precursor_mass": float,
        "precursor_mz": pl.Float64,
        "precursor_charge": int,
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
            "retention_time": pl.Float64,
            "Mass values": pl.List(pl.Float64),
            "Intensity": pl.List(pl.Float64),
        }

    df = pl.DataFrame(schema=schema)

    if source.is_file():
        filenames = [source]
    else:
        filenames = source.iterdir()

    for filepath in filenames:
        if not filepath.suffix.lower().endswith("mgf"):
            logger.info(f"Skipping {filepath}... Not a mgf file...")
            continue

        if verbose:
            logger.info(f"Processing {filepath}...")

        if not filepath.is_file():
            if verbose:
                logger.warning("File not found, skipping...")
            continue

        exp = load_from_mgf(str(filepath))

        data = []
        evidence_index = 0
        for spectrum in exp:
            meta = spectrum.metadata

            peptide = ""
            unmod_peptide = ""
            if "peptide_sequence" in meta:
                peptide = meta["peptide_sequence"]
                unmod_peptide = "".join([x[0] for x in re.split(r"(?<=.)(?=[A-Z])", peptide)])

            if "charge" not in meta or meta["charge"] > max_charge:
                continue

            scan_id = (
                int(re.findall(r":(\d+)", meta["scans"])[-1]) if "scans" in meta else evidence_index
            )

            data.append(
                [
                    Path(meta["title"].replace("\\", "/")).stem,
                    evidence_index,
                    scan_id,
                    unmod_peptide,
                    peptide if not use_old_schema else f"_{peptide}_",
                    meta["pepmass"][0],
                    meta["precursor_mz"],
                    meta["charge"],
                    meta["retention_time"],
                    list(spectrum.mz),
                    list(spectrum.intensities),
                ]
            )

            evidence_index += 1
        df = pl.concat([df, pl.DataFrame(data, schema=schema)])

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(target)

    return df


# flake8: noqa: CR001
def convert_mzml_ipc(
    source: Path,
    target: Path,
    max_charge: int = 10,
    use_old_schema: bool = False,
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

    if source.is_file():
        filenames = [source]
    else:
        filenames = source.iterdir()

    for filepath in filenames:
        if not filepath.suffix.lower().endswith("mzml"):
            logger.info(f"Skipping {filepath}... Not a mzml file...")
            continue

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
        "--source_type", default=None, choices=["mgf", "mzml", "csv"], help="type of input data"
    )
    parser.add_argument("--max_charge", default=10, help="maximum charge to filter out")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_old_schema", action="store_true")

    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    source_type = args.source_type

    if source_type is None:
        # Attempt to infer type from file
        if source.is_dir():
            raise ValueError(
                f"Cannot infer source type from a directory. Please specify with --source_type"
            )
        source_type = source.suffix[1:].lower()
    else:
        source_type = source_type.lower()

    if source_type == "mgf":
        convert_mgf_ipc(
            source,
            target,
            args.max_charge,
            use_old_schema=args.use_old_schema,
            verbose=args.verbose,
        )
    elif source_type == "mzml":
        convert_mzml_ipc(
            source,
            target,
            args.max_charge,
            use_old_schema=args.use_old_schema,
            verbose=args.verbose,
        )
    elif source_type == "csv":
        df = pd.read_csv(source)
        df = pl.from_pandas(df)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        df.write_ipc(target)
    else:
        raise ValueError(f"Source type {source_type} not supported.")


if __name__ == "__main__":
    main()
