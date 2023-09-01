from __future__ import annotations

import argparse
import logging
import os
import tempfile
import uuid
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from depthcharge.data.hdf5 import AnnotatedSpectrumIndex
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def convert_mgf_ipc(source: str, target: str, max_charge: int = 10) -> pl.DataFrame:
    """Convert .mgf file to Polars .ipc."""
    tmp_dir = tempfile.TemporaryDirectory()
    idx_fn = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    valid_charge = np.arange(1, max_charge + 1)
    index = AnnotatedSpectrumIndex(idx_fn, source, valid_charge=valid_charge)

    print(f"Loaded {index.n_spectra} spectra, converting...")

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

    os.makedirs("/".join(target.split("/")[:-1]), exist_ok=True)
    df.write_ipc(target)

    tmp_dir.cleanup()

    return df


def main() -> None:
    """Convert data to ipc."""
    parser = argparse.ArgumentParser()

    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--source_type", default="mgf")
    parser.add_argument("--max_charge", default=10)

    args = parser.parse_args()

    source = args.source
    target = args.target

    if args.source_type == "mgf":
        convert_mgf_ipc(source, target, args.max_charge)
    elif args.source_type == "csv":
        df = pd.read_csv(source)
        df = pl.from_pandas(df)
        os.makedirs("/".join(target.split("/")[:-1]), exist_ok=True)
        df.write_ipc(target)
    else:
        assert Exception(f"Source type {args.source_type} not supported.")


if __name__ == "__main__":
    main()
