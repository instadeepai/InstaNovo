from __future__ import annotations

import logging
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import polars as pl
import spectrum_utils.spectrum as sus
import torch
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset

from instanovo.constants import PROTON_MASS_AMU
from instanovo.types import Peptide
from instanovo.types import PeptideMask
from instanovo.types import PrecursorFeatures
from instanovo.types import Spectrum
from instanovo.types import SpectrumMask
from instanovo.utils.residues import ResidueSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SpectrumDataset(Dataset):
    """Spectrum dataset class supporting `.ipc` and `.csv`."""

    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame,
        residue_set: ResidueSet,
        n_peaks: int = 200,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        reverse_peptide: bool = True,
        annotated: bool = True,
        return_str: bool = False,
    ) -> None:
        super().__init__()
        self.df = df
        self.residue_set = residue_set
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity
        self.reverse_peptide = reverse_peptide
        self.annotated = annotated
        self.return_str = return_str

        if isinstance(df, pd.DataFrame):
            self.data_type = "pd"
        elif isinstance(df, pl.DataFrame):
            self.data_type = "pl"
        elif isinstance(df, datasets.Dataset):
            self.data_type = "hf"
        else:
            raise Exception(f"Unsupported data type {type(df)}")

        self._check_expected_cols()

    def __len__(self) -> int:
        return int(self.df.shape[0])

    def __getitem__(self, idx: int) -> tuple[Spectrum, float, int, Peptide | list[str]]:
        peptide = ""

        if self.data_type == "pl":
            mz_array = torch.Tensor(self.df[idx, "mz_array"].to_list())
            int_array = torch.Tensor(self.df[idx, "intensity_array"].to_list())
            precursor_mz = self.df[idx, "precursor_mz"]
            precursor_charge = self.df[idx, "precursor_charge"]
            if self.annotated:
                peptide = self.df[idx, "modified_sequence"]
        elif self.data_type == "pd":
            row = self.df.iloc[idx]
            mz_array = torch.Tensor(row["mz_array"])
            int_array = torch.Tensor(row["intensity_array"])
            precursor_mz = row["precursor_mz"]
            precursor_charge = row["precursor_charge"]
            if self.annotated:
                peptide = row["modified_sequence"]
        elif self.data_type == "hf":
            row = self.df[idx]
            mz_array = torch.Tensor(row["mz_array"])
            int_array = torch.Tensor(row["intensity_array"])
            precursor_mz = float(row["precursor_mz"])
            precursor_charge = float(row["precursor_charge"])
            if self.annotated:
                peptide = row["modified_sequence"]

        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )

        if not self.return_str:
            peptide_tokenized = self.residue_set.tokenize(peptide)
            if self.reverse_peptide:
                peptide_tokenized = peptide_tokenized[::-1]

            peptide_encoding = self.residue_set.encode(
                peptide_tokenized, add_eos=True, return_tensor="pt"
            )
            return spectrum, precursor_mz, precursor_charge, peptide_encoding

        return spectrum, precursor_mz, precursor_charge, peptide

    def _check_expected_cols(self) -> None:
        expected_cols = [
            "mz_array",
            "intensity_array",
            "precursor_mz",
            "precursor_charge",
            "modified_sequence",
        ]
        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        if missing_cols:
            plural_s = "s" if len(missing_cols) > 1 else ""
            missing_col_names = ", ".join(missing_cols)
            raise ValueError(
                f"Column{plural_s} missing! Missing column{plural_s}: {missing_col_names}"
            )

    def _process_peaks(
        self,
        mz_array: Float[Tensor, " peak"],
        int_array: Float[Tensor, " peak"],
        precursor_mz: float,
        precursor_charge: int,
        use_sus_preprocess: bool = True,
    ) -> Spectrum:
        """Preprocess the spectrum by removing noise peaks and scaling the peak intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak m/z values.
        int_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak intensity values.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        """
        if use_sus_preprocess:
            spectrum = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                np.array(mz_array).astype(np.float32),
                np.array(int_array).astype(np.float32),
            )
            try:
                spectrum.set_mz_range(self.min_mz, self.max_mz)
                if len(spectrum.mz) == 0:
                    raise ValueError
                spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
                if len(spectrum.mz) == 0:
                    raise ValueError
                spectrum.filter_intensity(self.min_intensity, self.n_peaks)
                if len(spectrum.mz) == 0:
                    raise ValueError
                spectrum.scale_intensity("root", 1)
                intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
                return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
            except ValueError:
                # Replace invalid spectra by a dummy spectrum.
                return torch.tensor([[0, 1]]).float()

        # Intensity norm
        int_array = int_array / int_array.max()

        # filters
        filter_idx = (
            (int_array < self.min_intensity)
            & (mz_array < self.min_mz)
            & (mz_array > self.max_mz)
        )

        int_array = int_array[~filter_idx]
        mz_array = mz_array[~filter_idx]

        # filter peaks
        idx = int_array.argsort(descending=True)
        idx = idx[: self.n_peaks]

        int_array = int_array[idx]
        mz_array = mz_array[idx]

        return torch.stack([mz_array, int_array]).T.float()


def collate_batch(
    batch: list[tuple[Spectrum, float, int, Peptide]],
) -> tuple[
    Float[Spectrum, " batch"],
    Float[PrecursorFeatures, " batch"],
    Bool[SpectrumMask, " batch"],
    Integer[Peptide, " batch"],
    Bool[PeptideMask, " batch"],
]:
    """Collate batch of samples."""
    spectra, precursor_mzs, precursor_charges, peptides_batch = zip(*batch)

    # Pad spectra
    ll = torch.tensor([x.shape[0] for x in spectra], dtype=torch.long)
    spectra = nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    spectra_mask = (
        torch.arange(spectra.shape[1], dtype=torch.long)[None, :] >= ll[:, None]
    )

    # Pad peptide
    if isinstance(peptides_batch[0], str):
        peptides_mask = None
    else:
        ll = torch.tensor([x.shape[0] for x in peptides_batch], dtype=torch.long)
        peptides = nn.utils.rnn.pad_sequence(peptides_batch, batch_first=True)
        peptides_mask = (
            torch.arange(peptides.shape[1], dtype=torch.long)[None, :] >= ll[:, None]
        )

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()

    return spectra, precursors, spectra_mask, peptides, peptides_mask


# TODO: move to generic utils
def load_ipc_shards(
    data_path: str | Path, split: str = "train", remap_cols: bool = True
) -> pl.DataFrame:
    """Load sharded polars dataframe."""
    data_path = Path(data_path)
    if data_path.is_file():
        return pl.read_ipc(str(data_path))

    df_list = []
    for file in data_path.iterdir():
        if f"{split}_shard" not in file.stem:
            continue
        logger.info(f"Reading shard {str(file)}")
        raw_df = pl.read_ipc(str(file))
        df = _clean_and_remap(raw_df)
        del raw_df
        df_list.append(df)
        logger.info(f"Read {df.shape[0]:,} spectra")

    if len(df_list) == 0:
        # No shards found, assume "{split}.ipc" name used
        file = data_path.joinpath(f"{split}.ipc")
        logger.info(f"No shards found, reading {str(file)}")
        if not file.exists():
            raise ValueError(
                f"No shards for split '{split}' in {data_path}. Fallback to {file} also failed."
            )
        raw_df = pl.read_ipc(str(file))
        df = _clean_and_remap(raw_df)
        del raw_df
        df_list.append(df)
        logger.info(f"Read {df.shape[0]:,} spectra")

    df = pl.concat(df_list)

    return df


def _clean_and_remap(df: pl.DataFrame) -> pl.DataFrame:
    col_map: dict[str, str] = {
        "Modified sequence": "modified_sequence",
        "MS/MS m/z": "precursor_mz",
        "Precursor m/z": "precursor_mz",
        "Theoretical m/z": "theoretical_mz",
        "Mass": "precursor_mass",
        "Charge": "precursor_charge",
        "Mass values": "mz_array",
        "Mass spectrum": "mz_array",
        "Intensity": "intensity_array",
        "Raw intensity spectrum": "intensity_array",
    }

    col_dtypes: dict[str, pl.DataType] = {
        "modified_sequence": str,
        "precursor_mz": pl.Float64,
        "precursor_charge": pl.Int32,
        "mz_array": pl.List(pl.Float32),
        "intensity_array": pl.List(pl.Float32),
    }

    df = df.rename({k: v for k, v in col_map.items() if k in df.columns})
    if df.select(pl.first("modified_sequence")).item()[0] == "_":
        df = df.with_columns(pl.col("modified_sequence").apply(lambda x: x[1:-1]))
    if df.select(pl.first("modified_sequence")).item()[0] == ".":
        df = df.with_columns(pl.col("modified_sequence").apply(lambda x: x[1:-1]))

    df = df.drop([col for col in df.columns if col not in list(col_dtypes.keys())])

    # cast fp64
    df = df.with_columns([pl.col(k).cast(v) for k, v in col_dtypes.items()])

    return df.select(list(col_dtypes.keys()))
