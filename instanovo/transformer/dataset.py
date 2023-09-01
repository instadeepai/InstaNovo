from __future__ import annotations

import re

import pandas as pd
import polars as pl
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset

PROTON_MASS_AMU = 1.007276


class SpectrumDataset(Dataset):
    """Spectrum dataset class supporting `.ipc` and `.csv`."""

    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame,
        s2i: dict[str, int],
        n_peaks: int = 200,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        reverse_peptide: bool = True,
        eos_symbol: str = "</s>",
        annotated: bool = True,
        return_str: bool = False,
    ) -> None:
        super().__init__()
        self.df = df
        self.s2i = s2i
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity
        self.reverse_peptide = reverse_peptide
        self.annotated = annotated
        self.return_str = return_str

        if eos_symbol in self.s2i:
            self.EOS_ID = self.s2i[eos_symbol]
        else:
            self.EOS_ID = -1

        if type(df) == pd.DataFrame:
            self.data_type = "pd"
        elif type(df) == pl.DataFrame:
            self.data_type = "pl"
        else:
            raise Exception(f"Unsupported data type {type(df)}")

    def __len__(self) -> int:
        return int(self.df.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, float, int, Tensor | list[str]]:
        peptide = ""

        if self.data_type == "pl":
            mz_array = torch.Tensor(self.df[idx, "Mass values"])
            int_array = torch.Tensor(self.df[idx, "Intensity"])
            precursor_mz = self.df[idx, "MS/MS m/z"]
            precursor_charge = self.df[idx, "Charge"]
            if self.annotated:
                peptide = self.df[idx, "Modified sequence"][1:-1]
        elif self.data_type == "pd":
            row = self.df.iloc[idx]
            mz_array = torch.Tensor(row["Mass values"])
            int_array = torch.Tensor(row["Intensity"])
            precursor_mz = row["MS/MS m/z"]
            precursor_charge = row["Charge"]
            if self.annotated:
                peptide = row["Modified sequence"][1:-1]

        # Split on amino acids allowing for modifications eg. AM(ox)Z -> [A, M(ox), Z]
        # Groups A-Z with any suffix
        if not self.return_str:
            peptide = re.split(r"(?<=.)(?=[A-Z])", peptide)  # type: ignore
            if self.reverse_peptide:
                peptide = peptide[::-1]

            peptide = torch.tensor([self.s2i[x] for x in peptide] + [self.EOS_ID]).long()

        spectrum = self._process_peaks(mz_array, int_array, precursor_mz, precursor_charge)

        return spectrum, precursor_mz, precursor_charge, peptide

    def _process_peaks(
        self,
        mz_array: Tensor,
        int_array: Tensor,
        precursor_mz: Tensor,
        precursor_charge: Tensor,
    ) -> Tensor:
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
        # # Intensity norm
        int_array = int_array / int_array.max()

        # filters
        filter_idx = (
            (int_array < self.min_intensity) & (mz_array < self.min_mz) & (mz_array > self.max_mz)
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
    batch: list[tuple[Tensor, float, int, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectra, precursor_mzs, precursor_charges, peptides = zip(*batch)

    # Pad spectra
    ll = torch.tensor([x.shape[0] for x in spectra], dtype=torch.long)
    spectra = nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    spectra_mask = torch.arange(spectra.shape[1], dtype=torch.long)[None, :] >= ll[:, None]

    # Pad peptide
    if type(peptides[0]) == str:
        peptides_mask = None
    else:
        ll = torch.tensor([x.shape[0] for x in peptides], dtype=torch.long)
        peptides = nn.utils.rnn.pad_sequence(peptides, batch_first=True)
        peptides_mask = torch.arange(peptides.shape[1], dtype=torch.long)[None, :] >= ll[:, None]

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()

    return spectra, precursors, spectra_mask, peptides, peptides_mask
