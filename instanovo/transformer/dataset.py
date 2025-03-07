from __future__ import annotations

import logging
import re
from pathlib import Path
import polars as pl

import numpy as np
import spectrum_utils.spectrum as sus
import torch
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset

from instanovo.constants import PROTON_MASS_AMU, MSColumns, ANNOTATED_COLUMN
from instanovo.types import Peptide
from instanovo.types import PeptideMask
from instanovo.types import PrecursorFeatures
from instanovo.types import Spectrum
from instanovo.types import SpectrumMask
from instanovo.utils import ResidueSet
from instanovo.utils import SpectrumDataFrame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SpectrumDataset(Dataset):
    """Spectrum dataset class supporting `.ipc` and `.csv`."""

    def __init__(
        self,
        df: SpectrumDataFrame,
        residue_set: ResidueSet,
        n_peaks: int = 200,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        pad_spectrum_max_length: bool = False,
        peptide_pad_length: int = 0,
        reverse_peptide: bool = True,
        annotated: bool = True,
        return_str: bool = False,
        bin_spectra: bool = False,
        bin_size: float = 0.01,
        diffusion: bool = False,
    ) -> None:
        super().__init__()
        self.df = df
        self.residue_set = residue_set
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity
        self.pad_spectrum_max_length = pad_spectrum_max_length
        self.peptide_pad_length = peptide_pad_length
        self.reverse_peptide = reverse_peptide
        self.annotated = annotated
        self.return_str = return_str
        self.bin_spectra = bin_spectra
        self.bin_size = bin_size
        self.diffusion = diffusion

        if self.bin_spectra:
            self.bins = torch.arange(0, self.max_mz + self.bin_size, self.bin_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Spectrum, float, int, Peptide | list[str]]:
        peptide = ""

        row = self.df[idx]

        mz_array = torch.Tensor(row[MSColumns.MZ_ARRAY.value])
        int_array = torch.Tensor(row[MSColumns.INTENSITY_ARRAY.value])
        precursor_mz = row[MSColumns.PRECURSOR_MZ.value]
        precursor_charge = row[MSColumns.PRECURSOR_CHARGE.value]

        if self.annotated:
            peptide = row[ANNOTATED_COLUMN]

        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )

        if self.bin_spectra:
            spectrum = torch.histogram(
                spectrum[:, 0], weight=spectrum[:, 1], bins=self.bins
            ).hist

        if self.pad_spectrum_max_length and not self.bin_spectra:
            spectrum_padded = torch.zeros(
                (self.n_peaks, 2), dtype=spectrum.dtype, device=spectrum.device
            )
            spectrum_padded[: spectrum.shape[0]] = spectrum
            spectrum = spectrum_padded

        if not self.return_str:
            peptide_tokenized = self.residue_set.tokenize(peptide)
            if self.reverse_peptide:
                peptide_tokenized = peptide_tokenized[::-1]

            peptide_encoding = self.residue_set.encode(
                peptide_tokenized, add_eos=not self.diffusion, return_tensor="pt"
            )
            # Does nothing when peptide_pad_length = 0 (default). This is used for torch.compile
            peptide_padded = torch.zeros(
                (max(self.peptide_pad_length, peptide_encoding.shape[0]),),
                dtype=peptide_encoding.dtype,
                device=peptide_encoding.device,
            )
            peptide_padded[: peptide_encoding.shape[0]] = peptide_encoding
            return spectrum, precursor_mz, precursor_charge, peptide_padded

        return spectrum, precursor_mz, precursor_charge, peptide

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
    Integer[Peptide, " batch"] | list[str],
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
    if not isinstance(peptides_batch[0], str):
        ll = torch.tensor([x.shape[0] for x in peptides_batch], dtype=torch.long)
        peptides = nn.utils.rnn.pad_sequence(peptides_batch, batch_first=True)
        peptides_mask = (
            torch.arange(peptides.shape[1], dtype=torch.long)[None, :] >= ll[:, None]  # type: ignore
        )
    else:
        peptides = peptides_batch
        peptides_mask = None

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


def remove_modifications(x: str) -> str:
    """Remove modifications and map I to L."""
    x = re.findall(r"[A-Z]", x)
    x = ["L" if y == "I" else y for y in x]
    return "".join(x)


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
        df = df.with_columns(
            pl.col("modified_sequence").map_elements(
                lambda x: x[1:-1], return_dtype=pl.Utf8
            )
        )
    if df.select(pl.first("modified_sequence")).item()[0] == ".":
        df = df.with_columns(
            pl.col("modified_sequence").map_elements(
                lambda x: x[1:-1], return_dtype=pl.Utf8
            )
        )

    df = df.drop([col for col in df.columns if col not in list(col_dtypes.keys())])

    # cast fp64
    df = df.with_columns([pl.col(k).cast(v) for k, v in col_dtypes.items()])

    return df.select(list(col_dtypes.keys()))
