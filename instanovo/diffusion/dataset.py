from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import polars
import torch

from instanovo.constants import PROTON_MASS_AMU
from instanovo.utils.residues import ResidueSet


class SpectrumBatch(NamedTuple):
    """Represents a batch of spectrum data without annotations.

    Attributes:
        spectra (torch.FloatTensor): The tensor containing the spectra data.
        spectra_padding_mask (torch.BoolTensor): A boolean tensor indicating the padding positions in the spectra tensor.
        precursors (torch.FloatTensor): The tensor containing precursor mass information.
    """

    spectra: torch.FloatTensor
    spectra_padding_mask: torch.BoolTensor
    precursors: torch.FloatTensor


class AnnotatedSpectrumBatch(NamedTuple):
    """Represents a batch of annotated spectrum data.

    Attributes:
        spectra (torch.FloatTensor): The tensor containing the spectra data.
        spectra_padding_mask (torch.BoolTensor): A boolean tensor indicating the padding positions in the spectra tensor.
        precursors (torch.FloatTensor): The tensor containing precursor mass information.
        peptides (torch.LongTensor): The tensor containing peptide sequence information.
        peptide_padding_mask (torch.BoolTensor): A boolean tensor indicating the padding positions in the peptides tensor.
    """

    spectra: torch.FloatTensor
    spectra_padding_mask: torch.BoolTensor
    precursors: torch.FloatTensor
    peptides: torch.LongTensor
    peptide_padding_mask: torch.BoolTensor


class PolarsSpectrumDataset(torch.utils.data.Dataset):
    """An Polars data frame index wrapper for `depthcharge`/`casanovo` datasets."""

    def __init__(self, data_frame: polars.DataFrame) -> None:
        self.data = data_frame

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, float]:
        row = self.data[idx]
        mz_array = torch.FloatTensor(row["mz_array"].to_numpy()[0])
        int_array = row["intensity_array"].to_numpy()[0]
        int_array = torch.FloatTensor(int_array / int_array.max())
        precursor_mz = row["precursor_mz"].to_numpy()[0]
        precursor_charge = row["precursor_charge"].to_numpy()[0]
        spectrum = torch.stack([mz_array, int_array]).T

        return spectrum, precursor_mz, precursor_charge


class AnnotatedPolarsSpectrumDataset(PolarsSpectrumDataset):
    """A dataset with a Polars index that includes peptides from an aligned list."""

    def __init__(self, data_frame: polars.DataFrame, peptides: list[str]) -> None:
        super().__init__(data_frame)
        self.peptides = peptides

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, float, str]:  # type: ignore[override]
        spectrum, precursor_mz, precursor_charge = super().__getitem__(idx)
        peptide = self.peptides[idx]
        return spectrum, precursor_mz, precursor_charge, peptide


def collate_batches(
    residues: ResidueSet, max_length: int, time_steps: int, annotated: bool
) -> Callable[
    [list[tuple[torch.FloatTensor, float, int, str]]], SpectrumBatch | AnnotatedSpectrumBatch
]:
    """Get batch collation function for given residue set, maximum length and time steps.

    The returned function combines spectra and precursor information for a batch into
    `torch` tensors. It also maps the residues in a peptide to their indices in
    `residues`, pads or truncates them all to `max_length` and returns this as a
    `torch` tensor.

    Args:
        residues (ResidueSet):
            The residues in the vocabulary together with their masses
            and index map.

        max_length (int):
            The maximum peptide sequence length. All sequences are
            padded to this length.

        time_steps (int):
            The number of diffusion time steps.

    Returns:
        Callable[ [list[tuple[torch.FloatTensor, float, int, str]]], SpectrumBatch | AnnotatedSpectrumBatch]:
            The function that combines examples into a batch given the parameters above.
    """

    def fn(
        batch: list[tuple[torch.Tensor, float, int, str]]
    ) -> SpectrumBatch | AnnotatedSpectrumBatch:
        if annotated:
            spectra, precursor_mz, precursor_charge, peptides = list(zip(*batch))
        else:
            spectra, precursor_mz, precursor_charge = list(zip(*batch))

        spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
        spectra_padding_mask = spectra[:, :, 0] == 0.0

        precursor_mz = torch.tensor(precursor_mz)
        precursor_charge = torch.FloatTensor(precursor_charge)
        precursor_masses = (precursor_mz - PROTON_MASS_AMU) * precursor_charge
        precursors = torch.stack([precursor_masses, precursor_charge, precursor_mz], -1).float()
        if annotated:
            peptides = [sequence if isinstance(sequence, str) else "$" for sequence in peptides]
            peptides = [sequence if len(sequence) > 0 else "$" for sequence in peptides]
            peptides = torch.stack(
                [
                    residues.encode(residues.tokenize(sequence)[:max_length], pad_length=max_length)
                    for sequence in peptides
                ]
            )
            peptide_padding_mask = peptides == residues.PAD_INDEX
            return AnnotatedSpectrumBatch(
                spectra, spectra_padding_mask, precursors, peptides, peptide_padding_mask
            )
        else:
            return SpectrumBatch(spectra, spectra_padding_mask, precursors)

    return fn
