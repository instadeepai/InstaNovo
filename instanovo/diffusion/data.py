from __future__ import annotations

from collections.abc import Callable

import polars
import torch

from instanovo.utils.residues import ResidueSet


class PolarsSpectrumDataset(torch.utils.data.Dataset):
    """An `polars` data frame index wrapper for `depthcharge`/`casanovo` datasets."""

    def __init__(self, data_frame: polars.DataFrame) -> None:
        self.data = data_frame

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, float]:
        row = self.data[idx]
        mz_array = torch.FloatTensor(row["Mass values"].to_numpy()[0])
        int_array = row["Intensity"].to_numpy()[0]
        int_array = torch.FloatTensor(int_array / int_array.max())
        precursor_mz = row["MS/MS m/z"].to_numpy()[0]
        precursor_charge = row["Charge"].to_numpy()[0]
        spectrum = torch.stack([mz_array, int_array]).T

        return spectrum, precursor_mz, precursor_charge


class AnnotatedPolarsSpectrumDataset(PolarsSpectrumDataset):
    """A dataset with a Polars index that includes peptides from an aligned list."""

    def __init__(self, data_frame: polars.DataFrame, peptides: list[str]) -> None:
        super().__init__(data_frame)
        self.peptides = peptides

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, float, str]:
        spectrum, precursor_mz, precursor_charge = super().__getitem__(idx)
        peptide = self.peptides[idx]
        return spectrum, precursor_mz, precursor_charge, peptide


def collate_batches(
    residues: ResidueSet, max_length: int, time_steps: int, annotated: bool
) -> Callable[
    [list[tuple[torch.FloatTensor, float, int, str]]],
    tuple[
        torch.FloatTensor,
        torch.BoolTensor,
        torch.FloatTensor,
    ]
    | tuple[
        torch.FloatTensor,
        torch.BoolTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.BoolTensor,
    ],
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
        Callable[ [list[tuple[torch.FloatTensor, float, int, str]]], tuple[ torch.FloatTensor, torch.BoolTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.BoolTensor, ], ]:
            The function that combines examples into a batch given the parameters above.
    """

    def fn(
        batch: list[tuple[torch.Tensor, float, int, str]]
    ) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.FloatTensor] | tuple[
        torch.FloatTensor,
        torch.BoolTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.BoolTensor,
    ]:
        if annotated:
            spectra, precursor_mz, precursor_charge, peptides = list(zip(*batch))
        else:
            spectra, precursor_mz, precursor_charge = list(zip(*batch))

        spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
        spectra_padding_mask = spectra[:, :, 0] == 0.0

        precursor_mz = torch.tensor(precursor_mz)
        precursor_charge = torch.FloatTensor(precursor_charge)
        precursor_masses = (precursor_mz - 1.007276) * precursor_charge
        precursors = torch.stack([precursor_masses, precursor_charge, precursor_mz], -1).float()
        if annotated:
            peptides = [sequence if isinstance(sequence, str) else "$" for sequence in peptides]
            peptides = torch.stack(
                [
                    residues.encode(residues.tokenize(sequence)[:max_length], pad_length=max_length)
                    for sequence in peptides
                ]
            )
            peptide_padding_mask = peptides == residues.pad_index
            return spectra, spectra_padding_mask, precursors, peptides, peptide_padding_mask
        else:
            return spectra, spectra_padding_mask, precursors

    return fn
