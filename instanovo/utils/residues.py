from __future__ import annotations

import re

import torch


class ResidueSet:
    """A class for managing sets of residues."""

    def __init__(self, residue_masses: dict[str, float]) -> None:
        self.residue_masses = residue_masses
        self.residue_to_index = {
            residue: index for index, residue in enumerate(self.residue_masses.keys())
        }
        self.index_to_residue = list(self.residue_to_index.keys())
        self.tokenizer_regex = r"(?<=.)(?=[A-Z])"
        self.eos_index = self.residue_to_index["$"]
        self.pad_index = self.eos_index

    def get_mass(self, residue: str) -> float:
        """Get the mass of a residue.

        Args:
            residue (str):
                The residue whose mass to fetch. This residue
                must be in the residue set or this will raise
                a `KeyError`.

        Returns:
            float: The mass of the residue in Daltons.
        """
        return self.residue_masses[residue]

    def tokenize(self, sequence: str) -> list[str]:
        """Split a peptide represented as a string into a list of residues.

        Args:
            sequence (str): The peptide to be split.

        Returns:
            list[str]: The sequence of residues forming the peptide.
        """
        return re.split(self.tokenizer_regex, sequence)

    def detokenize(self, sequence: list[str]) -> str:
        """Joining a list of residues into a string representing the peptide.

        Args:
            sequence (list[str]):
                The sequence of residues.

        Returns:
            str:
                The string representing the peptide.
        """
        return "".join(sequence)

    def encode(self, sequence: list[str], pad_length: int | None = None) -> torch.LongTensor:
        """Map a sequence of residues to their indices and optionally pad them to a fixed length.

        Args:
            sequence (list[str]):
                The sequence of residues.
            pad_length (int | None, optional):
                An optional fixed length to pad the encoded sequence to.
                If this is `None`, no padding is done.

        Returns:
            torch.LongTensor:
                A tensor with the indices of the residues.
        """
        encoded_list = [self.residue_to_index[residue] for residue in sequence]
        if pad_length:
            encoded_list.extend((pad_length - len(encoded_list)) * [self.pad_index])
        return torch.tensor(encoded_list)

    def decode(self, sequence: list[int]) -> list[str]:
        """Map a sequence of indices to the corresponding sequence of residues.

        Args:
            sequence (list[int]): The sequence of residue indices.

        Returns:
            list[str]: The corresponding sequence of residue strings.
        """
        return [self.index_to_residue[index] for index in sequence]

    def __len__(self) -> int:
        return len(self.residue_masses)
