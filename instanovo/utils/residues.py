from __future__ import annotations

import re

import numpy as np
import torch

from instanovo.constants import H2O_MASS, PROTON_MASS_AMU, SpecialTokens


class ResidueSet:
    """A class for managing sets of residues.

    Args:
        residue_masses (dict[str, float]):
            Dictionary of residues mapping to corresponding mass values.
        residue_remapping (dict[str, str] | None, optional):
            Dictionary of residues mapping to keys in `residue_masses`.
            This is used for dataset specific residue naming conventions.
            Residue remapping may be many-to-one.
    """

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str] | None = None,
    ) -> None:
        self.residue_masses = residue_masses
        self.residue_remapping = residue_remapping if residue_remapping else {}

        # Special tokens come first
        self.special_tokens = [
            SpecialTokens.PAD_TOKEN.value,
            SpecialTokens.SOS_TOKEN.value,
            SpecialTokens.EOS_TOKEN.value,
        ]
        self.vocab = self.special_tokens + list(self.residue_masses.keys())

        # Create mappings
        self.residue_to_index = {
            residue: index for index, residue in enumerate(self.vocab)
        }
        self.index_to_residue = {
            index: residue for index, residue in enumerate(self.vocab)
        }
        # Split on amino acids allowing for modifications eg. AM(ox)Z -> [A, M(ox), Z]
        # Supports brackets or unimod notation
        self.tokenizer_regex = (
            # First capture group: matches either:
            # - A UNIMOD annotation like [UNIMOD:123]
            # - Any text inside parentheses like (ox) or (+.98)
            r"(\[UNIMOD:\d+\]|\([^)]+\))|"
            # Second capture group: starts with a valid amino acid letter
            # (including U for selenocysteine and O for pyrrolysine)
            r"([A-Z]"
            # Optionally followed by a UNIMOD annotation
            r"(?:\[UNIMOD:\d+\]|"
            # Or optionally followed by text in parentheses
            r"\([^)]+\))?"
            # Close second capture group
            r")"
        )

        self.PAD_INDEX: int = self.residue_to_index[SpecialTokens.PAD_TOKEN.value]
        self.SOS_INDEX: int = self.residue_to_index[SpecialTokens.SOS_TOKEN.value]
        self.EOS_INDEX: int = self.residue_to_index[SpecialTokens.EOS_TOKEN.value]

        # TODO: Add support for specifying which residues are n-terminal only.

    def update_remapping(self, mapping: dict[str, str]) -> None:
        """Update the residue remapping for specific datasets.

        Args:
            mapping (dict[str, str]):
                The mapping from residues specific to a dataset
                to residues in the original `residue_masses`.
        """
        self.residue_remapping.update(mapping)

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
        if self.residue_remapping and residue in self.residue_remapping:
            residue = self.residue_remapping[residue]
        return self.residue_masses[residue]

    def get_sequence_mass(self, sequence: str, charge: int | None) -> float:
        """Get the mass of a residue sequence.

        Args:
            sequence (str):
                The residue sequence whose mass to calculate.
                All residues must be in the residue set or
                this will raise a `KeyError`.
            charge (int | None, optional):
                Charge of the sequence to calculate the mass.

        Returns:
            float: The mass of the residue in Daltons.
                   If a charge is specified, returns m/z.
        """
        mass = sum([self.get_mass(residue) for residue in sequence]) + H2O_MASS
        if charge:
            mass = (mass / charge) + PROTON_MASS_AMU
        return float(mass)

    def tokenize(self, sequence: str) -> list[str]:
        """Split a peptide represented as a string into a list of residues.

        Args:
            sequence (str): The peptide to be split.

        Returns:
            list[str]: The sequence of residues forming the peptide.
        """
        # return re.split(self.tokenizer_regex, sequence)
        # TODO: find a way to handle N-terminal PTMs appearing at any position
        return [
            item
            for sublist in re.findall(self.tokenizer_regex, sequence)
            for item in sublist
            if item
        ]

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

    def encode(
        self,
        sequence: list[str],
        add_eos: bool = False,
        return_tensor: str | None = None,
        pad_length: int | None = None,
    ) -> torch.LongTensor | np.ndarray:
        """Map a sequence of residues to their indices and optionally pad them to a fixed length.

        Args:
            sequence (list[str]):
                The sequence of residues.
            add_eos (bool):
                Add an EOS token when encoding.
                Defaults to `False`.
            return_tensor (str | None, optional):
                Return type of encoded tensor. Returns a list if integers
                if no return type is specified. Options: None, pt, np
            pad_length (int | None, optional):
                An optional fixed length to pad the encoded sequence to.
                If this is `None`, no padding is done.

        Returns:
            torch.LongTensor | np.ndarray:
                A tensor with the indices of the residues.
        """
        encoded_list = [
            self.residue_to_index[
                # remap the residue if possible
                self.residue_remapping[residue]
                if residue in self.residue_remapping
                else residue
            ]
            for residue in sequence
        ]

        if add_eos:
            encoded_list.extend([self.EOS_INDEX])
        if pad_length:
            encoded_list.extend((pad_length - len(encoded_list)) * [self.PAD_INDEX])

        if return_tensor == "pt":
            return torch.tensor(encoded_list, dtype=torch.long)
        elif return_tensor == "np":
            return np.array(encoded_list, dtype=np.int32)
        else:
            return encoded_list

    def decode(
        self, sequence: torch.LongTensor | list[int], reverse: bool = False
    ) -> list[str]:
        """Map a sequence of indices to the corresponding sequence of residues.

        Args:
            sequence (torch.LongTensor | list[int]):
                The sequence of residue indices.
            reverse (bool):
                Optionally reverse the decoded sequence.

        Returns:
            list[str]:
                The corresponding sequence of residue strings.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        residue_sequence = []
        for index in sequence:
            if index == self.EOS_INDEX:
                break
            if index == self.SOS_INDEX or index == self.PAD_INDEX:
                continue
            residue_sequence.append(index)

        if reverse:
            residue_sequence = residue_sequence[::-1]

        return [self.index_to_residue[index] for index in residue_sequence]

    def __len__(self) -> int:
        return len(self.index_to_residue)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueSet):
            return NotImplemented
        return self.vocab == other.vocab
