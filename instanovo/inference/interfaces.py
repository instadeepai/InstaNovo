from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Any

import torch
from jaxtyping import Float
from jaxtyping import Integer

from instanovo.types import Peptide
from instanovo.types import PrecursorFeatures
from instanovo.types import Spectrum


class Decodable(metaclass=ABCMeta):
    """An interface for models that can be decoded by algorithms that conform to the search interface."""

    @abstractmethod
    def init(  # type:ignore
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        *args,
        **kwargs,
    ) -> Any:
        """Initialize the search state.

        Args:
            spectra (torch.FloatTensor):
                The spectra to be sequenced.

            precursors (torch.FloatTensor[batch size, 3]):
                The precursor mass, charge and mass-to-charge ratio.
        """
        pass

    @abstractmethod
    def score_candidates(  # type:ignore
        self,
        sequences: Integer[Peptide, "..."],
        precursor_mass_charge: Float[PrecursorFeatures, "..."],
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """Generate and score the next set of candidates.

        Args:
            sequences (torch.LongTensor):
                Partial residue sequences in generated
                the course of decoding.

            precursor_mass_charge (torch.FloatTensor[batch size, 3]):
                The precursor mass, charge and mass-to-charge ratio.
        """
        pass

    @abstractmethod
    def get_residue_masses(self, mass_scale: int) -> torch.LongTensor:
        """Get residue masses for the model's residue vocabulary.

        Args:
            mass_scale (int):
                The scale in Daltons at which masses are
                calculated and rounded off. For example,
                a scale of 10000 would represent masses
                at a scale of 1e4 Da.
        """
        pass

    @abstractmethod
    def decode(self, sequence: Integer[Peptide, "..."]) -> list[str]:
        """Map sequences of indices to residues using the model's residue vocabulary.

        Args:
            sequence (torch.LongTensor):
                The sequence of residue indices to be mapped
                to the corresponding residue strings.
        """
        pass

    @abstractmethod
    def get_eos_index(self) -> int:
        """Get the end of sequence token's index in the model's residue vocabulary."""
        pass

    @abstractmethod
    def get_empty_index(self) -> int:
        """Get the empty token's index in the model's residue vocabulary."""
        pass


class Decoder(metaclass=ABCMeta):
    """A class that implements some search algorithm for decoding from a model that conforms to the `Decodable` interface.

    Args:
        model (Decodable):
            The model to predict residue sequences
            from using the implemented search
            algorithm.
    """

    def __init__(self, model: Decodable):
        self.model = model

    @abstractmethod
    def decode(  # type:ignore
        self,
        spectra: Float[Spectrum, "..."],
        precursors: Float[PrecursorFeatures, "..."],
        *args,
        **kwargs,
    ) -> list[list[str]]:
        """Generate the predicted residue sequence using the decoder's search algorithm.

        Args:
            spectra (torch.FloatTensor):
                The spectra to be sequenced.

            precursors (torch.FloatTensor):
                The precursor mass, charge and mass-to-charge ratio.

        """
        pass
