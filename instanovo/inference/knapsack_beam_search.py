from __future__ import annotations

import numpy
import torch
from jaxtyping import Float, Integer

from instanovo.constants import CARBON_MASS_DELTA
from instanovo.inference.beam_search import BeamSearchDecoder
from instanovo.inference.interfaces import Decodable
from instanovo.inference.knapsack import Knapsack
from instanovo.types import DiscretizedMass, ResidueLogProbabilities


class KnapsackBeamSearchDecoder(BeamSearchDecoder):
    """Class for decoding from de novo sequencing models using beam search & knapsack filtering."""

    def __init__(
        self,
        model: Decodable,
        knapsack: Knapsack,
    ):
        super().__init__(model=model, mass_scale=knapsack.mass_scale)
        self.knapsack = knapsack
        self.chart = torch.tensor(self.knapsack.chart)

    @classmethod
    def from_file(cls, model: Decodable, path: str) -> KnapsackBeamSearchDecoder:
        """Initialize a decoder by loading a saved knapsack.

        Args:
            model (Decodable): The model to be decoded from.
            path (str): The path to the directory where the knapsack
                        was saved to.

        Returns:
            KnapsackBeamSearchDecoder: The decoder.
        """
        knapsack = Knapsack.from_file(path=path)
        return cls(model=model, knapsack=knapsack)

    def prefilter_items(
        self,
        log_probabilities: Float[ResidueLogProbabilities, "batch beam residue"],
        remaining_masses: Integer[DiscretizedMass, "batch beam residue"],
        beam_masses: Integer[DiscretizedMass, "batch beam"],
        mass_buffer: Integer[DiscretizedMass, "batch 1 1"],
        max_isotope: int,
    ) -> Float[ResidueLogProbabilities, "batch beam residue"]:
        """Filter illegal next token by setting the corresponding log probabilities to `-inf`.

        Args:
            log_probabilities (torch.FloatTensor[batch size, beam size, number of residues]):
                The candidate log probabilities for each
                item on the beam and each potential next residue
                for batch spectrum in the batch.

            remaining_masses (torch.LongTensor[batch size, beam size, number of residues]):

            mass_buffer (torch.LongTensor[batch size, 1, 1]): _description_

        Returns:
            torch.FloatTensor: _description_
        """
        log_probabilities = super().prefilter_items(
            log_probabilities=log_probabilities,
            remaining_masses=remaining_masses,
            beam_masses=beam_masses,
            mass_buffer=mass_buffer,
            max_isotope=max_isotope,
        )
        mass_lower_bound = torch.clamp(beam_masses - mass_buffer.squeeze(-1), min=0)
        mass_upper_bound = beam_masses + mass_buffer.squeeze(-1)
        batch_size, beam_size, num_residues = log_probabilities.shape
        scaled_nucleon_mass = round(self.mass_scale * CARBON_MASS_DELTA)
        for batch in range(batch_size):
            for beam in range(beam_size):
                beam_lower_bound = mass_lower_bound[batch, beam].item()
                beam_upper_bound = mass_upper_bound[batch, beam].item()

                for residue in range(num_residues):
                    if log_probabilities[batch, beam, residue].isfinite().item():
                        valid_residue = self.chart[
                            beam_lower_bound : (beam_upper_bound + 1), residue
                        ].any()
                        if max_isotope > 0:
                            for num_nucleons in range(1, max_isotope + 1):
                                local_valid_residue = self.chart[
                                    beam_lower_bound - num_nucleons * scaled_nucleon_mass : (
                                        beam_upper_bound - num_nucleons * scaled_nucleon_mass + 1
                                    ),
                                    residue,
                                ].any()
                                valid_residue = valid_residue or local_valid_residue

                        if not valid_residue:
                            log_probabilities[batch, beam, residue] = -float("inf")

        return log_probabilities

    def _get_isotope_chart(
        self,
        beam_lower_bound: int,
        beam_upper_bound: int,
        scaled_nucleon_mass: int,
        num_nucleons: int,
    ) -> Float[numpy.ndarray, "mass residue"]:
        return self.chart[
            (beam_lower_bound - num_nucleons * scaled_nucleon_mass) : (
                beam_upper_bound - num_nucleons * scaled_nucleon_mass + 1
            )
        ].any(0)

    def _init_prefilter(
        self,
        precursor_masses: Integer[DiscretizedMass, " batch"],
        log_probabilities: Float[ResidueLogProbabilities, "batch beam"],
        mass_buffer: Integer[DiscretizedMass, " batch"],
    ) -> Float[ResidueLogProbabilities, "batch beam"]:
        mass_lower_bound = torch.clamp(precursor_masses - mass_buffer, min=0)
        mass_upper_bound = precursor_masses + mass_buffer
        for batch, (lower_bound, upper_bound) in enumerate(
            zip(mass_lower_bound, mass_upper_bound, strict=True)
        ):
            valid_residues = self.chart[lower_bound:upper_bound].any(0)
            log_probabilities[batch, ~valid_residues] = -float("inf")
        return log_probabilities
