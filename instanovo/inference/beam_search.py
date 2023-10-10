from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch.nn.functional import one_hot

from instanovo.inference.interfaces import Decodable
from instanovo.inference.interfaces import Decoder
from instanovo.constants import H2O_MASS, CARBON_MASS_DELTA
from instanovo.constants import MASS_SCALE, INTEGER, PrecursorDimension
from instanovo.constants import PRECURSOR_DIM 


@dataclass
class BeamState:
    """This class holds a specification of the beam state during beam search.

    Args:
        sequences (torch.LongTensor[beam size, num. steps]):
                    A tensor of the partial sequences on the beam
                    represented as indices into a residue vocabulary.

        log_probabilities (torch.FloatTensor[beam size]):
                    The log probabilities of each of the
                    partial sequences on the beam.

        remaining_masses (torch.LongTensor[beam size]):
                    The remaining mass still to be allocated
                    before each of the partial sequences are
                    complete, represented as parts per
                    MASS_SCALE. This is defined as the upper error limit
                    on the precursor mass minus the partial sequence's
                    theoretical mass.

        precursor_mass_charge (torch.FloatTensor[beam size, 3]):
                    The precursor mass, charge and mass-to-charge ratio
                    respectively.

        spectrum_encoding (torch.FloatTensor[batch size, sequence length, hidden dim]):
                    The spectrum embedding. These are the outputs of running
                    a transformer peptide encoder on the spectra.

        spectrum_mask (torch.BoolTensor[batch size, sequence length]):
                    The padding mask of the spectrum embedding. Indices
                    in the original sequence map to 1 and padding tokens map
                    to 0.
    """

    sequences: torch.LongTensor | None
    log_probabilities: torch.FloatTensor | None
    remaining_masses: torch.LongTensor | None
    precursor_mass_charge: torch.FloatTensor
    spectrum_encoding: torch.FloatTensor
    spectrum_mask: torch.BoolTensor

    def is_empty(self) -> bool:
        """Check whether the beam is empty."""
        if self.sequences is None:
            if (self.log_probabilities is not None) or (self.remaining_masses is not None):
                raise ValueError(
                    f"""Sequences, log_probabilities and remaining masses
                                     should all be None or all not None. Sequences
                                     is None while log_probabilities is
                                     {self.log_probabilities} and remaining masses is
                                     {self.remaining_masses}.
                                     """
                )
            return True
        else:
            if (self.log_probabilities is None) or (self.remaining_masses is None):
                raise ValueError(
                    f"""Sequences, log_probabilities and remaining masses
                                     should all be None or all not None. Sequences
                                     is not None while log_probabilities is
                                     {self.log_probabilities} and remaining masses is
                                     {self.remaining_masses}.
                                     """
                )
            return False


@dataclass
class ScoredSequence:
    """This class holds a residue sequence and its log probability."""

    sequence: list[str]
    mass_error: float
    log_probability: float


class BeamSearchDecoder(Decoder):
    """A class for decoding from de novo sequence models using beam search.

    This class conforms to the `Decoder` interface and decodes from
    models that conform to the `Decodable` interface.
    """

    def __init__(self, model: Decodable, mass_scale: int = MASS_SCALE):
        super().__init__(model=model)
        self.mass_scale = mass_scale

    @staticmethod
    def unravel_index(
        indices: torch.LongTensor, outer_dim: int
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Get row and column coordinates for indices on a pair of dimensions that have been flattened.

        Args:
            indices (torch.LongTensor): The flattened indices to unravel
            outer_dim (int): The outermost dimension of the pair that has been flattened

        Returns:
            tuple[torch.LongTensor, torch.LongTensor]:
                The rows and columns of the indices respectively
        """
        rows = indices.div(outer_dim, rounding_mode="floor")
        columns = indices.remainder(outer_dim)
        return rows, columns

    def expand_candidates(
        self, beam_state: BeamState, residue_masses: torch.LongTensor
    ) -> torch.FloatTensor:
        """Calculate log probabilities for all candidate next tokens for all sequences in the current beam.

        Args:
            beam_state (BeamState): The current beam state

        Returns:
            Result (torch.FloatTensor [beam size, vocabulary size]):
            The tensor of log probabilities on the candidate next tokens for
            each sequence in the beam for each spectrum in the batch.

            Result[i, j, k] is the log probability of token `k` in the vocabulary
            being the next token given sequence `i` in the beam for batch spectrum
            `j` is the prefix.

            `exp(Result[i, j]).sum() == 1` for all `i` and `j`.

        """
        assert beam_state.remaining_masses is not None
        remaining_masses = beam_state.remaining_masses.unsqueeze(-1) - residue_masses.unsqueeze(
            0
        ).unsqueeze(0)

        assert beam_state.sequences is not None
        sequence_length = beam_state.sequences.shape[-1]
        spectrum_length = beam_state.spectrum_encoding.shape[2]
        hidden_dim = beam_state.spectrum_encoding.shape[3]
        log_probabilities = self.model.score_candidates(
            beam_state.sequences.reshape(-1, sequence_length),
            beam_state.precursor_mass_charge.reshape(-1, PRECURSOR_DIM),
            beam_state.spectrum_encoding.reshape(-1, spectrum_length, hidden_dim),
            beam_state.spectrum_mask.reshape(-1, spectrum_length),
        )

        assert beam_state.log_probabilities is not None
        batch_size = beam_state.log_probabilities.shape[0]
        beam_size = beam_state.log_probabilities.shape[1]
        candidate_log_probabilities = log_probabilities.reshape(
            batch_size, beam_size, -1
        ) + beam_state.log_probabilities.unsqueeze(-1)
        return candidate_log_probabilities, remaining_masses

    def filter_items(
        self,
        beam_state: BeamState,
        log_probabilities: torch.FloatTensor,
        beam_size: int,
        remaining_masses: torch.LongTensor,
        mass_buffer: torch.LongTensor,
        max_isotope: int,
    ) -> tuple[list[list[ScoredSequence]], BeamState]:
        """Separate and prune incomplete and complete sequences.

        Separate candidate residues into those that lead to incomplete sequences and those that lead
        to complete sequences. Prune the ones leading to incomplete sequences down to the top `beam_size`
        and simply return the ones leading to complete sequences.

        Args:
            beam_state (BeamState):
                The current beam state.

            log_probabilities (torch.FloatTensor[batch size, beam size, number of residues]):
                The candidate log probabilities for each
                item on the beam and each potential next residue
                for batch spectrum in the batch.

            beam_size (int):
                The maximum size of the beam.

            residue_masses (torch.FloatTensor[number of residues]):
                The masses of the residues in the vocabulary
                as integers in units of the mass scale.

            mass_buffer (torch.FloatTensor[batch size]):
                The maximum absolute difference between
                the batch precursor masses and the
                theoretical masses of their predicted
                sequences.

        Returns:
            tuple[list[ScoredSequence], BeamState]:
                A (potentially empty) list of completed sequences
                and the next beam state resulting from pruning
                incomplete sequences.
        """
        assert beam_state.remaining_masses is not None
        reshaped_mass_buffer = mass_buffer.unsqueeze(-1).unsqueeze(-1)

        batch_size, beam_size, num_residues = log_probabilities.shape

        # Collect completed items
        completed_items: list[list[ScoredSequence]] = [[] for _ in range(batch_size)]

        is_eos = (
            one_hot(
                torch.tensor([self.model.get_eos_index()])
                .unsqueeze(0)
                .expand(batch_size, beam_size),
                num_classes=num_residues,
            )
            .bool()
            .to(log_probabilities.device)
        )
        item_is_complete = (reshaped_mass_buffer >= remaining_masses) & (
            remaining_masses >= -reshaped_mass_buffer
        )
        if max_isotope > 0:
            for num_nucleons in range(1, max_isotope + 1):
                isotope_is_complete = (
                    reshaped_mass_buffer
                    >= remaining_masses - num_nucleons * round(self.mass_scale * CARBON_MASS_DELTA)
                ) & (
                    remaining_masses - num_nucleons * round(self.mass_scale * CARBON_MASS_DELTA)
                    >= -reshaped_mass_buffer
                )
                item_is_complete = item_is_complete | isotope_is_complete
        item_is_complete = item_is_complete & ~is_eos & log_probabilities.isfinite()

        assert beam_state.sequences is not None
        local_variables = zip(
            item_is_complete, remaining_masses, log_probabilities, beam_state.sequences
        )
        for batch, (is_complete, mass_errors, local_log_probabilities, sequences) in enumerate(
            local_variables
        ):
            if is_complete.any().item():
                beam_index, residues = torch.where(is_complete)
                completed_sequences = torch.column_stack((sequences[beam_index], residues))

                eos_log_probabilities = self.model.score_candidates(
                    completed_sequences,
                    beam_state.precursor_mass_charge[batch, beam_index],
                    beam_state.spectrum_encoding[batch, beam_index],
                    beam_state.spectrum_mask[batch, beam_index],
                )
                completed_log_probabilities = (
                    local_log_probabilities[beam_index, residues]
                    + eos_log_probabilities[:, self.model.get_eos_index()]
                )
                completed_mass_errors = mass_errors[beam_index, residues]
                completed_items[batch].extend(
                    ScoredSequence(
                        sequence=self.model.decode(sequence),
                        mass_error=mass_error.item() / self.mass_scale,
                        log_probability=log_probability,
                    )
                    for sequence, mass_error, log_probability in zip(
                        completed_sequences,
                        completed_mass_errors,
                        completed_log_probabilities.tolist(),
                    )
                )

        # Filter invalid items
        log_probabilities = self.prefilter_items(
            log_probabilities=log_probabilities,
            remaining_masses=remaining_masses,
            beam_masses=beam_state.remaining_masses,
            mass_buffer=reshaped_mass_buffer,
            max_isotope=max_isotope,
        )

        # Prune incomplete items to form next beam
        beam_log_probabilities, beam_indices = log_probabilities.reshape(batch_size, -1).topk(
            k=beam_size
        )
        beam_sequences = self._append_next_token(
            indices=beam_indices, outer_dim=num_residues, sequences=beam_state.sequences
        )
        remaining_masses = remaining_masses.reshape(batch_size, -1)
        beam_remaining_masses = []
        for local_remaining_masses, local_indices in zip(remaining_masses, beam_indices):
            beam_remaining_masses.append(local_remaining_masses[local_indices])
        beam_remaining_masses = torch.stack(beam_remaining_masses)
        new_beam = BeamState(
            sequences=beam_sequences,
            log_probabilities=beam_log_probabilities,
            remaining_masses=beam_remaining_masses,
            precursor_mass_charge=beam_state.precursor_mass_charge,
            spectrum_encoding=beam_state.spectrum_encoding,
            spectrum_mask=beam_state.spectrum_mask,
        )
        return completed_items, new_beam

    def prefilter_items(
        self,
        log_probabilities: torch.FloatTensor,
        remaining_masses: torch.LongTensor,
        beam_masses: torch.LongTensor,
        mass_buffer: torch.LongTensor,
        max_isotope: int,
    ) -> torch.FloatTensor:
        """Filter illegal next token by setting the corresponding log probabilities to `-inf`.

        Args:
            log_probabilities (torch.FloatTensor[batch size, beam size, number of residues]):
                The candidate log probabilities for each
                item on the beam and each potential next residue
                for batch spectrum in the batch.

            remaining_masses (torch.LongTensor[batch size, beam size]):

            mass_buffer (torch.LongTensor[batch size, 1, 1]): _description_

        Returns:
            torch.FloatTensor: _description_
        """
        # Filter out the end of sequence token since
        # it is added manually when a sequence's mass
        # means it's complete
        EOS_TOKEN = self.model.get_eos_index()
        log_probabilities[:, :, EOS_TOKEN] = -float("inf")

        # Filter out the empty token
        EMPTY_TOKEN = self.model.get_empty_index()
        log_probabilities[:, :, EMPTY_TOKEN] = -float("inf")

        # Filter out large masses
        mass_is_invalid = remaining_masses < -mass_buffer
        log_probabilities[mass_is_invalid] = -float("inf")

        return log_probabilities

    def init_beam(
        self,
        spectra: torch.FloatTensor,
        precursor_mass_charge: torch.FloatTensor,
        residue_masses: torch.LongTensor,
        mass_buffers: torch.LongTensor,
        beam_size: int,
    ) -> BeamState:
        """Construct the initial beam state.

        This means precomputing the spectrum embeddings and
        adding the first set of candidate tokens to the beam.

        Args:
            spectra (torch.FloatTensor[]):
                The spectra to be sequenced.

            precursor_mass_charge (torch.FloatTensor[batch size, 3]):
                The precursor mass, charge and mass-to-charge ratio.

            residue_masses (torch.LongTensor[]):
                The masses of the residues in the vocabulary
                as integers in units of the mass scale.

            beam_size (int):
                The maximum size of the beam.

        Returns:
            BeamState:
                The initial beam state.
        """
        # 1. Compute spectrum encoding and masks
        (spectrum_encoding, spectrum_mask), log_probabilities = self.model.init(
            spectra, precursor_mass_charge
        )

        # 2. Calculate candidate residue masses and remaining mass budgets
        precursor_masses = (
            torch.round(
                self.mass_scale * precursor_mass_charge[:, PrecursorDimension.PRECURSOR_MASS.value]
            )
            .type(INTEGER)
            .to(spectra.device)
        )
        precursor_masses = precursor_masses - round(self.mass_scale * H2O_MASS)

        log_probabilities = self._init_prefilter(
            precursor_masses=precursor_masses,
            log_probabilities=log_probabilities,
            mass_buffer=mass_buffers,
        )

        beam_log_probabilities = log_probabilities.topk(k=beam_size)
        beam_masses = residue_masses.to(spectra.device).gather(-1, beam_log_probabilities.indices)
        remaining_masses = precursor_masses.unsqueeze(-1) - beam_masses

        # 3. Copy inputs for beam candidates
        beam_precursor_mass_charge = precursor_mass_charge.unsqueeze(1).expand(-1, beam_size, -1)
        beam_spectrum_encoding = spectrum_encoding.unsqueeze(1).expand(-1, beam_size, -1, -1)
        beam_spectrum_mask = spectrum_mask.unsqueeze(1).expand(-1, beam_size, -1)
        return BeamState(
            sequences=beam_log_probabilities.indices.unsqueeze(-1),
            log_probabilities=beam_log_probabilities.values,
            remaining_masses=remaining_masses,
            precursor_mass_charge=beam_precursor_mass_charge,
            spectrum_encoding=beam_spectrum_encoding,
            spectrum_mask=beam_spectrum_mask,
        )

    def decode(  # type:ignore
        self,
        spectra: torch.FloatTensor,
        precursors: torch.FloatTensor,
        beam_size: int,
        max_length: int,
        mass_tolerance: float = 5e-5,
        max_isotope: int = 1,
        return_all_beams: bool = False,
    ) -> list[Any]:
        """Decode predicted residue sequence for a batch of spectra using beam search.

        Args:
            spectra (torch.FloatTensor):
                The spectra to be sequenced.

            precursors (torch.FloatTensor[batch size, 3]):
                The precursor mass, charge and mass-to-charge ratio.

            beam_size (int):
                The maximum size of the beam.

            max_length (int):
                The maximum length of a residue sequence.

            mass_tolerance (float):
                The maximum relative error for which a predicted sequence
                is still considered to have matched the precursor mass.

            max_isotope (int):
                The maximum number of additional neutrons for isotopes
                whose mass a predicted sequence's mass is considered
                when comparing to the precursor mass.

                All additional nucleon numbers from 1 to `max_isotope` inclusive
                are considered.

            return_all_beams (bool):
                Optionally return all beam-search results, not only the best beam.

        Returns:
            list[list[str]]:
                The predicted sequence as a list of residue tokens.
                This method will return an empty list for each
                spectrum in the batch where
                decoding fails i.e. no sequence that fits the precursor mass
                to within a tolerance is found.
        """
        with torch.no_grad():
            batch_size = spectra.shape[0]
            complete_items: list[list[ScoredSequence]] = [[] for _ in range(batch_size)]

            # Precompute mass matrix and mass buffers
            residue_masses = self.model.get_residue_masses(mass_scale=self.mass_scale).to(
                spectra.device
            )
            num_residues = residue_masses.shape[-1]
            mass_buffers = (
                (
                    self.mass_scale
                    * mass_tolerance
                    * precursors[:, PrecursorDimension.PRECURSOR_MASS.value]
                )
                .round()
                .long()
            )

            # Initialize beam
            beam: BeamState = self.init_beam(
                spectra=spectra,
                precursor_mass_charge=precursors,
                residue_masses=residue_masses.unsqueeze(0).expand(batch_size, num_residues),
                beam_size=beam_size,
                mass_buffers=mass_buffers,
            )

            for _ in range(max_length):
                if beam.is_empty():
                    break

                assert beam.log_probabilities is not None
                if beam.log_probabilities.isinf().all():
                    break

                # 1. Expand candidates
                log_probabilities, remaining_masses = self.expand_candidates(
                    beam_state=beam, residue_masses=residue_masses
                )

                # 2. Filter complete items and prune incomplete ones to get the new beam
                complete_candidates, beam = self.filter_items(
                    log_probabilities=log_probabilities,
                    beam_state=beam,
                    beam_size=beam_size,
                    remaining_masses=remaining_masses,
                    mass_buffer=mass_buffers,
                    max_isotope=max_isotope,
                )
                for i, items in enumerate(complete_candidates):
                    complete_items[i].extend(items)

            for items in complete_items:
                items.sort(key=lambda item: item.log_probability, reverse=True)

            if not return_all_beams:
                sequences = [items[0] if len(items) > 0 else [] for items in complete_items]
            else:
                sequences = [items if len(items) > 0 else [] for items in complete_items]

            return sequences

    def _init_prefilter(
        self,
        precursor_masses: torch.LongTensor,
        log_probabilities: torch.FloatTensor,
        mass_buffer: torch.LongTensor,
    ) -> torch.FloatTensor:
        return log_probabilities

    def _append_next_token(
        self,
        indices: torch.LongTensor,
        outer_dim: int,
        sequences: torch.LongTensor,
    ) -> torch.LongTensor:
        beam_items, residues = self.unravel_index(indices, outer_dim)

        collected_sequences = []
        for beam_item, residue, sequence in zip(beam_items, residues, sequences):
            collected_sequences.append(torch.column_stack((sequence[beam_item], residue)))
        return torch.stack(collected_sequences, 0)
