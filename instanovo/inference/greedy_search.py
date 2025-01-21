from __future__ import annotations

import torch
from jaxtyping import Float

from instanovo.types import PrecursorFeatures
from instanovo.types import Spectrum
from instanovo.constants import CARBON_MASS_DELTA
from instanovo.constants import H2O_MASS
from instanovo.constants import MASS_SCALE
from instanovo.constants import PrecursorDimension
from instanovo.inference.interfaces import Decodable
from instanovo.inference.interfaces import Decoder
from instanovo.inference.interfaces import ScoredSequence


class GreedyDecoder(Decoder):
    """A class for decoding from de novo sequence models using beam search.

    This class conforms to the `Decoder` interface and decodes from
    models that conform to the `Decodable` interface.
    """

    def __init__(
        self,
        model: Decodable,
        suppressed_residues: list[str] | None = None,
        mass_scale: int = MASS_SCALE,
        disable_terminal_residues_anywhere: bool = True,
    ):
        super().__init__(model=model)
        self.mass_scale = mass_scale
        self.disable_terminal_residues_anywhere = disable_terminal_residues_anywhere

        suppressed_residues = suppressed_residues or []

        # NOTE: Greedy search requires `residue_set` class in the model, update all methods accordingly.
        if not hasattr(model, "residue_set"):
            raise AttributeError(
                "The model is missing the required attribute: residue_set"
            )

        # TODO: Check if this can be replaced with model.get_residue_masses(mass_scale=10000)/10000
        # We would need to divide the scaled masses as we use floating point masses.
        # These residue masses are per amino acid and include special tokens, special tokens have a mass of 0.
        self.residue_masses = torch.zeros(
            (len(self.model.residue_set),), dtype=torch.float64
        )
        terminal_residues_idx: list[int] = []
        suppressed_residues_idx: list[int] = []
        for i, residue in enumerate(model.residue_set.vocab):
            if residue in self.model.residue_set.special_tokens:
                continue
            self.residue_masses[i] = self.model.residue_set.get_mass(residue)
            # If no residue is attached, assume it is a n-terminal residue
            if not residue[0].isalpha():
                terminal_residues_idx.append(i)

            # Check if residue is suppressed
            if residue in suppressed_residues:
                suppressed_residues_idx.append(i)
                suppressed_residues.remove(residue)

        if len(suppressed_residues) > 0:
            raise ValueError(
                f"Suppressed residues not found in vocabulary: {suppressed_residues}"
            )

        self.terminal_residue_indices = torch.tensor(
            terminal_residues_idx, dtype=torch.long
        )
        self.suppressed_residue_indices = torch.tensor(
            suppressed_residues_idx, dtype=torch.long
        )

        self.vocab_size = len(self.model.residue_set)

    def decode(  # type:ignore
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        max_length: int,
        mass_tolerance: float = 5e-5,
        max_isotope: int = 1,
        min_log_prob: float = -float("inf"),
        **kwargs,
    ) -> list[ScoredSequence] | list[list[ScoredSequence]]:
        """Decode predicted residue sequence for a batch of spectra using beam search.

        Args:
            spectra (torch.FloatTensor):
                The spectra to be sequenced.

            precursors (torch.FloatTensor[batch size, 3]):
                The precursor mass, charge and mass-to-charge ratio.

            beam_size (int):
                The maximum size of the beam.
                Ignored in beam search.

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
                Ignored in greedy search.

            min_log_prob (float):
                Minimum log probability to stop decoding early. If a sequence
                probability is less than this value it is marked as complete.
                Defailts to -inf.

        Returns:
            list[list[str]]:
                The predicted sequence as a list of residue tokens.
                This method will return an empty list for each
                spectrum in the batch where
                decoding fails i.e. no sequence that fits the precursor mass
                to within a tolerance is found.
        """
        # Greedy search with precursor mass termination condition
        batch_size = spectra.shape[0]
        device = spectra.device

        # Masses of all residues in vocabulary, 0 for special tokens
        self.residue_masses = self.residue_masses.to(
            spectra.device
        )  # float32 (vocab_size, )

        # ppm equivalent of mass tolerance
        delta_ppm_tol = mass_tolerance * 10**6  # float (1, )

        # Residue masses expanded (repeated) across batch_size
        # This is used to quickly compute all possible remaining masses per vocab entry
        residue_mass_delta = self.residue_masses.expand(
            batch_size, self.residue_masses.shape[0]
        )  # float32 (batch_size, vocab_size)

        with torch.no_grad():
            # 1. Compute spectrum encoding and masks
            #    Encoder is only run once.
            (spectrum_encoding, spectrum_mask), _ = self.model.init(spectra, precursors)

            # 2. Initialise beams and other variables
            # The sequences decoded so far, grows on index 1 for every decoding pass. sequence_length is variable!
            sequences = torch.zeros(
                (batch_size, 0), device=device, dtype=torch.long
            )  # long (batch_size, sequence_length)

            # Log probabiliies of the sequences decoded so far, token probabilities are added at each step.
            log_probabilities = torch.zeros(
                (batch_size, 1), device=device, dtype=torch.float32
            )  # long (batch_size, 1)

            # Keeps track of which beams are completed, this allows the model to skip these
            complete_beams = torch.zeros(
                (batch_size), device=device, dtype=bool
            )  # bool (batch_size, )

            # Keeps track of which stopped early or terminated with a bad stop condition. These predictions will be deleted.
            # bad_stop_condition = torch.zeros((batch_size), device=device, dtype=bool) # bool (batch_size, )

            # Extract precursor mass from `precursors`
            precursor_mass = precursors[
                :, PrecursorDimension.PRECURSOR_MASS.value
            ]  # float32 (batch_size, )

            # Target mass delta, remaining mass x must be within `target > x > -target`. This target can shift with isotopes.
            # Mass targets = error_ppm * m_prec * 1e-6
            mass_target_delta = (
                delta_ppm_tol * precursor_mass.to(torch.float64) * 1e-6
            )  # float64 (batch_size, )

            # This keeps track of the remaining mass budget for the currently decoding sequence, starts at the precursor - H2O
            remaining_mass = (
                precursor_mass.to(torch.float64) - H2O_MASS
            )  # float64 (batch_size, )

            # TODO: only check when close to precursor mass? Might not be worth the overhead.
            # Idea is if remaining < check_zone, we do the valid mass and complete checks.
            # check_zone = self.residue_masses.max().expand(batch_size) + mass_target_delta

            # Store token probabilities
            token_log_probabilities = []  # list[float(batch_size)]

            # Start decoding
            for _ in range(max_length):
                # If all beams are complete, we can stop early.
                if complete_beams.all():
                    break

                # We only run the model on incomplete beams, note: we have to expand to the full batch size afterwards.
                minibatch = (
                    x[~complete_beams]
                    for x in (sequences, precursors, spectrum_encoding, spectrum_mask)
                )
                # Keep track of how large the minibatch is
                sub_batch_size = (~complete_beams).sum()

                # Step 3: score the next tokens
                # NOTE: SOS token is appended automatically in `score_candidates`. We do not have to add it.
                next_token_probabilities = self.model.score_candidates(*minibatch)

                # Step 4: Filter probabilities
                # If remaining mass is within tolerance, we force an EOS token.
                # All tokens that would set the remaining mass below the minimum cutoff `-mass_target_delta` including isotopes is set to -inf

                # Step 4.1: Check if remaining mass is within tolerance:
                # To keep it efficient we compute some of the indexed variables first:
                remaining_mass_incomplete = remaining_mass[
                    ~complete_beams
                ]  # float64 (sub_batch_size, )
                mass_target_incomplete = mass_target_delta[
                    ~complete_beams
                ]  # float64 (sub_batch_size, )

                # remaining_meets_precursor = (remaining_mass[~complete_beams] < mass_target_delta[~complete_beams])
                remaining_meets_precursor = torch.zeros(
                    (sub_batch_size,), device=device, dtype=bool
                )  # bool (sub_batch_size, )
                # This loop checks if mass is within tolerance for 0 to max_isotopes (inclusive)
                for j in range(0, max_isotope + 1, 1):
                    # TODO: Use vectorized approach for this
                    isotope = CARBON_MASS_DELTA * j  # float
                    remaining_lesser_isotope = (
                        remaining_mass_incomplete - isotope < mass_target_incomplete
                    )  # bool (sub_batch_size, )
                    remaining_greater_isotope = (
                        remaining_mass_incomplete - isotope > -mass_target_incomplete
                    )  # bool (sub_batch_size, )

                    # remaining mass is within the target tolerance
                    remaining_within_range = (
                        remaining_lesser_isotope & remaining_greater_isotope
                    )  # bool (sub_batch_size, )
                    remaining_meets_precursor = (
                        remaining_meets_precursor | remaining_within_range
                    )  # bool (sub_batch_size, )
                    if remaining_within_range.any() and j > 0:
                        # If we did hit an isotope, correct the remaining mass accordingly
                        # TODO check this
                        remaining_mass_incomplete[remaining_within_range] = (
                            remaining_mass_incomplete[remaining_within_range] - isotope
                        )

                # Step 4.2: Check which residues are valid
                # Expand incomplete remaining mass across vocabulary size
                remaining_mass_incomplete_expanded = remaining_mass_incomplete[
                    :, None
                ].expand(
                    sub_batch_size, self.vocab_size
                )  # float64 (sub_batch_size, vocab_size)
                mass_target_incomplete_expanded = mass_target_incomplete[
                    :, None
                ].expand(
                    sub_batch_size, self.vocab_size
                )  # float64 (sub_batch_size, vocab_size)
                residue_mass_delta_incomplete = residue_mass_delta[
                    ~complete_beams
                ]  # float64 (sub_batch_size, vocab_size)

                valid_mass = (
                    remaining_mass_incomplete_expanded - residue_mass_delta_incomplete
                    > -mass_target_incomplete_expanded
                )  # bool (sub_batch_size, vocab_size)
                # Check all isotopes for valid masses
                for j in range(1, max_isotope + 1, 1):
                    isotope = CARBON_MASS_DELTA * j  # float
                    mass_lesser_isotope = (
                        remaining_mass_incomplete_expanded
                        - residue_mass_delta_incomplete
                        < mass_target_incomplete_expanded + isotope
                    )  # bool (sub_batch_size, vocab_size)
                    mass_greater_isotope = (
                        remaining_mass_incomplete_expanded
                        - residue_mass_delta_incomplete
                        > -mass_target_incomplete_expanded + isotope
                    )  # bool (sub_batch_size, vocab_size)
                    valid_mass = valid_mass | (
                        mass_lesser_isotope & mass_greater_isotope
                    )  # bool (sub_batch_size, vocab_size)

                # Filtered probabilities:
                next_token_probabilities_filtered = (
                    next_token_probabilities.clone()
                )  # float32 (sub_batch_size, vocab_size)
                # If mass is invalid, set log_prob to -inf
                next_token_probabilities_filtered[~valid_mass] = -float("inf")
                next_token_probabilities_filtered[
                    :, self.model.residue_set.EOS_INDEX
                ] = -float("inf")
                # Allow the model to predict PAD when all residues are -inf
                # next_token_probabilities_filtered[
                #     :, self.model.residue_set.PAD_INDEX
                # ] = -float("inf")
                next_token_probabilities_filtered[
                    :, self.model.residue_set.SOS_INDEX
                ] = -float("inf")
                next_token_probabilities_filtered[
                    :, self.suppressed_residue_indices
                ] = -float("inf")
                # Set probability of n-terminal modifications to -inf when i > 0
                if self.disable_terminal_residues_anywhere:
                    # Check if adding terminal residues would result in a complete sequence
                    # First generate remaining mass matrix with isotopes
                    remaining_mass_incomplete_isotope = remaining_mass_incomplete[
                        :, None
                    ].expand(sub_batch_size, max_isotope + 1) - CARBON_MASS_DELTA * (
                        torch.arange(max_isotope + 1, device=device)
                    )
                    # Expand with terminal residues and subtract
                    remaining_mass_incomplete_isotope_delta = (
                        remaining_mass_incomplete_isotope[:, :, None].expand(
                            sub_batch_size,
                            max_isotope + 1,
                            self.terminal_residue_indices.shape[0],
                        )
                        - self.residue_masses[self.terminal_residue_indices]
                    )

                    # If within target delta, allow these residues to be predicted, otherwise set probability to -inf
                    allow_terminal = (
                        remaining_mass_incomplete_isotope_delta.abs()
                        < mass_target_incomplete[:, None, None]
                    ).any(dim=1)
                    allow_terminal_full = torch.ones(
                        (sub_batch_size, self.vocab_size),
                        device=spectra.device,
                        dtype=bool,
                    )
                    allow_terminal_full[:, self.terminal_residue_indices] = (
                        allow_terminal
                    )

                    # Set to -inf
                    next_token_probabilities_filtered[~allow_terminal_full] = -float(
                        "inf"
                    )

                # Step 5: Select next token:
                next_token = next_token_probabilities_filtered.argmax(-1).unsqueeze(
                    1
                )  # long (sub_batch_size, 1)
                next_token[remaining_meets_precursor] = self.model.residue_set.EOS_INDEX

                # Update sequences
                next_token_full = torch.full(
                    (batch_size, 1),
                    fill_value=self.model.residue_set.PAD_INDEX,
                    device=spectra.device,
                    dtype=sequences.dtype,
                )  # long (batch_size, 1)
                next_token_full[~complete_beams] = next_token
                sequences = torch.concat(
                    [sequences, next_token_full], axis=1
                )  # long (batch_size, 1)

                # Expand and update masses
                next_masses = self.residue_masses[
                    next_token
                ].squeeze()  # float64 (sub_batch_size, )
                next_masses_full = torch.zeros(
                    (batch_size), device=spectra.device, dtype=remaining_mass.dtype
                )  # float64 (batch_size, )
                next_masses_full[~complete_beams] = next_masses
                remaining_mass = (
                    remaining_mass - next_masses_full
                )  # float64 (batch_size, )

                # Expand and update probabilities
                next_probabilities = torch.gather(
                    next_token_probabilities, 1, next_token
                )
                next_probabilities_full = torch.zeros(
                    (batch_size, 1),
                    device=spectra.device,
                    dtype=log_probabilities.dtype,
                )
                next_probabilities_full[~complete_beams] = next_probabilities
                log_probabilities = log_probabilities + next_probabilities_full
                token_log_probabilities.append(next_probabilities_full[:, 0])

                # Step 6: Terminate complete beams

                # Check if complete:
                # Early stopping if beam log probability below threshold
                beam_confidence_filter = (
                    log_probabilities[~complete_beams, 0] < min_log_prob
                )
                # Stop if beam is forced to output an EOS
                next_token_is_eos = next_token[:, 0] == self.model.residue_set.EOS_INDEX
                next_is_complete = next_token_is_eos | beam_confidence_filter

                # Check for a bad stop
                # bad_stop_condition = beam_confidence_filter
                # bad_stop_condition_full = torch.zeros((batch_size,), device=spectra.device, dtype=bad_stop_condition.dtype)
                # bad_stop_condition_full[~complete_beams] = bad_stop_condition
                # bad_stop_condition = bad_stop_condition | bad_stop_condition_full

                # Expand and update complete beams
                next_is_complete_full = torch.zeros(
                    (batch_size,), device=spectra.device, dtype=complete_beams.dtype
                )
                next_is_complete_full[~complete_beams] = next_is_complete
                complete_beams = complete_beams | next_is_complete_full

                # Repeat from step 3.

        all_log_probabilities = torch.stack(token_log_probabilities, axis=1)

        # Convert to sequences
        # Clean up predictions that are empty
        result = []
        for i in range(batch_size):
            # if bad_prediction[i]:
            #     result.append([])
            #     continue
            sequence = self.model.decode(sequences[i])
            result.append(
                ScoredSequence(
                    sequence=sequence,  # list[str] (sequence_length)
                    mass_error=remaining_mass[i].item(),  # float - TODO check this
                    sequence_log_probability=log_probabilities[i, 0].item(),  # float
                    token_log_probabilities=[
                        x.cpu().item()
                        for x in all_log_probabilities[i, : len(sequence)]
                    ][::-1],  # list[float] (sequence_length) excludes EOS
                )
            )

        return result
