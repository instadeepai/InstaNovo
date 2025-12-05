from __future__ import annotations

from typing import Any, Literal

import torch
from jaxtyping import Float

from instanovo.__init__ import console
from instanovo.constants import CARBON_MASS_DELTA, H2O_MASS, MASS_SCALE, PrecursorDimension
from instanovo.inference.interfaces import Decodable, Decoder
from instanovo.types import PrecursorFeatures, Spectrum
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class BeamSearchDecoder(Decoder):
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
        keep_invalid_mass_sequences: bool = True,
        float_dtype: torch.dtype = torch.float64,
    ):
        super().__init__(model=model)
        self.mass_scale = mass_scale
        self.disable_terminal_residues_anywhere = disable_terminal_residues_anywhere
        self.keep_invalid_mass_sequences = keep_invalid_mass_sequences
        self.float_dtype = float_dtype

        suppressed_residues = suppressed_residues or []

        # NOTE: Greedy search requires `residue_set` class in the model,
        # update all methods accordingly.
        if not hasattr(model, "residue_set"):
            raise AttributeError("The model is missing the required attribute: residue_set")

        # TODO: Check if this can be replaced with model.get_residue_masses(mass_scale=10000)/10000
        # We would need to divide the scaled masses as we use floating point masses.
        # These residue masses are per amino acid and include special tokens,
        # special tokens have a mass of 0.
        self.residue_masses = torch.zeros((len(self.model.residue_set),), dtype=self.float_dtype)
        terminal_residues_idx: list[int] = []
        suppressed_residues_idx: list[int] = []

        # residue_target_offsets supports negative masses (overshoot the remaining mass)
        # This fixes a bug where the residue prior to a negative mass residue is always invalid.
        residue_target_offsets: list[float] = [0.0]

        for i, residue in enumerate(model.residue_set.vocab):
            if residue in self.model.residue_set.special_tokens:
                continue
            self.residue_masses[i] = self.model.residue_set.get_mass(residue)
            # If no residue is attached, assume it is a n-terminal residue
            if not residue[0].isalpha():
                terminal_residues_idx.append(i)
            if self.residue_masses[i] < 0:
                residue_target_offsets.append(self.residue_masses[i])

            # Check if residue is suppressed
            if residue in suppressed_residues:
                suppressed_residues_idx.append(i)
                suppressed_residues.remove(residue)

        if len(suppressed_residues) > 0:
            logger.warning(f"Some suppressed residues not found in vocabulary: {suppressed_residues}")

        self.terminal_residue_indices = torch.tensor(terminal_residues_idx, dtype=torch.long)
        self.suppressed_residue_indices = torch.tensor(suppressed_residues_idx, dtype=torch.long)
        self.residue_target_offsets = torch.tensor(residue_target_offsets, dtype=self.float_dtype)

        self.vocab_size = len(self.model.residue_set)

    def decode(  # type:ignore
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        beam_size: int,
        max_length: int,
        mass_tolerance: float = 5e-5,
        max_isotope: int = 1,
        min_log_prob: float = -float("inf"),
        return_encoder_output: bool = False,
        encoder_output_reduction: Literal["mean", "max", "sum", "full"] = "mean",
        return_beam: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
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

            min_log_prob (float):
                Minimum log probability to stop decoding early. If a sequence
                probability is less than this value it is marked as complete.
                Defaults to -inf.

            return_beam (bool):
                Optionally return beam-search results. Ignored in greedy search.

        Returns:
            list[list[str]]:
                The predicted sequence as a list of residue tokens.
                This method will return an empty list for each
                spectrum in the batch where
                decoding fails i.e. no sequence that fits the precursor mass
                to within a tolerance is found.
        """
        # Beam search with precursor mass termination condition
        batch_size = spectra.shape[0]
        effective_batch_size = batch_size * beam_size
        device = spectra.device

        # Masses of all residues in vocabulary, 0 for special tokens
        self.residue_masses = self.residue_masses.to(spectra.device)  # float32 (vocab_size, )

        # ppm equivalent of mass tolerance
        delta_ppm_tol = mass_tolerance * 10**6  # float (1, )

        # Residue masses expanded (repeated) across batch_size
        # This is used to quickly compute all possible remaining masses per vocab entry
        residue_mass_delta = self.residue_masses.expand(effective_batch_size, self.residue_masses.shape[0])  # float32 (batch_size, vocab_size)

        # completed_items: list[list[ScoredSequence]] = [[] for _ in range(batch_size)]
        completed_beams: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]

        with torch.no_grad():
            # 1. Compute spectrum encoding and masks
            #    Encoder is only run once.
            (spectrum_encoding, spectrum_mask), _ = self.model.init(spectra, precursors)

            # EXPAND FOR BEAM SIZE
            spectrum_encoding_expanded = spectrum_encoding.repeat_interleave(beam_size, dim=0)
            spectrum_mask_expanded = spectrum_mask.repeat_interleave(beam_size, dim=0)

            # 2. Initialise beams and other variables
            # The sequences decoded so far, grows on index 1 for every decoding pass.
            # sequence_length is variable!
            sequences = torch.zeros((effective_batch_size, 0), device=device, dtype=torch.long)  # long (batch_size, sequence_length)

            # Log probabilities of the sequences decoded so far,
            # token probabilities are added at each step.
            log_probabilities = torch.zeros((effective_batch_size, 1), device=device, dtype=torch.float32)  # long (batch_size, 1)

            # Keeps track of which beams are completed, this allows the model to skip these
            complete_beams = torch.zeros((effective_batch_size), device=device, dtype=bool)  # bool (batch_size, )
            is_first_complete = torch.zeros((effective_batch_size), device=device, dtype=bool)  # bool (batch_size, )

            # Extract precursor mass from `precursors`
            precursors_expanded = precursors.repeat_interleave(beam_size, dim=0)

            precursor_mass = precursors_expanded[:, PrecursorDimension.PRECURSOR_MASS.value]  # float32 (batch_size, )

            # Target mass delta, remaining mass x must be within `target > x > -target`.
            # This target can shift with isotopes.
            # Mass targets = error_ppm * m_prec * 1e-6
            mass_target_delta = delta_ppm_tol * precursor_mass.to(self.float_dtype) * 1e-6  # float_dtype (batch_size, )

            # This keeps track of the remaining mass budget for the currently decoding sequence,
            # starts at the precursor - H2O
            remaining_mass = precursor_mass.to(self.float_dtype) - H2O_MASS  # float_dtype (batch_size, )

            # TODO: only check when close to precursor mass? Might not be worth the overhead.
            # Idea is if remaining < check_zone, we do the valid mass and complete checks.
            # check_zone = self.residue_masses.max().expand(batch_size) + mass_target_delta

            # Constant beam indices for retaining beams on failed decoding
            constant_beam_indices = torch.arange(beam_size, device=device)[None, :].repeat_interleave(batch_size, dim=0)

            # Store token probabilities
            token_log_probabilities: dict[str, list[float]] = {}  # dict[str, list[float]]

            # Start decoding
            for i in range(max_length):
                # If all beams are complete, we can stop early.
                if complete_beams.all():
                    break

                # Step 3: score the next tokens
                # NOTE: SOS token is appended automatically in `score_candidates`.
                # We do not have to add it.
                batch = (sequences, precursors_expanded, spectrum_encoding_expanded, spectrum_mask_expanded)
                next_token_probabilities = self.model.score_candidates(*batch)

                # Step 4: Filter probabilities
                # If remaining mass is within tolerance, we force an EOS token.
                # All tokens that would set the remaining mass below the minimum
                # cutoff `-mass_target_delta` including isotopes is set to -inf

                # Step 4.1: Check if remaining mass is within tolerance:
                # To keep it efficient we compute some of the indexed variables first:
                remaining_meets_precursor = torch.zeros((effective_batch_size,), device=device, dtype=bool)  # bool (sub_batch_size, )
                # This loop checks if mass is within tolerance for 0 to max_isotopes (inclusive)
                for j in range(0, max_isotope + 1, 1):
                    # TODO: Use vectorized approach for this
                    isotope = CARBON_MASS_DELTA * j  # float
                    remaining_lesser_isotope = remaining_mass - isotope < mass_target_delta  # bool (sub_batch_size, )
                    remaining_greater_isotope = remaining_mass - isotope > -mass_target_delta  # bool (sub_batch_size, )

                    # remaining mass is within the target tolerance
                    remaining_within_range = remaining_lesser_isotope & remaining_greater_isotope  # bool (sub_batch_size, )
                    remaining_meets_precursor = remaining_meets_precursor | remaining_within_range  # bool (sub_batch_size, )
                    if remaining_within_range.any() and j > 0:
                        # If we did hit an isotope, correct the remaining mass accordingly
                        # TODO check this
                        remaining_mass[remaining_within_range] = remaining_mass[remaining_within_range] - isotope

                # Step 4.2: Check which residues are valid
                # Expand incomplete remaining mass across vocabulary size
                remaining_mass_expanded = remaining_mass[:, None].expand(
                    effective_batch_size, self.vocab_size
                )  # float64 (effective_batch_size, vocab_size)
                mass_target_expanded = mass_target_delta[:, None].expand(
                    effective_batch_size, self.vocab_size
                )  # float64 (effective_batch_size, vocab_size)

                valid_mass = remaining_mass_expanded - residue_mass_delta > -mass_target_expanded  # bool (effective_batch_size, vocab_size)
                # Check all isotopes for valid masses
                for mass_offset in self.residue_target_offsets:
                    for j in range(0, max_isotope + 1, 1):
                        isotope = CARBON_MASS_DELTA * j  # float
                        mass_lesser_isotope = (
                            remaining_mass_expanded - residue_mass_delta < mass_target_expanded + isotope + mass_offset
                        )  # bool (effective_batch_size, vocab_size)
                        mass_greater_isotope = (
                            remaining_mass_expanded - residue_mass_delta > -mass_target_expanded + isotope + mass_offset
                        )  # bool (effective_batch_size, vocab_size)
                        valid_mass = valid_mass | (mass_lesser_isotope & mass_greater_isotope)  # bool (effective_batch_size, vocab_size)

                # Filtered probabilities:
                next_token_probabilities_filtered = next_token_probabilities.clone()  # float32 (effective_batch_size, vocab_size)
                # If mass is invalid, set log_prob to -inf
                next_token_probabilities_filtered[~valid_mass] = -float("inf")

                next_token_probabilities_filtered[:, self.model.residue_set.EOS_INDEX] = -float("inf")
                # Allow the model to predict PAD when all residues are -inf
                next_token_probabilities_filtered[:, self.model.residue_set.PAD_INDEX] = -float("inf")
                next_token_probabilities_filtered[:, self.model.residue_set.SOS_INDEX] = -float("inf")
                next_token_probabilities_filtered[:, self.suppressed_residue_indices] = -float("inf")
                # Set probability of n-terminal modifications to -inf when i > 0
                if self.disable_terminal_residues_anywhere:
                    # Check if adding terminal residues would result in a complete sequence
                    # First generate remaining mass matrix with isotopes
                    remaining_mass_isotope = remaining_mass[:, None].expand(effective_batch_size, max_isotope + 1) - CARBON_MASS_DELTA * (
                        torch.arange(max_isotope + 1, device=device)
                    )
                    # Expand with terminal residues and subtract
                    remaining_mass_isotope_delta = (
                        remaining_mass_isotope[:, :, None].expand(
                            effective_batch_size,
                            max_isotope + 1,
                            self.terminal_residue_indices.shape[0],
                        )
                        - self.residue_masses[self.terminal_residue_indices]
                    )

                    # If within target delta, allow these residues to be predicted,
                    # otherwise set probability to -inf
                    allow_terminal = (remaining_mass_isotope_delta.abs() < mass_target_delta[:, None, None]).any(dim=1)
                    allow_terminal_full = torch.ones(
                        (effective_batch_size, self.vocab_size),
                        device=spectra.device,
                        dtype=bool,
                    )
                    allow_terminal_full[:, self.terminal_residue_indices] = allow_terminal

                    # Set to -inf
                    next_token_probabilities_filtered[~allow_terminal_full] = -float("inf")

                # Set to -inf for newly completed beams, only allow EOS
                # NEW WAY TO FORCE EOS
                # for beam_idx in remaining_meets_precursor:
                next_beam_no_predictions = next_token_probabilities_filtered.isinf().all(-1)

                if is_first_complete.any():
                    completed_idxs = is_first_complete.nonzero().squeeze(-1)
                    for beam_idx in completed_idxs:
                        sequence_probability = (
                            log_probabilities[beam_idx]  # + next_token_probabilities[beam_idx,
                        )
                        sequence_str = str((beam_idx // beam_size).item()) + "-" + ".".join([str(x) for x in sequences[beam_idx].cpu().tolist()])
                        sequence = self.model.decode(sequences[beam_idx])
                        seen_completed_sequences = {"".join(x["predictions"]) for x in completed_beams[beam_idx // beam_size]}
                        if "".join(sequence) in seen_completed_sequences:
                            continue
                        completed_beams[beam_idx // beam_size].append(
                            {
                                "predictions": sequence,
                                "mass_error": remaining_mass[beam_idx].item(),
                                "meets_precursor": remaining_meets_precursor[beam_idx].item(),
                                "prediction_log_probability": sequence_probability.item(),
                                "prediction_token_log_probabilities": token_log_probabilities[sequence_str][: len(sequence)][::-1],
                            }
                        )

                # print(sequences[:5])

                # For beams that already meet precursor, -inf them and force an EOS
                next_token_probabilities_filtered[remaining_meets_precursor, :] = -float("inf")
                if self.keep_invalid_mass_sequences:
                    # Allow EOS on beams that dont fit precursor
                    allow_eos = (remaining_meets_precursor | next_beam_no_predictions) & ~complete_beams
                else:
                    allow_eos = (remaining_meets_precursor) & ~complete_beams
                next_eos_probs = next_token_probabilities[allow_eos, self.model.residue_set.EOS_INDEX]
                next_token_probabilities_filtered[allow_eos, self.model.residue_set.EOS_INDEX] = next_eos_probs

                # Step 5: Select next token:
                log_probabilities_expanded = log_probabilities.repeat_interleave(self.vocab_size, dim=1)
                log_probabilities_expanded = log_probabilities_expanded + next_token_probabilities_filtered

                log_probabilities_beams = log_probabilities_expanded.view(-1, beam_size, self.vocab_size)
                if i == 0 and beam_size > 1:
                    # Nullify all beams except the first one
                    log_probabilities_beams[:, 1:] = -float("inf")

                log_probabilities_beams = log_probabilities_beams.view(-1, beam_size * self.vocab_size)

                topk_values, topk_indices = log_probabilities_beams.topk(beam_size, dim=-1)
                topk_is_inf = topk_values.isinf()

                beam_indices = topk_indices // self.vocab_size
                # Retain beams on failed decoding (when all beams are -inf)
                beam_indices[topk_is_inf] = constant_beam_indices[topk_is_inf]
                beam_indices_full = (beam_indices + torch.arange(batch_size, device=beam_indices.device)[:, None] * beam_size).view(-1)

                next_token = topk_indices % self.vocab_size
                next_token[topk_is_inf] = self.model.residue_set.PAD_INDEX
                next_token = next_token.view(-1, 1)  # long (sub_batch_size, 1)\

                # Update beams by indices
                sequences = sequences[beam_indices_full]
                log_probabilities = log_probabilities[beam_indices_full]
                next_token_probabilities = next_token_probabilities[beam_indices_full]
                remaining_mass = remaining_mass[beam_indices_full]
                complete_beams = complete_beams[beam_indices_full]

                sequences = torch.concat([sequences, next_token], axis=1)  # long (batch_size, 1)

                # Expand and update masses
                next_masses = self.residue_masses[next_token].squeeze()  # float64 (sub_batch_size, )
                remaining_mass = remaining_mass - next_masses  # float64 (batch_size, )

                # Expand and update probabilities
                next_token_probabilities[:, self.model.residue_set.PAD_INDEX] = 0
                next_probabilities = torch.gather(next_token_probabilities, 1, next_token)
                next_probabilities[complete_beams] = 0
                log_probabilities = log_probabilities + next_probabilities

                for batch_index in range(effective_batch_size):
                    # Create unique ID for the sequence
                    # Store beam token probabilities in a hash table
                    spectrum_index = batch_index // beam_size
                    sequence = [str(x) for x in sequences[batch_index].cpu().tolist()]
                    sequence_str = str(spectrum_index) + "-" + ".".join(sequence)
                    sequence_prev_str = str(spectrum_index) + "-" + ".".join(sequence[:-1])

                    if sequence_prev_str in token_log_probabilities:
                        previous_probabilities = list(token_log_probabilities[sequence_prev_str])
                    else:
                        previous_probabilities = []

                    previous_probabilities.append(next_probabilities[batch_index, 0].float().item())

                    token_log_probabilities[sequence_str] = previous_probabilities

                # Step 6: Terminate complete beams

                # Check if complete:
                # Early stopping if beam log probability below threshold
                beam_confidence_filter = log_probabilities[:, 0] < min_log_prob
                # Stop if beam is forced to output an EOS
                next_token_is_eos = next_token[:, 0] == self.model.residue_set.EOS_INDEX
                next_token_is_pad = next_token[:, 0] == self.model.residue_set.PAD_INDEX
                next_is_complete = next_token_is_eos | beam_confidence_filter  # | next_token_is_pad

                complete_beams = complete_beams | next_is_complete
                is_first_complete = next_is_complete

                if next_token_is_pad.all():
                    break

                # Repeat from step 3.

        # Check if any beams are complete at the end of the loop
        if is_first_complete.any():
            completed_idxs = is_first_complete.nonzero().squeeze(-1)
            for beam_idx in completed_idxs:
                sequence_probability = (
                    log_probabilities[beam_idx]  # + next_token_probabilities[beam_idx,
                    # self.model.residue_set.EOS_INDEX]
                )
                sequence_str = str((beam_idx // beam_size).item()) + "-" + ".".join([str(x) for x in sequences[beam_idx].cpu().tolist()])
                sequence = self.model.decode(sequences[beam_idx])
                seen_completed_sequences = {"".join(x["predictions"]) for x in completed_beams[beam_idx // beam_size]}
                if "".join(sequence) in seen_completed_sequences:
                    continue
                completed_beams[beam_idx // beam_size].append(
                    {
                        "predictions": sequence,
                        "mass_error": remaining_mass[beam_idx].item(),
                        "meets_precursor": remaining_meets_precursor[beam_idx].item(),
                        "prediction_log_probability": sequence_probability.item(),
                        "prediction_token_log_probabilities": token_log_probabilities[sequence_str][: len(sequence)][::-1],
                    }
                )

        # This loop forcefully adds all beams at the end, whether they are complete or not
        if self.keep_invalid_mass_sequences:
            for batch_idx in range(effective_batch_size):
                sequence_str = str(batch_idx // beam_size) + "-" + ".".join([str(x) for x in sequences[batch_idx].cpu().tolist()])
                sequence = self.model.decode(sequences[batch_idx])
                seen_completed_sequences = {"".join(x["predictions"]) for x in completed_beams[batch_idx // beam_size]}
                if "".join(sequence) in seen_completed_sequences:
                    # print(f"Skipping {sequence_str} because it is added")
                    continue
                completed_beams[batch_idx // beam_size].append(
                    {
                        "predictions": sequence,
                        "mass_error": remaining_mass[batch_idx].item(),
                        "meets_precursor": remaining_meets_precursor[batch_idx].item(),
                        "prediction_log_probability": log_probabilities[batch_idx, 0].item(),
                        "prediction_token_log_probabilities": token_log_probabilities[sequence_str][: len(sequence)][::-1],
                    }
                )

        # Get top n beams per batch
        # Filtered on meets_precursor and log_probability
        top_completed_beams = self._get_top_n_beams(completed_beams, beam_size)

        # Prepare result dictionary
        result: dict[str, Any] = {
            "predictions": [],
            # "mass_error": [],
            "prediction_log_probability": [],
            "prediction_token_log_probabilities": [],
        }
        if return_beam:
            for i in range(beam_size):
                result[f"predictions_beam_{i}"] = []
                # result[f"mass_error_beam_{i}"] = []
                result[f"predictions_log_probability_beam_{i}"] = []
                result[f"predictions_token_log_probabilities_beam_{i}"] = []

        for batch_idx in range(batch_size):
            if return_beam:
                for beam_idx in range(beam_size):
                    result[f"predictions_beam_{beam_idx}"].append("".join(top_completed_beams[batch_idx][beam_idx]["predictions"]))
                    # result[f"mass_error_beam_{beam_idx}"].append(top_completed_beams[batch_idx][beam_idx]["mass_error"])
                    result[f"predictions_log_probability_beam_{beam_idx}"].append(
                        top_completed_beams[batch_idx][beam_idx]["prediction_log_probability"]
                    )
                    result[f"predictions_token_log_probabilities_beam_{beam_idx}"].append(
                        top_completed_beams[batch_idx][beam_idx]["prediction_token_log_probabilities"]
                    )

            # Save best beam as main result
            result["predictions"].append(top_completed_beams[batch_idx][0]["predictions"])
            # result[f"mass_error"].append(top_completed_beams[batch_idx][0]["mass_error"])
            result["prediction_log_probability"].append(top_completed_beams[batch_idx][0]["prediction_log_probability"])
            result["prediction_token_log_probabilities"].append(top_completed_beams[batch_idx][0]["prediction_token_log_probabilities"])

        # Optionally include encoder output
        if return_encoder_output:
            # Reduce along sequence length dimension
            encoder_output = spectrum_encoding.float().cpu()
            encoder_mask = (1 - spectrum_mask.float()).cpu()
            encoder_output = encoder_output * encoder_mask.unsqueeze(-1)
            if encoder_output_reduction == "mean":
                count = encoder_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
                encoder_output = encoder_output.sum(dim=1) / count
            elif encoder_output_reduction == "max":
                encoder_output[encoder_output == 0] = -float("inf")
                encoder_output = encoder_output.max(dim=1)[0]
            elif encoder_output_reduction == "sum":
                encoder_output = encoder_output.sum(dim=1)
            elif encoder_output_reduction == "full":
                raise NotImplementedError("Full encoder output reduction is not yet implemented")
            else:
                raise ValueError(f"Invalid encoder output reduction: {encoder_output_reduction}")
            result["encoder_output"] = list(encoder_output.numpy())

        return result

    def _get_top_n_beams(self, completed_beams: list[list[dict[str, Any]]], beam_size: int) -> list[list[dict[str, Any]]]:
        """Get the top n beams from the completed beams.

        Args:
            completed_beams: The completed beams to get the top n beams from.
                Each beam is a dictionary with the following keys:
                - predictions: The predictions of the beam.
                - mass_error: The mass error of the beam.
                - meets_precursor: Whether the beam meets the precursor mass.
                - prediction_log_probability: The log probability of the beam.
                - prediction_token_log_probabilities: The log probabilities of the tokens in the beam.
            beam_size: The number of beams to keep per batch.

        Returns:
            A list of lists, each containing the top n beams for a batch.
        """
        default_beam = {
            "predictions": [],
            "mass_error": -float("inf"),
            "prediction_log_probability": -float("inf"),
            "prediction_token_log_probabilities": [],
        }

        top_beams_per_row = []
        for beams in completed_beams:
            # Sort first by error within tolerance, then by log_prob descending
            beams.sort(key=lambda x: (x["meets_precursor"], x["prediction_log_probability"]), reverse=True)

            # Keep top N beams
            top_beams = beams[:beam_size]

            # Pad with default beam if fewer than N
            while len(top_beams) < beam_size:
                top_beams.append(default_beam.copy())

            top_beams_per_row.append(top_beams)

        return top_beams_per_row
