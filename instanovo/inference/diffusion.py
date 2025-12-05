from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from jaxtyping import Bool, Float, Integer
from torch.distributions import Categorical

from instanovo.constants import CARBON_MASS_DELTA, DIFFUSION_EVAL_STEPS, DIFFUSION_START_STEP, H2O_MASS, PrecursorDimension
from instanovo.diffusion.multinomial_diffusion import DiffusionLoss, InstaNovoPlus
from instanovo.inference.interfaces import Decoder
from instanovo.types import Peptide, PrecursorFeatures, Spectrum, SpectrumMask


class DiffusionDecoder(Decoder):
    """Class for decoding from a diffusion model by forward sampling."""

    def __init__(self, model: InstaNovoPlus) -> None:
        super().__init__(model=model)
        # Override base class type annotation - this is actually InstaNovoPlus, not just Decodable
        self.model: InstaNovoPlus = model

        self.time_steps = self.model.time_steps
        self.residue_set = self.model.residue_set
        self.loss_function = DiffusionLoss(model=self.model)

    def decode(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        spectra_padding_mask: Bool[SpectrumMask, " batch"],
        initial_sequence: Optional[Integer[Peptide, " batch"]] = None,
        start_step: int = DIFFUSION_START_STEP,
        eval_steps: tuple[int, ...] = DIFFUSION_EVAL_STEPS,
        beam_size: int = 1,
        mass_tolerance: float = 5e-5,
        max_isotope: int = 1,
        return_encoder_output: bool = False,
        encoder_output_reduction: Literal["mean", "max", "sum", "full"] = "mean",
        return_beam: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Decoding predictions from a diffusion model by forward sampling.

        Args:
            spectra:
                A batch of spectra to be decoded.

            spectra_padding_mask:
                Padding mask for a batch of variable length spectra.

            precursors:
                Precursor mass, charge and m/z for a batch of spectra.

            initial_sequence:
                An initial sequence for the model to refine. If no initial sequence is
                provided (the value is None), will sample a random sequence from a uniform unigram
                model. Defaults to None.

            start_step:
                The step at which to insert the initial sequence and start refinement. If
                `initial_sequence` is not provided, this will be set to `time_steps - 1`.

            eval_steps:
                The steps at which to evaluate the loss and compute the log-probabilities.

            return_encoder_output:
                Whether to return the encoder output.

            encoder_output_reduction:
                The reduction to apply to the encoder output.
                Valid values are "mean", "max", "sum", "full".
                Defaults to "mean".

        Returns:
            dict[str, Any]:
                The decoded peptides and their log-probabilities for a batch of spectra.
                Required keys:
                    - "predictions": list[list[str]]
                    - "prediction_log_probability": list[float]
                    - "prediction_token_log_probabilities": list[list[float]]
                    - "encoder_output": list[float] (optional)
                Example additional keys:
                    - "prediction_beam_0": list[str]
        """
        device = spectra.device
        sequence_length = self.model.config.max_length
        batch_size, num_classes = spectra.size(0), len(self.residue_set)
        effective_batch_size = batch_size * beam_size

        spectra_expanded = spectra.repeat_interleave(beam_size, dim=0)
        spectra_padding_mask_expanded = spectra_padding_mask.repeat_interleave(beam_size, dim=0)
        precursors_expanded = precursors.repeat_interleave(beam_size, dim=0)

        if initial_sequence is None:
            # Sample uniformly
            initial_distribution = Categorical(torch.ones(effective_batch_size, sequence_length, num_classes) / num_classes)
            sample = initial_distribution.sample().to(device)
            start_step = self.time_steps - 1
        else:
            sample = initial_sequence.repeat_interleave(beam_size, dim=0)

        peptide_mask = torch.zeros(effective_batch_size, sequence_length).bool().to(device)

        log_probs = torch.zeros((effective_batch_size, sequence_length)).to(device)
        # Sample through reverse process
        for t in range(start_step, -1, -1):
            times = (t * torch.ones((effective_batch_size,))).long().to(spectra.device)
            distribution = Categorical(
                logits=self.model.reverse_distribution(
                    x_t=sample,
                    time=times,
                    spectra=spectra_expanded,
                    spectra_padding_mask=spectra_padding_mask_expanded,
                    precursors=precursors_expanded,
                    x_padding_mask=peptide_mask,
                )
            )
            sample = distribution.sample()

        # Calculate log-probabilities as average loss across `eval_steps`
        losses = []
        for t in eval_steps:
            times = (t * torch.ones((effective_batch_size,))).long().to(spectra.device)
            losses.append(
                self.loss_function._compute_loss(
                    x_0=sample,
                    t=times,
                    spectra=spectra_expanded,
                    spectra_padding_mask=spectra_padding_mask_expanded,
                    precursors=precursors_expanded,
                    x_padding_mask=peptide_mask,
                )
            )
        log_probs = (-torch.stack(losses).mean(axis=0).cpu()).tolist()
        sequences = self._extract_predictions(sample)

        # convert to batch_size, beam_size format
        completed_beams: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]

        for idx in range(effective_batch_size):
            batch_idx = idx // beam_size

            sequence = sequences[idx]
            sequence_mass = sum([self.model.residue_set.get_mass(residue) for residue in sequence])
            # Check precursor matching
            precursor_mass = precursors[batch_idx, PrecursorDimension.PRECURSOR_MASS.value]
            remaining_mass = precursor_mass - sequence_mass - H2O_MASS
            mass_target_delta = mass_tolerance * precursor_mass

            # Check if mass within range, including isotopes
            remaining_meets_precursor = False
            for j in range(0, max_isotope + 1, 1):
                isotope = CARBON_MASS_DELTA * j
                remaining_lesser_isotope = remaining_mass - isotope < mass_target_delta
                remaining_greater_isotope = remaining_mass - isotope > -mass_target_delta
                remaining_meets_precursor = remaining_meets_precursor | (remaining_lesser_isotope & remaining_greater_isotope)

            completed_beams[batch_idx].append(
                {
                    "predictions": sequences[idx],
                    "meets_precursor": remaining_meets_precursor,
                    "prediction_log_probability": log_probs[idx],
                    "prediction_token_log_probabilities": log_probs[idx],
                }
            )

        # Sort beams by meets_precursor and prediction_log_probability
        # Index 0 is the best beam
        for batch_idx in range(batch_size):
            completed_beams[batch_idx].sort(key=lambda x: (x["meets_precursor"], x["prediction_log_probability"]), reverse=True)

        result: dict[str, Any] = {
            "predictions": [],
            "meets_precursor": [],
            "prediction_log_probability": [],
            "prediction_token_log_probabilities": [None] * batch_size,
        }
        if return_beam:
            for i in range(beam_size):
                result[f"predictions_beam_{i}"] = []
                result[f"prediction_log_probability_beam_{i}"] = []
                result[f"prediction_token_log_probabilities_beam_{i}"] = ([None] * batch_size,)

        for batch_idx in range(batch_size):
            if return_beam:
                for beam_idx in range(beam_size):
                    result[f"predictions_beam_{beam_idx}"].append(completed_beams[batch_idx][beam_idx]["predictions"])
                    result[f"prediction_log_probability_beam_{beam_idx}"].append(completed_beams[batch_idx][beam_idx]["prediction_log_probability"])

            result["predictions"].append(completed_beams[batch_idx][0]["predictions"])
            result["meets_precursor"].append(completed_beams[batch_idx][0]["meets_precursor"])
            result["prediction_log_probability"].append(completed_beams[batch_idx][0]["prediction_log_probability"])

        if return_encoder_output:
            # Extract encoder output from cache
            encoder_output = self.model.transition_model.cache_cond_emb
            encoder_mask = self.model.transition_model.cache_cond_padding_mask

            if encoder_output is None or encoder_mask is None:
                raise ValueError("Could not extract encoder output from model to return as encoder output.")

            encoder_output = encoder_output.float().cpu()
            encoder_mask = (1 - encoder_mask.float()).cpu()
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

    def _extract_predictions(self, sample: Integer[Peptide, " batch"]) -> list[list[str]]:
        output = []
        for sequence in sample:
            tokens = sequence.tolist()
            if self.residue_set.EOS_INDEX in sequence:
                peptide = tokens[: tokens.index(self.residue_set.EOS_INDEX)]
            else:
                peptide = tokens
            output.append(self.residue_set.decode(peptide, reverse=False))  # we do not reverse peptide for diffusion
        return output
