from __future__ import annotations

from typing import Optional

import torch
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer
from torch.distributions import Categorical

from instanovo.constants import DIFFUSION_EVAL_STEPS
from instanovo.constants import DIFFUSION_START_STEP
from instanovo.diffusion.multinomial_diffusion import DiffusionLoss
from instanovo.diffusion.multinomial_diffusion import MultinomialDiffusion
from instanovo.types import Peptide
from instanovo.types import PrecursorFeatures
from instanovo.types import Spectrum
from instanovo.types import SpectrumMask


class DiffusionDecoder:
    """Class for decoding from a diffusion model by forward sampling."""

    def __init__(self, model: MultinomialDiffusion) -> None:
        self.model = model
        self.time_steps = model.time_steps
        self.residues = model.residues
        self.loss_function = DiffusionLoss(model=self.model)

    def decode(
        self,
        spectra: Float[Spectrum, " batch"],
        spectra_padding_mask: Bool[SpectrumMask, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        initial_sequence: Optional[Integer[Peptide, " batch"]] = None,
        start_step: int = DIFFUSION_START_STEP,
        eval_steps: tuple[int, ...] = DIFFUSION_EVAL_STEPS,
    ) -> tuple[list[list[str]], list[float]]:
        """Decoding predictions from a diffusion model by forward sampling.

        Args:
            spectra (torch.FloatTensor[batch_size, sequence_length, 2]):
                A batch of spectra to be decoded.

            spectra_padding_mask (torch.BoolTensor[batch_size, sequence_length]):
                Padding mask for a batch of variable length spectra.

            precursors (torch.FloatTensor[batch_size, 3]):
                Precursor mass, charge and m/z for a batch of spectra.

            initial_sequence (None | torch.LongTensor[batch_size, output_sequence_length], optional):
                An initial sequence for the model to refine. If no initial sequence is provided (the value
                is None), will sample a random sequence from a uniform unigram model. Defaults to None.

            start_step (int):
                The step at which to insert the initial sequence and start refinement. If
                `initial_sequence` is not provided, this will be set to `time_steps - 1`.

        Returns:
            tuple[list[list[str]], list[float]]:
                The decoded peptides and their log-probabilities for a batch of spectra.
        """
        device = spectra.device
        sequence_length = self.model.config.max_length
        batch_size, num_classes = spectra.size(0), len(self.model.residues)
        if initial_sequence is None:
            # Sample uniformly
            initial_distribution = Categorical(
                torch.ones(batch_size, sequence_length, num_classes) / num_classes
            )
            sample = initial_distribution.sample().to(device)
            start_step = self.time_steps - 1
        else:
            sample = initial_sequence

        peptide_mask = torch.zeros(batch_size, sequence_length).bool().to(device)

        log_probs = torch.zeros((batch_size, sequence_length)).to(device)
        # Sample through reverse process
        for t in range(start_step, -1, -1):
            times = (t * torch.ones((batch_size,))).long().to(spectra.device)
            distribution = Categorical(
                logits=self.model.reverse_distribution(
                    x_t=sample,
                    time=times,
                    spectra=spectra,
                    spectra_padding_mask=spectra_padding_mask,
                    precursors=precursors,
                    x_padding_mask=peptide_mask,
                )
            )
            sample = distribution.sample()

        # Calculate log-probabilities as average loss across `eval_steps`
        losses = []
        for t in eval_steps:
            times = (t * torch.ones((batch_size,))).long().to(spectra.device)
            losses.append(
                self.loss_function._compute_loss(
                    x_0=sample,
                    t=times,
                    spectra=spectra,
                    spectra_padding_mask=spectra_padding_mask,
                    precursors=precursors,
                    x_padding_mask=peptide_mask,
                )
            )
        log_probs = (-torch.stack(losses).mean(axis=0).cpu()).tolist()
        sequences = self._extract_predictions(sample)
        return sequences, log_probs

    def _extract_predictions(
        self, sample: Integer[Peptide, " batch"]
    ) -> list[list[str]]:
        output = []
        for sequence in sample:
            tokens = sequence.tolist()
            if self.residues.EOS_INDEX in sequence:
                peptide = tokens[: tokens.index(self.residues.EOS_INDEX)]
            else:
                peptide = tokens
            output.append(
                self.residues.decode(peptide, reverse=False)
            )  # we do not reverse peptide for diffusion
        return output
