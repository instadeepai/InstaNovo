from __future__ import annotations

import torch
from torch.distributions import Categorical

from instanovo.diffusion.multinomial_diffusion import MultinomialDiffusion


NUCLEON_MASS = 1.00335


class DiffusionDecoder:
    """Class for decoding from a diffusion model by forward sampling."""

    def __init__(self, model: MultinomialDiffusion) -> None:
        self.model = model
        self.time_steps = model.time_steps
        self.residues = model.residues

    def decode(
        self,
        spectra: torch.FloatTensor,
        spectra_padding_mask: torch.BoolTensor,
        precursors: torch.FloatTensor,
        initial_sequence: None | torch.LongTensor = None,
        start_step: int | None = None,
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

            start_step (int | None, optional):
                The step at which to insert the initial sequence and start refinement. If `initial_sequence` is
                provided, this must be as well. Defaults to None.

        Returns:
            tuple[list[list[str]], list[float]]:
                The decoded peptides and their log-probabilities for a batch of spectra.
        """
        # Sample uniformly
        device = spectra.device
        sequence_length = self.model.config.max_length
        batch_size, num_classes = spectra.size(0), len(self.model.residues)
        if initial_sequence is None:
            initial_distribution = Categorical(
                torch.ones(batch_size, sequence_length, num_classes) / num_classes
            )
            sample = initial_distribution.sample().to(device)
        else:
            sample = initial_sequence

        if start_step is not None:
            msg = "`start_step` can only be set if there is an initial sequence"
            assert initial_sequence is not None, msg

        peptide_mask = torch.zeros(batch_size, sequence_length).bool().to(device)

        start_step = self.time_steps - 1 if start_step is None else start_step

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
            log_probs += distribution.log_prob(sample)

        log_probs = log_probs.sum(-1).cpu().tolist()
        sequences = self._extract_predictions(sample)
        return sequences, log_probs

    def _extract_predictions(self, sample: torch.LongTensor) -> list[list[str]]:
        output = []
        for sequence in sample:
            tokens = sequence.tolist()
            if self.residues.eos_index in sequence:
                peptide = tokens[: tokens.index(self.residues.eos_index)]
            else:
                peptide = tokens
            output.append(self.residues.decode(peptide))
        return output
