from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from instanovo.transformer.model import InstaNovo


class BaseDecoder:
    """Base model decoder."""

    def __init__(
        self,
        model: nn.Module,
        i2s: dict[int, str],
        max_length: int = 30,
        eos_id: int = 2,
        bos_id: int = 1,
        pad_id: int = 0,
    ) -> None:
        self.model = model
        self.i2s = i2s
        self.max_length = max_length
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    @abstractmethod
    def decode(self, *args: Any, **kwds: Any) -> Any:
        """Abstract decoding method."""
        pass

    def idx_to_aa(self, idx: Tensor) -> list[str]:
        """Decode a single sample of indices to aa list."""
        idx = idx.cpu().numpy()
        t = []
        for i in idx:
            if i == self.eos_id:
                break
            if i == self.bos_id or i == self.pad_id:
                continue
            t.append(i)
        return [self.i2s[x.item()] for x in t]

    def batch_idx_to_aa(self, idx: Tensor) -> list[list[str]]:
        """Decode a batch of indices to aa lists."""
        return [self.idx_to_aa(i) for i in idx]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Decoder call."""
        return self.decode(*args, **kwargs)


class GreedyDecoder(BaseDecoder):
    """Greedy model decoder."""

    def __init__(
        self,
        model: InstaNovo,
        i2s: dict[int, str],
        max_length: int = 30,
        eos_id: int = 2,
        bos_id: int = 1,
        pad_id: int = 0,
    ) -> None:
        super().__init__(model, i2s, max_length, eos_id, bos_id, pad_id)

    # TODO score function WIP, needs to be checked!
    def score(
        self,
        spectra: Tensor,
        precursors: Tensor,
        peptide: Tensor,
        spectra_mask: Tensor,
        y_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score peptide sequences."""
        with torch.no_grad():
            logits = self.model(spectra, precursors, peptide, spectra_mask, y_mask, add_bos=True)

        logits = logits[:, :-1].softmax(dim=-1)  # check this
        aa_scores = torch.gather(logits, peptide, dim=2)

        return aa_scores, aa_scores.mean(dim=1)

    def decode(
        self, spectra: Tensor, precursors: Tensor, spectra_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Greedy model decode."""
        return self.greedy(spectra, precursors, spectra_mask)

    def greedy(
        self, spectra: Tensor, precursors: Tensor, spectra_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Greedy decoding strategy."""
        device = spectra.device
        bs = spectra.shape[0]

        y = torch.ones((bs, 1)).to(device).long()
        y_mask = torch.zeros((bs, 1), dtype=bool, device=device)  # BS, N
        eos_reached = torch.zeros(bs, dtype=bool, device=device)  # BS

        for _ in range(self.max_length):
            with torch.no_grad():
                logits = self.model(spectra, precursors, y, spectra_mask, y_mask, add_bos=False)
            preds = logits[:, -1].argmax(dim=-1)
            y = torch.cat([y, preds[:, None]], dim=-1)

            eos_reached = eos_reached | (y[:, -1] == self.eos_id)
            y_mask = torch.cat(
                [
                    y_mask,
                    eos_reached[:, None],
                ],
                axis=1,
            ).bool()

            if eos_reached.all():
                break

        # TODO check if logits is corrects shape
        return y, logits
