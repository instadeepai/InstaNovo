from __future__ import annotations

import math

import numpy as np
import torch
from jaxtyping import Float
from torch import nn
from torch import Tensor

from instanovo.types import SpectrumEmbedding, Spectrum


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: Float[Tensor, "token batch embedding"]
    ) -> Float[Tensor, "token batch embedding"]:
        """Positional encoding forward pass.

        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiScalePeakEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding based on Voronov et. al."""

    def __init__(self, h_size: int, dropout: float = 0) -> None:
        super().__init__()
        self.h_size = h_size

        self.mlp = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(h_size + 1, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

        freqs = 2 * np.pi / torch.logspace(-2, -3, int(h_size / 2), dtype=torch.float64)
        self.register_buffer("freqs", freqs)

    def forward(
        self, spectra: Float[Spectrum, " batch"]
    ) -> Float[SpectrumEmbedding, " batch"]:
        """Encode peaks."""
        mz_values, intensities = spectra[:, :, [0]], spectra[:, :, [1]]
        x = self.encode_mass(mz_values)
        x = self.mlp(x)
        x = torch.cat([x, intensities], axis=2)
        return self.head(x)

    def encode_mass(
        self, x: Float[Tensor, " batch"]
    ) -> Float[Tensor, "batch embedding"]:
        """Encode mz."""
        x = self.freqs[None, None, :] * x
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=2)
        return x.float()


class ConvPeakEmbedding(nn.Module):
    """Convolutional peak embedding."""

    def __init__(self, h_size: int, dropout: float = 0) -> None:
        super().__init__()
        self.h_size = h_size

        self.conv = nn.Sequential(
            nn.Conv1d(
                1, h_size // 4, kernel_size=40_000, stride=100, padding=40_000 // 2 - 1
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(h_size // 4, h_size, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Conv peak embedding."""
        x = x.unsqueeze(1)
        return self.conv(x).transpose(-1, -2)
