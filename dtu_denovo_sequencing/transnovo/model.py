"""TransNovo model."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from dtu_denovo_sequencing.transnovo.config import ModelConfig
from dtu_denovo_sequencing.utils.layers import Transformer


class TransNovo(nn.Module):
    """TransNovo model."""

    def __init__(
        self,
        cfg: ModelConfig,
    ) -> None:
        """Initialize the model."""
        super().__init__()

        # Encoder
        self.input_embed = nn.Linear(cfg.input_size, cfg.dim)

        # Decoder
        self.seq_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_enc = PositionalEncoding(cfg.dim)

        self.transformer = Transformer(
            cfg.dim,
            cfg.nheads,
            cfg.layers,
            batch_first=True,
            pos_encoding_type=cfg.encoder_positional_embedding,
            pos_encoding_freq=cfg.positional_embedding_frequency,
        )

        # self.transformer = nn.Transformer(
        #     cfg.dim,
        #     cfg.nheads,
        #     cfg.layers,
        #     batch_first=True,
        # )

        self.head = nn.Linear(cfg.dim, cfg.vocab_size)

    def forward(self, x: Tensor, x_pad: Tensor, y: Tensor, y_pad: Tensor | None = None) -> Tensor:
        """Defines the computation performed at every call."""
        # x.shape: (batch, peaks, dim)
        bias = x[:, :, 0]

        x = self.input_embed(x)
        y = self.pos_enc(self.seq_embed(y))

        out = self.transformer(
            x,
            y,
            bias=bias,
            tgt_mask=self.transformer.generate_square_subsequent_mask(y.shape[1]).to(y.device),
            src_key_padding_mask=x_pad,
            tgt_key_padding_mask=y_pad,
            memory_key_padding_mask=x_pad,
        )

        # feedforward + add and norm
        return self.head(out)


class PositionalEncoding(nn.Module):
    """PositionalEncoding module."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """Initialize the model.

        Args:
            d_model (int):
            dropout (float, optional): Defaults to 0.1.
            max_len (int, optional): Defaults to 100.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): shape [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
