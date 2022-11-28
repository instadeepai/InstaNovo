import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from dtu_denovo_sequencing.config import ModelConfig


class TransNovo(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()

        # Encoder
        self.input_embed = nn.Linear(cfg.input_size, cfg.dim)

        # Decoder
        self.seq_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_enc = PositionalEncoding(cfg.dim)

        self.transformer = nn.Transformer(
            cfg.dim,
            cfg.nheads,
            cfg.layers,
            batch_first=True,
        )

        self.head = nn.Linear(cfg.dim, cfg.vocab_size)

    def forward(
        self, x: Tensor, x_pad: Tensor, y: Tensor, y_pad: Optional[Tensor] = None
    ) -> Tensor:
        x = self.input_embed(x)  # /10 is fp16 fix
        # FiX THIS POSITIONAL ENCODING
        y = self.pos_enc(self.seq_embed(y).transpose(0, 1)).transpose(0, 1)

        out = self.transformer(
            x,
            y,
            tgt_mask=self.transformer.generate_square_subsequent_mask(y.shape[1]).to(
                y.device
            ),
            src_key_padding_mask=x_pad,
            tgt_key_padding_mask=y_pad,
            memory_key_padding_mask=x_pad,
        )

        # feedforward + add and norm
        return self.head(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
