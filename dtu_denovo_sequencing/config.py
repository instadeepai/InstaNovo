from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration of the model."""

    # general params
    vocab_size: int = 24
    max_len: int = 15

    # main transformer params
    dim: int = 512  # 768 512 256
    nheads: int = 16  # 16
    layers: int = 6  # 12 6
    input_size: int = 4
    dropout: float = 0.1

    # which positional encoding to use ["none", "transnovo", "casanovo"]
    encoder_positional_embedding: str = "casanovo"
    # at which encoder layers should we apply positional embeddings
    positional_embedding_frequency: int = 0  # 0 only applies to first layer


i2s: list[str] = [
    "PAD",
    "SOS",
    "EOS",
    "#",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
