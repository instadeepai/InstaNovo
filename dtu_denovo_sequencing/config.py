from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    # general params
    vocab_size: int = 24
    max_len: int = 20

    # main transformer params
    dim: int = 256  # 768
    nheads: int = 16  # 16
    layers: int = 6  # 12
    input_size: int = 4
    dropout: float = 0.1


i2s: List[str] = [
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
