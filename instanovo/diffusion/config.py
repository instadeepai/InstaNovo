from __future__ import annotations

from dataclasses import dataclass

from transfusion.config import ModelConfig


@dataclass
class MassSpectrumModelConfig(ModelConfig):
    """Configuration for a `MassSpectrumTransfusion` model."""

    # waveform encoder
    wav_encoder: str = "wavlm"  # only 'wavlm' currently supported

    dim_feedforward: int = 1024

    # precursor embedding
    max_charge: int = 10

    # either "casanovo" or "linear"
    mass_encoding: str = "casanovo"

    relative_peaks: bool = False
    localised_attn: bool = False
    window_size: int = 1000


mass_dict: dict[str, float] = {
    "$": 0.0,
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C(+57.02)": 160.030649,  # 103.009185 + 57.021464
    # "C": 160.030649,  # C+57.021 V1
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    "M(+15.99)": 147.035400,  # Met oxidation:   131.040485 + 15.994915
    # "M(ox)": 147.035400,  # Met oxidation:   131.040485 + 15.994915 V1
    "N(+.98)": 115.026943,  # Asn deamidation: 114.042927 +  0.984016
    "Q(+.98)": 129.042594,  # Gln deamidation: 128.058578 +  0.984016
}
i2s: list[str] = list(mass_dict.keys())
s2i: dict[str, int] = {k: v for v, k in enumerate(i2s)}
