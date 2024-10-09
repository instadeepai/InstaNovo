from __future__ import annotations

from transfusion.config import ModelConfig

from instanovo.diffusion.config import i2s
from instanovo.diffusion.config import mass_dict
from instanovo.diffusion.config import MassSpectrumModelConfig
from instanovo.diffusion.config import s2i


def test_ms_config() -> None:
    """Test mass spectrum model configuration initialisation."""
    ms_config = MassSpectrumModelConfig(ModelConfig)

    assert ms_config.wav_encoder == "wavlm"
    assert ms_config.dim_feedforward == 1024
    assert ms_config.max_charge == 10
    assert ms_config.mass_encoding == "casanovo"
    assert not ms_config.relative_peaks
    assert not ms_config.localised_attn
    assert ms_config.window_size == 1000


def test_mass_dict_types() -> None:
    """Test mass dictionary types."""
    assert all(isinstance(key, str) for key in mass_dict.keys())
    assert all(isinstance(value, float) for value in mass_dict.values())


def test_mass_dict_contents() -> None:
    """Test mass dictionary contents."""
    assert mass_dict == {
        "$": 0.0,
        "G": 57.021464,
        "A": 71.037114,
        "S": 87.032028,
        "P": 97.052764,
        "V": 99.068414,
        "T": 101.047670,
        "C(+57.02)": 160.030649,
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
        "M(+15.99)": 147.035400,
        "N(+.98)": 115.026943,
        "Q(+.98)": 129.042594,
    }


def test_i2s() -> None:
    """Test i2s types and contents."""
    assert isinstance(i2s, list)
    assert all(isinstance(x, str) for x in i2s)
    assert i2s == [
        "$",
        "G",
        "A",
        "S",
        "P",
        "V",
        "T",
        "C(+57.02)",
        "L",
        "I",
        "N",
        "D",
        "Q",
        "K",
        "E",
        "M",
        "H",
        "F",
        "R",
        "Y",
        "W",
        "M(+15.99)",
        "N(+.98)",
        "Q(+.98)",
    ]


def test_s2i() -> None:
    """Test s2i types and contents."""
    assert isinstance(mass_dict, dict)
    assert all(isinstance(key, str) for key in s2i.keys())
    assert all(isinstance(value, int) for value in s2i.values())
    assert s2i == {
        "$": 0,
        "G": 1,
        "A": 2,
        "S": 3,
        "P": 4,
        "V": 5,
        "T": 6,
        "C(+57.02)": 7,
        "L": 8,
        "I": 9,
        "N": 10,
        "D": 11,
        "Q": 12,
        "K": 13,
        "E": 14,
        "M": 15,
        "H": 16,
        "F": 17,
        "R": 18,
        "Y": 19,
        "W": 20,
        "M(+15.99)": 21,
        "N(+.98)": 22,
        "Q(+.98)": 23,
    }
