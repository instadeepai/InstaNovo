from __future__ import annotations

from typing import Any

import pytest
import torch

from instanovo.constants import MASS_SCALE
from instanovo.inference.greedy_search import GreedyDecoder
from instanovo.utils.device_handler import check_device


def test_greedy_decoder(instanovo_model: tuple[Any, Any]) -> None:
    """Test greedy decoder initialisation."""
    model, _ = instanovo_model
    gd = GreedyDecoder(model, mass_scale=MASS_SCALE)

    assert gd.mass_scale == MASS_SCALE
    assert gd.vocab_size == 8
    assert torch.allclose(
        gd.residue_masses,
        torch.tensor(
            [0.0000, 0.0000, 0.0000, 10.5000, 20.7500, 15.6800, 18.2500, 12.3300],
            dtype=torch.float64,
        ),
    )


def test_decode(instanovo_model: tuple[Any, Any]) -> None:
    """Test greedy decoder decode function."""
    device = check_device()
    model, _ = instanovo_model
    model = model.to(device)
    gd = GreedyDecoder(model, mass_scale=MASS_SCALE, float_dtype=torch.float32 if device == "mps" else torch.float64)

    spectra = torch.tensor(
        [
            [
                [104.2779, 1.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
            ],
        ],
        device=device,
    )
    precursors = torch.tensor(
        [
            [121.5206, 2.0000, 61.7676],
        ],
        device=device,
    )

    result: dict[str, Any] = gd.decode(spectra, precursors, beam_size=5, max_length=6)

    assert result["predictions"][0] == ["C", "B", "E", "D", "D", "D"]
    assert result["prediction_log_probability"][0] == pytest.approx(-4.22208309173584, rel=1e-1)
    assert result["prediction_token_log_probabilities"][0] == pytest.approx(
        [
            -0.006113164126873016,
            -0.6898691058158875,
            -1.1169137954711914,
            -0.9948660135269165,
            -0.7691261768341064,
            -0.645194947719574,
        ],
        rel=1e-1,
    )
