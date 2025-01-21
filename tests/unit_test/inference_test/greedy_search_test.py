from __future__ import annotations

from typing import Any

import pytest
import torch

from instanovo.constants import MASS_SCALE
from instanovo.inference.greedy_search import GreedyDecoder
from instanovo.inference.interfaces import ScoredSequence


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = instanovo_model
    model = model.to(device)
    gd = GreedyDecoder(model, mass_scale=MASS_SCALE)

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

    scored_sequence = gd.decode(spectra, precursors, beam_size=5, max_length=6)

    expected_scored_sequence = [
        ScoredSequence(
            sequence=["C", "B", "E", "D", "D", "D"],
            mass_error=-6.347656196226126e-07,
            sequence_log_probability=-4.22208309173584,
            token_log_probabilities=[
                -0.006113164126873016,
                -0.6898691058158875,
                -1.1169137954711914,
                -0.9948660135269165,
                -0.7691261768341064,
                -0.645194947719574,
            ],
        )
    ]

    if isinstance(scored_sequence[0], ScoredSequence):
        assert scored_sequence[0].sequence == expected_scored_sequence[0].sequence
        assert scored_sequence[0].mass_error == pytest.approx(
            expected_scored_sequence[0].mass_error, abs=1e-2
        )
        assert scored_sequence[0].sequence_log_probability == pytest.approx(
            expected_scored_sequence[0].sequence_log_probability, abs=1e-2
        )
        assert scored_sequence[0].token_log_probabilities == pytest.approx(
            expected_scored_sequence[0].token_log_probabilities, abs=1e-2
        )
