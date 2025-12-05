from typing import Any

import torch

from instanovo.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
from instanovo.transformer.predict import _setup_knapsack


def test_init(instanovo_model: tuple[Any, Any]) -> None:
    """Test knapsack beam search decoder initialisation."""
    model, _ = instanovo_model
    knapsack = _setup_knapsack(model)
    knapsack_decoder = KnapsackBeamSearchDecoder(model, knapsack)

    assert knapsack_decoder.model == model
    assert knapsack_decoder.knapsack is not None
    assert torch.equal(knapsack_decoder.chart, torch.tensor(knapsack_decoder.knapsack.chart))


def test_from_file(
    knapsack_dir: str,
    instanovo_model: tuple[Any, Any],
) -> None:
    """Test knapsack beam search decoder loader."""
    model, _ = instanovo_model
    knapsack_decoder = KnapsackBeamSearchDecoder.from_file(model, knapsack_dir)

    assert knapsack_decoder.model == model
    assert knapsack_decoder.knapsack is not None
    assert torch.equal(knapsack_decoder.chart, torch.tensor(knapsack_decoder.knapsack.chart))
