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


def test_prefilter_items(
    setup_knapsack_decoder: KnapsackBeamSearchDecoder,
) -> None:
    """Test knapsack beam search decoder prefilter items."""
    knapsack_decoder = setup_knapsack_decoder

    log_probabilities = torch.tensor(
        [
            [
                [
                    -13.7944,
                    -13.8530,
                    -11.5678,
                    -3.8042,
                    -5.0835,
                    -3.8764,
                    -5.3554,
                    -3.5288,
                ],
                [
                    -13.7159,
                    -13.6573,
                    -11.7588,
                    -7.5664,
                    -4.4161,
                    -3.5616,
                    -4.2823,
                    -4.0713,
                ],
                [
                    -13.9910,
                    -13.7879,
                    -11.7996,
                    -4.1902,
                    -3.9851,
                    -4.1531,
                    -3.8953,
                    -7.8836,
                ],
                [
                    -14.0309,
                    -14.1090,
                    -11.7613,
                    -3.9723,
                    -5.2506,
                    -4.4791,
                    -5.4635,
                    -3.7008,
                ],
                [
                    -13.9431,
                    -14.1423,
                    -11.8923,
                    -4.2614,
                    -8.8415,
                    -4.2809,
                    -4.3298,
                    -3.7712,
                ],
            ]
        ]
    )
    remaining_masses = torch.tensor(
        [
            [
                [592600, 592600, 592600, 487600, 385100, 435800, 410100, 469300],
                [670100, 670100, 670100, 565100, 462600, 513300, 487600, 546800],
                [651800, 651800, 651800, 546800, 444300, 495000, 469300, 528500],
                [592600, 592600, 592600, 487600, 385100, 435800, 410100, 469300],
                [567600, 567600, 567600, 462600, 360100, 410800, 385100, 444300],
            ]
        ]
    )
    beam_masses = torch.tensor([[592600, 670100, 651800, 592600, 567600]])
    mass_buffer = torch.tensor([[[56]]])
    max_isotope = 1

    result = knapsack_decoder.prefilter_items(
        log_probabilities, remaining_masses, beam_masses, mass_buffer, max_isotope
    )

    assert torch.allclose(
        result,
        torch.tensor(
            [
                [
                    [
                        -float("inf"),
                        -float("inf"),
                        -float("inf"),
                        -3.8042,
                        -5.0835,
                        -3.8764,
                        -float("inf"),
                        -3.5288,
                    ],
                    [
                        -float("inf"),
                        -float("inf"),
                        -float("inf"),
                        -float("inf"),
                        -4.4161,
                        -3.5616,
                        -4.2823,
                        -4.0713,
                    ],
                    [
                        -float("inf"),
                        -float("inf"),
                        -float("inf"),
                        -4.1902,
                        -3.9851,
                        -4.1531,
                        -3.8953,
                        -float("inf"),
                    ],
                    [
                        -float("inf"),
                        -float("inf"),
                        -float("inf"),
                        -3.9723,
                        -5.2506,
                        -4.4791,
                        -float("inf"),
                        -3.7008,
                    ],
                    [
                        -float("inf"),
                        -float("inf"),
                        -float("inf"),
                        -4.2614,
                        -float("inf"),
                        -4.2809,
                        -4.3298,
                        -3.7712,
                    ],
                ]
            ]
        ),
    )


def test_get_isotope_chart(setup_knapsack_decoder: KnapsackBeamSearchDecoder) -> None:
    """Test knapsack beam search decoder isotope chart."""
    knapsack_decoder = setup_knapsack_decoder

    beam_lower_bound = 10
    beam_upper_bound = 20
    scaled_nucleon_mass = 1
    num_nucleons = 2

    result = knapsack_decoder._get_isotope_chart(
        beam_lower_bound, beam_upper_bound, scaled_nucleon_mass, num_nucleons
    )
    assert torch.equal(
        result, torch.tensor([False, False, False, False, False, False, False, False])
    )
    assert result.shape == torch.Size([8])


def test_init_prefilter(setup_knapsack_decoder: KnapsackBeamSearchDecoder) -> None:
    """Test knapsack beam search decoder prefilter initialisation."""
    knapsack_decoder = setup_knapsack_decoder

    log_probabilities = torch.tensor(
        [
            [-12.5467, -12.2993, -1.8574, -2.3155, -1.4480, -5.4448, -1.5925, -1.1959],
            [-12.5608, -12.1632, -2.0155, -1.8701, -1.3412, -2.0690, -3.9073, -1.1885],
        ],
    )
    precursor_masses = torch.tensor(
        [924100, 923400],
    )
    mass_buffer = torch.tensor(
        [55, 55],
    )

    result = knapsack_decoder._init_prefilter(precursor_masses, log_probabilities, mass_buffer)

    expected_tensor = torch.tensor(
        [
            [
                float("-inf"),
                float("-inf"),
                float("-inf"),
                -2.3155,
                -1.4480,
                float("-inf"),
                -1.5925,
                -1.1959,
            ],
            [
                float("-inf"),
                float("-inf"),
                float("-inf"),
                -1.8701,
                -1.3412,
                -2.0690,
                float("-inf"),
                -1.1885,
            ],
        ],
    )

    assert torch.equal(result, expected_tensor)
    assert result.shape == log_probabilities.shape
