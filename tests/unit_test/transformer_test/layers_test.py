from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from instanovo.transformer.layers import MultiScalePeakEmbedding, PositionalEncoding


@pytest.mark.usefixtures("_reset_seed")
def test_pos_enc() -> None:
    """Test positional encoding default initialisation."""
    pe = PositionalEncoding(d_model=12)
    pe.eval()  # Set to eval mode to avoid dropout during testing, makes test deterministic
    pe.dropout.p = 0.0

    x = torch.ones(4, 2, 12)
    y = pe(x)

    expected = torch.tensor(
        [
            [
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                [
                    1.8414709568023682,
                    1.5403022766113281,
                    1.2137806415557861,
                    1.976881742477417,
                    1.0463992357254028,
                    1.9989229440689087,
                    1.0099998712539673,
                    1.9999499320983887,
                    1.0021544694900513,
                    1.999997615814209,
                    1.0004642009735107,
                    1.9999998807907104,
                ],
            ],
            [
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                [
                    1.8414709568023682,
                    1.5403022766113281,
                    1.2137806415557861,
                    1.976881742477417,
                    1.0463992357254028,
                    1.9989229440689087,
                    1.0099998712539673,
                    1.9999499320983887,
                    1.0021544694900513,
                    1.999997615814209,
                    1.0004642009735107,
                    1.9999998807907104,
                ],
            ],
            [
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                [
                    1.8414709568023682,
                    1.5403022766113281,
                    1.2137806415557861,
                    1.976881742477417,
                    1.0463992357254028,
                    1.9989229440689087,
                    1.0099998712539673,
                    1.9999499320983887,
                    1.0021544694900513,
                    1.999997615814209,
                    1.0004642009735107,
                    1.9999998807907104,
                ],
            ],
            [
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                [
                    1.8414709568023682,
                    1.5403022766113281,
                    1.2137806415557861,
                    1.976881742477417,
                    1.0463992357254028,
                    1.9989229440689087,
                    1.0099998712539673,
                    1.9999499320983887,
                    1.0021544694900513,
                    1.999997615814209,
                    1.0004642009735107,
                    1.9999998807907104,
                ],
            ],
        ]
    )

    assert torch.allclose(y, expected, rtol=1e-2, atol=1e-4)


def test_pos_enc_spec() -> None:
    """Test positional encoding specified initialisation."""
    pe = PositionalEncoding(d_model=12, dropout=0.2, max_len=10)
    assert pe.dropout.p == 0.2


def test_pos_enc_errors() -> None:
    """Test positional encoding errors."""
    with pytest.raises(RuntimeError):
        _ = PositionalEncoding(d_model=-1)

    with pytest.raises(ValueError, match="dropout probability has to be between 0 and 1, but got 2"):
        _ = PositionalEncoding(d_model=12, dropout=2)


def test_peak_embed() -> None:
    """Test multi scale peak embedding initialisation."""
    mspe = MultiScalePeakEmbedding(h_size=8, dropout=0.2)

    assert mspe.h_size == 8
    assert isinstance(mspe.mlp, nn.Sequential)
    assert isinstance(mspe.head, nn.Sequential)
    assert mspe.mlp[2].p == 0.2
    assert mspe.head[2].p == 0.2

    mz_values = torch.ones(3, 5, 1)
    intensities = torch.ones(3, 5, 1)
    spectra = torch.cat([mz_values, intensities], dim=2)

    y = mspe(spectra)
    assert y.shape == torch.Size([3, 5, 8])


def test_peak_embed_errors() -> None:
    """Test multi scale peak embedding errors."""
    with pytest.raises(RuntimeError):
        _ = MultiScalePeakEmbedding(h_size=-1)

    with pytest.raises(ValueError, match="dropout probability has to be between 0 and 1, but got 2"):
        _ = MultiScalePeakEmbedding(h_size=12, dropout=2)

    mspe = MultiScalePeakEmbedding(h_size=5, dropout=0.1)
    mz_values = torch.ones(3, 5, 1)
    intensities = torch.ones(3, 5, 1)
    spectra = torch.cat([mz_values, intensities], dim=2)

    with pytest.raises(RuntimeError):
        _ = mspe(spectra)
