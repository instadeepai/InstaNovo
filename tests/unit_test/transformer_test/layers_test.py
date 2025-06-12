from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from instanovo.transformer.layers import MultiScalePeakEmbedding, PositionalEncoding


@pytest.mark.usefixtures("_reset_seed")
def test_pos_enc() -> None:
    """Test positional encoding default initialisation."""
    pe = PositionalEncoding(d_model=12)
    assert pe.dropout.p == 0.1
    pe.eval() # Disable dropout

    x = torch.ones(4, 2, 12)
    y = pe(x)

    assert torch.allclose(
        y,
        torch.tensor(
            [
                [
                    [
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                    ],
                    [
                        2.0461,
                        1.7114,
                        1.3486,
                        2.1965,
                        1.1627,
                        0.0000,
                        1.1222,
                        2.2222,
                        1.1135,
                        2.2222,
                        1.1116,
                        2.2222,
                    ],
                ],
                [
                    [
                        1.1111,
                        0.0000,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        0.0000,
                        1.1111,
                        2.2222,
                    ],
                    [
                        0.0000,
                        1.7114,
                        1.3486,
                        2.1965,
                        0.0000,
                        2.2210,
                        1.1222,
                        2.2222,
                        1.1135,
                        2.2222,
                        1.1116,
                        2.2222,
                    ],
                ],
                [
                    [
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                    ],
                    [
                        2.0461,
                        1.7114,
                        1.3486,
                        2.1965,
                        1.1627,
                        2.2210,
                        1.1222,
                        0.0000,
                        1.1135,
                        2.2222,
                        0.0000,
                        2.2222,
                    ],
                ],
                [
                    [
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                        1.1111,
                        0.0000,
                        1.1111,
                        2.2222,
                        1.1111,
                        2.2222,
                    ],
                    [
                        2.0461,
                        1.7114,
                        1.3486,
                        2.1965,
                        1.1627,
                        2.2210,
                        1.1222,
                        2.2222,
                        1.1135,
                        2.2222,
                        1.1116,
                        2.2222,
                    ],
                ],
            ]
        ),
        rtol=1e-2,
    )


def test_pos_enc_spec() -> None:
    """Test positional encoding specified initialisation."""
    pe = PositionalEncoding(d_model=12, dropout=0.2, max_len=10)
    assert pe.dropout.p == 0.2


def test_pos_enc_errors() -> None:
    """Test positional encoding errors."""
    with pytest.raises(RuntimeError):
        _ = PositionalEncoding(d_model=-1)

    with pytest.raises(
        ValueError, match="dropout probability has to be between 0 and 1, but got 2"
    ):
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

    with pytest.raises(
        ValueError, match="dropout probability has to be between 0 and 1, but got 2"
    ):
        _ = MultiScalePeakEmbedding(h_size=12, dropout=2)

    mspe = MultiScalePeakEmbedding(h_size=5, dropout=0.1)
    mz_values = torch.ones(3, 5, 1)
    intensities = torch.ones(3, 5, 1)
    spectra = torch.cat([mz_values, intensities], dim=2)

    with pytest.raises(RuntimeError):
        _ = mspe(spectra)
