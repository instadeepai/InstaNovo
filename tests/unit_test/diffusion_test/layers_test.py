from __future__ import annotations

import pytest
import torch
from depthcharge.components.encoders import MassEncoder

from instanovo.diffusion.layers import CustomPeakEncoder
from instanovo.diffusion.layers import CustomSpectrumEncoder
from instanovo.diffusion.layers import LocalisedEncoderLayer
from instanovo.diffusion.layers import LocalisedEncoding
from instanovo.diffusion.layers import LocalisedSpectrumEncoder
from instanovo.diffusion.layers import LocalisedTransformerEncoder


def test_custom_peak_encoder_default_init() -> None:
    """Test custom peak encoder default initialisation."""
    cpe = CustomPeakEncoder(dim_model=15)

    assert cpe.dim_model == 15
    assert cpe.dim_intensity is None
    assert cpe.partial_encode == 1
    assert cpe.dim_mz == cpe.dim_model
    assert isinstance(cpe.int_encoder, torch.nn.Linear)

    spectrum = torch.randn(1, 5, 2)

    fp = cpe.forward(x=spectrum, mass=None, precursor_mass=None)

    assert fp.shape == (spectrum.shape[0], spectrum.shape[1], cpe.dim_model)


def test_custom_peak_encoder_spec_init() -> None:
    """Test custom peak encoder specified initialisation."""
    cpe = CustomPeakEncoder(dim_model=16, dim_intensity=8, partial_encode=0.75)

    assert cpe.dim_intensity == 8
    assert cpe.dim_model == 16
    assert cpe.partial_encode == 0.75
    assert cpe.dim_mz == 4
    assert isinstance(cpe.int_encoder, MassEncoder)

    spectrum = torch.randn(1, 5, 2)
    mass = torch.tensor([0.5]).unsqueeze(-1)
    precursor_mass = torch.tensor([1.0])

    fp = cpe.forward(x=spectrum, mass=mass, precursor_mass=precursor_mass)

    assert fp.shape == (spectrum.shape[0], spectrum.shape[1], cpe.dim_model + cpe.dim_intensity)


def test_custom_peak_encoder_errors() -> None:
    """Test custom peak encoder specified initialisation error catches."""
    with pytest.raises(RuntimeError):
        _ = CustomPeakEncoder(dim_model=15, dim_intensity=16)

    cpe = CustomPeakEncoder(dim_model=16, dim_intensity=8, partial_encode=2.05)
    spectrum = torch.randn(1, 5, 2)

    with pytest.raises(RuntimeError):
        _ = cpe.forward(x=spectrum, mass=None, precursor_mass=None)


def test_custom_spectrum_encoder_default_init() -> None:
    """Test custom spectrum encoder specified initialisation."""
    cse = CustomSpectrumEncoder()

    assert cse.peak_encoder(torch.randn(10, 2)).shape == torch.Size([10, 128])
    assert cse.linear_encoder is True

    spectra = torch.randn(3, 5, 2)
    spectra_padding_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )

    latent, mem_mask = cse.forward(spectra)

    assert latent.shape == (3, 6, 128)
    assert mem_mask.shape == (3, 6)

    latent, mem_mask = cse.forward(spectra, spectra_padding_mask=spectra_padding_mask)

    assert not mem_mask[:, 0].all().item()


def test_custom_spectrum_encoder_spec_init() -> None:
    """Test custom spectrum encoder specific initialisation."""
    cse = CustomSpectrumEncoder(dim_model=16, dim_intensity=8)

    assert cse.peak_encoder(torch.randn(10, 2)).shape == torch.Size([10, 16])
    assert cse.linear_encoder is True

    spectra = torch.randn(3, 5, 2)
    spectra_padding_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )

    latent, mem_mask = cse.forward(spectra)

    assert latent.shape == (3, 6, 16)
    assert mem_mask.shape == (3, 6)

    latent, mem_mask = cse.forward(spectra, spectra_padding_mask=spectra_padding_mask)

    assert not mem_mask[:, 0].all().item()


def test_custom_spectrum_encoder_errors() -> None:
    """Test custom spectrum encoder specific initialisation."""
    with pytest.raises(AssertionError):
        _ = CustomSpectrumEncoder(dim_model=16, dim_intensity=8, n_head=5)

    cse = CustomSpectrumEncoder(dim_model=16, dim_intensity=8, n_head=4, n_layers=-1)
    with pytest.raises(IndexError):
        cse.forward(spectra=torch.randn(3, 5, 2))


def test_localised_spectrum_encoder_spec_init() -> None:
    """Test localised spectrum encoder specified initialisation."""
    lse = LocalisedSpectrumEncoder(
        dim_model=768,
        n_head=16,
        dim_feedforward=1024,
        n_layers=9,
        dropout=0,
        window_size=1000,
        mass_encoding="casanovo",
    )

    assert lse.latent_spectrum.shape == torch.Size([1, 1, 768])
    assert isinstance(lse.peak_encoder, CustomPeakEncoder)
    assert isinstance(lse.transformer_encoder, LocalisedTransformerEncoder)
    assert isinstance(lse.transformer_encoder.layers[0].pos_enc, LocalisedEncoding)

    spectra = torch.randn(3, 5, 2)
    spectra_padding_mask = torch.tensor(
        [
            [False, True, True, True, True, True],
            [False, False, True, True, True, False],
            [False, False, False, True, True, True],
        ]
    )

    latent, mem_mask = lse.forward(spectra, spectra_padding_mask)

    assert latent.shape == (3, 6, 768)
    assert mem_mask.shape == (3, 6)
    assert not mem_mask[:, 0].all().item()


def test_localised_encoder_layer_default_init() -> None:
    """Test localised transformer encoder default initialisation."""
    lel = LocalisedEncoderLayer(d_model=16, nhead=8, mass_encoding="linear")

    assert lel.pos_enc is None
    assert lel.mass_encoding == "linear"

    fp = lel.forward(src=torch.randn(3, 5, 16))

    assert fp.shape == torch.Size([3, 5, 16])


def test_localised_encoder_layer_spec_init() -> None:
    """Test localised transformer encoder specified initialisation."""
    lel = LocalisedEncoderLayer(d_model=8, nhead=2, mass_encoding="casanovo")

    assert lel.pos_enc is None
    assert lel.mass_encoding == "casanovo"

    src_key_padding_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    ).T

    fp = lel.forward(
        src=torch.randn(3, 5, 8),
        mass=torch.randn(3),
        src_mask=torch.randn(3, 3),
        src_key_padding_mask=src_key_padding_mask,
    )

    assert fp.shape == torch.Size([3, 5, 8])


def test_localised_encoding_default_init() -> None:
    """Test localised encoding default initialisation."""
    le = LocalisedEncoding(d_model=128)

    assert le.min_wavelength == 0.001
    assert le.window_size == 100
    assert le.d_model == 128

    fp = le.forward(mass=torch.randn(3, 20, 1))

    assert fp.shape == torch.Size([3, 20, 1, 128])
