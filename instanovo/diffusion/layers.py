from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from instanovo.__init__ import console
from instanovo.transformer.layers import ConvPeakEmbedding, MultiScalePeakEmbedding
from instanovo.types import Spectrum, SpectrumEmbedding, SpectrumMask
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class TransformerEncoder(nn.Module):
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    """

    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.0,
        use_flash_attention: bool = False,
        conv_peak_encoder: bool = False,
    ) -> None:
        """Initialise a TransformerEncoder."""
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.conv_peak_encoder = conv_peak_encoder

        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        if self.use_flash_attention:
            # All input spectra are padded to some max length
            # Pad spectrum replaces zeros in input spectra
            # This is for flash attention (no masks allowed)
            self.pad_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        # Encoder
        self.peak_encoder = MultiScalePeakEmbedding(dim_model, dropout=dropout)
        if self.conv_peak_encoder:
            self.conv_encoder = ConvPeakEmbedding(dim_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            # enable_nested_tensor=False, TODO: Figure out the correct way to handle this
        )

    def forward(
        self,
        x: Float[Spectrum, " batch"],
        x_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
    ) -> Tuple[Float[SpectrumEmbedding, " batch"], Bool[SpectrumMask, " batch"]]:
        """The forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        x_mask: torch.Tensor
            Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)

        Returns:
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        if self.conv_peak_encoder:
            x = self.conv_encoder(x)
            x_mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device).bool()
        else:
            if x_mask is None:
                x_mask = ~x.sum(dim=2).bool()
            x = self.peak_encoder(x)

        # Self-attention on latent spectra AND peaks
        latent_spectra = self.latent_spectrum.expand(x.shape[0], -1, -1)
        x = torch.cat([latent_spectra, x], dim=1)
        latent_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
        x_mask = torch.cat([latent_mask, x_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=x_mask)

        return x, x_mask
