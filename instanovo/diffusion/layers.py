from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from depthcharge.components import SpectrumEncoder
from depthcharge.components.encoders import MassEncoder
from torch import Tensor

logger = logging.getLogger("casanovo")


class CustomPeakEncoder(MassEncoder):
    """Encode m/z values in a mass spectrum using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    dim_intensity : int, optional
        The number of features to use for intensity. The remaining features
        will be used to encode the m/z values.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.
    """

    def __init__(
        self,
        dim_model: int,
        dim_intensity: int | None = None,
        min_wavelength: float = 0.001,
        max_wavelength: float = 10000,
        partial_encode: float = 1.0,
    ):
        """Initialize the MzEncoder."""
        self.dim_intensity = dim_intensity
        self.dim_model = dim_model
        self.dim_mz = int(dim_model * partial_encode)
        self.partial_encode = partial_encode
        if self.dim_intensity is not None:
            self.dim_mz -= self.dim_intensity

        super().__init__(
            dim_model=self.dim_mz,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

        self.int_encoder: nn.Module
        if self.dim_intensity is None:
            self.int_encoder = torch.nn.Linear(1, dim_model, bias=False)
        else:
            self.int_encoder = MassEncoder(
                dim_model=dim_intensity,
                min_wavelength=0,
                max_wavelength=1,
            )

    def forward(
        self, x: torch.Tensor, mass: torch.Tensor, precursor_mass: torch.Tensor
    ) -> torch.Tensor:
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        x : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        mass : optional torch.Tensor of shape (n_spectra, )
            The mass of the sequence decoded so far
        precursor_mass : optional torch.Tensor of shape (n_spectra, )
            The mass of the parent ion

        Returns
        -------
        torch.Tensor of shape (n_spectr, n_peaks, dim_model)
            The encoded features for the mass spectra.
        """
        m_over_z = x[:, :, [0]]
        encoded = torch.zeros(
            (x.shape[0], x.shape[1], self.dim_model), device=x.device, dtype=x.dtype
        )
        encoded[:, :, : self.dim_mz] = super().forward(m_over_z)

        if mass is not None:
            encoded[:, :, : self.dim_mz] += super().forward(m_over_z - mass[:, :, None])

        if precursor_mass is not None:
            encoded[:, :, : self.dim_mz] += super().forward(
                precursor_mass[:, None, None] - m_over_z
            )

        if self.dim_intensity is None:
            intensity = self.int_encoder(x[:, :, [1]])
            return encoded + intensity

        intensity = self.int_encoder(x[:, :, [1]])

        return torch.cat([encoded, intensity], dim=2)


class CustomSpectrumEncoder(SpectrumEncoder):
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
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value.
    """

    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.0,
        peak_encoder: bool = True,
        dim_intensity: int | None = None,
        mass_encoding: str = "linear",
    ):
        """Initialize a CustomSpectrumEncoder."""
        super().__init__(
            dim_model, n_head, dim_feedforward, n_layers, dropout, peak_encoder, dim_intensity
        )

        if peak_encoder and mass_encoding == "casanovo":
            self.peak_encoder = CustomPeakEncoder(dim_model, dim_intensity=dim_intensity)
            self.linear_encoder = False
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)
            self.linear_encoder = True

    def forward(
        self,
        spectra: torch.Tensor,
        spectra_padding_mask: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        precursor_mass: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward pass.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        mass : torch.Tensor of shape (n_spectra, )
            The mass of the sequence decoded so far
        precursor_mass : torch.Tensor of shape (n_spectra, )
            The mass of the parent ion

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        zeros = ~spectra.sum(dim=2).bool()
        if spectra_padding_mask is not None:
            mask = spectra_padding_mask
            mask = torch.cat(
                [torch.tensor([[False]] * spectra.shape[0]).type_as(zeros), mask], dim=1
            )
        else:
            mask = [
                torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
        if not self.linear_encoder:
            peaks = self.peak_encoder(spectra, mass, precursor_mass)
        else:
            peaks = self.peak_encoder(spectra)
        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        peaks = torch.cat([latent_spectra, peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask


class LocalisedSpectrumEncoder(torch.nn.Module):
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
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value.
    """

    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: int = 0,
        peak_encoder: bool = True,
        dim_intensity: int | None = None,
        window_size: int = 400,
        device: str | None = None,
        mass_encoding: str = "linear",
    ):
        """Initialize a SpectrumEncoder."""
        super().__init__()

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))

        if peak_encoder and mass_encoding == "casanovo":
            self.peak_encoder = CustomPeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
                partial_encode=0.5,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        # The Transformer layers:
        layer = LocalisedEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            mass_encoding=mass_encoding,
        )

        self.transformer_encoder = LocalisedTransformerEncoder(
            layer,
            num_layers=n_layers,
        )

        # Only add positional encoding to first layer!
        if mass_encoding == "casanovo":
            self.transformer_encoder.layers[0].pos_enc = LocalisedEncoding(
                d_model=dim_model // 2, window_size=window_size, device=device
            )
        elif mass_encoding == "linear":
            self.transformer_encoder.layers[0].pos_enc = nn.Sequential(
                nn.Linear(3, dim_model // 2),
            )

    @property
    def device(self) -> str:
        """The current device for the model."""
        device: str = next(self.parameters()).device
        return device

    def forward(
        self, spectra: torch.Tensor, spectra_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The forward pass.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        zeros = ~spectra.sum(dim=2).bool()
        if spectra_padding_mask is not None:
            mask = spectra_padding_mask
        else:
            mask = [
                torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
        peaks = self.peak_encoder(spectra, None, None)
        m_over_z = spectra[:, :, 0]

        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
        latent_mz = torch.zeros((m_over_z.shape[0], 1), device=spectra.device, dtype=m_over_z.dtype)

        peaks = torch.cat([latent_spectra, peaks], dim=1)
        m_over_z = torch.cat([latent_mz, m_over_z], dim=1)
        return self.transformer_encoder(peaks, mass=m_over_z, src_key_padding_mask=mask), mask


class LocalisedTransformerEncoder(torch.nn.TransformerEncoder):
    """Localised transformer encoder."""

    def forward(
        self,
        src: Tensor,
        mass: Tensor | None = None,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute representations using localised transformer.

        Args:
            src (Tensor): The source tensor.
            mass (Tensor | None, optional): Masses of the batch. Defaults to None.
            mask (Tensor | None, optional): The self-attention mask. Defaults to None.
            src_key_padding_mask (Tensor | None, optional): The padding mask for th source sequence. Defaults to None.

        Returns:
            Tensor: The encoding representation.
        """
        src_key_padding_mask_for_layers = src_key_padding_mask

        output = self.layers[0](
            src, mass=mass, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers
        )

        for mod in self.layers[1:]:
            output = mod(
                output,
                mass=None,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )

        return output


class LocalisedEncoderLayer(torch.nn.TransformerEncoderLayer):
    """Layer in a localised transformer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        mass_encoding: str = "linear",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        self.pos_enc = None
        self.mass_encoding = mass_encoding

    def forward(
        self,
        src: Tensor,
        mass: Tensor | None = None,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute localised transformer encoding for one layer.

        Args:
            src (Tensor): The source tensor.
            mass (Tensor | None, optional): Masses of the batch. Defaults to None.
            src_mask (Tensor | None, optional): The source self-attention mask. Defaults to None.
            src_key_padding_mask (Tensor | None, optional): The source padding mask. Defaults to None.

        Returns:
            Tensor: The encoding representation for one layer.
        """
        x = src
        x = self.norm1(x + self._sa_block(x, mass, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    # flake8: noqa: CR001
    def _sa_block(
        self,
        x: Tensor,
        mass: Tensor | None,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        # full_attention: bool = True,
    ) -> Tensor:

        if self.pos_enc and mass is not None:
            q = x.reshape(-1, 1, x.shape[-1])  # apply queries in one big batch
            zero_enc = self.pos_enc(torch.zeros((q.shape[0], 1), device=x.device, dtype=x.dtype))
            q[:, :, -self.pos_enc.d_model :] += zero_enc  # add encoding to query

            kv = x.repeat_interleave(x.shape[1], axis=0)  # kv values
            mask = key_padding_mask.repeat_interleave(x.shape[1], axis=0)  # mask
            # calculate pairwise distance matrix
            if self.mass_encoding == "casanovo":
                mass_mat = torch.ones(mass.shape[0], mass.shape[1], 1) @ mass[:, None, :]
                mass_mat = mass_mat - mass_mat.transpose(1, 2)

                enc = self.pos_enc(mass_mat.reshape(-1, x.shape[1]))
            elif self.mass_encoding == "linear":
                mass = mass.repeat((1, x.shape[1]))
                m1 = mass.reshape(2, 2, 2).transpose(1, 2).reshape(4, 2)
                m2 = mass.reshape(4, 2)
                mass = torch.stack([m1, m2, m1 - m2], dim=-1)
                enc = self.pos_enc(mass)

            kv[:, :, -self.pos_enc.d_model :] += enc  # add positional encoding to kv values

            y = self.self_attn(
                q, kv, kv, attn_mask=attn_mask, key_padding_mask=mask, need_weights=False
            )[0]
            y = y.reshape(x.shape)  # return big batch to original shape

        else:
            y = self.self_attn(
                x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
            )[0]
        return self.dropout1(y)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class LocalisedEncoding(nn.Module):
    """LocalisedEncoding module."""

    def __init__(
        self,
        d_model: int,
        casanovo_style: bool = False,
        window_size: int = 100,
        min_wavelength: float = 0.001,
        device: str | None = None,
    ) -> None:
        """Custom localised positional encoder.

        Args:
            d_model (int):
            window_size (int, optional): Defaults to 100.
            min_wavelength (float, optional): Defaults to 0.001.
        """
        super().__init__()
        self.min_wavelength = min_wavelength
        self.window_size = window_size

        self.d_model = d_model

        max_len = int((window_size + 1) * 2 / min_wavelength)  # +2 to be inclusive of window bounds

        logging.info(
            f"Pre-computing localised encoding matrix on device={'cpu' if not device else device}, \
                       total {(window_size * d_model * 4)/min_wavelength/(1024**2):.2f} MB"
        )
        position = torch.arange(max_len, device=device).unsqueeze(1) - max_len // 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, device=device)
        if casanovo_style:
            pe[:, : pe.shape[1] // 2] = torch.sin(position * div_term, device=device)
            pe[:, pe.shape[1] // 2 :] = torch.cos(position * div_term, device=device)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        logging.info(f"Pre-computing complete")
        self.register_buffer("pe", pe)

    def forward(self, mass: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            mass (Tensor): shape [batch_size, seq_len, 1]

        Returns:
            Tensor
        """
        mass = mass.clamp(-self.window_size, self.window_size)
        mass_idx = (((mass + 1) + self.window_size) / self.min_wavelength).to(torch.long)

        return self.pe[mass_idx]
