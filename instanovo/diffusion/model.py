from __future__ import annotations

from typing import Optional

import torch
from depthcharge.components.encoders import MassEncoder
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer
from torch import nn
from torch import Tensor
from transfusion.model import Pogfuse
from transfusion.model import timestep_embedding
from transfusion.model import TransFusion

from instanovo.diffusion.config import MassSpectrumModelConfig
from instanovo.diffusion.layers import CustomSpectrumEncoder
from instanovo.diffusion.layers import LocalisedSpectrumEncoder
from instanovo.types import Peptide
from instanovo.types import PeptideEmbedding
from instanovo.types import PeptideMask
from instanovo.types import PrecursorFeatures
from instanovo.types import ResidueLogits
from instanovo.types import Spectrum
from instanovo.types import SpectrumEmbedding
from instanovo.types import SpectrumMask
from instanovo.types import TimeEmbedding
from instanovo.types import TimeStep


class MassSpectrumTransformer(Pogfuse):
    """A transformer model specialised for encoding mass spectra."""

    def forward(
        self,
        x: Float[SpectrumEmbedding, " batch"],
        t_emb: Float[TimeEmbedding, " batch"],
        precursor_emb: Float[Tensor, "..."],
        cond_emb: Optional[Float[PeptideEmbedding, " batch"]] = None,
        x_padding_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
        cond_padding_mask: Optional[Bool[PeptideMask, " batch"]] = None,
        pos_bias: Optional[Float[Tensor, "..."]] = None,
    ) -> Float[Tensor, "batch token embedding"]:
        """Compute encodings with the model.

        Forward with `x` (bs, seq_len, dim), summing `t_emb` (bs, dim) before the transformer layer,
        and appending `conditioning_emb` (bs, seq_len2, dim) to the key/value pairs of the attention.
        Also `pooled_conv_emb` (bs, 1, dim) is summed with the timestep embeddings

        Optionally specify key/value padding for input `x` with `x_padding_mask` (bs, seq_len), and optionally
        specify key/value padding mask for conditional embedding with `cond_padding_mask` (bs, seq_len2).
        By default no padding is used. Good idea to use cond padding but not x padding.

        `pos_bias` is positional bias for wavlm-style attention gated relative position bias.

        Returns `x` of same shape (bs, seq_len, dim)
        """
        # -----------------------
        # 1. Get and add timestep embedding
        t = self.t_layers(t_emb)[:, None]  # (bs, 1, dim)
        p = self.cond_pooled_layers(precursor_emb)  # (bs, 1, dim)
        x += t + p  # (bs, seq_len, dim)
        # -----------------------
        # 2. Get and append conditioning embeddings
        if self.add_cond_seq:
            c = self.cond_layers(cond_emb)  # (bs, seq_len2, dim)
        else:
            c = None
        # -----------------------
        # 3. Do transformer layer
        # -- Self-attention block
        x1, pos_bias = self._sa_block(
            x,
            c,
            x_padding_mask=x_padding_mask,
            c_padding_mask=cond_padding_mask,
            pos_bias=pos_bias,
        )

        # -- Layer-norm with residual connection
        x = self.norm1(x + x1)

        # -- Layer-norm with feedfoward block and residual connection
        x = self.norm2(x + self._ff_block(x))

        return x, pos_bias


class MassSpectrumTransFusion(TransFusion):
    """Diffusion reconstruction model conditioned on mass spectra."""

    def __init__(
        self, cfg: MassSpectrumModelConfig, max_transcript_len: int = 200
    ) -> None:
        super().__init__(cfg, max_transcript_len)
        layers = []
        for i in range(cfg.layers):
            add_cond_cross_attn = i in list(self.cfg.cond_cross_attn_layers)
            layer = MassSpectrumTransformer(
                self.cfg.dim,
                self.cfg.t_emb_dim,
                self.cfg.cond_emb_dim,
                self.cfg.nheads,
                add_cond_seq=add_cond_cross_attn,
                dropout=self.cfg.dropout,
                use_wavlm_attn=cfg.attention_type == "wavlm"
                and not add_cond_cross_attn,
                wavlm_num_bucket=cfg.wavlm_num_bucket,
                wavlm_max_dist=cfg.wavlm_max_dist,
                has_rel_attn_bias=(cfg.attention_type == "wavlm" and i == 1),
            )
            # add relative attn bias at i=1 as that is first attn where we do not use
            # cross attention.
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.conditioning_pos_emb = None
        if cfg.localised_attn:
            self.encoder = LocalisedSpectrumEncoder(
                dim_model=cfg.dim,
                n_head=cfg.nheads,
                dim_feedforward=cfg.dim,
                n_layers=cfg.layers,
                dropout=cfg.dropout,
                window_size=cfg.window_size,
                mass_encoding=cfg.mass_encoding,
            )
        else:
            self.encoder = CustomSpectrumEncoder(
                dim_model=cfg.dim,
                n_head=cfg.nheads,
                dim_feedforward=cfg.dim_feedforward,
                n_layers=cfg.layers,
                dropout=cfg.dropout,
                mass_encoding=cfg.mass_encoding,
            )

        # precursor embedding
        self.charge_encoder = torch.nn.Embedding(cfg.max_charge, cfg.dim)
        self.mass_encoder = MassEncoder(cfg.dim)

        self.cache_spectra = None
        self.cache_cond_emb = None
        self.cache_cond_padding_mask = None

    def forward(
        self,
        x: Integer[Peptide, " batch"],
        t: Integer[TimeStep, " batch"],
        spectra: Float[Spectrum, " batch"],
        spectra_padding_mask: Bool[SpectrumMask, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        x_padding_mask: Optional[Bool[PeptideMask, " batch"]] = None,
    ) -> Float[ResidueLogits, "batch token"]:
        """Transformer with conditioning cross attention.

        - `x`: (bs, seq_len) long tensor of character indices
            or (bs, seq_len, vocab_size) if cfg.diffusion_type == 'continuous'
        - `t`: (bs, ) long tensor of timestep indices
        - `cond_emb`: (bs, seq_len2, cond_emb_dim) if using wavlm encoder, else (bs, T)
        - `x_padding_mask`: (bs, seq_len) if using wavlm encoder, else (bs, T)
        - `cond_padding_mask`: (bs, seq_len2)

        Returns logits (bs, seq_len, vocab_size)
        """
        # 1. Base: character, timestep embeddings and zeroing
        bs = x.shape[0]
        x = self.char_embedding(x)  # (bs, seq_len, dim)

        if self.cfg.pos_encoding == "relative":
            x = self.pos_embedding(x)
        else:
            pos_emb = self.pos_embedding.weight[None].expand(
                bs, -1, -1
            )  # (seq_len, dim) --> (bs, seq_len, dim)
            x = x + pos_emb

        t_emb = timestep_embedding(
            t, self.cfg.t_emb_dim, self.cfg.t_emb_max_period, dtype=spectra.dtype
        )  # (bs, t_dim)
        # 2. Classifier-free guidance: with prob cfg.drop_cond_prob, zero out and drop conditional probability
        if self.training:
            zero_cond_inds = (
                torch.rand_like(t, dtype=spectra.dtype) < self.cfg.drop_cond_prob
            )
        else:
            # never randomly zero when in eval mode
            zero_cond_inds = torch.zeros_like(t, dtype=torch.bool)
            if spectra_padding_mask.all():
                # BUT, if all cond information is padded then we are obviously doing unconditional synthesis,
                # so, force zero_cond_inds to be all ones
                zero_cond_inds = ~zero_cond_inds

        # 3. DENOVO calculate spectrum embedding here
        if self.training:
            cond_emb, cond_padding_mask = self.encoder(spectra, spectra_padding_mask)
        else:
            if self.cache_spectra is not None and torch.equal(
                self.cache_spectra, spectra
            ):
                cond_emb, cond_padding_mask = (
                    self.cache_cond_emb,
                    self.cache_cond_padding_mask,
                )
            else:
                cond_emb, cond_padding_mask = self.encoder(
                    spectra, spectra_padding_mask
                )
                self.cache_spectra = spectra
                self.cache_cond_emb = cond_emb
                self.cache_cond_padding_mask = cond_padding_mask

        # set mask for these conditional entries to true everywhere (i.e. mask them out)
        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursor_emb = masses + charges[:, None, :]

        cond_padding_mask[zero_cond_inds] = True
        cond_emb[zero_cond_inds] = 0

        # 4. Iterate through layers
        pos_bias = None
        for layer in self.layers:
            x, pos_bias = layer(
                x,
                t_emb,
                precursor_emb,
                cond_emb,
                x_padding_mask,
                cond_padding_mask,
                pos_bias=pos_bias,
            )
        # 5. Pass through head to get logits
        x = self.head(x)  # (bs, seq_len, vocab size)

        return x
