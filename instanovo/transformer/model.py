from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch
import json
import os
import requests
from urllib.parse import urlsplit
from tqdm import tqdm
from pathlib import Path
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer
from omegaconf import DictConfig
from torch import nn
from torch import Tensor

from instanovo.constants import MAX_SEQUENCE_LENGTH
from instanovo.transformer.layers import ConvPeakEmbedding
from instanovo.transformer.layers import MultiScalePeakEmbedding
from instanovo.transformer.layers import PositionalEncoding
from instanovo.inference import Decodable
from instanovo.types import DiscretizedMass
from instanovo.types import Peptide
from instanovo.types import PeptideMask
from instanovo.types import PrecursorFeatures
from instanovo.types import ResidueLogits
from instanovo.types import ResidueLogProbabilities
from instanovo.types import Spectrum
from instanovo.types import SpectrumEmbedding
from instanovo.types import SpectrumMask
from instanovo.utils import ResidueSet

MODELS_PATH = Path(__file__).parent.parent / "models.json"
MODEL_TYPE = "transformer"


class InstaNovo(nn.Module, Decodable):
    """The Instanovo model."""

    def __init__(
        self,
        residue_set: ResidueSet,
        dim_model: int = 768,
        n_head: int = 16,
        dim_feedforward: int = 2048,
        n_layers: int = 9,
        dropout: float = 0.1,
        max_charge: int = 5,
        use_flash_attention: bool = False,
        conv_peak_encoder: bool = False,
    ) -> None:
        super().__init__()
        self._residue_set = residue_set
        self.vocab_size = len(residue_set)
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

        # Decoder
        self.aa_embed = nn.Embedding(self.vocab_size, dim_model, padding_idx=0)

        self.aa_pos_embed = PositionalEncoding(
            dim_model, dropout, max_len=MAX_SEQUENCE_LENGTH
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        self.head = nn.Linear(dim_model, self.vocab_size)
        self.charge_encoder = nn.Embedding(max_charge, dim_model)

    @property
    def residue_set(self) -> ResidueSet:
        """Every model must have a `residue_set` attribute."""
        return self._residue_set

    @staticmethod
    def _get_causal_mask(seq_len: int, return_float: bool = False) -> PeptideMask:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        if return_float:
            return (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
        return ~mask.bool()

    @staticmethod
    def get_pretrained() -> list[str]:
        """Get a list of pretrained model ids."""
        # Load the models.json file
        with open(MODELS_PATH, "r") as f:
            models_config = json.load(f)

        if MODEL_TYPE not in models_config:
            return []

        return list(models_config[MODEL_TYPE].keys())

    @classmethod
    def load(cls, path: str) -> Tuple["InstaNovo", "DictConfig"]:
        """Load model from checkpoint path."""
        # Add  to allow list
        _whitelist_torch_omegaconf()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        config = ckpt["config"]

        # check if PTL checkpoint
        if all([x.startswith("model") for x in ckpt["state_dict"].keys()]):
            ckpt["state_dict"] = {
                k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
            }

        residue_set = ResidueSet(config["residues"])

        model = cls(
            residue_set=residue_set,
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            max_charge=config["max_charge"],
            use_flash_attention=config.get("use_flash_attention", False),
            conv_peak_encoder=config.get("conv_peak_encoder", False),
        )
        model.load_state_dict(ckpt["state_dict"])

        return model, config

    @classmethod
    def from_pretrained(cls, model_id: str) -> Tuple["InstaNovo", "DictConfig"]:
        """Download and load by model id or model path."""
        # Check if model_id is a local file path
        if "/" in model_id or "\\" in model_id or "." in model_id:
            if os.path.isfile(model_id):
                return cls.load(model_id)
            else:
                raise FileNotFoundError(f"No file found at path: {model_id}")

        # Load the models.json file
        with open(MODELS_PATH, "r") as f:
            models_config = json.load(f)

        # Find the model in the config
        if MODEL_TYPE not in models_config or model_id not in models_config[MODEL_TYPE]:
            raise ValueError(
                f"Model {model_id} not found in models.json, options are [{', '.join(models_config[MODEL_TYPE].keys())}]"
            )

        model_info = models_config[MODEL_TYPE][model_id]
        url = model_info["url"]

        # Create cache directory if it doesn't exist
        cache_dir = Path.home() / ".cache" / "instanovo"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate a filename for the cached model
        file_name = urlsplit(url).path.split("/")[-1]
        cached_file = cache_dir / file_name

        # Check if the file is already cached
        if not cached_file.exists():
            # If not cached, download the file with a progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with open(cached_file, "wb") as file, tqdm(
                desc=file_name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
        # else:
        #     TODO: Optional verbose logging
        #     print(f"Model {model_id} already cached at {cached_file}")

        # Load and return the model
        return cls.load(str(cached_file))

    def forward(
        self,
        x: Float[Spectrum, " batch"],
        p: Float[PrecursorFeatures, " batch"],
        y: Integer[Peptide, " batch"],
        x_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
        y_mask: Optional[Bool[PeptideMask, " batch"]] = None,
        add_bos: bool = True,
    ) -> Float[ResidueLogits, "batch token+1"]:
        """Model forward pass.

        Args:
            x: Spectra, float Tensor (batch, n_peaks, 2)
            p: Precursors, float Tensor (batch, 3)
            y: Peptide, long Tensor (batch, seq_len, vocab)
            x_mask: Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)
            y_mask: Peptide padding mask, bool Tensor (batch, seq_len)
            add_bos: Force add a <s> prefix to y, bool

        Returns:
            logits: float Tensor (batch, n, vocab_size),
            (batch, n+1, vocab_size) if add_bos==True.
        """
        if self.use_flash_attention:
            x, x_mask = self._flash_encoder(x, p, x_mask)
            return self._flash_decoder(x, y, x_mask, y_mask, add_bos)

        x, x_mask = self._encoder(x, p, x_mask)
        return self._decoder(x, y, x_mask, y_mask, add_bos)

    def init(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        spectra_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
    ) -> Tuple[
        Tuple[Float[Spectrum, " batch"], Bool[SpectrumMask, " batch"]],
        Float[ResidueLogProbabilities, "batch token"],
    ]:
        """Initialise model encoder."""
        if self.use_flash_attention:
            spectra, _ = self._encoder(spectra, precursors, None)
            logits = self._decoder(spectra, None, None, None, add_bos=False)
            return (
                spectra,
                torch.zeros(spectra.shape[0], spectra.shape[1]).to(spectra.device),
            ), torch.log_softmax(logits[:, -1, :], -1)

        spectra, spectra_mask = self._encoder(spectra, precursors, spectra_mask)
        logits = self._decoder(spectra, None, spectra_mask, None, add_bos=False)
        return (spectra, spectra_mask), torch.log_softmax(logits[:, -1, :], -1)

    def score_candidates(
        self,
        sequences: Integer[Peptide, " batch"],
        precursor_mass_charge: Float[PrecursorFeatures, " batch"],
        spectra: Float[Spectrum, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"],
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """Score a set of candidate sequences."""
        if self.use_flash_attention:
            logits = self._flash_decoder(spectra, sequences, None, None, add_bos=True)
        else:
            logits = self._decoder(spectra, sequences, spectra_mask, None, add_bos=True)

        return torch.log_softmax(logits[:, -1, :], -1)

    def get_residue_masses(
        self, mass_scale: int
    ) -> Integer[DiscretizedMass, " residue"]:
        """Get the scaled masses of all residues."""
        residue_masses = torch.zeros(len(self.residue_set), dtype=torch.int64)
        for index, residue in self.residue_set.index_to_residue.items():
            if residue in self.residue_set.residue_masses:
                residue_masses[index] = round(
                    mass_scale * self.residue_set.get_mass(residue)
                )
        return residue_masses

    def get_eos_index(self) -> int:
        """Get the EOS token ID."""
        return int(self.residue_set.EOS_INDEX)

    def get_empty_index(self) -> int:
        """Get the PAD token ID."""
        return int(self.residue_set.PAD_INDEX)

    def decode(self, sequence: Peptide) -> list[str]:
        """Decode a single sequence of AA IDs."""
        # Note: Sequence is reversed as InstaNovo predicts right-to-left.
        # We reverse the sequence again when decoding to ensure the decoder outputs forward sequences.
        return self.residue_set.decode(sequence, reverse=True)  # type: ignore

    def idx_to_aa(self, idx: Peptide) -> list[str]:
        """Decode a single sample of indices to aa list."""
        idx = idx.cpu().numpy()
        t = []
        for i in idx:
            if i == self.eos_id:
                break
            if i == self.bos_id or i == self.pad_id:
                continue
            t.append(i)
        return [self.i2s[x.item()] for x in t]

    def batch_idx_to_aa(
        self, idx: Integer[Peptide, " batch"], reverse: bool
    ) -> list[list[str]]:
        """Decode a batch of indices to aa lists."""
        return [self.residue_set.decode(i, reverse=reverse) for i in idx]

    def _encoder(
        self,
        x: Float[Spectrum, " batch"],
        p: Float[PrecursorFeatures, " batch"],
        x_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
    ) -> Tuple[Float[SpectrumEmbedding, " batch"], Bool[SpectrumMask, " batch"]]:
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
        latent_mask = torch.zeros(
            (x_mask.shape[0], 1), dtype=bool, device=x_mask.device
        )
        x_mask = torch.cat([latent_mask, x_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=x_mask)

        # Prepare precursors
        masses = self.peak_encoder.encode_mass(p[:, None, [0]])
        charges = self.charge_encoder(p[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Concatenate precursors
        x = torch.cat([precursors, x], dim=1)
        prec_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
        x_mask = torch.cat([prec_mask, x_mask], dim=1)

        return x, x_mask

    def _decoder(
        self,
        x: Float[Spectrum, " batch"],
        y: Integer[Peptide, " batch"],
        x_mask: Bool[SpectrumMask, " batch"],
        y_mask: Optional[Bool[PeptideMask, " batch"]] = None,
        add_bos: bool = True,
    ) -> Float[ResidueLogits, " batch"]:
        if y is None:
            y = torch.full((x.shape[0], 1), self.residue_set.SOS_INDEX, device=x.device)
        elif add_bos:
            bos = (
                torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device)
                * self.residue_set.SOS_INDEX
            )
            y = torch.cat([bos, y], dim=1)

            if y_mask is not None:
                bos_mask = torch.zeros(
                    (y_mask.shape[0], 1), dtype=bool, device=y_mask.device
                )
                y_mask = torch.cat([bos_mask, y_mask], dim=1)

        y = self.aa_embed(y)
        if y_mask is None:
            y_mask = ~y.sum(axis=2).bool()

        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        y_hat = self.decoder(
            y,
            x,
            tgt_mask=c_mask,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
        )

        return self.head(y_hat)

    def _flash_encoder(
        self, x: Tensor, p: Tensor, x_mask: Tensor = None
    ) -> tuple[Tensor, Tensor]:
        # Special mask for zero-indices
        # One is padded, zero is normal
        x_mask = (~x.sum(dim=2).bool()).float()

        x = self.peak_encoder(x[:, :, [0]], x[:, :, [1]])
        pad_spectrum = self.pad_spectrum.expand(x.shape[0], x.shape[1], -1)

        # torch.compile doesn't allow dynamic sizes (returned by mask indexing)
        # x[x_mask] = pad_spectrum[x_mask].to(x.dtype)
        x = x * (1 - x_mask[:, :, None]) + pad_spectrum * (x_mask[:, :, None])

        # Self-attention on latent spectra AND peaks
        latent_spectra = self.latent_spectrum.expand(x.shape[0], -1, -1)
        x = torch.cat([latent_spectra, x], dim=1).contiguous()

        try:
            from torch.nn.attention import sdpa_kernel
            from torch.nn.attention import SDPBackend
        except ImportError:
            raise ImportError(
                "Training InstaNovo with Flash attention enabled requires at least pytorch v2.3. Please upgrade your pytorch version"
            )

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = self.encoder(x)

        # Prepare precursors
        masses = self.peak_encoder.encode_mass(p[:, None, [0]])
        charges = self.charge_encoder(p[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Concatenate precursors
        x = torch.cat([precursors, x], dim=1).contiguous()

        return x, None

    def _flash_decoder(
        self,
        x: Tensor,
        y: Tensor,
        x_mask: Tensor,
        y_mask: Tensor = None,
        add_bos: bool = True,
    ) -> Tensor:
        if y is None:
            y = torch.full((x.shape[0], 1), self.residue_set.SOS_INDEX, device=x.device)
        elif add_bos:
            bos = (
                torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device)
                * self.residue_set.SOS_INDEX
            )
            y = torch.cat([bos, y], dim=1)

        y = self.aa_embed(y)

        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        try:
            from torch.nn.attention import sdpa_kernel
            from torch.nn.attention import SDPBackend
        except ImportError:
            raise ImportError(
                "Training InstaNovo with Flash attention enabled requires at least pytorch v2.3. Please upgrade your pytorch version"
            )

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y_hat = self.decoder(y, x, tgt_mask=c_mask)

        return self.head(y_hat)


def _whitelist_torch_omegaconf() -> None:
    """Whitelist specific modules for loading configs from checkpoints."""
    # This is done to safeguard against arbitrary code execution from checkpoints.
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode
    from typing import Any
    from collections import defaultdict

    torch.serialization.add_safe_globals(
        [
            DictConfig,
            ContainerMetadata,
            Metadata,
            ListConfig,
            AnyNode,
            Any,  # Only used for type hinting in omegaconf.
            defaultdict,
            dict,
            list,
            int,
        ]
    )
