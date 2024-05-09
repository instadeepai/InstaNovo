from __future__ import annotations

import torch
from torch import nn
from torch import Tensor

from instanovo.constants import MAX_SEQUENCE_LENGTH
from instanovo.transformer.layers import MultiScalePeakEmbedding
from instanovo.transformer.layers import PositionalEncoding
from instanovo.utils.residues import ResidueSet


class InstaNovo(nn.Module):
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
    ) -> None:
        super().__init__()
        self.residue_set = residue_set
        self.vocab_size = len(residue_set)

        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        # Encoder
        self.peak_encoder = MultiScalePeakEmbedding(dim_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Decoder
        self.aa_embed = nn.Embedding(self.vocab_size, dim_model, padding_idx=0)

        self.aa_pos_embed = PositionalEncoding(dim_model, dropout, max_len=MAX_SEQUENCE_LENGTH)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        self.head = nn.Linear(dim_model, self.vocab_size)
        self.charge_encoder = nn.Embedding(max_charge, dim_model)

    @staticmethod
    def _get_causal_mask(seq_len: int) -> Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    @classmethod
    def load(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location="cpu")

        config = ckpt["config"]

        # check if PTL checkpoint
        if all([x.startswith("model") for x in ckpt["state_dict"].keys()]):
            ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}

        residue_set = ResidueSet(config["residues"])

        model = cls(
            residue_set=residue_set,
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            max_charge=config["max_charge"],
        )
        model.load_state_dict(ckpt["state_dict"])

        return model, config

    def forward(
        self,
        x: Tensor,
        p: Tensor,
        y: Tensor,
        x_mask: Tensor = None,
        y_mask: Tensor = None,
        add_bos: bool = True,
    ) -> Tensor:
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
        x, x_mask = self._encoder(x, p, x_mask)
        return self._decoder(x, y, x_mask, y_mask, add_bos)

    def init(
        self, x: Tensor, p: Tensor, x_mask: Tensor = None
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Initialise model encoder."""
        x, x_mask = self._encoder(x, p, x_mask)
        logits = self._decoder(x, None, x_mask, None, add_bos=False)
        return (x, x_mask), torch.log_softmax(logits[:, -1, :], -1)

    def score_candidates(
        self,
        y: torch.LongTensor,
        p: torch.FloatTensor,
        x: torch.FloatTensor,
        x_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Score a set of candidate sequences."""
        logits = self._decoder(x, y, x_mask, None, add_bos=True)

        return torch.log_softmax(logits[:, -1, :], -1)

    def get_residue_masses(self, mass_scale: int) -> torch.LongTensor:
        """Get the scaled masses of all residues."""
        residue_masses = torch.zeros(len(self.residue_set), dtype=torch.int64)
        for index, residue in self.residue_set.index_to_residue.items():
            if residue in self.residue_set.residue_masses:
                residue_masses[index] = round(mass_scale * self.residue_set.get_mass(residue))
        return residue_masses

    def get_eos_index(self) -> int:
        """Get the EOS token ID."""
        return int(self.residue_set.EOS_INDEX)

    def get_empty_index(self) -> int:
        """Get the PAD token ID."""
        return int(self.residue_set.PAD_INDEX)

    def decode(self, sequence: torch.LongTensor) -> list[str]:
        """Decode a single sequence of AA IDs."""
        # return self.decoder.detokenize(sequence)  # type: ignore
        # Note: Sequence is reversed as InstaNovo predicts right-to-left.
        # We reverse the sequence again when decoding to ensure the decoder outputs forward sequences.
        return self.residue_set.decode(sequence, reverse=True)  # type: ignore

    def batch_idx_to_aa(self, idx: torch.LongTensor, reverse: bool = False) -> list[list[str]]:
        """Decode a batch of indices to aa lists."""
        return [self.residue_set.decode(i, reverse=reverse) for i in idx]

    def _encoder(self, x: Tensor, p: Tensor, x_mask: Tensor = None) -> tuple[Tensor, Tensor]:
        if x_mask is None:
            x_mask = ~x.sum(dim=2).bool()

        x = self.peak_encoder(x[:, :, [0]], x[:, :, [1]])

        # Self-attention on latent spectra AND peaks
        latent_spectra = self.latent_spectrum.expand(x.shape[0], -1, -1)
        x = torch.cat([latent_spectra, x], dim=1)
        latent_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
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

            if y_mask is not None:
                bos_mask = torch.zeros((y_mask.shape[0], 1), dtype=bool, device=y_mask.device)
                y_mask = torch.cat([bos_mask, y_mask], dim=1)

        y = self.aa_embed(y)
        if y_mask is None:
            y_mask = ~y.sum(axis=2).bool()

        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        y_hat = self.decoder(
            y, x, tgt_mask=c_mask, tgt_key_padding_mask=y_mask, memory_key_padding_mask=x_mask
        )

        return self.head(y_hat)
