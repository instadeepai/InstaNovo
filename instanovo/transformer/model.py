from __future__ import annotations

import depthcharge.masses
import torch
from depthcharge.components import PeptideDecoder
from depthcharge.components import SpectrumEncoder
from depthcharge.components.encoders import MassEncoder
from depthcharge.components.encoders import PeakEncoder
from depthcharge.components.encoders import PositionalEncoder
from torch import nn
from torch import Tensor

from instanovo.transformer.layers import MultiScalePeakEmbedding
from instanovo.transformer.layers import PositionalEncoding


class InstaNovo(nn.Module):
    """The Instanovo model."""

    def __init__(
        self,
        i2s: dict[int, str],
        residues: dict[str, float],
        dim_model: int = 768,
        n_head: int = 16,
        dim_feedforward: int = 2048,
        n_layers: int = 9,
        dropout: float = 0.1,
        max_length: int = 30,
        max_charge: int = 5,
        bos_id: int = 1,
        eos_id: int = 2,
        use_depthcharge: bool = True,
        enc_type: str = "depthcharge",
        dec_type: str = "depthcharge",
        dec_precursor_sos: bool = False,
    ) -> None:
        super().__init__()
        self.i2s = i2s
        self.n_vocab = len(self.i2s)
        self.residues = residues
        self.bos_id = bos_id  # beginning of sentence ID, prepend to y
        self.eos_id = eos_id  # stop token
        self.pad_id = 0
        self.use_depthcharge = use_depthcharge

        self.enc_type = enc_type
        self.dec_type = dec_type
        self.dec_precursor_sos = dec_precursor_sos
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(self.residues)

        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        # Encoder
        if self.enc_type == "depthcharge":
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                dim_intensity=None,
            )
            if not self.dec_precursor_sos:
                self.mass_encoder = MassEncoder(dim_model)
                self.charge_encoder = nn.Embedding(max_charge, dim_model)

        else:
            if not self.use_depthcharge:
                self.peak_encoder = MultiScalePeakEmbedding(dim_model, dropout=dropout)
                self.mass_encoder = self.peak_encoder.encode_mass
            else:
                self.mass_encoder = MassEncoder(dim_model)
                self.peak_encoder = PeakEncoder(dim_model, dim_intensity=None)

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

            self.head = nn.Linear(dim_model, self.n_vocab)
            self.charge_encoder = nn.Embedding(max_charge, dim_model)

        # Decoder
        if dec_type == "depthcharge":
            self.decoder = PeptideDecoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                residues=residues,
                max_charge=max_charge,
            )

            if not dec_precursor_sos:
                del self.decoder.charge_encoder
                self.decoder.charge_encoder = lambda x: torch.zeros(
                    x.shape[0], dim_model, device=x.device
                )
                self.sos_embedding = nn.Parameter(torch.randn(1, 1, dim_model))
                del self.decoder.mass_encoder
                self.decoder.mass_encoder = lambda x: self.sos_embedding.expand(x.shape[0], -1, -1)
        else:
            self.aa_embed = nn.Embedding(self.n_vocab, dim_model, padding_idx=0)
            if not self.use_depthcharge:
                self.aa_pos_embed = PositionalEncoding(dim_model, dropout, max_len=200)
                assert max_length <= 200  # update value if necessary
            else:
                self.aa_pos_embed = PositionalEncoder(dim_model)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=dim_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
                # norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=n_layers,
            )

        if self.dec_type == "depthcharge":
            self.eos_id = self.decoder._aa2idx["$"]

    @staticmethod
    def _get_causal_mask(seq_len: int) -> Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    @classmethod
    def load(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        ckpt = torch.load(path)

        config = ckpt["config"]

        i2s = {i: v for i, v in enumerate(config["vocab"])}

        model = cls(
            i2s=i2s,
            residues=config["residues"],
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            max_length=config["max_length"],
            max_charge=config["max_charge"],
            use_depthcharge=config["use_depthcharge"],
        )
        model.load_state_dict(ckpt["model"])

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
        return self._decoder(x, p, y, x_mask, y_mask, add_bos)

    def init(
        self, x: Tensor, p: Tensor, x_mask: Tensor = None
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Initialise model encoder."""
        x, x_mask = self._encoder(x, p, x_mask)
        # y = torch.ones((x.shape[0], 1), dtype=torch.long, device=x.device) * self.bos_id
        logits, _ = self._decoder(x, p, None, x_mask, None, add_bos=False)
        return (x, x_mask), torch.log_softmax(logits[:, -1, :], -1)

    def score_candidates(
        self,
        y: torch.LongTensor,
        p: torch.FloatTensor,
        x: torch.FloatTensor,
        x_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Score a set of candidate sequences."""
        logits, _ = self._decoder(x, p, y, x_mask, None, add_bos=True)

        return torch.log_softmax(logits[:, -1, :], -1)

    def get_residue_masses(self, mass_scale: int) -> torch.LongTensor:
        """Get the scaled masses of all residues."""
        residue_masses = torch.zeros(max(self.decoder._idx2aa.keys()) + 1).type(torch.int64)
        for index, residue in self.decoder._idx2aa.items():
            if residue in self.peptide_mass_calculator.masses:
                residue_masses[index] = round(
                    mass_scale * self.peptide_mass_calculator.masses[residue]
                )
        return residue_masses

    def get_eos_index(self) -> int:
        """Get the EOS token ID."""
        return self.eos_id

    def get_empty_index(self) -> int:
        """Get the PAD token ID."""
        return 0

    def decode(self, sequence: torch.LongTensor) -> list[str]:
        """Decode a single sequence of AA IDs."""
        return self.decoder.detokenize(sequence)  # type: ignore

    def idx_to_aa(self, idx: Tensor) -> list[str]:
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

    def batch_idx_to_aa(self, idx: Tensor) -> list[list[str]]:
        """Decode a batch of indices to aa lists."""
        return [self.idx_to_aa(i) for i in idx]

    def _encoder(self, x: Tensor, p: Tensor, x_mask: Tensor = None) -> tuple[Tensor, Tensor]:
        if self.enc_type == "depthcharge":
            x, x_mask = self.encoder(x)

            if not self.dec_precursor_sos:
                # Prepare precursors
                masses = self.mass_encoder(p[:, None, [0]])
                charges = self.charge_encoder(p[:, 1].int() - 1)
                precursors = masses + charges[:, None, :]
                x = torch.cat([precursors, x], dim=1)
                prec_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
                x_mask = torch.cat([prec_mask, x_mask], dim=1)
            return x, x_mask

        # Peak encoding
        if not self.use_depthcharge:
            x = self.peak_encoder(x[:, :, [0]], x[:, :, [1]])
        else:
            x = self.peak_encoder(x)
        # x = self.peak_norm(x)

        if x_mask is None:
            x_mask = ~x.sum(dim=2).bool()

        # Prepare precursors
        masses = self.mass_encoder(p[:, None, [0]])
        charges = self.charge_encoder(p[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Self-attention on latent spectra AND peaks
        latent_spectra = self.latent_spectrum.expand(x.shape[0], -1, -1)
        x = torch.cat([latent_spectra, x], dim=1)
        latent_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
        x_mask = torch.cat([latent_mask, x_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=x_mask)

        if not self.dec_precursor_sos:
            # Concatenate precursors
            x = torch.cat([precursors, x], dim=1)
            prec_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
            x_mask = torch.cat([prec_mask, x_mask], dim=1)

        return x, x_mask

    def _decoder(
        self,
        x: Tensor,
        p: Tensor,
        y: Tensor,
        x_mask: Tensor,
        y_mask: Tensor = None,
        add_bos: bool = True,
    ) -> Tensor:
        if self.dec_type == "depthcharge":
            return self.decoder(y, p, x, x_mask)

        if add_bos:
            bos = torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device) * self.bos_id
            y = torch.cat([bos, y], dim=1)

            if y_mask is not None:
                bos_mask = torch.zeros((y_mask.shape[0], 1), dtype=bool, device=y_mask.device)
                y_mask = torch.cat([bos_mask, y_mask], dim=1)

        y = self.aa_embed(y)
        if y_mask is None:
            y_mask = y.sum(axis=2) == 0
        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        y_hat = self.decoder(
            y, x, tgt_mask=c_mask, tgt_key_padding_mask=y_mask, memory_key_padding_mask=x_mask
        )

        return self.head(y_hat)
