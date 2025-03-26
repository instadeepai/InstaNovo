from __future__ import annotations

import os
from typing import Any

import polars as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from instanovo.constants import MASS_SCALE
from instanovo.transformer.dataset import SpectrumDataset, collate_batch
from instanovo.transformer.layers import MultiScalePeakEmbedding, PositionalEncoding
from instanovo.transformer.model import InstaNovo
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.residues import ResidueSet


def test_model_init(instanovo_config: DictConfig, dir_paths: tuple[Any, Any]) -> None:
    """Test InstaNovo model initialisation and methods."""
    residue_set = ResidueSet(
        residue_masses=instanovo_config["residues"],
        residue_remapping=instanovo_config["residue_remapping"],
    )

    model = InstaNovo(
        residue_set=residue_set,
        dim_model=instanovo_config["dim_model"],
        n_head=instanovo_config["n_head"],
        dim_feedforward=instanovo_config["dim_feedforward"],
        n_layers=instanovo_config["n_layers"],
        dropout=instanovo_config["dropout"],
        max_charge=instanovo_config["max_charge"],
    )

    assert model.residue_set.residue_masses == instanovo_config["residues"]
    assert model.residue_set.residue_remapping == instanovo_config["residue_remapping"]
    assert model.vocab_size == 8

    assert isinstance(model.latent_spectrum, nn.Parameter)
    assert isinstance(model.peak_encoder, MultiScalePeakEmbedding)
    assert isinstance(model.encoder, nn.TransformerEncoder)
    assert isinstance(model.aa_embed, nn.Embedding)
    assert isinstance(model.aa_pos_embed, PositionalEncoding)
    assert isinstance(model.decoder, nn.TransformerDecoder)
    assert isinstance(model.head, nn.Linear)
    assert isinstance(model.charge_encoder, nn.Embedding)

    assert model._get_causal_mask(seq_len=instanovo_config["max_length"]).shape == torch.Size(
        [6, 6]
    )

    _, data_dir = dir_paths
    df = pl.read_ipc(os.path.join(data_dir, "test.ipc"))
    sdf = SpectrumDataFrame(df=df, is_annotated=True)
    sd = SpectrumDataset(
        df=sdf,
        residue_set=residue_set,
        n_peaks=instanovo_config["n_peaks"],
        min_mz=instanovo_config["min_mz"],
        max_mz=instanovo_config["max_mz"],
        peptide_pad_length=instanovo_config["max_length"],
    )

    batch = list(
        zip(
            [sd[i][0] for i in range(3)],
            [sd[i][1] for i in range(3)],
            [sd[i][2] for i in range(3)],
            [sd[i][3] for i in range(3)],
            strict=True,
        )
    )
    spectra, precursors, spectra_mask, peptides, peptides_mask = collate_batch(batch)

    assert spectra.shape == torch.Size([3, 10, 2])
    assert precursors.shape == torch.Size([3, 3])
    assert spectra_mask.shape == torch.Size([3, 10])

    if isinstance(peptides, torch.Tensor):
        assert peptides.shape == torch.Size([3, 7])
    assert peptides_mask.shape == torch.Size([3, 7])

    assert model.forward(x=spectra, p=precursors, y=peptides, add_bos=True).shape == torch.Size(
        [3, 8, 8]
    )

    (x, x_mask), probs = model.init(
        spectra=spectra, precursors=precursors, spectra_mask=spectra_mask
    )

    assert x.shape == torch.Size([3, 12, 320])
    assert x_mask.shape == torch.Size([3, 12])
    assert probs.shape == torch.Size([3, 8])

    assert model.score_candidates(
        sequences=peptides,
        precursor_mass_charge=precursors,
        spectra=x,
        spectra_mask=x_mask,
    ).shape == torch.Size([3, 8])

    assert torch.allclose(
        model.get_residue_masses(MASS_SCALE),
        torch.tensor([0, 0, 0, 105000, 207500, 156800, 182500, 123300]),
    )

    assert model.get_eos_index() == 2
    assert model.get_empty_index() == 0
