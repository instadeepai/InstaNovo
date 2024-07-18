from __future__ import annotations

import os
from typing import Any

import polars as pl
import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

from instanovo.constants import MASS_SCALE
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.layers import MultiScalePeakEmbedding
from instanovo.transformer.layers import PositionalEncoding
from instanovo.transformer.model import InstaNovo
from instanovo.utils.residues import ResidueSet


@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
@pytest.mark.usefixtures("_get_gcp_test_bucket")
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

    assert model._get_causal_mask(
        seq_len=instanovo_config["max_length"]
    ).shape == torch.Size([6, 6])

    root_dir, data_dir = dir_paths
    df = pl.read_ipc(os.path.join(data_dir, "test.ipc"))
    sd = SpectrumDataset(
        df=df,
        residue_set=residue_set,
        n_peaks=instanovo_config["n_peaks"],
        min_mz=instanovo_config["min_mz"],
        max_mz=instanovo_config["max_mz"],
    )

    batch = list(
        zip(
            [sd[i][0] for i in range(3)],
            [sd[i][1] for i in range(3)],
            [sd[i][2] for i in range(3)],
            [sd[i][3] for i in range(3)],
        )
    )
    spectra, precursors, spectra_mask, peptides, peptides_mask = collate_batch(batch)

    assert spectra.shape == torch.Size([3, 10, 2])
    assert precursors.shape == torch.Size([3, 3])
    assert spectra_mask.shape == torch.Size([3, 10])
    assert peptides.shape == torch.Size([3, 7])
    assert peptides_mask.shape == torch.Size([3, 7])

    assert model.forward(
        x=spectra, p=precursors, y=peptides, add_bos=True
    ).shape == torch.Size([3, 8, 8])

    (x, x_mask), probs = model.init(x=spectra, p=precursors, x_mask=spectra_mask)

    assert x.shape == torch.Size([3, 12, 320])
    assert x_mask.shape == torch.Size([3, 12])
    assert probs.shape == torch.Size([3, 8])

    assert model.score_candidates(
        y=peptides, p=precursors, x=x, x_mask=x_mask
    ).shape == torch.Size([3, 8])

    assert torch.allclose(
        model.get_residue_masses(MASS_SCALE),
        torch.tensor([0, 0, 0, 105000, 207500, 156800, 182500, 123300]),
    )

    assert model.get_eos_index() == 2
    assert model.get_empty_index() == 0
