from __future__ import annotations

import os
from typing import Any

import polars as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from instanovo.constants import MASS_SCALE
from instanovo.transformer.data import TransformerDataProcessor
from instanovo.transformer.layers import MultiScalePeakEmbedding, PositionalEncoding
from instanovo.transformer.model import InstaNovo
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.residues import ResidueSet


def test_model_init(instanovo_config: DictConfig, dir_paths: tuple[Any, Any]) -> None:
    """Test InstaNovo model initialisation and methods."""
    residue_set = ResidueSet(
        residue_masses=instanovo_config.residues["residues"],
        residue_remapping=instanovo_config.residues.get("residue_remapping", {}),
    )

    model = InstaNovo(
        residue_set=residue_set,
        dim_model=instanovo_config.model["dim_model"],
        n_head=instanovo_config.model["n_head"],
        dim_feedforward=instanovo_config.model["dim_feedforward"],
        encoder_layers=instanovo_config.model["n_layers"],
        decoder_layers=instanovo_config.model["n_layers"],
        dropout=instanovo_config.model["dropout"],
        max_charge=instanovo_config.model["max_charge"],
    )

    assert model.residue_set.residue_masses == instanovo_config.residues["residues"]
    assert model.residue_set.residue_remapping == instanovo_config.residues.get("residue_remapping", {})
    assert model.vocab_size == 8

    assert isinstance(model.latent_spectrum, nn.Parameter)
    assert isinstance(model.peak_encoder, MultiScalePeakEmbedding)
    assert isinstance(model.encoder, nn.TransformerEncoder)
    assert isinstance(model.aa_embed, nn.Embedding)
    assert isinstance(model.aa_pos_embed, PositionalEncoding)
    assert isinstance(model.decoder, nn.TransformerDecoder)
    assert isinstance(model.head, nn.Linear)
    assert isinstance(model.charge_encoder, nn.Embedding)

    assert model._get_causal_mask(seq_len=instanovo_config.model["max_length"]).shape == torch.Size([6, 6])

    _, data_dir = dir_paths
    df = pl.read_ipc(os.path.join(data_dir, "test.ipc"))
    sdf = SpectrumDataFrame(df=df, is_annotated=True)
    ds = sdf.to_dataset(in_memory=True)
    processor = TransformerDataProcessor(
        residue_set=residue_set,
        n_peaks=instanovo_config.model["n_peaks"],
        min_mz=instanovo_config.model["min_mz"],
        max_mz=instanovo_config.model["max_mz"],
    )
    ds = ds.map(processor.process_row)
    ds.set_format(type="torch", columns=processor.get_expected_columns())

    batch = [ds[i] for i in range(3)]
    # spectra, precursors, spectra_mask, peptides, peptides_mask = collate_batch(batch)
    batch = processor.collate_fn(batch)

    assert batch["spectra"].shape == torch.Size([3, 10, 2])
    assert batch["precursors"].shape == torch.Size([3, 3])
    assert batch["spectra_mask"].shape == torch.Size([3, 10])

    if isinstance(batch["peptides"], torch.Tensor):
        assert batch["peptides"].shape == torch.Size([3, 7])
    assert batch["peptides_mask"].shape == torch.Size([3, 7])

    assert model.forward(x=batch["spectra"], p=batch["precursors"], y=batch["peptides"], add_bos=True).shape == torch.Size([3, 8, 8])

    (x, x_mask), probs = model.init(spectra=batch["spectra"], precursors=batch["precursors"], spectra_mask=batch["spectra_mask"])

    assert x.shape == torch.Size([3, 12, 320])
    assert x_mask.shape == torch.Size([3, 12])
    assert probs.shape == torch.Size([3, 8])

    assert model.score_candidates(
        sequences=batch["peptides"],
        precursor_mass_charge=batch["precursors"],
        spectra=x,
        spectra_mask=x_mask,
    ).shape == torch.Size([3, 8])

    assert torch.allclose(
        model.get_residue_masses(MASS_SCALE),
        torch.tensor([0, 0, 0, 105000, 207500, 156800, 182500, 123300]),
    )

    assert model.get_eos_index() == 2
    assert model.get_empty_index() == 0
