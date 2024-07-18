from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from instanovo.inference.beam_search import BeamSearchDecoder
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset


@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
@pytest.mark.usefixtures("_reset_seed")
def test_model(
    instanovo_model: tuple[Any, Any],
    instanovo_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test loading an InstaNovo model and doing inference end-to-end."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = instanovo_model

    assert model.residue_set.residue_masses == instanovo_config["residues"]
    assert model.residue_set.residue_remapping == instanovo_config["residue_remapping"]
    assert model.vocab_size == 8

    assert instanovo_config["n_peaks"] == config["n_peaks"]
    assert instanovo_config["min_mz"] == config["min_mz"]
    assert instanovo_config["max_mz"] == config["max_mz"]
    assert instanovo_config["max_length"] == config["max_length"]

    root_dir, data_dir = dir_paths
    df = pl.read_ipc(os.path.join(data_dir, "test.ipc"))
    sd = SpectrumDataset(
        df=df,
        residue_set=model.residue_set,
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
    )
    assert len(sd) == 1938

    spectrum, precursor_mz, precursor_charge, peptide = sd[0]

    assert torch.allclose(
        spectrum,
        torch.Tensor(
            [
                [13.0941, 0.2582],
                [13.9275, 0.3651],
                [17.4275, 0.3651],
                [22.4308, 0.2582],
                [22.6541, 0.3651],
                [27.6575, 0.2582],
                [41.3176, 0.3651],
                [49.5979, 0.2582],
                [91.4579, 0.2582],
                [93.9579, 0.3651],
            ]
        ),
        rtol=1.5e-04,
    )
    assert precursor_mz == 112.207876
    assert precursor_charge == 1.0
    assert torch.allclose(
        peptide,
        torch.tensor([6, 7, 5, 5, 3, 4, 2]),
        rtol=1e-04,
    )

    dl = DataLoader(sd, batch_size=2, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(dl))
    spectra, precursors, spectra_mask, peptides, _ = batch

    assert torch.allclose(
        spectra,
        torch.tensor(
            [
                [
                    [13.0941, 0.2582],
                    [13.9275, 0.3651],
                    [17.4275, 0.3651],
                    [22.4308, 0.2582],
                    [22.6541, 0.3651],
                    [27.6575, 0.2582],
                    [41.3176, 0.3651],
                    [49.5979, 0.2582],
                    [91.4579, 0.2582],
                    [93.9579, 0.3651],
                ],
                [
                    [20.3876, 0.2582],
                    [24.0176, 0.3651],
                    [25.6376, 0.2582],
                    [31.3479, 0.3651],
                    [40.5576, 0.3651],
                    [42.1776, 0.2582],
                    [50.0176, 0.2582],
                    [62.5979, 0.2582],
                    [67.7779, 0.3651],
                    [90.6079, 0.3651],
                ],
            ]
        ),
        rtol=1.5e-04,
    )
    assert torch.allclose(
        precursors,
        torch.tensor([[111.2006, 1.0000, 112.2079], [110.3506, 1.0000, 111.3579]]),
    )
    assert torch.equal(
        spectra_mask,
        torch.tensor(
            [
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False],
            ]
        ),
    )
    assert torch.allclose(
        peptides, torch.tensor([[6, 7, 5, 5, 3, 4, 2], [4, 3, 7, 4, 5, 7, 2]])
    )

    spectra = spectra.to(device)
    precursors = precursors.to(device)

    model = model.to(device)
    model.eval()
    decoder = BeamSearchDecoder(model=model)

    with torch.no_grad():
        p = decoder.decode(
            spectra=spectra,
            precursors=precursors,
            beam_size=config["n_beams"],
            max_length=config["max_length"],
        )
    preds = ["".join(x.sequence) if not isinstance(x, list) else "" for x in p]
    probs = [x.sequence_log_probability if not isinstance(x, list) else -1 for x in p]

    assert preds == ["DCECBA", "AEBEBC"]
    assert np.allclose(probs, [-5.17131233215332, -4.346163272857666], rtol=1e-2)
