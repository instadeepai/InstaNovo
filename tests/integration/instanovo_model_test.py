from __future__ import annotations

import logging
import os
from typing import Any

import polars as pl
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from instanovo.transformer.dataset import SpectrumDataset, collate_batch
from instanovo.transformer.predict import get_preds
from instanovo.utils.data_handler import SpectrumDataFrame

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_model(
    instanovo_model: tuple[Any, Any],
    instanovo_config: DictConfig,
    dir_paths: tuple[str, str],
    instanovo_inference_config: DictConfig,
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

    _, data_dir = dir_paths
    df = pl.read_ipc(os.path.join(data_dir, "test.ipc"))
    sdf = SpectrumDataFrame(df=df)
    sd = SpectrumDataset(
        df=sdf,
        residue_set=model.residue_set,
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        peptide_pad_length=config["max_length"],
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
    assert torch.allclose(peptides, torch.tensor([[6, 7, 5, 5, 3, 4, 2], [4, 3, 7, 4, 5, 7, 2]]))

    instanovo_inference_config["device"] = device

    get_preds(
        config=instanovo_inference_config,
        model=model,
        model_config=config,
    )

    pred_df = pl.read_csv(instanovo_inference_config["output_path"])

    assert instanovo_inference_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["preds"][0] == "CADD"
    assert pred_df["log_probs"][0] == pytest.approx(-2.01, rel=1e-1)
