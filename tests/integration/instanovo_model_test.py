from __future__ import annotations

import copy
import logging
import os

import polars as pl
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from instanovo.transformer.data import TransformerDataProcessor
from instanovo.transformer.model import InstaNovo
from instanovo.transformer.predict import TransformerPredictor
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.device_handler import check_device

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_transformer_model(
    instanovo_model: tuple[InstaNovo, DictConfig],
    instanovo_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test loading an InstaNovo model and doing inference end-to-end."""
    temp_config = copy.deepcopy(instanovo_config)
    device = check_device(temp_config)
    model, config = instanovo_model

    assert model.residue_set.residue_masses == temp_config.residues["residues"]
    assert model.residue_set.residue_remapping == temp_config.residues.get("residue_remapping", {})
    assert model.vocab_size == 8

    assert temp_config.model["n_peaks"] == config["n_peaks"]
    assert temp_config.model["min_mz"] == config["min_mz"]
    assert temp_config.model["max_mz"] == config["max_mz"]
    assert temp_config.model["max_length"] == config["max_length"]

    _, data_dir = dir_paths
    df = pl.read_ipc(os.path.join(data_dir, "test.ipc"))
    sdf = SpectrumDataFrame(df=df, is_lazy=False)
    processor = TransformerDataProcessor(
        model.residue_set,
        n_peaks=config.get("n_peaks", 200),
        min_mz=config.get("min_mz", 50.0),
        max_mz=config.get("max_mz", 2500.0),
        min_intensity=config.get("min_intensity", 0.01),
        remove_precursor_tol=config.get("remove_precursor_tol", 2.0),
        return_str=False,
        use_spectrum_utils=True,
    )
    ds = sdf.to_dataset(in_memory=True)
    ds = ds.map(processor.process_row)
    ds.set_format(type="torch", columns=processor.get_expected_columns())

    assert len(ds) == 1938

    batch = ds[0]

    assert torch.allclose(
        batch["spectra"],
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
    assert batch["precursor_mz"] == 112.207876
    assert batch["precursor_charge"] == 1.0
    assert torch.allclose(
        batch["peptide"],
        torch.tensor([6, 7, 5, 5, 3, 4, 2]),
        rtol=1e-04,
    )

    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=processor.collate_fn)
    batch = next(iter(dl))

    assert torch.allclose(
        batch["spectra"],
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
        batch["precursors"],
        torch.tensor([[111.2006, 1.0000, 112.2079], [110.3506, 1.0000, 111.3579]]),
    )
    assert torch.equal(
        batch["spectra_mask"],
        torch.tensor(
            [
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False],
            ]
        ),
    )
    assert torch.allclose(batch["peptides"], torch.tensor([[6, 7, 5, 5, 3, 4, 2], [4, 3, 7, 4, 5, 7, 2]]))

    with torch.no_grad():
        output = model(
            x=batch["spectra"].to(device),
            p=batch["precursors"].to(device),
            y=batch["peptides"].to(device),
            x_mask=batch["spectra_mask"].to(device),
            y_mask=batch["peptides_mask"].to(device),
        )

    predictions = output.argmax(dim=-1).cpu()
    log_probs = torch.log_softmax(output, dim=-1).cpu()
    gathered_log_probs = log_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    sequence_log_probs = gathered_log_probs.sum(dim=-1)

    assert torch.allclose(predictions, torch.tensor([[3, 5, 5, 4, 4, 4, 2, 2], [7, 7, 7, 7, 5, 7, 2, 2]]))

    pred_str = "".join(model.residue_set.decode(predictions[0], reverse=True))
    target_str = "".join(model.residue_set.decode(batch["peptides"][0], reverse=True))

    assert pred_str == "BBBCCA"
    assert target_str == "BACCED"
    assert sequence_log_probs[0] == pytest.approx(-3.94, rel=1e-1)


def test_transformer_model_inference(
    instanovo_inference_config: DictConfig,
) -> None:
    """Test transformer model inference."""
    temp_config = copy.deepcopy(instanovo_inference_config)
    check_device(temp_config)

    predictor = TransformerPredictor(temp_config)
    predictor.predict()

    pred_df = pl.read_csv(temp_config["output_path"])

    assert temp_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df[temp_config["prediction_col"]][0] == "CADD"
    assert pred_df[temp_config["log_probs_col"]][0] == pytest.approx(-2.01, rel=1e-1)
