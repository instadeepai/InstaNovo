import copy

import polars as pl
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
from instanovo.diffusion.predict import get_preds
from instanovo.inference.diffusion import DiffusionDecoder
from instanovo.transformer.dataset import SpectrumDataset, collate_batch
from instanovo.utils.data_handler import SpectrumDataFrame
from tests.conftest import reset_seed


@pytest.mark.usefixtures("_reset_seed")
def test_model(
    instanovoplus_model: tuple[InstaNovoPlus, DiffusionDecoder],
    instanovoplus_config: DictConfig,
    dir_paths: tuple[str, str],
    instanovoplus_inference_config: DictConfig,
    instanovo_output_path: str,
) -> None:
    """Test loading an InstaNovo+ model and doing inference end-to-end."""
    temp_inference_config = copy.deepcopy(instanovoplus_inference_config)

    diffusion_model, _ = instanovoplus_model

    assert diffusion_model.residues.residue_masses == instanovoplus_config["residues"]
    assert (
        diffusion_model.residues.residue_remapping
        == instanovoplus_config["residue_remapping"]
    )
    assert diffusion_model.config["vocab_size"] == 8

    assert instanovoplus_config["n_peaks"] == diffusion_model.config["n_peaks"]
    assert instanovoplus_config["min_mz"] == diffusion_model.config["min_mz"]
    assert instanovoplus_config["max_mz"] == diffusion_model.config["max_mz"]
    assert instanovoplus_config["max_length"] == diffusion_model.config["max_length"]

    root_dir, data_dir = dir_paths
    sdf = SpectrumDataFrame(
        file_paths=[data_dir + "/test.ipc"], shuffle=False, is_lazy=False
    )

    sd = SpectrumDataset(
        df=sdf,
        residue_set=diffusion_model.residues,
        n_peaks=diffusion_model.config["n_peaks"],
        min_mz=diffusion_model.config["min_mz"],
        max_mz=diffusion_model.config["max_mz"],
        peptide_pad_length=diffusion_model.config["max_length"],
        diffusion=True,
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
        torch.tensor([6, 7, 5, 5, 3, 4]),
        rtol=1e-04,
    )

    dl = DataLoader(
        sd, batch_size=2, num_workers=0, shuffle=False, collate_fn=collate_batch
    )
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
        peptides, torch.tensor([[6, 7, 5, 5, 3, 4], [4, 3, 7, 4, 5, 7]])
    )

    get_preds(
        config=temp_inference_config,
        model=diffusion_model,
        model_config=diffusion_model.config,
    )

    pred_df = pl.read_csv(temp_inference_config["output_path"])

    assert temp_inference_config["subset"] == 1
    assert (
        temp_inference_config["instanovo_plus_model"]
        == "./tests/instanovo_test_resources/instanovoplus"
    )
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["diffusion_predictions"][0] == "EADCAD"
    assert (
        pred_df["diffusion_predictions_tokenised"][0]
        == "['E', 'A', 'D', 'C', 'A', 'D']"
    )
    assert pred_df["diffusion_log_probabilities"][0] == pytest.approx(-0.267, rel=1e-1)

    reset_seed()

    # InstaNovo+ refinement case
    temp_inference_config["refine"] = True

    temp_inference_config["instanovo_predictions_path"] = instanovo_output_path

    with pytest.raises(
        ValueError,
        match="All rows were dropped from the dataframe. No ID matches / predictions to refine were present.",
    ):
        get_preds(
            config=temp_inference_config,
            model=diffusion_model,
            model_config=diffusion_model.config,
        )

    reset_seed()

    temp_inference_config["instanovo_predictions_path"] = (
        root_dir + "/instanovoplus/sample_pred.csv"
    )

    get_preds(
        config=temp_inference_config,
        model=diffusion_model,
        model_config=diffusion_model.config,
    )

    pred_df = pl.read_csv(temp_inference_config["output_path"])

    assert temp_inference_config["subset"] == 1
    assert pred_df["targets"][0] == "DDCA"
    assert pred_df["diffusion_predictions"][0] == "BEAAAD"
    assert (
        pred_df["diffusion_predictions_tokenised"][0]
        == "['B', 'E', 'A', 'A', 'A', 'D']"
    )
    assert pred_df["diffusion_log_probabilities"][0] == pytest.approx(-0.402, rel=1e-1)
