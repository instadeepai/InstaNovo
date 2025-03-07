from __future__ import annotations

from typing import Any

import polars as pl
import pytest
import torch

from instanovo.transformer.dataset import _clean_and_remap
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import load_ipc_shards
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.utils.data_handler import SpectrumDataFrame


def test_dataset_default_init(residue_set: Any) -> None:
    """Test spectrum dataset default initialisation."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    sdf = SpectrumDataFrame(df)

    sd = SpectrumDataset(df=sdf, residue_set=residue_set, peptide_pad_length=11)

    assert sd.n_peaks == 200
    assert sd.min_mz == 50
    assert sd.max_mz == 2500
    assert sd.min_intensity == 0.01
    assert sd.remove_precursor_tol == 2.0
    assert sd.reverse_peptide
    assert sd.annotated
    assert not sd.return_str
    assert len(sd) == 1

    spectrum, precursor_mz, precursor_charge, peptide = sd[0]

    assert torch.allclose(
        spectrum,
        torch.Tensor([[51.2500, 0.7071], [66.6000, 0.7071]]),
        rtol=1e-04,
    )
    assert precursor_mz == 35.83
    assert precursor_charge == 2
    assert torch.allclose(peptide, torch.tensor([5, 4, 3, 3, 3, 7, 6, 5, 4, 3, 2]))


def test_dataset_spec_init(residue_set: Any) -> None:
    """Test spectrum dataset specified initialisation."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    sdf = SpectrumDataFrame(df)

    sd = SpectrumDataset(
        df=sdf,
        residue_set=residue_set,
        n_peaks=100,
        min_mz=25,
        max_mz=1000,
        min_intensity=0.01,
        remove_precursor_tol=1.0,
        reverse_peptide=False,
        annotated=False,
        return_str=True,
    )

    assert sd.n_peaks == 100
    assert sd.min_mz == 25
    assert sd.max_mz == 1000
    assert sd.min_intensity == 0.01
    assert sd.remove_precursor_tol == 1.0
    assert not sd.reverse_peptide
    assert not sd.annotated
    assert sd.return_str

    assert len(sd) == 1

    _, _, _, peptide = sd[0]

    assert peptide == ""


def test_process_peaks(residue_set: Any) -> None:
    """Test spectrum preprocessing."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }

    df = pl.DataFrame(data)

    sdf = SpectrumDataFrame(df)

    sd = SpectrumDataset(
        df=sdf, residue_set=residue_set, n_peaks=10, min_mz=25, peptide_pad_length=11
    )

    assert len(sd) == 1

    proc_spectrum = sd._process_peaks(
        mz_array=torch.Tensor(df["mz_array"][0]),
        int_array=torch.Tensor(df["intensity_array"][0]),
        precursor_mz=df["precursor_mz"][0],
        precursor_charge=df["precursor_charge"][0],
    )

    assert torch.allclose(
        proc_spectrum,
        torch.Tensor(
            [
                [27.2500, 0.3780],
                [28.6400, 0.3780],
                [29.8100, 0.3780],
                [38.5500, 0.3780],
                [49.9650, 0.3780],
                [51.2500, 0.3780],
                [66.6000, 0.3780],
            ]
        ),
        rtol=1e-04,
    )

    spectrum, precursor_mz, precursor_charge, peptide = sd[0]

    assert torch.allclose(
        spectrum,
        proc_spectrum,
        rtol=1e-04,
    )
    assert precursor_mz == 35.83
    assert precursor_charge == 2
    assert torch.allclose(peptide, torch.tensor([5, 4, 3, 3, 3, 7, 6, 5, 4, 3, 2]))


def test_spectrum_errors(residue_set: Any) -> None:
    """Test spectrum dataset error catching."""
    data = {
        "mz_array": [10, 20, 30],
        "intensity_array": [0.67, 1.0, 0.5],
        "precursor_mz": [150, 250, 350],
        "precursor_charge": [1, 2, 3],
    }
    df = pl.DataFrame(data)
    sdf = SpectrumDataFrame(df)
    sd = SpectrumDataset(df=sdf, residue_set=residue_set)

    with pytest.raises(KeyError, match="sequence"):
        sd[0]


def test_dataset_collate() -> None:
    """Test spectrum dataset collate function."""
    spectra = [torch.randn(10, 2), torch.randn(8, 2), torch.randn(15, 2)]
    precursor_mzs = [100.0, 200.0, 150.0]
    precursor_charges = [1, 2, 1]
    peptides = [
        torch.tensor([1, 5, 2, 1]),
        torch.tensor([3, 5, 9, 8, 7]),
        torch.tensor([9, 7, 3]),
    ]

    batch = list(zip(spectra, precursor_mzs, precursor_charges, peptides))
    spectra, precursors, spectra_mask, peptides_tensor, peptides_mask = collate_batch(
        batch
    )
    assert spectra.shape == torch.Size([3, 15, 2])
    assert spectra_mask.shape == torch.Size([3, 15])
    if isinstance(peptides_tensor, torch.Tensor):
        assert peptides_tensor.shape == torch.Size([3, 5])
    assert peptides_mask.shape == torch.Size([3, 5])
    assert precursors.shape == torch.Size([3, 3])


def test_clean_and_remap() -> None:
    """Test clean and remap function."""
    data = {
        "Mass spectrum": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "Raw intensity spectrum": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "MS/MS m/z": [35.83],
        "Charge": [2],
        "Modified sequence": ".ABCDEFGABC.",
    }
    old_df = pl.DataFrame(data)
    new_df = _clean_and_remap(old_df)
    assert new_df.columns == [
        "modified_sequence",
        "precursor_mz",
        "precursor_charge",
        "mz_array",
        "intensity_array",
    ]


def test_load_shards(dir_paths: tuple[str, str]) -> None:
    """Test the loading of a sharded polars dataframe."""
    _, data_dir = dir_paths
    df = load_ipc_shards(data_path=data_dir, split="test")
    assert len(df) == 1938


def test_dataset_collate_diffusion(residue_set: Any) -> None:
    """Test batch collation function for diffusion."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCD",
    }

    df = pl.DataFrame(data)
    sdf = SpectrumDataFrame(df)
    sd = SpectrumDataset(
        df=sdf,
        residue_set=residue_set,
        n_peaks=10,
        min_mz=25,
        peptide_pad_length=6,
        diffusion=True,
    )

    output = collate_batch([sd[0]])

    expected_output = (
        torch.tensor(
            [
                [
                    [27.2500, 0.3780],
                    [28.6400, 0.3780],
                    [29.8100, 0.3780],
                    [38.5500, 0.3780],
                    [49.9650, 0.3780],
                    [51.2500, 0.3780],
                    [66.6000, 0.3780],
                ]
            ]
        ),
        torch.tensor([[69.6455, 2.0000, 35.8300]]),
        torch.tensor([[False, False, False, False, False, False, False]]),
        torch.tensor([[6, 5, 4, 3, 0, 0]]),
        torch.tensor(
            [[False, False, False, False, False, False]],
        ),
    )

    assert torch.allclose(output[0], expected_output[0], atol=1e-3), "Spectra mismatch"
    assert torch.allclose(
        output[1], expected_output[1], atol=1e-6
    ), "Precursors mismatch"
    assert torch.equal(output[2], expected_output[2]), "Spectra mask mismatch"
    assert torch.equal(output[3], expected_output[3]), "Peptides mismatch"
    assert torch.equal(output[4], expected_output[4]), "Peptides mask mismatch"
