from __future__ import annotations

import os
from typing import Any

import pandas as pd
import polars as pl
import pytest
import torch

from instanovo.transformer.dataset import _clean_and_remap
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import load_ipc_shards
from instanovo.transformer.dataset import SpectrumDataset


@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
def test_dataset_default_init(residue_set: Any) -> None:
    """Test spectrum dataset default initialisation."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "modified_sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    sd = SpectrumDataset(df=df, residue_set=residue_set)

    assert sd.n_peaks == 200
    assert sd.min_mz == 50
    assert sd.max_mz == 2500
    assert sd.min_intensity == 0.01
    assert sd.remove_precursor_tol == 2.0
    assert sd.reverse_peptide
    assert sd.annotated
    assert not sd.return_str
    assert sd.data_type == "pl"

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


@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
def test_dataset_spec_init(residue_set: Any) -> None:
    """Test spectrum dataset specified initialisation."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "modified_sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    sd = SpectrumDataset(
        df=df,
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
    assert sd.data_type == "pl"

    assert len(sd) == 1

    _, _, _, peptide = sd[0]

    assert peptide == ""


@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
def test_process_peaks_with_pandas(residue_set: Any) -> None:
    """Test spectrum dataset default initialisation."""
    data = {
        "mz_array": [
            [7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]
        ],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "modified_sequence": "ABCDEAAABC",
    }

    df = pd.DataFrame(data)

    sd = SpectrumDataset(df=df, residue_set=residue_set, n_peaks=10, min_mz=25)
    assert sd.data_type == "pd"

    assert len(sd) == 1

    row = df.iloc[0]
    proc_spectrum = sd._process_peaks(
        mz_array=torch.Tensor(row["mz_array"]),
        int_array=torch.Tensor(row["intensity_array"]),
        precursor_mz=row["precursor_mz"],
        precursor_charge=row["precursor_charge"],
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


@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
def test_dataset_file(residue_set: Any) -> None:
    """Test spectrum dataset error catching."""
    data = [1, 1, 1]

    with pytest.raises(Exception, match=r"Unsupported data type <class 'list'>"):
        _ = SpectrumDataset(df=data, residue_set=residue_set)

    data = {
        "mz_array": [10, 20, 30],
        "intensity_array": [0.67, 1.0, 0.5],
        "precursor_mz": [150, 250, 350],
        "precursor_charge": [1, 2, 3],
    }
    df = pl.DataFrame(data)

    with pytest.raises(
        ValueError, match="Column missing! Missing column: modified_sequence"
    ):
        _ = SpectrumDataset(df=df, residue_set=residue_set)


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


@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="Google application credentials are required to run this test.",
)
@pytest.mark.usefixtures("_get_gcp_test_bucket")
def test_load_shards(dir_paths: tuple[str, str]) -> None:
    """A pytest fixture to check the loading of a sharded polards dataframe."""
    root_dir, data_dir = dir_paths
    df = load_ipc_shards(data_path=data_dir, split="test")
    assert len(df) == 1938
