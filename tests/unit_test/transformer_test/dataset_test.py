from __future__ import annotations

import logging
from typing import Any

import polars as pl
import pytest
import torch

# from instanovo.common.dataset import (
#     SpectrumDataset,
#     _clean_and_remap,
#     collate_batch,
#     load_ipc_shards,
# )
from instanovo.transformer.data import TransformerDataProcessor
from instanovo.utils.data_handler import SpectrumDataFrame

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_dataset_default_init(residue_set: Any) -> None:
    """Test spectrum dataset default initialisation."""
    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    sdf = SpectrumDataFrame(df)
    ds = sdf.to_dataset(in_memory=True)

    processor = TransformerDataProcessor(residue_set)

    assert processor.n_peaks == 200
    assert processor.min_mz == 50
    assert processor.max_mz == 2500
    assert processor.min_intensity == 0.01
    assert processor.remove_precursor_tol == 2.0
    assert processor.reverse_peptide
    assert processor.annotated
    assert not processor.return_str

    ds = processor.process_dataset(ds)
    assert len(ds) == 1

    batch = ds[0]

    assert torch.allclose(
        batch["spectra"],
        torch.Tensor([[51.2500, 0.7071], [66.6000, 0.7071]]),
        rtol=1e-04,
    )
    assert batch["precursor_mz"] == 35.83
    assert batch["precursor_charge"] == 2
    assert torch.allclose(batch["peptide"], torch.tensor([5, 4, 3, 3, 3, 7, 6, 5, 4, 3, 2]))


def test_dataset_spec_init(residue_set: Any) -> None:
    """Test spectrum dataset specified initialisation."""
    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    sdf = SpectrumDataFrame(df)
    ds = sdf.to_dataset(in_memory=True)

    processor = TransformerDataProcessor(
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

    assert processor.n_peaks == 100
    assert processor.min_mz == 25
    assert processor.max_mz == 1000
    assert processor.min_intensity == 0.01
    assert processor.remove_precursor_tol == 1.0
    assert not processor.reverse_peptide
    assert not processor.annotated
    assert processor.return_str

    ds = ds.map(processor.process_row)
    ds.set_format(type="torch", columns=processor.get_expected_columns())
    assert len(ds) == 1

    batch = ds[0]

    # No peptide when annotated is False
    assert "peptide" not in batch


def test_process_peaks(residue_set: Any) -> None:
    """Test spectrum preprocessing."""
    data = {
        "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
        "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "precursor_mz": [35.83],
        "precursor_charge": [2],
        "sequence": "ABCDEAAABC",
    }
    df = pl.DataFrame(data)

    sdf = SpectrumDataFrame(df)
    ds = sdf.to_dataset(in_memory=True)

    processor = TransformerDataProcessor(residue_set=residue_set, n_peaks=10, min_mz=25, use_spectrum_utils=True)

    assert len(ds) == 1

    # With spectrum utils
    proc_spectrum = processor._process_spectrum(
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

    # Without spectrum utils
    processor.use_spectrum_utils = False
    proc_spectrum = processor._process_spectrum(
        mz_array=torch.Tensor(df["mz_array"][0]),
        int_array=torch.Tensor(df["intensity_array"][0]),
        precursor_mz=df["precursor_mz"][0],
        precursor_charge=df["precursor_charge"][0],
    )

    assert torch.allclose(
        proc_spectrum,
        torch.Tensor(
            [
                [28.6400, 0.3780],
                [66.6000, 0.3780],
                [38.5500, 0.3780],
                [29.8100, 0.3780],
                [49.9650, 0.3780],
                [51.2500, 0.3780],
                [27.2500, 0.3780],
            ]
        ),
        rtol=1e-04,
    )

    ds = ds.map(processor.process_row)
    ds.set_format(type="torch", columns=processor.get_expected_columns())

    batch = ds[0]

    assert torch.allclose(
        batch["spectra"],
        proc_spectrum,
        rtol=1e-04,
    )
    assert batch["precursor_mz"] == 35.83
    assert batch["precursor_charge"] == 2
    assert torch.allclose(batch["peptide"], torch.tensor([5, 4, 3, 3, 3, 7, 6, 5, 4, 3, 2]))


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

    processor = TransformerDataProcessor(residue_set=residue_set, annotated=True, use_spectrum_utils=False)
    ds = sdf.to_dataset(in_memory=True)

    with pytest.raises(KeyError, match="sequence"):
        ds = ds.map(processor.process_row)


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

    processor = TransformerDataProcessor(None)  # type: ignore

    batch = [
        {
            "spectra": spectra[i],
            "precursor_mz": precursor_mzs[i],
            "precursor_charge": precursor_charges[i],
            "peptide": peptides[i],
        }
        for i in range(len(spectra))
    ]

    batch = processor.collate_fn(batch)
    assert batch["spectra"].shape == torch.Size([3, 15, 2])
    assert batch["spectra_mask"].shape == torch.Size([3, 15])
    if isinstance(batch["peptides"], torch.Tensor):
        assert batch["peptides"].shape == torch.Size([3, 5])
    assert batch["peptides_mask"].shape == torch.Size([3, 5])
    assert batch["precursors"].shape == torch.Size([3, 3])


# def test_dataset_collate_diffusion(residue_set: Any) -> None:
#     """Test batch collation function for diffusion."""
#     data = {
#         "mz_array": [[7.84, 18.215, 20.8, 28.64, 66.6, 38.55, 29.81, 49.965, 51.25, 27.25]],
#         "intensity_array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
#         "precursor_mz": [35.83],
#         "precursor_charge": [2],
#         "sequence": "ABCD",
#     }

#     df = pl.DataFrame(data)
#     sdf = SpectrumDataFrame(df)
#     sd = SpectrumDataset(
#         df=sdf,
#         residue_set=residue_set,
#         n_peaks=10,
#         min_mz=25,
#         peptide_pad_length=6,
#         add_eos=False,
#     )

#     output = collate_batch([sd[0]])

#     expected_output = (
#         torch.tensor(
#             [
#                 [
#                     [27.2500, 0.3780],
#                     [28.6400, 0.3780],
#                     [29.8100, 0.3780],
#                     [38.5500, 0.3780],
#                     [49.9650, 0.3780],
#                     [51.2500, 0.3780],
#                     [66.6000, 0.3780],
#                 ]
#             ]
#         ),
#         torch.tensor([[69.6455, 2.0000, 35.8300]]),
#         torch.tensor([[False, False, False, False, False, False, False]]),
#         torch.tensor([[6, 5, 4, 3, 0, 0]]),
#         torch.tensor(
#             [[False, False, False, False, False, False]],
#         ),
#     )

#     assert torch.allclose(output[0], expected_output[0], atol=1e-3), "Spectra mismatch"
#     assert torch.allclose(output[1], expected_output[1], atol=1e-6), "Precursors mismatch"
#     assert torch.equal(output[2], expected_output[2]), "Spectra mask mismatch"
#     assert torch.equal(output[3], expected_output[3]), "Peptides mismatch"
#     assert torch.equal(output[4], expected_output[4]), "Peptides mask mismatch"
