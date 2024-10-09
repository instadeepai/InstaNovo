from __future__ import annotations

import os
from typing import Any

import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig

from instanovo.diffusion.dataset import AnnotatedPolarsSpectrumDataset
from instanovo.diffusion.dataset import AnnotatedSpectrumBatch
from instanovo.diffusion.dataset import collate_batches
from instanovo.diffusion.dataset import PolarsSpectrumDataset
from instanovo.diffusion.dataset import SpectrumBatch


def test_spectrum_batch() -> None:
    """Test spectrum batch initialisation."""
    spectra = torch.tensor(
        [[[1.0, 1.0], [3.0, 0.7], [2.0, 0.75]], [[4.0, 0.3], [14.0, 0.35], [2.0, 0.9]]]
    )
    spectra_padding_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, True]],
            [[True, True], [True, True], [True, True]],
        ]
    )
    precursors = torch.tensor([100.0, 200.0])

    sb = SpectrumBatch(
        spectra=spectra,
        spectra_padding_mask=spectra_padding_mask,
        precursors=precursors,
    )

    assert torch.equal(sb.spectra, spectra)
    assert torch.equal(sb.spectra_padding_mask, spectra_padding_mask)
    assert torch.equal(sb.precursors, precursors)


def test_annotated_spectrum_batch() -> None:
    """Test annotated spectrum batch initialisation."""
    spectra = torch.tensor(
        [[[1.0, 1.0], [3.0, 0.7], [2.0, 0.75]], [[4.0, 0.3], [14.0, 0.35], [2.0, 0.9]]]
    )
    spectra_padding_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, True]],
            [[True, True], [True, True], [True, True]],
        ]
    )
    precursors = torch.tensor([100.0, 200.0])
    peptides = torch.tensor([[1, 2, 3], [4, 5, 6]])
    peptide_padding_mask = torch.tensor([[True, True, False], [True, True, True]])

    asb = AnnotatedSpectrumBatch(
        spectra=spectra,
        spectra_padding_mask=spectra_padding_mask,
        precursors=precursors,
        peptides=peptides,
        peptide_padding_mask=peptide_padding_mask,
    )

    assert torch.equal(asb.spectra, spectra)
    assert torch.equal(asb.spectra_padding_mask, spectra_padding_mask)
    assert torch.equal(asb.precursors, precursors)
    assert torch.equal(asb.peptides, peptides)
    assert torch.equal(asb.peptide_padding_mask, peptide_padding_mask)


def test_polar_spectrum(dir_paths: tuple[str, str]) -> None:
    """Test polars spectrum dataset."""
    _, data_dir = dir_paths
    diffusion_dataset = PolarsSpectrumDataset(
        pl.read_ipc(os.path.join(data_dir, "train.ipc"))
    )
    assert len(diffusion_dataset) == 15500

    spectrum, precursor_mz, precursor_charge = diffusion_dataset[0]

    assert torch.allclose(
        spectrum,
        torch.tensor(
            [
                [34.6979, 1.0000],
                [24.0176, 1.0000],
                [67.7779, 1.0000],
                [32.7176, 0.5000],
                [18.0375, 0.5000],
                [31.3479, 0.5000],
                [27.3741, 0.2500],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
            ]
        ),
        rtol=1e-04,
    )
    assert precursor_mz == 27.374142666666668
    assert precursor_charge == 3.0


def test_ann_polar_spectrum(dir_paths: tuple[str, str]) -> None:
    """Test annotated polars spectrum dataset."""
    root_dir, data_dir = dir_paths
    df = pd.read_csv(os.path.join(root_dir, "predictions.csv"))
    preds = df["preds"].tolist()
    diffusion_dataset = AnnotatedPolarsSpectrumDataset(
        pl.read_ipc(os.path.join(data_dir, "train.ipc")), peptides=preds
    )
    assert len(diffusion_dataset) == 15500

    spectrum, precursor_mz, precursor_charge, peptide = diffusion_dataset[0]

    assert torch.allclose(
        spectrum,
        torch.tensor(
            [
                [34.6979, 1.0000],
                [24.0176, 1.0000],
                [67.7779, 1.0000],
                [32.7176, 0.5000],
                [18.0375, 0.5000],
                [31.3479, 0.5000],
                [27.3741, 0.2500],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
            ]
        ),
        rtol=1e-04,
    )
    assert precursor_mz == 27.374142666666668
    assert precursor_charge == 3.0
    assert peptide == "DCECAB"


def test_collate_batches(residue_set: Any, instanovo_config: DictConfig) -> None:
    """Test batch collation function."""
    spectra = torch.tensor(
        [[[1.0, 1.0], [3.0, 0.7], [2.0, 0.75]], [[4.0, 0.3], [14.0, 0.35], [2.0, 0.9]]]
    )
    precursors_mz = [100.0, 200.0]
    precursors_charge = [1, 2]
    peptides = ["ABCD", "EDAB"]

    collate_fn = collate_batches(
        residues=residue_set,
        max_length=instanovo_config["max_length"],
        time_steps=10,
        annotated=True,
    )

    batch = [
        (spectra[i], precursors_mz[i], precursors_charge[i], peptides[i])
        for i in range(len(precursors_mz))
    ]

    output = collate_fn(batch)

    expected_output = AnnotatedSpectrumBatch(
        spectra=torch.tensor(
            [
                [[1.0000, 1.0000], [3.0000, 0.7000], [2.0000, 0.7500]],
                [[4.0000, 0.3000], [14.0000, 0.3500], [2.0000, 0.9000]],
            ]
        ),
        spectra_padding_mask=torch.tensor(
            [[False, False, False], [False, False, False]]
        ),
        precursors=torch.tensor(
            [[98.9927, 1.0000, 100.0000], [397.9854, 2.0000, 200.0000]]
        ),
        peptides=torch.tensor([[3, 4, 5, 6, 0, 0], [7, 6, 3, 4, 0, 0]]),
        peptide_padding_mask=torch.tensor(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
            ]
        ),
    )

    assert isinstance(output, AnnotatedSpectrumBatch)
    torch.equal(output.spectra, expected_output.spectra)
    torch.equal(output.spectra_padding_mask, expected_output.spectra_padding_mask)
    torch.equal(output.precursors, expected_output.precursors)
    torch.equal(output.peptides, expected_output.peptides)
    torch.equal(output.peptide_padding_mask, expected_output.peptide_padding_mask)

    collate_fn = collate_batches(
        residues=residue_set,
        max_length=instanovo_config["max_length"],
        time_steps=10,
        annotated=False,
    )

    batch = [
        (spectra[i], precursors_mz[i], precursors_charge[i])
        for i in range(len(precursors_mz))
    ]

    output = collate_fn(batch)

    expected_output = SpectrumBatch(
        spectra=torch.tensor(
            [
                [[1.0000, 1.0000], [3.0000, 0.7000], [2.0000, 0.7500]],
                [[4.0000, 0.3000], [14.0000, 0.3500], [2.0000, 0.9000]],
            ]
        ),
        spectra_padding_mask=torch.tensor(
            [[False, False, False], [False, False, False]]
        ),
        precursors=torch.tensor(
            [[98.9927, 1.0000, 100.0000], [397.9854, 2.0000, 200.0000]]
        ),
    )

    assert isinstance(output, SpectrumBatch)
    torch.equal(output.spectra, expected_output.spectra)
    torch.equal(output.spectra_padding_mask, expected_output.spectra_padding_mask)
    torch.equal(output.precursors, expected_output.precursors)
