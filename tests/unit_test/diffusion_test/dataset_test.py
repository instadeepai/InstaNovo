from __future__ import annotations

import pandas as pd
import polars as pl
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset

from instanovo.diffusion.dataset import AnnotatedPolarsSpectrumDataset
from instanovo.diffusion.dataset import PolarsSpectrumDataset


def test_preds(load_preds: list[str]) -> None:
    """Test InstaNovo predictions type."""
    assert isinstance(load_preds, list)
    assert all(isinstance(x, str) for x in load_preds)


def test_polar_spectrum(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
) -> None:
    """Test polars sepectrum dataset."""
    diffusion_dataset = PolarsSpectrumDataset(pl.from_pandas(pd.DataFrame(dataset)))
    assert len(diffusion_dataset) == 271

    spectrum, precursor_mz, precursor_charge = diffusion_dataset[0]

    assert torch.allclose(
        spectrum,
        torch.tensor(
            [
                [1.0096e02, 6.4273e-03],
                [1.1006e02, 6.0129e-03],
                [1.1646e02, 5.7488e-03],
                [1.2910e02, 2.5723e-02],
                [1.3009e02, 2.5281e-02],
                [1.4711e02, 3.0318e-02],
                [1.7309e02, 6.7769e-03],
                [1.8612e02, 1.3652e-02],
                [2.0413e02, 2.9710e-02],
                [2.7303e02, 7.5393e-03],
                [2.8318e02, 1.7115e-02],
                [3.0119e02, 3.4304e-01],
                [3.2845e02, 8.4199e-03],
                [3.7222e02, 4.9525e-02],
                [4.7877e02, 7.6899e-03],
                [5.2873e02, 9.7643e-03],
                [5.7176e02, 1.0661e-02],
                [5.7975e02, 1.1832e-02],
                [6.1527e02, 9.5337e-03],
                [6.5630e02, 1.2352e-02],
                [7.7837e02, 2.2391e-02],
                [7.7887e02, 7.0299e-02],
                [7.7938e02, 5.1397e-03],
                [7.9137e02, 1.6256e-02],
                [7.9189e02, 1.6157e-02],
                [8.0088e02, 6.5557e-02],
                [1.0365e03, 1.1454e-02],
                [1.1015e03, 1.2202e-02],
                [1.1395e03, 5.5224e-02],
                [1.1895e03, 1.2289e-02],
                [1.2285e03, 2.2203e-02],
                [1.2556e03, 1.8892e-02],
                [1.2565e03, 2.0319e-02],
                [1.2716e03, 1.0596e-01],
                [1.2996e03, 3.2774e-01],
                [1.5998e03, 1.0000e00],
            ]
        ),
        rtol=1e-04,
    )
    assert precursor_mz == 800.38427734375
    assert precursor_charge == 2.0


def test_ann_polar_spectrum(
    load_preds: list[str],
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
) -> None:
    """Test annotated polars sepectrum dataset."""
    diffusion_dataset = AnnotatedPolarsSpectrumDataset(
        pl.from_pandas(pd.DataFrame(dataset)), peptides=load_preds
    )
    assert len(diffusion_dataset) == 271

    spectrum, precursor_mz, precursor_charge, peptide = diffusion_dataset[0]

    assert torch.allclose(
        spectrum,
        torch.tensor(
            [
                [1.0096e02, 6.4273e-03],
                [1.1006e02, 6.0129e-03],
                [1.1646e02, 5.7488e-03],
                [1.2910e02, 2.5723e-02],
                [1.3009e02, 2.5281e-02],
                [1.4711e02, 3.0318e-02],
                [1.7309e02, 6.7769e-03],
                [1.8612e02, 1.3652e-02],
                [2.0413e02, 2.9710e-02],
                [2.7303e02, 7.5393e-03],
                [2.8318e02, 1.7115e-02],
                [3.0119e02, 3.4304e-01],
                [3.2845e02, 8.4199e-03],
                [3.7222e02, 4.9525e-02],
                [4.7877e02, 7.6899e-03],
                [5.2873e02, 9.7643e-03],
                [5.7176e02, 1.0661e-02],
                [5.7975e02, 1.1832e-02],
                [6.1527e02, 9.5337e-03],
                [6.5630e02, 1.2352e-02],
                [7.7837e02, 2.2391e-02],
                [7.7887e02, 7.0299e-02],
                [7.7938e02, 5.1397e-03],
                [7.9137e02, 1.6256e-02],
                [7.9189e02, 1.6157e-02],
                [8.0088e02, 6.5557e-02],
                [1.0365e03, 1.1454e-02],
                [1.1015e03, 1.2202e-02],
                [1.1395e03, 5.5224e-02],
                [1.1895e03, 1.2289e-02],
                [1.2285e03, 2.2203e-02],
                [1.2556e03, 1.8892e-02],
                [1.2565e03, 2.0319e-02],
                [1.2716e03, 1.0596e-01],
                [1.2996e03, 3.2774e-01],
                [1.5998e03, 1.0000e00],
            ]
        ),
        rtol=1e-04,
    )
    assert precursor_mz == 800.38427734375
    assert precursor_charge == 2.0
    assert peptide == "NRNVGDQNGC(+57.02)LAPGK"
