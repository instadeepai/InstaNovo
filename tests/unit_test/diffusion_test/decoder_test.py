from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader

from instanovo.diffusion.dataset import AnnotatedPolarsSpectrumDataset
from instanovo.diffusion.dataset import collate_batches
from instanovo.diffusion.multinomial_diffusion import MultinomialDiffusion
from instanovo.inference.diffusion import DiffusionDecoder


def test_decoder_errors(
    instanovoplus_model: tuple[MultinomialDiffusion, DiffusionDecoder],
    load_preds: list[str],
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
) -> None:
    """Test diffusion decoder initialisation."""
    diffusion_model, diffusion_decoder = instanovoplus_model

    diffusion_dataset = AnnotatedPolarsSpectrumDataset(
        pl.from_pandas(pd.DataFrame(dataset)), peptides=[""]
    )

    with pytest.raises(
        ValueError, match="batch_size should be a positive integer value, but got batch_size=-1"
    ):
        _ = DataLoader(
            diffusion_dataset,
            batch_size=-1,
            shuffle=False,
            collate_fn=collate_batches(
                residues=diffusion_model.residues,
                max_length=diffusion_model.config.max_length,
                time_steps=diffusion_decoder.time_steps,
                annotated=True,
            ),
        )

    diffusion_data_loader = DataLoader(
        diffusion_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_batches(
            residues=diffusion_model.residues,
            max_length=diffusion_model.config.max_length,
            time_steps=diffusion_decoder.time_steps,
            annotated=True,
        ),
    )

    with pytest.raises(IndexError):
        batch = next(iter(diffusion_data_loader))

    with torch.no_grad():
        with pytest.raises(AttributeError):
            pred_aa, probs = diffusion_decoder.decode(
                spectra=-1,
                spectra_padding_mask=-1,
                precursors=-1,
                initial_sequence=-1,
            )

    diffusion_dataset = AnnotatedPolarsSpectrumDataset(
        pl.from_pandas(pd.DataFrame(dataset)), peptides=load_preds
    )

    diffusion_data_loader = DataLoader(
        diffusion_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_batches(
            residues=diffusion_model.residues,
            max_length=diffusion_model.config.max_length,
            time_steps=diffusion_decoder.time_steps,
            annotated=True,
        ),
    )

    batch = next(iter(diffusion_data_loader))
    spectra, spectra_padding_mask, precursors, peptides, peptide_padding_mask = batch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spectra = torch.tensor([1, 2, 3]).to(device)
    spectra_padding_mask = spectra_padding_mask.to(device)
    precursors = precursors.to(device)
    peptides = peptides.to(device)
    peptide_padding_mask = peptide_padding_mask.to(device)

    with torch.no_grad():
        with pytest.raises(IndexError):
            pred_aa, probs = diffusion_decoder.decode(
                spectra=spectra,
                spectra_padding_mask=spectra_padding_mask,
                precursors=precursors,
                initial_sequence=peptides,
            )
