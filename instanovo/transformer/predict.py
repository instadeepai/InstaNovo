from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import polars as pl
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from instanovo.inference.beam_search import ScoredSequence
from instanovo.inference.knapsack import Knapsack
from instanovo.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo
from instanovo.utils.metrics import Metrics

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# flake8: noqa: CR001
def get_preds(
    data_path: str,
    model: InstaNovo,
    config: dict[str, Any],
    denovo: bool = False,
    output_path: str | None = None,
    knapsack_path: str | None = None,
    save_beams: bool = False,
    device: str = "cuda",
) -> None:
    """Get predictions from a trained model."""
    if denovo and output_path is None:
        raise ValueError(
            "Must specify an output path in denovo mode. Specify an output csv file with --output_path"
        )

    if Path(data_path).suffix.lower() != ".ipc":
        raise ValueError(
            f"Unknown filetype of {data_path}. Only Polars .ipc is currently supported."
        )

    logging.info(f"Loading data from {data_path}")
    df = pl.read_ipc(data_path)
    col_map = {
        "Modified sequence": "modified_sequence",
        "MS/MS m/z": "precursor_mz",
        "m/z": "precursor_mz",
        "Mass": "precursor_mass",
        "Charge": "precursor_charge",
        "Mass values": "mz_array",
        "Mass spectrum": "mz_array",
        "Intensity": "intensity_array",
        "Raw intensity spectrum": "intensity_array",
    }
    if "m/z" in df.columns or "MS/MS m/z" in df.columns:
        if "MS/MS m/z" in df.columns:
            col_map["m/z"] = "calc_precursor_mz"
        df = df.rename({k: v for k, v in col_map.items() if k in df.columns})
        df = df.with_columns(pl.col("modified_sequence").apply(lambda x: x[1:-1]))

    df = df.sample(fraction=config["subset"], seed=0)
    logging.info(
        f"Data loaded, evaluating {config['subset']*100:.1f}%, {df.shape[0]} samples in total."
    )

    if not denovo and (df["modified_sequence"] == "").all():
        raise ValueError(
            "The modified_sequence column is empty, are you trying to run de novo prediction? Add the --denovo flag"
        )

    residue_set = model.residue_set
    logging.info(f"Vocab: {residue_set.index_to_residue}")

    # TODO: find a better place for this, maybe config?
    residue_set.update_remapping(
        {
            "M(+15.99)": "M(ox)",
            "C(+57.02)": "C",
        }
    )

    if not denovo:
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        df = df.with_columns(
            pl.col("modified_sequence")
            .map_elements(
                lambda x: all(
                    [y in supported_residues for y in residue_set.tokenize(x)]
                )
            )  # TODO: set return_dtype
            .alias("supported")
        )
        if (~df["supported"]).sum() > 0:
            logger.warning(
                "Unsupported residues found in evaluation set! These rows will be dropped."
            )
            df_residues = set()
            for x in df["modified_sequence"]:
                df_residues.update(set(residue_set.tokenize(x)))
            logger.warning(f"Residues found: \n{df_residues-supported_residues}")
            logger.warning(f"Residues supported: \n{supported_residues}")
            logger.warning(
                "Please check residue remapping if a different convention has been used."
            )
            original_size = df.shape[0]
            df = df.filter(pl.col("supported"))
            logger.warning(f"{original_size-df.shape[0]:,d} rows have been dropped.")
            logger.warning("Peptide recall should be manually updated accordingly.")

    ds = SpectrumDataset(
        df, residue_set, config["n_peaks"], return_str=True, annotated=not denovo
    )

    dl = DataLoader(
        ds,
        batch_size=config["predict_batch_size"],
        num_workers=config["n_workers"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = model.to(device)
    model = model.eval()

    # Setup decoder
    # TODO: Add flag to choose decoding type (greedy, beam, knapsack beam)
    if knapsack_path is None or not os.path.exists(knapsack_path):
        logging.info("Knapsack path missing or not specified, generating...")
        knapsack = _setup_knapsack(model)
        decoder = KnapsackBeamSearchDecoder(model, knapsack)
        if knapsack_path is not None:
            logging.info(f"Saving knapsack to {knapsack_path}")
            knapsack.save(knapsack_path)
    else:
        logging.info("Knapsack path found. Loading...")
        decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_path)
    # decoder = BeamSearchDecoder(model=model)

    index_cols = [
        "id",
        "experiment_name",
        "evidence_index",
        "scan_number",
        "global_index",
        "spectrum_index",
        "file_index",
        "sample",
        "file",
        "index",
        "fileno",
    ]
    cols = [x for x in df.columns if x in index_cols]

    pred_df = df.to_pandas()[cols].copy()

    preds: dict[int, list[str]] = {i: [] for i in range(config["n_beams"])}
    targs: list[str] = []
    probs: dict[int, list[float]] = {i: [] for i in range(config["n_beams"])}

    start = time.time()
    for _, batch in tqdm(enumerate(dl), total=len(dl)):
        spectra, precursors, spectra_mask, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)
        spectra_mask = spectra_mask.to(device)

        with torch.no_grad():
            p = decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=config["n_beams"],
                max_length=config["max_length"],
                return_beam=save_beams,
            )

        if save_beams:
            p = cast(list[list[ScoredSequence]], p)
            for x in p:
                for i in range(config["n_beams"]):
                    if i >= len(x):
                        preds[i].append("")
                        probs[i].append(-1e6)
                    else:
                        preds[i].append("".join(x[i].sequence))
                        probs[i].append(x[i].sequence_log_probability)
        else:
            p = cast(list[ScoredSequence], p)
            preds[0] += ["".join(x.sequence) if isinstance(x, list) else "" for x in p]
            probs[0] += [
                x.sequence_log_probability if isinstance(x, list) else -1e6 for x in p
            ]
            targs += list(peptides)

    delta = time.time() - start

    logging.info(f"Time taken for {data_path} is {delta:.1f} seconds")
    logging.info(
        f"Average time per batch (bs={config['predict_batch_size']}): {delta/len(dl):.1f} seconds"
    )

    if not denovo:
        pred_df["targets"] = targs
    pred_df["preds"] = ["".join(x) for x in preds[0]]
    pred_df["log_probs"] = probs[0]

    if save_beams:
        for i in range(config["n_beams"]):
            pred_df[f"preds_beam_{i}"] = ["".join(x) for x in preds[i]]
            pred_df[f"log_probs_beam_{i}"] = probs[i]

    if output_path is not None:
        pred_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    # calculate metrics
    if not denovo:
        metrics = Metrics(residue_set, config["isotope_error_range"])

        # Make sure we pass preds[0] without joining on ""
        # This is to handle cases where n-terminus modifications could be accidentally joined
        aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
            pred_df["targets"], preds[0]
        )
        aa_er = metrics.compute_aa_er(pred_df["targets"], preds[0])
        auc = metrics.calc_auc(
            pred_df["targets"], preds[0], np.exp(pred_df["log_probs"])
        )

        logging.info(f"Performance on {data_path}:")
        logging.info(f"aa_er       {aa_er}")
        logging.info(f"aa_prec     {aa_prec}")
        logging.info(f"aa_recall   {aa_recall}")
        logging.info(f"pep_prec    {pep_prec}")
        logging.info(f"pep_recall  {pep_recall}")
        logging.info(f"auc         {auc}")


@click.command()
@click.argument("data-path")
@click.argument("model-path")
@click.option("--output-path", "-o", default=None)
@click.option("--denovo", "-n", is_flag=True, default=False)
@click.option("--subset", "-s", default=1.0)
@click.option("--knapsack-path", "-k", default=None)
@click.option("--n-workers", "-w", default=16)
@click.option("--save-beams", "-b", is_flag=True, default=False)
def main(
    data_path: str,
    model_path: str,
    output_path: str,
    denovo: bool,
    subset: float,
    knapsack_path: str,
    n_workers: int,
    save_beams: bool,
) -> None:
    """Predict with the model."""
    logging.info("Initializing inference.")

    logging.info(f"Loading model from {model_path}")
    model, config = InstaNovo.load(model_path)
    logging.info(f"Config:\n{config}")

    with open_dict(config):
        config["n_workers"] = int(n_workers)
        config["subset"] = float(subset)

    get_preds(data_path, model, config, denovo, output_path, knapsack_path, save_beams)


def _setup_knapsack(model: InstaNovo) -> Knapsack:
    MASS_SCALE = 10000
    residue_masses = dict(model.residue_set.residue_masses.copy())
    for special_residue in list(model.residue_set.residue_to_index.keys())[:3]:
        residue_masses[special_residue] = 0
    residue_indices = model.residue_set.residue_to_index
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=4000.00,
        mass_scale=MASS_SCALE,
    )


if __name__ == "__main__":
    main()
