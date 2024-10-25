from __future__ import annotations

import os
import logging
import time
from typing import cast

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from omegaconf import open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from instanovo.inference import BeamSearchDecoder
from instanovo.inference import GreedyDecoder
from instanovo.inference import KnapsackBeamSearchDecoder
from instanovo.inference import Decoder
from instanovo.inference import ScoredSequence
from instanovo.inference import Knapsack

from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo
from instanovo.transformer.train import _set_author_neptune_api_token
from instanovo.utils import s3
from instanovo.utils import Metrics
from instanovo.utils import SpectrumDataFrame
from instanovo.constants import MASS_SCALE, ANNOTATION_ERROR, ANNOTATED_COLUMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "inference"


# flake8: noqa: CCR001
def get_preds(
    config: DictConfig,
    model: InstaNovo,
    model_config: DictConfig,
) -> None:
    """Get predictions from a trained model."""
    if config.get("denovo", False) and config.get("output_path", None) is None:
        raise ValueError(
            "Must specify an output csv path in denovo mode. Please specify in config or with the cli flag output_path=`path/to/output.csv`"
        )

    data_path = config["data_path"]
    output_path = config.get("output_path", None)

    # Some commomly used config variables
    denovo = config.get("denovo", False)
    num_beams = config.get("num_beams", 1)
    use_basic_logging = config.get("use_basic_logging", True)
    save_beams = config.get("save_beams", False)
    device = config.get("device", "cuda")
    fp16 = config.get("fp16", True)

    if fp16 and device.lower() == "cpu":
        logger.warning("fp16 is enabled but device type is cpu. fp16 will be disabled.")
        fp16 = False

    logger.info(f"Loading data from {data_path}")

    try:
        sdf = SpectrumDataFrame.load(
            data_path,
            lazy=False,
            is_annotated=not denovo,
            column_mapping=config.get("column_map", None),
            shuffle=False,
        )
    except ValueError as e:
        # More descriptive error message in predict mode.
        if str(e) == ANNOTATION_ERROR:
            raise ValueError(
                "The sequence column is missing annotations, are you trying to run de novo prediction? Add the `denovo=True` flag"
            )
        else:
            raise

    # Check max charge values:
    original_size = len(sdf)
    max_charge = config.get("max_charge", 10)
    model_max_charge = model_config.get("max_charge", 10)
    if max_charge > model_max_charge:
        logger.warning(
            f"Inference has been configured with max_charge={max_charge}, but model has max_charge={model_max_charge}."
        )
        logger.warning(
            f"Overwriting max_charge config to model value: {model_max_charge}."
        )
        max_charge = model_max_charge

    sdf.filter_rows(
        lambda row: (row["precursor_charge"] <= max_charge)
        and (row["precursor_charge"] > 0)
    )
    if len(sdf) < original_size:
        logger.warning(
            f"Found {original_size - len(sdf)} rows with charge > {max_charge}. These rows will be skipped."
        )

    sdf.sample_subset(fraction=config.get("subset", 1.0), seed=42)
    logger.info(
        f"Data loaded, evaluating {config.get('subset', 1.0)*100:.1f}%, {len(sdf):,} samples in total."
    )

    residue_set = model.residue_set
    logger.info(f"Vocab: {residue_set.index_to_residue}")

    residue_set.update_remapping(config.get("residue_remapping", {}))

    if not denovo:
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        data_residues = sdf.get_vocabulary(residue_set.tokenize)
        if len(data_residues - supported_residues) > 0:
            logger.warning(
                "Unsupported residues found in evaluation set! These rows will be dropped."
            )
            logger.warning(f"Residues found: \n{data_residues-supported_residues}")
            logger.warning(
                "Please check residue remapping if a different convention has been used."
            )
            original_size = len(sdf)
            sdf.filter_rows(
                lambda row: all(
                    [
                        residue in supported_residues
                        for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                    ]
                )
            )
            logger.warning(f"{original_size-len(sdf):,d} rows have been dropped.")
            logger.warning("Peptide recall should recalculated accordingly.")

    ds = SpectrumDataset(
        sdf,
        residue_set,
        model_config.get("n_peaks", 200),
        return_str=True,
        annotated=not denovo,
    )

    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        num_workers=0,  # sdf requirement, handled internally
        shuffle=False,  # sdf requirement, handled internally
        collate_fn=collate_batch,
    )

    model = model.to(device)
    model = model.eval()

    # Setup decoder

    if config.get("use_knapsack", False):
        knapsack_path = config.get("knapsack_path", None)
        if knapsack_path is None or not os.path.exists(knapsack_path):
            logger.info("Knapsack path missing or not specified, generating...")
            knapsack = _setup_knapsack(model)
            decoder: Decoder = KnapsackBeamSearchDecoder(model, knapsack)
            if knapsack_path is not None:
                logger.info(f"Saving knapsack to {knapsack_path}")
                knapsack.save(knapsack_path)
        else:
            logger.info("Knapsack path found. Loading...")
            decoder = KnapsackBeamSearchDecoder.from_file(
                model=model, path=knapsack_path
            )
    elif num_beams > 1:
        decoder = BeamSearchDecoder(model=model)
    else:
        decoder = GreedyDecoder(
            model=model,
            suppressed_residues=config.get("suppressed_residues", None),
            disable_terminal_residues_anywhere=config.get(
                "disable_terminal_residues_anywhere", True
            ),
        )

    index_cols = config.get("index_columns", ["precursor_mz", "precursor_charge"])
    cols = [x for x in sdf.df.columns if x in index_cols]

    pred_df = sdf.df.to_pandas()[cols].copy()

    preds: dict[int, list[list[str]]] = {i: [] for i in range(num_beams)}
    targs: list[str] = []
    sequence_log_probs: dict[int, list[float]] = {i: [] for i in range(num_beams)}
    token_log_probs: dict[int, list[list[float]]] = {i: [] for i in range(num_beams)}

    start = time.time()

    iter_dl = enumerate(dl)
    if not use_basic_logging:
        iter_dl = tqdm(enumerate(dl), total=len(dl))

    logger.info("Starting evaluation...")

    for i, batch in iter_dl:
        spectra, precursors, spectra_mask, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)
        spectra_mask = spectra_mask.to(device)

        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=fp16
        ):
            batch_predictions = decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=num_beams,
                max_length=config.get("max_length", 40),
                return_beam=save_beams,
            )

        if save_beams:
            batch_predictions = cast(list[list[ScoredSequence]], batch_predictions)
            for predictions in batch_predictions:
                for j in range(num_beams):
                    if j >= len(predictions):
                        preds[j].append([])
                        sequence_log_probs[j].append(-float("inf"))
                        token_log_probs[j].append([])
                    else:
                        preds[j].append(predictions[j].sequence)
                        sequence_log_probs[j].append(
                            predictions[j].sequence_log_probability
                        )
                        token_log_probs[j].append(
                            predictions[j].token_log_probabilities
                        )
        else:
            batch_predictions = cast(list[ScoredSequence], batch_predictions)
            for prediction in batch_predictions:
                if isinstance(prediction, ScoredSequence):
                    preds[0].append(prediction.sequence)
                    sequence_log_probs[0].append(prediction.sequence_log_probability)
                    token_log_probs[0].append(prediction.token_log_probabilities)
                else:
                    preds[0].append([])
                    sequence_log_probs[0].append(-float("inf"))
                    token_log_probs[0].append([])
        targs += list(peptides)

        if use_basic_logging and (
            (i + 1) % config.get("log_interval", 50) == 0 or (i + 1) == len(dl)
        ):
            delta = time.time() - start
            est_total = delta / (i + 1) * (len(dl) - i - 1)
            logger.info(
                f"Batch {i+1:05d}/{len(dl):05d}, [{_format_time(delta)}/{_format_time(est_total)}, {(delta / (i + 1)):.3f}s/it]"
            )

    delta = time.time() - start

    logger.info(f"Time taken for {data_path} is {delta:.1f} seconds")
    logger.info(
        f"Average time per batch (bs={config['batch_size']}): {delta/len(dl):.1f} seconds"
    )

    if not denovo:
        pred_df["targets"] = targs
    pred_df["preds"] = ["".join(x) for x in preds[0]]
    pred_df["preds_tokenised"] = [", ".join(x) for x in preds[0]]
    pred_df["log_probs"] = sequence_log_probs[0]
    pred_df["token_log_probs"] = token_log_probs[0]

    if save_beams:
        for i in range(num_beams):
            pred_df[f"preds_beam_{i}"] = ["".join(x) for x in preds[i]]
            pred_df[f"log_probs_beam_{i}"] = sequence_log_probs[i]
            pred_df[f"token_log_probs_{i}"] = token_log_probs[i]

    # Calculate metrics
    if not denovo:
        metrics = Metrics(residue_set, config.get("isotope_error_range", [0, 1]))

        # Make sure we pass preds[0] without joining on ""
        # This is to handle cases where n-terminus modifications could be accidentally joined
        aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
            pred_df["targets"], preds[0]
        )
        aa_er = metrics.compute_aa_er(pred_df["targets"], preds[0])
        auc = metrics.calc_auc(
            pred_df["targets"], preds[0], np.exp(pred_df["log_probs"])
        )

        logger.info(f"Performance on {data_path}:")
        logger.info(f"  aa_er       {aa_er:.5f}")
        logger.info(f"  aa_prec     {aa_prec:.5f}")
        logger.info(f"  aa_recall   {aa_recall:.5f}")
        logger.info(f"  pep_prec    {pep_prec:.5f}")
        logger.info(f"  pep_recall  {pep_recall:.5f}")
        logger.info(f"  auc         {auc:.5f}")

        fdr = config.get("filter_fdr_threshold", None)
        if fdr:
            _, threshold = metrics.find_recall_at_fdr(
                pred_df["targets"], preds[0], np.exp(pred_df["log_probs"]), fdr=fdr
            )
            aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                pred_df["targets"],
                preds[0],
                np.exp(pred_df["log_probs"]),
                threshold=threshold,
            )
            logger.info(f"Performance at {fdr*100:.1f}% FDR:")
            logger.info(f"  aa_prec     {aa_prec:.5f}")
            logger.info(f"  aa_recall   {aa_recall:.5f}")
            logger.info(f"  pep_prec    {pep_prec:.5f}")
            logger.info(f"  pep_recall  {pep_recall:.5f}")
            logger.info(f"  confidence  {threshold:.5f}")

        # Calculate some additional information for filtering:
        pred_df["delta_mass_ppm"] = pred_df.apply(
            lambda row: np.min(
                np.abs(
                    metrics.matches_precursor(
                        preds[0][row.name], row["precursor_mz"], row["precursor_charge"]
                    )[1]
                )
            ),
            axis=1,
        )

        filter_precursor_ppm = config.get("filter_precursor_ppm", None)
        if filter_precursor_ppm:
            idx = pred_df["delta_mass_ppm"] < filter_precursor_ppm
            filtered_preds = pd.Series(preds[0])
            filtered_preds[~idx] = ""
            aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                pred_df["targets"], filtered_preds
            )
            logger.info(
                f"Performance with filtering at {filter_precursor_ppm} ppm delta mass:"
            )
            logger.info(f"  aa_prec     {aa_prec:.5f}")
            logger.info(f"  aa_recall   {aa_recall:.5f}")
            logger.info(f"  pep_prec    {pep_prec:.5f}")
            logger.info(f"  pep_recall  {pep_recall:.5f}")
            logger.info(
                f"Rows filtered: {len(sdf)-np.sum(idx)} ({(len(sdf)-np.sum(idx))/len(sdf)*100:.2f}%)"
            )

        model_confidence_no_pred = config.get("filter_confidence", None)
        if model_confidence_no_pred:
            idx = np.exp(pred_df["log_probs"]) > model_confidence_no_pred
            filtered_preds = pd.Series(preds[0])
            filtered_preds[~idx] = ""
            aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                pred_df["targets"], filtered_preds
            )
            logger.info(
                f"Performance with filtering confidence < {model_confidence_no_pred}"
            )
            logger.info(f"  aa_prec     {aa_prec:.5f}")
            logger.info(f"  aa_recall   {aa_recall:.5f}")
            logger.info(f"  pep_prec    {pep_prec:.5f}")
            logger.info(f"  pep_recall  {pep_recall:.5f}")
            logger.info(
                f"Rows filtered: {len(sdf)-np.sum(idx)} ({(len(sdf)-np.sum(idx))/len(sdf)*100:.2f}%)"
            )

    # Save output
    if output_path is not None:
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Upload to Aichor
        if s3._s3_enabled():
            s3.upload(output_path, s3.convert_to_s3_output(output_path))


@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="default")
def main(config: DictConfig) -> None:
    """Predict with the model."""
    logger.info("Initializing inference.")
    _set_author_neptune_api_token()

    # Check config inputs
    if not config.get("data_path", None):
        raise ValueError(
            "Expected data_path but found None. Please specify in predict config or with the cli flag `data_path=path/to/data.ipc`"
        )

    model_path = config.get("model_path", None)
    if not model_path:
        raise ValueError(
            "Expected model_path but found None. Please specify in predict config or with the cli flag `model_path=path/to/model.ckpt`"
        )

    logger.info(f"Loading model from {model_path}")
    model, model_config = InstaNovo.from_pretrained(model_path)
    logger.info(f"Config:\n{config}")
    logger.info(f"Model params: {np.sum([p.numel() for p in model.parameters()]):,d}")

    if config.get("save_beams", False) and config.get("num_beams", 1) == 1:
        logger.warning(
            "num_beams is 1 and will override save_beams. Only use save_beams in beam search."
        )
        with open_dict(config):
            config["save_beams"] = False

    logger.info(f"Performing search with {config.get('num_beams', 1)} beams")
    get_preds(config, model, model_config)


def _setup_knapsack(model: InstaNovo) -> Knapsack:
    residue_masses = dict(model.residue_set.residue_masses.copy())
    if any([x < 0 for x in residue_masses.values()]):
        raise NotImplementedError(
            "Negative mass found in residues, this will break the knapsack graph. Either disable knapsack or use strictly positive masses"
        )
    for special_residue in list(model.residue_set.residue_to_index.keys())[:3]:
        residue_masses[special_residue] = 0
    residue_indices = model.residue_set.residue_to_index
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=4000.00,
        mass_scale=MASS_SCALE,
    )


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"


if __name__ == "__main__":
    main()
