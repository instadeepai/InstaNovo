from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo.constants import ANNOTATED_COLUMN, ANNOTATION_ERROR, MASS_SCALE, MAX_MASS
from instanovo.inference import (
    BeamSearchDecoder,
    Decoder,
    GreedyDecoder,
    Knapsack,
    KnapsackBeamSearchDecoder,
    ScoredSequence,
)
from instanovo.transformer.dataset import SpectrumDataset, collate_batch
from instanovo.transformer.model import InstaNovo
from instanovo.utils import Metrics, SpectrumDataFrame
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.device_handler import check_device
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "inference"


def get_preds(
    config: DictConfig,
    model: InstaNovo,
    model_config: DictConfig,
    s3: S3FileHandler,
) -> None:
    """Get predictions from a trained model."""
    if config.get("denovo", False) and config.get("output_path", None) is None:
        raise ValueError(
            "Must specify an output csv path in denovo mode. Please specify in config "
            "or with the cli flag --output-path `path/to/output.csv`"
        )

    data_path = config["data_path"]

    if OmegaConf.is_list(data_path):
        _new_data_paths = []
        group_mapping = {}
        group_output = {}
        for group in data_path:
            path = group.get("input_path")
            name = group.get("result_name")
            for fp in SpectrumDataFrame._convert_file_paths(path):
                group_mapping[fp] = name
            _new_data_paths.append(path)
            group_output[name] = group.get("output_path")
        data_path = _new_data_paths
    else:
        group_mapping = None
        group_output = None

    output_path = config.get("output_path", None)

    # Some commomly used config variables
    denovo = config.get("denovo", False)
    num_beams = config.get("num_beams", 1)
    use_basic_logging = config.get("use_basic_logging", True)
    save_beams = config.get("save_beams", False)
    device = check_device(config=config)
    logger.info(f"Using device: {device} for InstaNovo predictions")
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
            add_spectrum_id=True,
            add_source_file_column=True,
        )
    except ValueError as e:
        # More descriptive error message in predict mode.
        if str(e) == ANNOTATION_ERROR:
            raise ValueError(
                "The sequence column is missing annotations, "
                "are you trying to run de novo prediction? Add the `denovo=True` flag"
            ) from e
        else:
            raise

    # Check max charge values:
    original_size = len(sdf)
    max_charge = config.get("max_charge", 10)
    model_max_charge = model_config.get("max_charge", 10)
    if max_charge > model_max_charge:
        logger.warning(
            f"Inference has been configured with max_charge={max_charge}, "
            f"but model has max_charge={model_max_charge}."
        )
        logger.warning(f"Overwriting max_charge config to model value: {model_max_charge}.")
        max_charge = model_max_charge

    sdf.filter_rows(
        lambda row: (row["precursor_charge"] <= max_charge) and (row["precursor_charge"] > 0)
    )
    if len(sdf) < original_size:
        logger.warning(
            f"Found {original_size - len(sdf)} rows with charge > {max_charge}. "
            "These rows will be skipped."
        )

    subset = config.get("subset", 1.0)
    if not 0 < subset <= 1:
        raise ValueError(
            f"Invalid subset value: {subset}. Must be a float greater than 0 and less than or equal to 1."  # noqa: E501
        )

    sdf.sample_subset(fraction=subset, seed=42)
    logger.info(f"Data loaded, evaluating {subset * 100:.2f}%, {len(sdf):,} samples in total.")

    assert sdf.df is not None
    if sdf.df.is_empty():
        logger.warning("No data found, exiting.")
        sys.exit()

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
            logger.warning(f"Residues found: \n{data_residues - supported_residues}")
            logger.warning(
                "Please check residue remapping if a different convention has been used."
            )
            original_size = len(sdf)
            sdf.filter_rows(
                lambda row: all(
                    residue in supported_residues
                    for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                )
            )
            logger.warning(f"{original_size - len(sdf):,d} rows have been dropped.")
            logger.warning("Peptide recall should recalculated accordingly.")

    # Used to group validation outputs
    if group_mapping is not None:
        logger.info("Computing validation groups.")
        sequence_groups = pd.Series(
            [
                group_mapping[row["source_file"]]
                if row.get("source_file", None) in group_mapping
                else "no_group"
                for row in iter(sdf)
            ]
        )
        logger.info("Sequences per validation group:")
        for group in sequence_groups.unique():
            logger.info(f" - {group}: {(sequence_groups == group).sum():,d}")
    else:
        sequence_groups = None

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
        logger.info(f"Using Knapsack Beam Search with {num_beams} beam(s)")
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
            decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_path)
    elif num_beams > 1:
        logger.info(f"Using Beam Search with {num_beams} beam(s)")
        decoder = BeamSearchDecoder(model=model)
    else:
        logger.info(f"Using Greedy Search with  {num_beams} beam(s)")
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

    iter_dl: enumerate[Any] | tqdm[tuple[int, Any]] = enumerate(dl)
    if not use_basic_logging:
        iter_dl = tqdm(enumerate(dl), total=len(dl))

    logger.info("Starting evaluation...")

    for i, batch in iter_dl:
        spectra, precursors, spectra_mask, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)
        spectra_mask = spectra_mask.to(device)

        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16),
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
                        sequence_log_probs[j].append(predictions[j].sequence_log_probability)
                        token_log_probs[j].append(predictions[j].token_log_probabilities)
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
                f"Batch {i + 1:05d}/{len(dl):05d}, [{_format_time(delta)}/"
                f"{_format_time(est_total)}, {(delta / (i + 1)):.3f}s/it]"
            )

    delta = time.time() - start

    logger.info(f"Time taken for {data_path} is {delta:.1f} seconds")
    if len(dl) > 0:
        logger.info(
            f"Average time per batch (bs={config['batch_size']}): {delta / len(dl):.1f} seconds"
        )

    if not denovo:
        pred_df["targets"] = targs
    pred_df[config.get("pred_col", "predictions")] = ["".join(x) for x in preds[0]]
    pred_df[config.get("pred_tok_col", "predictions_tokenised")] = [", ".join(x) for x in preds[0]]
    pred_df[config.get("log_probs_col", "log_probabilities")] = sequence_log_probs[0]
    pred_df[config.get("token_log_probs_col", "token_log_probabilities")] = token_log_probs[0]

    if save_beams:
        for i in range(num_beams):
            pred_df[f"preds_beam_{i}"] = ["".join(x) for x in preds[i]]
            pred_df[f"log_probs_beam_{i}"] = sequence_log_probs[i]
            pred_df[f"token_log_probs_{i}"] = token_log_probs[i]

    # Always calculate delta_mass_ppm, even in de novo mode
    metrics = Metrics(residue_set, config.get("isotope_error_range", [0, 1]))

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

    # Calculate metrics
    if not denovo:
        # Make sure we pass preds[0] without joining on ""
        # This is to handle cases where n-terminus modifications could be accidentally joined
        aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
            pred_df["targets"], preds[0]
        )
        aa_er = metrics.compute_aa_er(pred_df["targets"], preds[0])
        auc = metrics.calc_auc(
            pred_df["targets"],
            preds[0],
            np.exp(pred_df[config.get("log_probs_col", "log_probabilities")]),
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
                pred_df["targets"],
                preds[0],
                np.exp(pred_df[config.get("log_probs_col", "log_probabilities")]),
                fdr=fdr,
            )
            aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                pred_df["targets"],
                preds[0],
                np.exp(pred_df[config.get("log_probs_col", "log_probabilities")]),
                threshold=threshold,
            )
            logger.info(f"Performance at {fdr * 100:.1f}% FDR:")
            logger.info(f"  aa_prec     {aa_prec:.5f}")
            logger.info(f"  aa_recall   {aa_recall:.5f}")
            logger.info(f"  pep_prec    {pep_prec:.5f}")
            logger.info(f"  pep_recall  {pep_recall:.5f}")
            logger.info(f"  confidence  {threshold:.5f}")

        filter_precursor_ppm = config.get("filter_precursor_ppm", None)
        if filter_precursor_ppm:
            idx = pred_df["delta_mass_ppm"] < filter_precursor_ppm
            logger.info(f"Performance with filtering at {filter_precursor_ppm} ppm delta mass:")
            if np.sum(idx) > 0:
                filtered_preds = pd.Series(preds[0])
                filtered_preds[~idx] = ""
                aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                    pred_df["targets"], filtered_preds
                )
                logger.info(f"  aa_prec     {aa_prec:.5f}")
                logger.info(f"  aa_recall   {aa_recall:.5f}")
                logger.info(f"  pep_prec    {pep_prec:.5f}")
                logger.info(f"  pep_recall  {pep_recall:.5f}")
                logger.info(
                    f"Rows filtered: {len(sdf) - np.sum(idx)} "
                    f"({(len(sdf) - np.sum(idx)) / len(sdf) * 100:.2f}%)"
                )
                if np.sum(idx) < 1000:
                    logger.info(
                        f"Metrics calculated on a small number of samples ({np.sum(idx)}), "
                        "interpret with care!"
                    )
            else:
                logger.info("No predictions met criteria, skipping metrics.")

        model_confidence_no_pred = config.get("filter_confidence", None)
        if model_confidence_no_pred:
            idx = (
                np.exp(pred_df[config.get("log_probs_col", "log_probabilities")])
                > model_confidence_no_pred
            )
            logger.info(f"Performance with filtering confidence < {model_confidence_no_pred}")
            if np.sum(idx) > 0:
                filtered_preds = pd.Series(preds[0])
                filtered_preds[~idx] = ""
                aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                    pred_df["targets"], filtered_preds
                )
                logger.info(f"  aa_prec     {aa_prec:.5f}")
                logger.info(f"  aa_recall   {aa_recall:.5f}")
                logger.info(f"  pep_prec    {pep_prec:.5f}")
                logger.info(f"  pep_recall  {pep_recall:.5f}")
                logger.info(
                    f"Rows filtered: {len(sdf) - np.sum(idx)} "
                    f"({(len(sdf) - np.sum(idx)) / len(sdf) * 100:.2f}%)"
                )
                if np.sum(idx) < 1000:
                    logger.info(
                        f"Metrics calculated on a small number of samples ({np.sum(idx)}), "
                        "interpret with care!"
                    )
            else:
                logger.info("No predictions met criteria, skipping metrics.")

    # Evaluate individual result files
    if sequence_groups is not None and not denovo:
        _preds = pd.Series(preds[0])
        _targs = pd.Series(pred_df["targets"])
        _probs = pd.Series(pred_df[config.get("log_probs_col", "log_probabilities")])

        results = {
            "run_name": config.get("run_name"),
            "instanovo_model": config.get("instanovo_model"),
            "num_beams": num_beams,
            "use_knapsack": config.get("use_knapsack", False),
        }
        for group in sequence_groups.unique():
            if group == "no_group":
                continue
            idx = sequence_groups == group
            _group_preds = _preds[idx].reset_index(drop=True)
            _group_targs = _targs[idx].reset_index(drop=True)
            _group_probs = _probs[idx].reset_index(drop=True)
            aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                _group_targs, _group_preds
            )
            aa_er = metrics.compute_aa_er(_group_targs, _group_preds)
            auc = metrics.calc_auc(_group_targs, _group_preds, _group_probs)

            results.update(
                {
                    f"{group}_aa_prec": [aa_prec],
                    f"{group}_aa_recall": [aa_recall],
                    f"{group}_pep_recall": [pep_recall],
                    f"{group}_pep_prec": [pep_prec],
                    f"{group}_aa_er": [aa_er],
                    f"{group}_auc": [auc],
                }
            )

            fdr = config.get("filter_fdr_threshold", None)
            if fdr:
                _, threshold = metrics.find_recall_at_fdr(
                    _group_targs, _group_preds, np.exp(_group_probs), fdr=fdr
                )
                _, _, pep_recall_at_fdr, _ = metrics.compute_precision_recall(
                    _group_targs,
                    _group_preds,
                    np.exp(_group_probs),
                    threshold=threshold,
                )

                results.update(
                    {
                        f"{group}_pep_recall_at_{fdr:.3f}_fdr": [pep_recall_at_fdr],
                    }
                )

        result_path = config.get("result_file_path")
        local_path = s3.get_local_path(result_path, missing_ok=True)
        if local_path is not None and os.path.exists(local_path):
            results_df = pd.read_csv(local_path)
            results_df = pd.concat(
                [results_df, pd.DataFrame(results)], ignore_index=True, join="outer"
            )
        else:
            results_df = pd.DataFrame(results)

        s3.upload_to_s3_wrapper(results_df.to_csv, config.get("result_file_path"), index=False)

    # Save individual result files per group
    if sequence_groups is not None and group_output is not None:
        for group in sequence_groups.unique():
            idx = sequence_groups == group
            if group_output[group] is not None:
                s3.upload_to_s3_wrapper(pred_df[idx].to_csv, group_output[group], index=False)

    # Save output
    if output_path is not None:
        s3.upload_to_s3_wrapper(pred_df.to_csv, output_path, index=False)
        # pred_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Upload to Aichor
        if S3FileHandler._aichor_enabled() and not output_path.startswith("s3://"):
            s3.upload(output_path, S3FileHandler.convert_to_s3_output(output_path))


def _setup_knapsack(model: InstaNovo) -> Knapsack:
    residue_masses = dict(model.residue_set.residue_masses.copy())
    negative_residues = [k for k, v in residue_masses.items() if v < 0]
    if len(negative_residues) > 0:
        logger.warning(f"Negative mass found in residues: {negative_residues}.")
        logger.warning(
            "These residues will be disabled when using knapsack decoding. "
            "A future release is planned to support negative masses."
        )
        residue_masses.update(dict.fromkeys(negative_residues, MAX_MASS))
    for special_residue in list(model.residue_set.residue_to_index.keys())[:3]:
        residue_masses[special_residue] = 0
    residue_indices = model.residue_set.residue_to_index
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=MAX_MASS,
        mass_scale=MASS_SCALE,
    )


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
