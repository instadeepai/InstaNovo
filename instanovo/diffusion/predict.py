from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo.constants import ANNOTATED_COLUMN, ANNOTATION_ERROR
from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
from instanovo.inference.diffusion import DiffusionDecoder
from instanovo.transformer.dataset import SpectrumDataset, collate_batch
from instanovo.transformer.train import _format_time
from instanovo.utils import Metrics, SpectrumDataFrame
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.device_handler import check_device
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "inference"


def get_preds(
    config: DictConfig,
    model: InstaNovoPlus,
    model_config: DictConfig,
    s3: S3FileHandler,
) -> None:
    """Predict peptides from spectra using the diffusion model for iterative refinement."""
    if config.get("denovo", False) and config.get("output_path", None) is None:
        raise ValueError(
            "Must specify an output csv path in denovo mode. Please specify in config "
            "or with the cli flag --output-path `path/to/output.csv`"
        )

    data_path = config["data_path"]
    refine = config.get("refine", False)

    if OmegaConf.is_list(data_path):
        _new_data_paths = []
        group_mapping = {}
        group_output = {}
        if refine:
            _new_instanovo_predictions_paths = []

        for group in data_path:
            path = group.get("input_path")
            name = group.get("result_name")
            for fp in SpectrumDataFrame._convert_file_paths(path):  # e.g. expands list of globs
                group_mapping[fp] = name
            _new_data_paths.append(path)
            group_output[name] = group.get("output_path")
            if refine:
                _new_instanovo_predictions_paths.append(group.get("instanovo_predictions_path"))
        data_path = _new_data_paths
        if refine:
            instanovo_predictions_path = _new_instanovo_predictions_paths
    else:
        group_mapping = None
        group_output = None
        if refine:
            instanovo_predictions_path = config.get("instanovo_predictions_path", None)

    output_path = config.get("output_path", None)

    if refine and not instanovo_predictions_path:
        raise ValueError("The InstaNovo predictions csv path is missing.")

    # Some commomly used config variables
    denovo = config.get("denovo", False)
    use_basic_logging = config.get("use_basic_logging", True)
    device = check_device(config=config)
    logger.info(f"Using device {device} for InstaNovo+ predictions")
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

    assert sdf.df is not None, "SpectrumDataFrame should be in non-lazy mode"
    if sdf.df.is_empty():
        logger.warning("No data found, exiting.")
        sys.exit()

    if refine:
        if not config.get("refine_all", True) and config.get("refine_threshold") is None:
            raise ValueError("Config variable refine_threshold must be set if refine_all is False.")

        def load_prediction_file(path: str) -> pl.DataFrame:
            """Load a single prediction file, handling S3 paths."""
            logger.info(f"Loading InstaNovo predictions from {path}...")
            df = pl.read_csv(s3._download_from_s3(path) if "s3://" in path else path)

            # Convert columns to consistent types
            for col in ["global_index", "fileno"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.String))
            return df

        # Load and combine predictions
        instanovo_preds_df = (
            pl.concat(
                [load_prediction_file(path) for path in instanovo_predictions_path], how="vertical"
            )
            if isinstance(instanovo_predictions_path, list)
            else load_prediction_file(instanovo_predictions_path)
        )

        if len(instanovo_preds_df) != len(sdf.df):
            logger.warning("Length mismatch between input data and predictions.")

        # Get ID columns
        instanovo_id_col = config.get("instanovo_id_col", "spectrum_id")
        instanovoplus_id_col = config.get("instanovoplus_id_col", "spectrum_id")

        for df, col, name in [
            (sdf.df, instanovoplus_id_col, "input data"),
            (instanovo_preds_df, instanovo_id_col, "InstaNovo predictions"),
        ]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' does not exist in {name}.")
            if df[col].n_unique() != df[col].len():
                logger.warning(f"There are duplicate IDs in the {name}. This may lead to errors.")

        initial_rows = sdf.df.height
        instanovo_preds_df = instanovo_preds_df.rename({instanovo_id_col: instanovoplus_id_col})
        sdf.df = sdf.df.join(instanovo_preds_df, on=instanovoplus_id_col, how="inner")

        if sdf.df.height == 0:
            raise ValueError(
                "All rows were dropped from the dataframe. "
                "No ID matches / predictions to refine were present."
            )

        if (dropped_rows := initial_rows - sdf.df.height) > 0:
            logger.warning(
                f"{dropped_rows} rows were dropped due to unmatched IDs in the inner join."
            )

        # Handle targets and predictions
        if not denovo:
            targets = sdf.df[ANNOTATED_COLUMN].to_list()
            sdf.df = sdf.df.with_columns(pl.Series("original_peptide", targets))

        # Replace sequence column with InstaNovo predictions for dataloader
        sdf.df = sdf.df.with_columns(
            pl.col(config.get("pred_refine_col", "predictions")).alias(ANNOTATED_COLUMN)
        )

        logger.info("Successfully merged InstaNovo predictions into input data.")
        logger.info(f"There are {sdf.df.height} rows remaining in the input data.")

        if not config.get("refine_all", True):
            log_threshold = math.log(config.get("refine_threshold", 0.9))
            needs_refinement = (
                sdf.df[config.get("log_probs_col", "log_probabilities")] < log_threshold
            )
            logger.info(
                f"Refining {needs_refinement.sum()} predictions below "
                f"confidence threshold {config.get('refine_threshold', 0.9)}"
            )

    else:
        if not denovo:
            targets = sdf.df[ANNOTATED_COLUMN].to_list()
            sdf.df = sdf.df.with_columns(pl.Series("original_peptide", targets))

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
            targets = sdf.df["original_peptide"].to_list()  # update targets post filter
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
        return_str=False,  # encoded and padded
        annotated=not denovo,
        peptide_pad_length=config.get("max_length", 30),
        reverse_peptide=False,  # we do not reverse peptides for diffusion
        add_eos=False,
        tokenize_peptide=True,
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

    # Initialize decoder
    logger.info("Initializing decoder.")
    decoder = DiffusionDecoder(model=model)

    index_cols = config.get("index_columns", ["precursor_mz", "precursor_charge"])
    cols = [x for x in sdf.df.columns if x in index_cols]

    pred_df = sdf.df.to_pandas()[cols].copy()

    # Elicit predictions
    logger.info(
        f"Decoding for {config.get('n_preds', 1)} rounds of "
        f"predictions with T={model_config.get('time_steps')}..."
    )

    inference_start = time.time()

    results_n = []
    all_log_probs_n = []
    for j in range(config.get("n_preds", 1)):
        iter_start = time.time()

        iter_dl: enumerate[Any] | tqdm[tuple[int, Any]] = enumerate(dl)
        if not use_basic_logging:
            iter_dl = tqdm(enumerate(dl), total=len(dl))

        results = []
        all_log_probs = []

        logger.info(f"Starting evaluation round {j + 1} of {config.get('n_preds', 1)}...")

        for i, batch in iter_dl:
            spectra, precursors, spectra_mask, peptides, _ = batch
            spectra = spectra.to(device)
            precursors = precursors.to(device)
            spectra_mask = spectra_mask.to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                predictions, log_probs = decoder.decode(
                    initial_sequence=peptides.to(device) if refine else None,
                    spectra=spectra,
                    spectra_padding_mask=spectra_mask,
                    precursors=precursors,
                )

                results.extend(predictions)
                all_log_probs.extend(log_probs)

            if use_basic_logging and (
                (i + 1) % config.get("log_interval", 50) == 0 or (i + 1) == len(dl)
            ):
                delta = time.time() - iter_start
                est_total = delta / (i + 1) * (len(dl) - i - 1)
                logger.info(
                    f"Batch {i + 1:05d}/{len(dl):05d}, "
                    f"[{_format_time(delta)}/{_format_time(est_total)}, {(delta / (i + 1)):.3f}s/it]"  # noqa: E501
                )

        results_n.append(results)
        all_log_probs_n.append(all_log_probs)

    delta = time.time() - inference_start

    logger.info(f"Time taken for {data_path} is {delta:.1f} seconds")
    if len(dl) > 0:
        logger.info(
            f"Average time per batch (bs={config['batch_size']}): {delta / len(dl):.1f} seconds"
        )

    # Filter predictions based on precursor mass matching and log probabilities
    metrics = Metrics(residue_set, config.get("isotope_error_range", [0, 1]))
    filtered_preds = []
    filtered_probs = []

    for i in range(len(sdf.df)):
        row = sdf.df.row(i, named=True)
        prec_mz = row["precursor_mz"]
        prec_charge = row["precursor_charge"]

        # Get all predictions and their probabilities for this spectrum
        preds = [results_n[n][i] for n in range(config.get("n_preds", 1))]
        probs = [all_log_probs_n[n][i] for n in range(config.get("n_preds", 1))]

        # Get unique predictions and their counts
        unique_preds = []
        unique_probs = []
        for pred, prob in zip(preds, probs, strict=False):
            if pred not in unique_preds:
                unique_preds.append(pred)
                unique_probs.append(prob)

        # Find best prediction that matches precursor mass
        best_prob = None
        best_pred = None
        best_matches = False

        for pred, prob in zip(unique_preds, unique_probs, strict=False):
            matches, _ = metrics.matches_precursor(pred, prec_mz, prec_charge)
            if (
                best_prob is None
                or matches > best_matches
                or (matches >= best_matches and prob > best_prob)
            ):
                best_matches = matches
                best_pred = pred
                best_prob = prob

        filtered_preds.append(best_pred)
        filtered_probs.append(best_prob)

    # Save output
    pred_df["diffusion_predictions_tokenised"] = filtered_preds
    pred_df["diffusion_predictions"] = [
        "".join(pred) if pred is not None else "" for pred in filtered_preds
    ]
    pred_df["diffusion_log_probabilities"] = filtered_probs
    if not denovo:
        pred_df["targets"] = targets
    if refine:
        pred_df["transformer_predictions"] = sdf.df[config.get("pred_col", "predictions")]
        pred_df["transformer_predictions_tokenised"] = sdf.df[
            config.get("pred_tok_col", "predictions_tokenised")
        ]
        pred_df["transformer_log_probabilities"] = sdf.df[
            config.get("log_probs_col", "log_probabilities")
        ]
        if config.get("token_log_probs_col", "token_log_probabilities") in sdf.df.columns:
            pred_df["transformer_token_log_probabilities"] = sdf.df[
                config.get("token_log_probs_col", "token_log_probabilities")
            ]

        # Create final prediction columns based on refinement threshold
        if config.get(
            "refine_all", True
        ):  # TODO if diffusion prediction doesn't fit precursor mass use transformer
            # If refine_all is True, compare all predictions
            replacement_mask = (
                pred_df["diffusion_log_probabilities"] > pred_df["transformer_log_probabilities"]
            )
        else:
            # If refine_all is False,
            # use transformer predictions for predictions above confidence threshold
            replacement_mask = (pred_df["transformer_log_probabilities"] < log_threshold) & (
                pred_df["diffusion_log_probabilities"] > pred_df["transformer_log_probabilities"]
            )

        # Post-process final predictions
        pred_df["final_prediction"] = pred_df["diffusion_predictions"].where(
            replacement_mask, pred_df["transformer_predictions"]
        )
        pred_df["final_prediction_tokenised"] = pred_df["diffusion_predictions_tokenised"].where(
            replacement_mask, pred_df["transformer_predictions_tokenised"]
        )
        pred_df["final_log_probabilities"] = pred_df["diffusion_log_probabilities"].where(
            replacement_mask, pred_df["transformer_log_probabilities"]
        )
        pred_df["selected_model"] = pd.Series("diffusion", index=pred_df.index).where(
            replacement_mask, pd.Series("transformer", index=pred_df.index)
        )

    # Add precursor mass match information
    pred_df["precursor_mass_match"] = [
        metrics.matches_precursor(pred, row.precursor_mz, row.precursor_charge)[0]
        for pred, row in zip(filtered_preds, pred_df.itertuples(), strict=False)
    ]
    # TODO add verbose mode for minimal vs verbose info
    if not config.get("use_basic_logging", True):
        # Add individual prediction information
        for n in range(config.get("n_preds", 1)):
            pred_df[f"prediction_{n}"] = ["".join(pred) for pred in results_n[n]]
            pred_df[f"log_probability_{n}"] = all_log_probs_n[n]
            pred_df[f"precursor_mass_match_{n}"] = [
                metrics.matches_precursor(pred, row.precursor_mz, row.precursor_charge)[0]
                for pred, row in zip(results_n[n], pred_df.itertuples(), strict=False)
            ]

    # Calculate metrics
    if not denovo:
        metrics = Metrics(residue_set, config.get("isotope_error_range", [0, 1]))

        targets = [
            "".join(residue_set.residue_remapping.get(aa, aa) for aa in residue_set.tokenize(seq))
            for seq in targets
        ]

        # Use highest confidence prediction of transformer and diffusion models for metrics
        # if in refine mode, else use diffusion predictions
        if refine:
            results = pred_df["final_prediction_tokenised"].tolist()
            all_log_probs = pred_df["final_log_probabilities"].tolist()

        aa_prec, aa_recall, pep_recall, _ = metrics.compute_precision_recall(targets, results)
        aa_er = metrics.compute_aa_er(targets, results)

        logger.info(
            "aa_prec: %.4f, aa_recall: %.4f, pep_recall: %.4f, aa_er: %.4f",
            aa_prec,
            aa_recall,
            pep_recall,
            aa_er,
        )

    # Evaluate individual result files
    if sequence_groups is not None and not denovo:
        logger.info("Evaluating individual result files.")
        _preds = pd.Series(results)
        _targs = pd.Series(targets)
        _probs = pd.Series(all_log_probs)

        results_dict = {
            "instanovo_run_name": config.get("instanovo_run_name"),
            "instanovoplus_run_name": config.get("instanovoplus_run_name"),
            "instanovo_plus_model": config.get("instanovo_plus_model"),
            "n_preds": config.get("n_preds", 1),
            "refine_all": config.get("refine_all", True),
        }
        if not config.get("refine_all", True):
            results_dict["refine_threshold"] = config.get("refine_threshold", 0.9)

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

            results_dict.update(
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

                results_dict.update(
                    {
                        f"{group}_pep_recall_at_{fdr:.3f}_fdr": [pep_recall_at_fdr],
                    }
                )

        logger.info("Saving predictions.")
        result_path = config.get("result_file_path")
        local_path = s3.get_local_path(result_path, missing_ok=True)  # type: ignore
        if local_path is not None and os.path.exists(local_path):
            results_df = pd.read_csv(local_path)
            results_df = pd.concat(
                [results_df, pd.DataFrame(results_dict)], ignore_index=True, join="outer"
            )
        else:
            results_df = pd.DataFrame(results_dict)

        s3.upload_to_s3_wrapper(results_df.to_csv, config.get("result_file_path"), index=False)

    # Save individual result files per group
    if sequence_groups is not None and group_output is not None:
        for group in sequence_groups.unique():
            idx = sequence_groups == group
            if group_output[group] is not None:
                s3.upload_to_s3_wrapper(pred_df[idx].to_csv, group_output[group], index=False)

    # Save output
    if output_path is not None:
        logger.info("Saving predictions.")
        s3.upload_to_s3_wrapper(pred_df.to_csv, output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Upload to Aichor
        if S3FileHandler._aichor_enabled() and not output_path.startswith("s3://"):
            s3.upload(output_path, S3FileHandler.convert_to_s3_output(output_path))
