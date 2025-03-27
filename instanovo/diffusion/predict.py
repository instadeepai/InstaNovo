from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo.constants import ANNOTATED_COLUMN, ANNOTATION_ERROR, DIFFUSION_START_STEP
from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
from instanovo.inference.diffusion import DiffusionDecoder
from instanovo.transformer.dataset import SpectrumDataset, collate_batch
from instanovo.transformer.predict import _format_time
from instanovo.utils import Metrics, SpectrumDataFrame, s3
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "inference"


def get_preds(
    config: DictConfig,
    model: InstaNovoPlus,
    model_config: DictConfig,
) -> None:
    """Predict peptides from spectra using the diffusion model for iterative refinement."""
    if config.get("denovo", False) and config.get("output_path", None) is None:
        raise ValueError(
            "Must specify an output csv path in denovo mode. Please specify in config "
            "or with the cli flag --output-path `path/to/output.csv`"
        )

    data_path = config["data_path"]
    output_path = config.get("output_path", None)

    if config.get("refine", False) and config.get("instanovo_predictions_path", None) is None:
        raise ValueError("The InstaNovo predictions csv path is missing.")

    # Some commomly used config variables
    denovo = config.get("denovo", False)
    use_basic_logging = config.get("use_basic_logging", True)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
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

    sdf.sample_subset(fraction=config.get("subset", 1.0), seed=42)
    logger.info(
        f"Data loaded, evaluating {config.get('subset', 1.0) * 100:.1f}%, {len(sdf):,} "
        "samples in total."
    )

    if sdf.df.is_empty():
        logger.warning("No data found, exiting.")
        sys.exit()

    if config.get("refine", False):
        logger.info("Loading InstaNovo predictions.")
        instanovo_preds_df = pl.read_csv(config.get("instanovo_predictions_path"))

        if len(instanovo_preds_df) != len(sdf.df):
            logger.warning("Length mismatch between input data and predictions.")

        initial_rows = sdf.df.height
        id_col = config.get("id_col", "spectrum_id")
        if id_col not in sdf.df.columns:
            raise ValueError(
                f"The column '{id_col}' does not exist in the input data. "
                "Set an appropriate 'id_col' value in the config."
            )
        if id_col not in instanovo_preds_df.columns:
            raise ValueError(
                f"The column '{id_col}' does not exist in one InstaNovo predictions data. "
                "Set an appropriate 'id_col' value in the config."
            )

        sdf.df = sdf.df.join(instanovo_preds_df, on=id_col, how="inner")
        sdf.df = sdf.df.filter(pl.col(config.get("pred_col", "predictions")) != "")

        if sdf.df.height == 0:
            raise ValueError(
                "All rows were dropped from the dataframe. "
                "No ID matches / predictions to refine were present."
            )

        dropped_rows = initial_rows - sdf.df.height
        if dropped_rows > 0:
            logger.warning(
                f"{dropped_rows} rows were dropped due to unmatched IDs, "
                "or empty predictions, in the inner join."
            )

        if not denovo:
            targets = sdf.df[ANNOTATED_COLUMN].to_list()  # these are not reversed
            sdf.df = sdf.df.with_columns(
                pl.Series("original_peptide", targets)
            )  # add back to dataframe for temp workaround

        sdf.df = sdf.df.with_columns(
            pl.coalesce(config.get("pred_col", "predictions"), ANNOTATED_COLUMN).alias(
                ANNOTATED_COLUMN
            )
        )  # replace sequence column with instanovo predictions
        sdf.df = sdf.df.drop(config.get("pred_col", "predictions"))

        logger.info("Successfully merged InstaNovo predictions into input data.")
        logger.info(f"There are {sdf.df.height} rows remaining in the input data.")

    else:
        if not denovo:
            targets = sdf.df[ANNOTATED_COLUMN].to_list()  # these are not reversed
            sdf.df = sdf.df.with_columns(
                pl.Series("original_peptide", targets)
            )  # add back to dataframe for temp workaround

    residue_set = model.residues
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

    ds = SpectrumDataset(  # peptides are not reversed here
        sdf,
        residue_set,
        model_config.get("n_peaks", 200),
        return_str=False,  # encoded and padded
        annotated=not denovo,
        peptide_pad_length=config.get("max_length", 30),
        diffusion=True,
        reverse_peptide=False,  # we do not reverse peptide for diffusion
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

    start = time.time()

    iter_dl: enumerate[Any] | tqdm[tuple[int, Any]] = enumerate(dl)
    if not use_basic_logging:
        iter_dl = tqdm(enumerate(dl), total=len(dl))

    # Elicit predictions
    logger.info("Performing decoding.")
    results = []
    all_log_probs = []

    for i, batch in iter_dl:
        spectra, precursors, spectra_mask, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)
        spectra_mask = spectra_mask.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
            predictions, log_probs = decoder.decode(
                initial_sequence=peptides.to(device) if config["refine"] else None,
                spectra=spectra,
                spectra_padding_mask=spectra_mask,
                precursors=precursors,
                start_step=DIFFUSION_START_STEP if config["refine"] else None,  # type: ignore
            )

            results.extend(predictions)
            all_log_probs.extend(log_probs)

        if use_basic_logging and (
            (i + 1) % config.get("log_interval", 50) == 0 or (i + 1) == len(dl)
        ):
            delta = time.time() - start
            est_total = delta / (i + 1) * (len(dl) - i - 1)
            logger.info(
                f"Batch {i + 1:05d}/{len(dl):05d}, "
                f"[{_format_time(delta)}/{_format_time(est_total)}, {(delta / (i + 1)):.3f}s/it]"
            )

    # print("targets ", targets)
    # print("predictions ", results)
    # print("log_probs ", all_log_probs)

    delta = time.time() - start

    logger.info(f"Time taken for {data_path} is {delta:.1f} seconds")
    if len(dl) > 0:
        logger.info(
            f"Average time per batch (bs={config['batch_size']}): {delta / len(dl):.1f} seconds"
        )

    # Calculate metrics
    if not denovo:
        metrics = Metrics(residue_set, config.get("isotope_error_range", [0, 1]))

        targets = [
            "".join(residue_set.residue_remapping.get(aa, aa) for aa in residue_set.tokenize(seq))
            for seq in targets
        ]

        aa_prec, aa_recall, pep_recall, _ = metrics.compute_precision_recall(targets, results)
        aa_er = metrics.compute_aa_er(targets, results)

        # TODO add neptune logging
        logger.info(
            "aa_prec: %.4f, aa_recall: %.4f, pep_recall: %.4f, aa_er: %.4f",
            aa_prec,
            aa_recall,
            pep_recall,
            aa_er,
        )

    # Save output
    logger.info("Saving predictions.")
    pred_df["diffusion_predictions_tokenised"] = results
    pred_df["diffusion_predictions"] = ["".join(pred) for pred in results]
    pred_df["diffusion_log_probabilities"] = all_log_probs
    if not denovo:
        pred_df["targets"] = targets
    if config["refine"]:
        pred_df["transformer_predictions"] = sdf.df[ANNOTATED_COLUMN].to_list()
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

    if output_path is not None:  # TODO this doesn't seem to overwrite previous csvs
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Upload to Aichor
        if s3._s3_enabled():
            s3.upload(output_path, s3.convert_to_s3_output(output_path))
