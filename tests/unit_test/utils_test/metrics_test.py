from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from omegaconf import DictConfig

from instanovo.utils.metrics import Metrics


def test_init_metrics(residue_set: Any, instanovo_config: DictConfig) -> None:
    """Test peptide metric class initialisation."""
    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    assert metric.residue_set == residue_set
    assert metric.isotope_error_range == instanovo_config["isotope_error_range"]
    assert metric.cum_mass_threshold == 0.5
    assert metric.ind_mass_threshold == 0.1


def test_precursor_match(
    residue_set: Any, instanovo_config: DictConfig, dir_paths: tuple[str, str]
) -> None:
    """Test function that checks if a sequence matches the precursor mass within some tolerance."""
    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    _, data_dir = dir_paths
    test_df = pl.read_ipc(os.path.join(data_dir, "test.ipc"))

    status, delta_mass_ppm = metric.matches_precursor(
        seq=test_df[0]["sequence"][0],
        prec_mz=test_df[0]["precursor_mass"][0],
        prec_charge=test_df[0]["precursor_charge"][0],
    )

    assert status is True
    assert np.allclose(delta_mass_ppm, [0.0, 8941.88568367516], rtol=1e-2)

    status, delta_mass_ppm = metric.matches_precursor(seq="A", prec_mz=5500.0, prec_charge=1)

    assert status is False
    assert np.allclose(delta_mass_ppm, [-994633.1134545455, -994450.6861818183], rtol=1e-2)


def test_compute_aa_er(
    residue_set: Any, instanovo_config: DictConfig, instanovo_output: pd.DataFrame
) -> None:
    """Tests calculation of amino-acid level error-rate."""
    target = instanovo_output["targets"][0]
    pred = instanovo_output["preds"][0]

    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    aa_er = metric.compute_aa_er([target], [pred])

    assert aa_er == pytest.approx(0.8333333333333334, rel=1e-2)


def test_compute_precision_recall(
    residue_set: Any,
    instanovo_config: DictConfig,
) -> None:
    """Tests calculation of precision and recall at peptide- and AA-level."""
    target = "ABCDE"
    pred = "ACCDE"

    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    aa_prec, aa_recall, pep_recall, pep_precision = metric.compute_precision_recall(
        [target], [pred]
    )

    assert aa_prec == pytest.approx(0.2, rel=1e-2)
    assert aa_recall == pytest.approx(0.2, rel=1e-2)
    assert pep_recall == pytest.approx(0.0, rel=1e-2)
    assert pep_precision == pytest.approx(0.0, rel=1e-2)

    target = "ABCDE"
    pred = "ABCDE"

    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    aa_prec, aa_recall, pep_recall, pep_precision = metric.compute_precision_recall(
        [target], [pred]
    )

    assert aa_prec == pytest.approx(1.0, rel=1e-2)
    assert aa_recall == pytest.approx(1.0, rel=1e-2)
    assert pep_recall == pytest.approx(1.0, rel=1e-2)
    assert pep_precision == pytest.approx(1.0, rel=1e-2)


def test_calc_auc(
    residue_set: Any, instanovo_config: DictConfig, instanovo_output: pd.DataFrame
) -> None:
    """Tests calculation of peptide-level AUC."""
    targets = instanovo_output["targets"][:100]
    preds = instanovo_output["preds"][:100]
    log_probs = instanovo_output["log_probs"][:100]

    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    auc = metric.calc_auc(targets, preds, np.exp(log_probs))
    assert auc == pytest.approx(0.0002371794871794872, rel=1e-4)


def test_find_recall_at_fdr(
    residue_set: Any, instanovo_config: DictConfig, instanovo_output: pd.DataFrame
) -> None:
    """Tests recall calculation at specified FDR."""
    targets = instanovo_output["targets"][:100]
    preds = instanovo_output["preds"][:100]
    log_probs = instanovo_output["log_probs"][:100]

    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    recall, threshold = metric.find_recall_at_fdr(targets, preds, np.exp(log_probs))
    assert recall == pytest.approx(0.01, rel=1e-4)
    assert threshold == pytest.approx(0.19803493795417956, rel=1e-2)

    aa_prec, aa_recall, pep_recall, pep_precision = metric.compute_precision_recall(
        targets, preds, np.exp(log_probs), threshold
    )
    assert aa_prec == pytest.approx(1.0, rel=1e-2)
    assert aa_recall == pytest.approx(0.010380622837370242, rel=1e-4)
    assert pep_recall == pytest.approx(0.01, rel=1e-2)
    assert pep_precision == pytest.approx(1.0, rel=1e-2)


def test_split_sequences(residue_set: Any, instanovo_config: DictConfig) -> None:
    """Tests splitting of sequence."""
    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    seq = metric._split_sequences(seq=["ABABAD"])
    assert seq == [["A", "B", "A", "B", "A", "D"]]


def test_split_peptide(residue_set: Any, instanovo_config: DictConfig) -> None:
    """Tests splitting of peptide."""
    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    peptide = metric._split_peptide(peptide="ABAB")
    assert peptide == ["A", "B", "A", "B"]


def test_get_pr_curve(
    residue_set: Any, instanovo_config: DictConfig, instanovo_output: pd.DataFrame
) -> None:
    """Tests precision-recall curve calculations."""
    targets = instanovo_output["targets"][:100]
    preds = instanovo_output["preds"][:100]
    log_probs = instanovo_output["log_probs"][:100]

    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    x, y = metric._get_pr_curve(targets, preds, np.exp(log_probs))

    assert np.allclose(
        x,
        [
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ],
        rtol=1e-2,
    )
    assert np.allclose(
        y,
        [
            0.02,
            0.021052631578947368,
            0.022222222222222223,
            0.023529411764705882,
            0.025,
            0.02666666666666667,
            0.02857142857142857,
            0.03076923076923077,
            0.016666666666666666,
            0.01818181818181818,
            0.02,
            0.022222222222222223,
            0.025,
            0.02857142857142857,
            0.03333333333333333,
            0.04,
            0.05,
            0.06666666666666667,
            0.1,
            0.2,
            1.0,
        ],
        rtol=1e-2,
    )


def test_mass(residue_set: Any, instanovo_config: DictConfig) -> None:
    """Tests amino acid mass sum."""
    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    mass = metric._mass(seq=["A", "B", "A", "B"], charge=1)
    assert mass == pytest.approx(81.517876, rel=1e-2)


def test_calc_mass_error(residue_set: Any, instanovo_config: DictConfig) -> None:
    """Tests mass error calculation."""
    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    mass_error = metric._calc_mass_error(mz_theoretical=120.0, mz_measured=109.0, charge=3)
    assert mass_error == pytest.approx(100917.43119266056, rel=1e-1)


def test_novor_match(residue_set: Any, instanovo_config: DictConfig) -> None:
    """Tests novor amino acid match algorithm."""
    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    novor_match = metric._novor_match(["A", "B", "A", "B"], ["A", "B", "E", "B"])
    assert novor_match == 2


def test_get_peptide_matches(
    residue_set: Any, instanovo_config: DictConfig, instanovo_output: pd.DataFrame
) -> None:
    """Tests peptide matches calculation."""
    targets = instanovo_output["targets"][:10]
    preds = instanovo_output["preds"][:10]

    metric = Metrics(residue_set, instanovo_config["isotope_error_range"])

    peptide_matches = metric._get_peptide_matches(targets, preds)
    assert peptide_matches == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
