from __future__ import annotations

import jiwer
import numpy as np

from instanovo.constants import CARBON_MASS_DELTA
from instanovo.utils.residues import ResidueSet


class Metrics:
    """Peptide metrics class."""

    def __init__(
        self,
        residue_set: ResidueSet,
        isotope_error_range: list[int],
        cum_mass_threshold: float = 0.5,
        ind_mass_threshold: float = 0.1,
    ) -> None:
        self.residue_set = residue_set
        self.isotope_error_range = isotope_error_range
        self.cum_mass_threshold = cum_mass_threshold
        self.ind_mass_threshold = ind_mass_threshold

    def matches_precursor(
        self,
        seq: str | list[str],
        prec_mass: float,
        prec_charge: int,
        prec_tol: int = 50,
    ) -> tuple[bool, list[float]]:
        """Check if a sequence matches the precursor mass within some tolerance."""
        seq_mass = self._mass(seq, charge=prec_charge)
        delta_mass_ppm = [
            self._calc_mass_error(seq_mass, prec_mass, prec_charge, isotope)
            for isotope in range(
                self.isotope_error_range[0],
                self.isotope_error_range[1] + 1,
            )
        ]
        return any(abs(d) < prec_tol for d in delta_mass_ppm), delta_mass_ppm

    def compute_aa_er(
        self,
        peptides_truth: list[str] | list[list[str]],
        peptides_predicted: list[str] | list[list[str]],
    ) -> float:
        """Compute amino-acid level error-rate."""
        # Ensure amino acids are separated
        peptides_truth = self._split_sequences(peptides_truth)
        peptides_predicted = self._split_sequences(peptides_predicted)

        return float(
            jiwer.wer(
                [" ".join(x).replace("I", "L") for x in peptides_truth],
                [" ".join(x).replace("I", "L") for x in peptides_predicted],
            )
        )

    # Adapted from https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/evaluate.py
    def compute_precision_recall(
        self,
        targets: list[str] | list[list[str]],
        predictions: list[str] | list[list[str]],
        confidence: list[float] | None = None,
        threshold: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Calculate precision and recall at peptide- and AA-level.

        Args:
            targets (list[str] | list[list[str]]): Target peptides.
            predictions (list[str] | list[list[str]]): Model predicted peptides.
            confidence (list[float] | None): Optional model confidence.
            threshold (float | None): Optional confidence threshold.
        """
        targets = self._split_sequences(targets)
        predictions = self._split_sequences(predictions)

        n_targ_aa, n_pred_aa, n_match_aa = 0, 0, 0
        n_pred_pep, n_match_pep = 0, 0

        if confidence is None or threshold is None:
            threshold = 0
            confidence = np.ones(len(predictions))

        for i in range(len(targets)):
            targ = self._split_peptide(targets[i])
            pred = self._split_peptide(predictions[i])
            conf = confidence[i]  # type: ignore

            # Legacy for old regex, may be removed
            if len(pred) > 0 and pred[0] == "":
                pred = []

            n_targ_aa += len(targ)
            if conf >= threshold and len(pred) > 0:
                n_pred_aa += len(pred)
                n_pred_pep += 1

                # pred = [x.replace('I', 'L') for x in pred]
                # n_match_aa += np.sum([m[0]==' ' for m in difflib.ndiff(targ,pred)])
                n_match = self._novor_match(targ, pred)
                n_match_aa += n_match

                if len(pred) == len(targ) and len(targ) == n_match:
                    n_match_pep += 1

        pep_recall = n_match_pep / len(targets)
        aa_recall = n_match_aa / n_targ_aa

        if n_pred_pep == 0:
            pep_precision = 1.0
            aa_prec = 1.0
        else:
            pep_precision = n_match_pep / n_pred_pep
            aa_prec = n_match_aa / n_pred_aa

        return aa_prec, aa_recall, pep_recall, pep_precision

    def calc_auc(
        self,
        targs: list[str] | list[list[str]],
        preds: list[str] | list[list[str]],
        conf: list[float],
    ) -> float:
        """Calculate the peptide-level AUC."""
        x, y = self._get_pr_curve(targs, preds, conf)
        recall, precision = np.array(x)[::-1], np.array(y)[::-1]

        width = recall[1:] - recall[:-1]
        height = np.minimum(precision[1:], precision[:-1])
        top = np.maximum(precision[1:], precision[:-1])
        side = top - height
        return (width * height).sum() + 0.5 * (side * width).sum()  # type: ignore

    def _split_sequences(self, seq: list[str] | list[list[str]]) -> list[list[str]]:
        return [self.residue_set.tokenize(x) if isinstance(x, str) else x for x in seq]

    def _split_peptide(self, peptide: str | list[str]) -> list[str]:
        if not isinstance(peptide, str):
            return peptide
        return self.residue_set.tokenize(peptide)  # type: ignore

    def _get_pr_curve(
        self,
        targs: list[str] | list[list[str]],
        preds: list[str] | list[list[str]],
        conf: np.ndarray,
        N: int = 20,  # noqa: N803
    ) -> tuple[list[float], list[float]]:
        x, y = [], []
        t_idx = np.argsort(np.array(conf))
        t_idx = t_idx[~conf[t_idx].isna()]
        t_idx = list(t_idx[(t_idx.shape[0] * np.arange(N) / N).astype(int)]) + [
            t_idx[-1]
        ]
        for t in conf[t_idx]:
            _, _, recall, precision = self.compute_precision_recall(
                targs, preds, conf, t
            )
            x.append(recall)
            y.append(precision)
        return x, y

    def _mass(self, seq: str | list[str], charge: int | None = None) -> float:
        """Calculate a peptide's mass or m/z."""
        seq = self._split_peptide(seq)
        return self.residue_set.get_sequence_mass(seq, charge)  # type: ignore
        # calc_mass = sum([self.residues[aa] for aa in seq]) + H2O_MASS

        # if charge is not None:
        #     # Neutral mass
        #     calc_mass = (calc_mass / charge) + PROTON_MASS_AMU

        # return calc_mass

    def _calc_mass_error(
        self, mz_theoretical: float, mz_measured: float, charge: int, isotope: int = 0
    ) -> float:
        """Calculate the mass error between theoretical and actual mz in ppm."""
        return float(
            (mz_theoretical - (mz_measured - isotope * CARBON_MASS_DELTA / charge))
            / mz_measured
            * 10**6
        )

    # Adapted from https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/evaluate.py
    def _novor_match(
        self,
        a: list[str],
        b: list[str],
    ) -> int:
        """Number of AA matches with novor method."""
        n = 0

        mass_a: list[float] = [self.residue_set.get_mass(x) for x in a]
        mass_b: list[float] = [self.residue_set.get_mass(x) for x in b]
        cum_mass_a = np.cumsum(mass_a)
        cum_mass_b = np.cumsum(mass_b)

        i, j = 0, 0
        while i < len(a) and j < len(b):
            if abs(cum_mass_a[i] - cum_mass_b[j]) < self.cum_mass_threshold:
                n += int(abs(mass_a[i] - mass_b[j]) < self.ind_mass_threshold)
                i += 1
                j += 1
            elif cum_mass_a[i] > cum_mass_b[j]:
                i += 1
            else:
                j += 1
        return n
