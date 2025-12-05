from __future__ import annotations

from enum import Enum

import polars as pl

# CLI constants

DEFAULT_TRAIN_CONFIG_PATH = "../configs"
DEFAULT_INFERENCE_CONFIG_PATH = "../configs/inference"
DEFAULT_INFERENCE_CONFIG_NAME = "default"

# Logging

USE_RICH_HANDLER = True
LOGGING_SHOW_PATH = False
LOGGING_SHOW_TIME = True

# Constants

H2O_MASS = 18.0106
CARBON_MASS_DELTA = 1.00335
PROTON_MASS_AMU = 1.007276
MASS_SCALE = 10000
MAX_MASS = 4000.0

MAX_SEQUENCE_LENGTH = 200

# Buffer size for shuffling the training dataset
SHUFFLE_BUFFER_SIZE = 100_000


class PrecursorDimension(Enum):
    """Names corresponding to indices in the precursor tensor."""

    PRECURSOR_MASS = 0
    PRECURSOR_CHARGE = 1
    PRECURSOR_MZ = 2


class SpecialTokens(Enum):
    """Special tokens used by the ResidueSet and model."""

    PAD_TOKEN = "[PAD]"  # Padding token
    EOS_TOKEN = "[EOS]"  # End of sequence
    SOS_TOKEN = "[SOS]"  # Start of sequence


PRECURSOR_DIM = 3

DIFFUSION_BASE_STEPS = 20
DIFFUSION_START_STEP = 15
DIFFUSION_EVAL_STEPS = (3, 8, 13, 18)


# Data handler constants


class MSColumns(Enum):
    """Columns names used by SpectrumDataFrame."""

    MZ_ARRAY = "mz_array"
    INTENSITY_ARRAY = "intensity_array"
    PRECURSOR_MZ = "precursor_mz"
    PRECURSOR_CHARGE = "precursor_charge"
    PRECURSOR_MASS = "precursor_mass"
    RETENTION_TIME = "retention_time"


ANNOTATED_COLUMN = "sequence"

MS_TYPES: dict[MSColumns, pl.DataType] = {
    MSColumns.MZ_ARRAY: pl.List(pl.Float64),
    MSColumns.INTENSITY_ARRAY: pl.List(pl.Float64),
    MSColumns.PRECURSOR_MZ: pl.Float64,
    MSColumns.PRECURSOR_CHARGE: pl.Int64,
    MSColumns.PRECURSOR_MASS: pl.Float64,
    MSColumns.RETENTION_TIME: pl.Float64,
}

DATASETS_COLUMNS = [
    MSColumns.MZ_ARRAY.value,
    MSColumns.INTENSITY_ARRAY.value,
    MSColumns.PRECURSOR_MZ.value,
    MSColumns.PRECURSOR_CHARGE.value,
    MSColumns.PRECURSOR_MASS.value,
]

ANNOTATION_ERROR = "Attempting to load annotated dataset, but some or all sequence annotations are missing."

LEGACY_PTM_TO_UNIMOD: dict[str, str] = {
    "M(ox)": "M[UNIMOD:35]",
    "S(p)": "S[UNIMOD:21]",
    "T(p)": "T[UNIMOD:21]",
    "Y(p)": "Y[UNIMOD:21]",
    "Q(+0.98)": "Q[UNIMOD:7]",
    "N(+0.98)": "N[UNIMOD:7]",
    "M(+15.99)": "M[UNIMOD:35]",
    "C(+57.02)": "C[UNIMOD:4]",
    "S(+79.97)": "S[UNIMOD:21]",
    "T(+79.97)": "T[UNIMOD:21]",
    "Y(+79.97)": "Y[UNIMOD:21]",
    "Q(+.98)": "Q[UNIMOD:7]",
    "N(+.98)": "N[UNIMOD:7]",
    "(+42.01)": "[UNIMOD:1]",
    "(+43.01)": "[UNIMOD:5]",
    "(-17.03)": "[UNIMOD:385]",
}

# Required output columns
PREDICTION_COLUMNS = [
    "prediction_id",
    "predictions",
    "targets",
    "prediction_log_probability",
    "prediction_token_log_probabilities",
    "group",
]

REFINEMENT_COLUMN = "input_predictions"
REFINEMENT_PROBABILITY_COLUMN = "input_log_probabilities"
