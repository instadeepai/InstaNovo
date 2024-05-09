from __future__ import annotations

from enum import Enum

import torch

H2O_MASS = 18.0106
CARBON_MASS_DELTA = 1.00335
PROTON_MASS_AMU = 1.007276
MASS_SCALE = 10000

MAX_SEQUENCE_LENGTH = 200


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

INTEGER = torch.int64
DIFFUSION_START_STEP = 15
DIFFUSION_EVAL_STEPS = (3, 8, 13, 18)
