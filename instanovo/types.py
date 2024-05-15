from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Integer

SequenceLogProbabilities = Float[torch.Tensor, ""]
TokenLogProbabilities = Float[torch.Tensor, "token"]
ResidueLogits = Float[torch.Tensor, " residue"]
ResidueLogProbabilities = Float[torch.Tensor, " residue"]
PrecursorFeatures = Float[torch.Tensor, "3"]

Peptide = Integer[torch.Tensor, "token"]
PeptideMask = Bool[torch.Tensor, "token"]
PeptideEmbedding = Float[torch.Tensor, "token peptide_embedding"]

SpectrumEmbedding = Float[torch.Tensor, "peak embedding"]
Spectrum = Float[torch.Tensor, "peak 2"]
SpectrumMask = Bool[torch.Tensor, " peak"]


KnapsackChart = Bool[torch.Tensor, "mass residue"]

Mass = Float[torch.Tensor, ""]
DiscretizedMass = Integer[torch.Tensor, ""]
MassArray = Integer[np.ndarray, "mass_item"]

TimeStep = Integer[torch.Tensor, ""]
TimeEmbedding = Float[torch.Tensor, "embedding"]
