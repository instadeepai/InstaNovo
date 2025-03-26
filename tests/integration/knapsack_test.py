import os
from pathlib import Path
from typing import Any

import pytest

from instanovo.constants import MASS_SCALE
from instanovo.inference.knapsack import Knapsack
from instanovo.transformer.predict import _setup_knapsack


def test_knapsack(instanovo_model: tuple[Any, Any], knapsack_dir: str, tmp_path: Path) -> None:
    """Test knapsack creation, loading, and functionality."""
    model, _ = instanovo_model

    knapsack = _setup_knapsack(model)

    assert isinstance(knapsack, Knapsack)
    assert knapsack.max_mass == 4000.0
    assert knapsack.mass_scale == MASS_SCALE
    assert knapsack.residues == ["", "A", "B", "C", "D", "E", "[PAD]", "[SOS]", "[EOS]"]
    assert knapsack.residue_indices == {
        "[PAD]": 0,
        "[SOS]": 1,
        "[EOS]": 2,
        "A": 3,
        "B": 4,
        "C": 5,
        "D": 6,
        "E": 7,
    }
    assert knapsack.masses.shape == (384853,)
    assert knapsack.chart.shape == (40000001, 8)

    with pytest.raises(FileExistsError):
        knapsack.save(knapsack_dir)

    trial_path = tmp_path / "test_knapsack"
    trial_str = str(trial_path)

    knapsack.save(trial_str)

    assert os.path.exists(os.path.join(trial_str, "parameters.pkl"))
    assert os.path.exists(os.path.join(trial_str, "masses.npy"))
    assert os.path.exists(os.path.join(trial_str, "chart.npy"))

    knapsack = knapsack.from_file(trial_str)

    feasible_masses = knapsack.get_feasible_masses(target_mass=46.76, tolerance=1.0)
    assert feasible_masses == [462600, 469300, 470000, 470400, 471800, 474900]
