from typing import Any

import numpy as np
import pytest
import torch

from instanovo.utils.residues import ResidueSet


def test_init(residue_set: Any) -> None:
    """Test residue set initialisation."""
    rs = ResidueSet(
        residue_masses=residue_set.residue_masses, residue_remapping={"E": "F"}
    )

    assert rs.residue_masses == {
        "A": 10.5,
        "B": 20.75,
        "C": 15.68,
        "D": 18.25,
        "E": 12.33,
    }
    assert rs.residue_remapping == {"E": "F"}
    assert rs.special_tokens == ["[PAD]", "[SOS]", "[EOS]"]
    assert rs.vocab == ["[PAD]", "[SOS]", "[EOS]", "A", "B", "C", "D", "E"]
    assert rs.residue_to_index == {
        "[PAD]": 0,
        "[SOS]": 1,
        "[EOS]": 2,
        "A": 3,
        "B": 4,
        "C": 5,
        "D": 6,
        "E": 7,
    }
    assert rs.index_to_residue == {
        0: "[PAD]",
        1: "[SOS]",
        2: "[EOS]",
        3: "A",
        4: "B",
        5: "C",
        6: "D",
        7: "E",
    }
    assert (
        rs.tokenizer_regex
        == r"(\[UNIMOD:\d+\]|\([^)]+\))|([A-Z](?:\[UNIMOD:\d+\]|\([^)]+\))?)"
    )
    assert rs.PAD_INDEX == 0
    assert rs.SOS_INDEX == 1
    assert rs.EOS_INDEX == 2


def test_update_remapping(residue_set: Any) -> None:
    """Test residue set remapping update."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    rs.update_remapping(mapping={"A": "A(+0.57)"})

    assert rs.residue_remapping == {"A": "A(+0.57)"}


def test_get_mass(residue_set: Any) -> None:
    """Test residue mass."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    assert np.isclose(rs.get_mass("A"), 10.5)

    rs.update_remapping(mapping={"A(+0.57)": "A"})
    assert np.isclose(rs.get_mass("A(+0.57)"), 10.5)

    with pytest.raises(KeyError):
        rs.get_mass("K")


def test_get_sequence_mass(residue_set: Any) -> None:
    """Test residue sequence mass."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    assert np.isclose(rs.get_sequence_mass("BACDA", charge=None), 93.6906, rtol=1e-2)
    assert np.isclose(rs.get_sequence_mass("BACDA", charge=1), 94.697876, rtol=1e-2)
    assert np.isclose(rs.get_sequence_mass("BACDA", charge=2), 47.852576, rtol=1e-2)

    with pytest.raises(KeyError):
        rs.get_sequence_mass("ABCK", charge=None)


def test_tokenize(residue_set: Any) -> None:
    """Test peptide tokenizer."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    tokens = rs.tokenize("BACDE")
    assert tokens == ["B", "A", "C", "D", "E"]

    tokens = rs.tokenize("BA(+57.02)CDE")
    assert tokens == ["B", "A(+57.02)", "C", "D", "E"]

    tokens = rs.tokenize("KA(+57.02)CDE")
    assert tokens == ["K", "A(+57.02)", "C", "D", "E"]


def test_detokenize(residue_set: Any) -> None:
    """Test residue detokenizer."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    peptide = rs.detokenize(["B", "A", "C", "D", "E"])
    assert peptide == "BACDE"

    peptide = rs.detokenize(["B", "A(+57.02)", "C", "D", "E"])
    assert peptide == "BA(+57.02)CDE"

    peptide = rs.detokenize(["K", "A(+57.02)", "C", "D", "E"])
    assert peptide == "KA(+57.02)CDE"


def test_encoder(residue_set: Any) -> None:
    """Test residue encoder."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    tokens = rs.encode(["B", "A", "C", "D", "E"])
    assert tokens == [4, 3, 5, 6, 7]

    tokens = rs.encode(["B", "A", "C", "D", "E"], add_eos=True)
    assert tokens == [4, 3, 5, 6, 7, 2]

    tokens = rs.encode(["B", "A", "C", "D", "E"], add_eos=True, pad_length=7)
    assert tokens == [4, 3, 5, 6, 7, 2, 0]

    tokens = rs.encode(
        ["B", "A", "C", "D", "E"], add_eos=True, pad_length=7, return_tensor="np"
    )
    assert np.array_equal(tokens, np.array([4, 3, 5, 6, 7, 2, 0]))

    tokens = rs.encode(
        ["B", "A", "C", "D", "E"], add_eos=True, pad_length=7, return_tensor="pt"
    )
    assert torch.equal(tokens, torch.tensor([4, 3, 5, 6, 7, 2, 0]))


def test_decoder(residue_set: Any) -> None:
    """Test residue decoder."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    tokens = rs.decode([4, 3, 5, 6, 7])
    assert tokens == ["B", "A", "C", "D", "E"]

    tokens = rs.decode([4, 3, 5, 6, 7], reverse=True)
    assert tokens == ["E", "D", "C", "A", "B"]

    tokens = rs.decode([4, 3, 5, 6, 7, 2, 0])
    assert tokens == ["B", "A", "C", "D", "E"]


def test_length(residue_set: Any) -> None:
    """Test residue sequence length."""
    rs = ResidueSet(residue_masses=residue_set.residue_masses)

    assert len(rs) == 8
