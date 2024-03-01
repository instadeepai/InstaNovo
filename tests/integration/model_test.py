from __future__ import annotations

import os

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader

from instanovo.constants import MASS_SCALE
from instanovo.inference.knapsack import Knapsack
from instanovo.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo


def _setup_knapsack(model: InstaNovo) -> Knapsack:
    residue_masses = model.peptide_mass_calculator.masses
    residue_masses["$"] = 0
    residue_indices = model.decoder._aa2idx
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=4000.00,
        mass_scale=MASS_SCALE,
    )


def test_model(
    instanovo_checkpoint: str,
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    knapsack_dir: str,
) -> None:
    """Test loading an InstaNovo model and doing inference end-to-end."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = InstaNovo.load(instanovo_checkpoint)
    model = model.to(device).eval()

    s2i = {v: k for k, v in model.i2s.items()}
    assert s2i == {
        "A": 1,
        "C(+57.02)": 6,
        "D": 10,
        "E": 13,
        "F": 16,
        "G": 0,
        "H": 15,
        "I": 8,
        "K": 12,
        "L": 7,
        "M": 14,
        "M(+15.99)": 20,
        "N": 9,
        "N(+.98)": 21,
        "P": 3,
        "Q": 11,
        "Q(+.98)": 22,
        "R": 17,
        "S": 2,
        "T": 5,
        "V": 4,
        "W": 19,
        "Y": 18,
    }

    n_peaks = config["n_peaks"]
    assert n_peaks == 200

    ds = SpectrumDataset(dataset, s2i, config["n_peaks"], return_str=True)
    assert len(ds) == 271
    spectrum, precursor_mz, precursor_charge, peptide = ds[0]
    assert torch.allclose(
        spectrum,
        torch.Tensor(
            [
                [1.0096e02, 6.8907e-02],
                [1.1006e02, 6.6649e-02],
                [1.1646e02, 6.5169e-02],
                [1.2910e02, 1.3785e-01],
                [1.3009e02, 1.3666e-01],
                [1.4711e02, 1.4966e-01],
                [1.7309e02, 7.0756e-02],
                [1.8612e02, 1.0042e-01],
                [2.0413e02, 1.4815e-01],
                [2.7303e02, 7.4630e-02],
                [2.8318e02, 1.1245e-01],
                [3.0119e02, 5.0341e-01],
                [3.2845e02, 7.8869e-02],
                [3.7222e02, 1.9128e-01],
                [4.7877e02, 7.5372e-02],
                [5.2873e02, 8.4931e-02],
                [5.7176e02, 8.8744e-02],
                [5.7975e02, 9.3491e-02],
                [6.1527e02, 8.3923e-02],
                [6.5630e02, 9.5524e-02],
                [7.7837e02, 1.2861e-01],
                [7.7887e02, 2.2789e-01],
                [7.7938e02, 6.1620e-02],
                [7.9137e02, 1.0959e-01],
                [7.9189e02, 1.0925e-01],
                [1.0365e03, 9.1989e-02],
                [1.1015e03, 9.4942e-02],
                [1.1395e03, 2.0198e-01],
                [1.1895e03, 9.5279e-02],
                [1.2285e03, 1.2807e-01],
                [1.2556e03, 1.1814e-01],
                [1.2565e03, 1.2252e-01],
                [1.2716e03, 2.7978e-01],
                [1.2996e03, 4.9205e-01],
            ]
        ),
        rtol=1e-04,
    )
    assert precursor_mz == 800.38427734375
    assert precursor_charge == 2.0
    assert peptide == "TPGREDAAEETAAPGK"

    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(dl))

    spectra, precursors, spectra_mask, peptides, _ = batch
    assert torch.allclose(
        spectra,
        torch.Tensor(
            [
                [
                    [1.0096e02, 6.8907e-02],
                    [1.1006e02, 6.6649e-02],
                    [1.1646e02, 6.5169e-02],
                    [1.2910e02, 1.3785e-01],
                    [1.3009e02, 1.3666e-01],
                    [1.4711e02, 1.4966e-01],
                    [1.7309e02, 7.0756e-02],
                    [1.8612e02, 1.0042e-01],
                    [2.0413e02, 1.4815e-01],
                    [2.7303e02, 7.4630e-02],
                    [2.8318e02, 1.1245e-01],
                    [3.0119e02, 5.0341e-01],
                    [3.2845e02, 7.8869e-02],
                    [3.7222e02, 1.9128e-01],
                    [4.7877e02, 7.5372e-02],
                    [5.2873e02, 8.4931e-02],
                    [5.7176e02, 8.8744e-02],
                    [5.7975e02, 9.3491e-02],
                    [6.1527e02, 8.3923e-02],
                    [6.5630e02, 9.5524e-02],
                    [7.7837e02, 1.2861e-01],
                    [7.7887e02, 2.2789e-01],
                    [7.7938e02, 6.1620e-02],
                    [7.9137e02, 1.0959e-01],
                    [7.9189e02, 1.0925e-01],
                    [1.0365e03, 9.1989e-02],
                    [1.1015e03, 9.4942e-02],
                    [1.1395e03, 2.0198e-01],
                    [1.1895e03, 9.5279e-02],
                    [1.2285e03, 1.2807e-01],
                    [1.2556e03, 1.1814e-01],
                    [1.2565e03, 1.2252e-01],
                    [1.2716e03, 2.7978e-01],
                    [1.2996e03, 4.9205e-01],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00],
                ],
                [
                    [1.0206e02, 1.5659e-01],
                    [1.2910e02, 8.5658e-02],
                    [1.3009e02, 6.9173e-02],
                    [1.4711e02, 9.2828e-02],
                    [1.5508e02, 7.6935e-02],
                    [1.6910e02, 4.4613e-02],
                    [1.7111e02, 1.0253e-01],
                    [1.7309e02, 1.0162e-01],
                    [1.9508e02, 4.1588e-02],
                    [1.9911e02, 7.1197e-02],
                    [2.0109e02, 7.8843e-02],
                    [2.0413e02, 1.0829e-01],
                    [2.1309e02, 6.7321e-02],
                    [2.3110e02, 5.2743e-02],
                    [2.3909e02, 4.1613e-02],
                    [2.4108e02, 5.1726e-02],
                    [2.5909e02, 7.4543e-02],
                    [2.8318e02, 9.6649e-02],
                    [2.8412e02, 4.9769e-02],
                    [3.0119e02, 4.0183e-01],
                    [3.1365e02, 6.9522e-02],
                    [3.2216e02, 5.1740e-02],
                    [3.2816e02, 4.3714e-02],
                    [3.2865e02, 8.4462e-02],
                    [3.4213e02, 4.7763e-02],
                    [3.5017e02, 8.2215e-02],
                    [3.5421e02, 5.3792e-02],
                    [3.5517e02, 1.0384e-01],
                    [3.7222e02, 1.6867e-01],
                    [3.7669e02, 5.1280e-02],
                    [3.8569e02, 7.6477e-02],
                    [4.1317e02, 7.4231e-02],
                    [4.2092e02, 4.0213e-02],
                    [4.4084e02, 5.2411e-02],
                    [4.4326e02, 1.5205e-01],
                    [4.5521e02, 7.7468e-02],
                    [4.5819e02, 4.3700e-02],
                    [4.7821e02, 1.0422e-01],
                    [4.8857e02, 7.0330e-02],
                    [5.4431e02, 2.1082e-01],
                    [5.5525e02, 1.6596e-01],
                    [5.5725e02, 4.2561e-02],
                    [5.6125e02, 5.4916e-02],
                    [6.0027e02, 7.5823e-02],
                    [6.2629e02, 9.8458e-02],
                    [6.5534e02, 9.4754e-02],
                    [6.5629e02, 1.0460e-01],
                    [6.6933e02, 1.2577e-01],
                    [6.7335e02, 1.8197e-01],
                    [6.9733e02, 1.7234e-01],
                    [7.0133e02, 1.1745e-01],
                    [7.0182e02, 5.1286e-02],
                    [7.0933e02, 5.0761e-02],
                    [7.2733e02, 2.1210e-01],
                    [7.2932e02, 5.3202e-02],
                    [7.4085e02, 4.9344e-02],
                    [7.4136e02, 5.5698e-02],
                    [7.4987e02, 1.0906e-01],
                    [7.5434e02, 4.2837e-02],
                    [7.7037e02, 7.6015e-02],
                    [7.8037e02, 1.1569e-01],
                    [7.8438e02, 7.9805e-02],
                    [7.9837e02, 2.4075e-01],
                    [8.0239e02, 1.4020e-01],
                    [8.2636e02, 1.7631e-01],
                    [8.7343e02, 4.6804e-02],
                    [9.2741e02, 2.3908e-01],
                    [9.3891e02, 4.2673e-02],
                    [9.5541e02, 5.4546e-02],
                    [1.0385e03, 1.2572e-01],
                    [1.0564e03, 1.6373e-01],
                    [1.1395e03, 1.2800e-01],
                    [1.1985e03, 5.0744e-02],
                    [1.1996e03, 4.8810e-02],
                    [1.3836e03, 7.1560e-02],
                    [1.4807e03, 1.4328e-01],
                    [1.4997e03, 1.1361e-01],
                ],
            ]
        ),
        rtol=1e-04,
    )
    assert torch.allclose(
        precursors, torch.Tensor([[1598.7540, 2.0000, 800.3843], [1598.7551, 3.0000, 533.9257]])
    )
    assert torch.equal(
        spectra_mask,
        torch.Tensor(
            [
                [
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
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
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
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ]
        ),
    )
    assert peptides == ("TPGREDAAEETAAPGK", "TPGREDAAEETAAPGK")

    spectra = spectra.to(device)
    precursors = precursors.to(device)

    if not os.path.exists(knapsack_dir):
        knapsack = _setup_knapsack(model)
        decoder = KnapsackBeamSearchDecoder(model, knapsack)
        knapsack.save(knapsack_dir)
    else:
        decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_dir)

    assert os.path.isfile(os.path.join(knapsack_dir, "parameters.pkl"))
    assert os.path.isfile(os.path.join(knapsack_dir, "chart.npy"))
    assert os.path.isfile(os.path.join(knapsack_dir, "masses.npy"))
    assert isinstance(decoder, KnapsackBeamSearchDecoder)

    with torch.no_grad():
        p = decoder.decode(
            spectra=spectra,
            precursors=precursors,
            beam_size=config["n_beams"],
            max_length=config["max_length"],
        )
    preds = ["".join(x.sequence) if not isinstance(x, list) else "" for x in p]
    probs = [x.log_probability if not isinstance(x, list) else -1 for x in p]

    assert preds == ["NRNVGDQNGC(+57.02)LAPGK", "TDRPGEAAEETAAPGK"]
    assert np.allclose(probs, [-8.156049728393555, -3.1159517765045166])
