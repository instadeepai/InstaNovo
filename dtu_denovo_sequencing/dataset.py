from __future__ import annotations

import os
import pickle
import zipfile
from typing import Any
from typing import Callable

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


class SpecDataset(Dataset):
    """Dataset of Mass-spectometry data."""

    def __init__(
        self,
        df: pd.DataFrame,
        s2i: dict[str, int],
        i2s: list,
        normalise_intensity: bool = True,
        force_eos: bool = True,
        ox_symbol: str = "#",
    ) -> None:
        super().__init__()
        self.df = df
        self.s2i: Callable[[str], int] = lambda x: s2i[x]
        self.i2s: Callable[[int], Any] = lambda x: i2s[x]
        self.normalise_intensity = normalise_intensity
        self.force_eos = force_eos

        self.ox_symbol = ox_symbol
        self.PAD = i2s.index("PAD")
        self.SOS = i2s.index("SOS")
        self.EOS = i2s.index("EOS")

        # TODO: the line below gives a SettingWithCopyWarning
        # A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_indexer,col_indexer] = value instead
        # See the caveats in the documentation:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        self.df["Sequence"] = self.df["Modified sequence"].map(
            lambda x: [self.s2i(y) for y in x[1:-1].replace("M(ox)", "#")]
        )
        if self.force_eos:
            # TODO: Same SettingWithCopyWarning here
            self.df["Sequence"] = self.df["Sequence"].map(lambda x: x + [self.EOS])

    def seq_to_aa(self, seq: Tensor) -> str:
        """Convert a sequence to amino acids."""
        aa = []
        for p in seq:
            if p == self.EOS:
                break
            aa.append(self.i2s(p))
        return "".join(aa)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.df.iloc[index]

        mass = torch.from_numpy(row["Mass values"])
        intensity = torch.from_numpy(row["Intensity"])
        mz = torch.tensor(row["MS/MS m/z"]).repeat_interleave(mass.shape[0])
        rt = torch.tensor(row["Retention time"]).repeat_interleave(mass.shape[0])

        if self.normalise_intensity:
            intensity /= intensity.max()
        x = torch.stack([mass, intensity, mz, rt]).T

        # seq = row["Modified sequence"][1:-1]
        # seq = seq.replace("M(ox)", "#")
        # seq = [self.s2i(x) for x in seq]
        seq = row["Sequence"]
        # if self.force_eos:
        #     seq += [self.EOS]
        y = torch.tensor(seq)

        return x, y


def collate_batch(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate lists of samples into batches."""
    x, y = zip(*batch)

    llx = torch.tensor([z.shape[0] for z in x], dtype=torch.long)
    lly = torch.tensor([z.shape[0] for z in y], dtype=torch.long)

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)

    x_padding_mask = torch.arange(x.shape[1], dtype=torch.long)[None, :] >= llx[:, None]
    y_padding_mask = torch.arange(y.shape[1], dtype=torch.long)[None, :] >= lly[:, None]

    return x, y, x_padding_mask, y_padding_mask


def load_all(path: str, verbose: bool = True) -> pd.DataFrame:
    """Load all zipped pickle files."""
    df_list = []

    enum = os.listdir(path)[:]
    if verbose:
        enum = tqdm(enum)

    for filename in enum:
        dst = f"{path}{filename}"
        if verbose:
            print(dst)
        with zipfile.ZipFile(dst, "r") as f:
            df_list.append(pickle.loads(f.read(f.namelist()[0])))
    return pd.concat(df_list)
