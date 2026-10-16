"""Torch-only pieces kept out of data.py so the ONNX service never imports torch."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import normalize


class ChipDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        if self.transform is not None:
            img = self.transform(img)
        t = torch.from_numpy(normalize(img[None])[0])
        return t, int(self.y[i])


def seed_everything(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
