"""EuroSAT RGB loading, the known/novel class split, and torch datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"

ALL_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
# Held out of training entirely: the model must flag these as "never seen", not force them into a known class.
NOVEL_CLASSES = ["Highway", "Industrial"]
KNOWN_CLASSES = [c for c in ALL_CLASSES if c not in NOVEL_CLASSES]

MEAN = np.array([0.344, 0.380, 0.408], dtype=np.float32)
STD = np.array([0.203, 0.137, 0.116], dtype=np.float32)


def prepare(out: Path = DATA / "eurosat.npz") -> Path:
    """Download EuroSAT RGB from the Hugging Face hub and cache it as uint8 arrays."""
    from datasets import load_dataset

    ds = load_dataset("timm/eurosat-rgb")
    arrs = {}
    for split in ("train", "validation", "test"):
        part = ds[split]
        arrs[f"x_{split}"] = np.stack([np.asarray(im.convert("RGB")) for im in part["image"]]).astype(np.uint8)
        arrs[f"y_{split}"] = np.asarray(part["label"], dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrs)
    return out


def load(path: Path = DATA / "eurosat.npz") -> dict[str, np.ndarray]:
    if not path.exists():
        prepare(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def known_mask(y_all: np.ndarray) -> np.ndarray:
    novel_ids = [ALL_CLASSES.index(c) for c in NOVEL_CLASSES]
    return ~np.isin(y_all, novel_ids)


def to_known_index(y_all: np.ndarray) -> np.ndarray:
    """Map 10-class labels to 0..7 for known classes, -1 for novel."""
    lut = np.full(len(ALL_CLASSES), -1, dtype=np.int64)
    for i, c in enumerate(KNOWN_CLASSES):
        lut[ALL_CLASSES.index(c)] = i
    return lut[y_all]


def normalize(x_uint8: np.ndarray) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,3,H,W) float32 normalized."""
    x = x_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
