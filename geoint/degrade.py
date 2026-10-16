"""Synthetic sensor / collection degradations applied to uint8 RGB chips.

Each function takes an (H,W,3) uint8 chip, a severity in 1..5 and an RNG, and returns a uint8 chip.
These stand in for things an operational imagery feed actually does: haze and thin cloud,
defocus/motion blur, sensor noise, aggressive compression on a constrained link, and coarser GSD.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def haze(img, s, rng):
    a = [0.15, 0.3, 0.45, 0.6, 0.75][s - 1]
    h, w, _ = img.shape
    # low-frequency cloud mask so haze is patchy, not a uniform fade
    coarse = rng.random((4, 4)).astype(np.float32)
    mask = np.asarray(Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255
    alpha = np.clip(a * (0.5 + mask), 0, 0.95)[..., None]
    out = img.astype(np.float32) * (1 - alpha) + 235.0 * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img, s, rng):
    r = [0.6, 1.0, 1.5, 2.0, 3.0][s - 1]
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(r)))


def noise(img, s, rng):
    sd = [6, 12, 20, 30, 45][s - 1]
    out = img.astype(np.float32) + rng.normal(0, sd, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(img, s, rng):
    q = [40, 25, 15, 10, 5][s - 1]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def lowres(img, s, rng):
    f = [1.5, 2, 3, 4, 6][s - 1]
    h, w, _ = img.shape
    small = Image.fromarray(img).resize((max(2, int(w / f)), max(2, int(h / f))), Image.BILINEAR)
    return np.asarray(small.resize((w, h), Image.BILINEAR))


DEGRADATIONS = {"haze": haze, "blur": blur, "noise": noise, "jpeg": jpeg, "lowres": lowres}


def apply(x: np.ndarray, kind: str, severity: int, seed: int = 0) -> np.ndarray:
    """Degrade a batch (N,H,W,3)."""
    rng = np.random.default_rng(seed)
    fn = DEGRADATIONS[kind]
    return np.stack([fn(im, severity, rng) for im in x])


class RandomDegrade:
    """Training-time augmentation: with prob p, apply a random degradation at a random severity."""

    def __init__(self, p: float = 0.5, max_severity: int = 4, seed: int = 0):
        self.p, self.max_severity = p, max_severity
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        img = flip_rot(img, self.rng)
        if self.rng.random() < self.p:
            kind = self.rng.choice(list(DEGRADATIONS))
            img = DEGRADATIONS[kind](img, int(self.rng.integers(1, self.max_severity + 1)), self.rng)
        return img


def flip_rot(img, rng):
    """Overhead imagery has no canonical up: random dihedral transform."""
    k = int(rng.integers(0, 4))
    img = np.rot90(img, k)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


class FlipRot:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        return flip_rot(img, self.rng)
