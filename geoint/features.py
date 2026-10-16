"""Hand-built spectral/texture features for the scikit-learn baseline (no deep learning)."""
from __future__ import annotations

import numpy as np


def chip_features(x: np.ndarray) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,F) float: per-band stats, color histograms, ExG/brightness, gradient-orientation histogram."""
    xf = x.astype(np.float32) / 255.0
    n = len(x)
    r, g, b = xf[..., 0], xf[..., 1], xf[..., 2]
    feats = []
    for band in (r, g, b):
        flat = band.reshape(n, -1)
        feats += [flat.mean(1), flat.std(1), np.percentile(flat, 10, axis=1), np.percentile(flat, 90, axis=1)]
    exg = (2 * g - r - b).reshape(n, -1)  # excess-green vegetation index (RGB proxy for NDVI)
    feats += [exg.mean(1), exg.std(1)]
    hist = [np.stack([np.histogram(c, bins=8, range=(0, 1))[0] for c in band.reshape(n, -1)]) / band[0].size
            for band in (r, g, b)]
    gray = xf.mean(-1)
    gy, gx = np.gradient(gray, axis=(1, 2))
    mag = np.hypot(gx, gy)
    # fold orientation to [0, pi): roads/rivers/field rows show up as peaked orientation histograms
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    hog = np.stack([np.histogram(a.ravel(), bins=9, range=(0, np.pi), weights=m.ravel())[0]
                    for a, m in zip(ang, mag)])
    hog = hog / (hog.sum(1, keepdims=True) + 1e-8)
    feats += [mag.reshape(n, -1).mean(1), mag.reshape(n, -1).std(1)]
    return np.column_stack(feats + hist + [hog]).astype(np.float32)
