"""Batch-level drift monitoring on embeddings: kernel MMD with a permutation test, plus per-class PSI on predictions."""
from __future__ import annotations

import numpy as np


def _rbf(a, b, gamma):
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    kxx, kyy, kxy = _rbf(x, x, gamma), _rbf(y, y, gamma), _rbf(x, y, gamma)
    n, m = len(x), len(y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


def median_gamma(x: np.ndarray) -> float:
    d = (x * x).sum(1)[:, None] + (x * x).sum(1)[None] - 2 * x @ x.T
    return 1.0 / np.median(d[np.triu_indices(len(x), 1)])


def mmd_test(ref: np.ndarray, batch: np.ndarray, n_perm: int = 200, seed: int = 0, gamma: float | None = None):
    rng = np.random.default_rng(seed)
    gamma = gamma or median_gamma(ref)
    stat = mmd2(ref, batch, gamma)
    pool = np.concatenate([ref, batch])
    n = len(ref)
    null = []
    for _ in range(n_perm):
        p = rng.permutation(len(pool))
        null.append(mmd2(pool[p[:n]], pool[p[n:]], gamma))
    p_value = (1 + np.sum(np.asarray(null) >= stat)) / (n_perm + 1)
    return {"mmd2": stat, "p_value": float(p_value)}


def psi(ref_pred: np.ndarray, batch_pred: np.ndarray, n_classes: int, eps: float = 1e-4) -> float:
    p = np.bincount(ref_pred, minlength=n_classes) / len(ref_pred) + eps
    q = np.bincount(batch_pred, minlength=n_classes) / len(batch_pred) + eps
    return float(np.sum((q - p) * np.log(q / p)))
