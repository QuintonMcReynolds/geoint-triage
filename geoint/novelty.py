"""Novelty (out-of-distribution) scores on logits / embeddings. Higher score = more likely novel."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax


def msp(logits):
    return -softmax(logits, axis=1).max(1)


def energy(logits, t: float = 1.0):
    return -t * logsumexp(logits / t, axis=1)


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance (Lee et al. 2018) on the penultimate embedding."""

    def fit(self, z: np.ndarray, y: np.ndarray, shrink: float = 1e-3):
        self.classes = np.unique(y)
        self.mu = np.stack([z[y == c].mean(0) for c in self.classes])
        zc = z - self.mu[np.searchsorted(self.classes, y)]
        cov = zc.T @ zc / len(z)
        cov += shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        d = np.stack([np.einsum("ij,jk,ik->i", z - m, self.prec, z - m) for m in self.mu], 1)
        return d.min(1)


class KNN:
    """Distance to the k-th nearest training embedding after L2 normalization (Sun et al. 2022)."""

    def fit(self, z: np.ndarray, k: int = 10):
        self.bank = z / np.linalg.norm(z, axis=1, keepdims=True)
        self.k = k
        return self

    def score(self, z: np.ndarray, chunk: int = 2048) -> np.ndarray:
        zn = z / np.linalg.norm(z, axis=1, keepdims=True)
        out = []
        for i in range(0, len(zn), chunk):
            sim = zn[i:i + chunk] @ self.bank.T
            # partition of -sim puts minus the k-th largest similarity at column k-1: a distance-like score
            out.append(np.partition(-sim, self.k - 1, axis=1)[:, self.k - 1])
        return np.concatenate(out)

    def neighbors(self, z: np.ndarray, k: int = 5):
        zn = z / np.linalg.norm(z, axis=1, keepdims=True)
        sim = zn @ self.bank.T
        idx = np.argsort(-sim, axis=1)[:, :k]
        return idx, np.take_along_axis(sim, idx, 1)


class RelativeMahalanobis(Mahalanobis):
    """Class Mahalanobis minus distance under one class-agnostic Gaussian (Ren et al. 2021), for near-OOD."""

    def fit(self, z, y, shrink: float = 1e-3):
        super().fit(z, y, shrink)
        self.bg = Mahalanobis().fit(z, np.zeros(len(z), int), shrink)
        return self

    def score(self, z):
        return super().score(z) - self.bg.score(z)
