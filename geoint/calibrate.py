"""Calibration (temperature scaling, ECE) and selective prediction."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(t):
        return -log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean()

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Error rate among the auto-accepted chips as the acceptance fraction grows (most confident first)."""
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    k = np.arange(1, len(c) + 1)
    return k / len(c), 1 - np.cumsum(c) / k


def threshold_for_risk(conf: np.ndarray, correct: np.ndarray, target_risk: float) -> float:
    """Lowest confidence threshold whose accepted set has error <= target_risk (fit on validation)."""
    cov, risk = risk_coverage(conf, correct)
    order = np.sort(conf)[::-1]
    ok = np.where(risk <= target_risk)[0]
    if len(ok) == 0:
        return 1.0
    return float(order[ok.max()])


def probs(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    return softmax(logits / t, axis=1)
