"""The routing policy: AUTO-accept a label, or send the chip to an ANALYST (low confidence or never-seen content).

Every threshold is fit on known-class validation data only. Novel classes are never used to tune anything,
because in deployment you don't get examples of what you haven't seen yet.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .calibrate import fit_temperature, probs, threshold_for_risk
from .novelty import Mahalanobis


@dataclass
class TriagePolicy:
    temperature: float
    conf_threshold: float
    novelty_threshold: float
    target_risk: float
    novelty_fpr: float

    def to_dict(self):
        return asdict(self)


def fit_policy(val_logits, val_z, val_y, maha: Mahalanobis, target_risk=0.01, novelty_fpr=0.05) -> TriagePolicy:
    t = fit_temperature(val_logits, val_y)
    p = probs(val_logits, t)
    nov = maha.score(val_z)
    nov_thr = float(np.quantile(nov, 1 - novelty_fpr))
    keep = nov <= nov_thr
    conf_thr = threshold_for_risk(p.max(1)[keep], (p.argmax(1) == val_y)[keep], target_risk)
    return TriagePolicy(t, conf_thr, nov_thr, target_risk, novelty_fpr)


def route(logits, z, policy: TriagePolicy, maha: Mahalanobis):
    p = probs(logits, policy.temperature)
    conf, pred = p.max(1), p.argmax(1)
    nov = maha.score(z)
    novel = nov > policy.novelty_threshold
    auto = (~novel) & (conf >= policy.conf_threshold)
    reason = np.where(novel, "novel", np.where(auto, "auto", "low_confidence"))
    return {"pred": pred, "conf": conf, "novelty": nov, "auto": auto, "reason": reason}


def summarize(r, y_known_idx):
    """y_known_idx: 0..7 for known, -1 for novel chips."""
    known = y_known_idx >= 0
    auto = r["auto"]
    wrong_auto = auto & (~known | (r["pred"] != y_known_idx))
    return {
        "n": int(len(auto)),
        "auto_rate": float(auto.mean()),
        "auto_error_rate": float(wrong_auto.sum() / max(auto.sum(), 1)),
        "novel_to_analyst": float((~auto[~known]).mean()) if (~known).any() else None,
        "novel_flagged_as_novel": float((r["reason"][~known] == "novel").mean()) if (~known).any() else None,
        "known_auto_rate": float(auto[known].mean()),
        "known_auto_acc": float((r["pred"][known & auto] == y_known_idx[known & auto]).mean()),
    }
