"""TriageEngine: the one object the REST API, the MCP server and the tests all share.

Loads only the ONNX model, the embedding bank and the policy, with no PyTorch import, so the service image stays small.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import data as D
from .drift import mmd_test, psi

FEED = Path(os.environ.get("GEOINT_FEED", D.ROOT / "samples" / "feed"))


def _softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def decode_chip(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (64, 64):
        img = img.resize((64, 64), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class TriageEngine:
    def __init__(self, art: Path = D.ART, feed: Path = FEED):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(art / "chipnet.onnx"), providers=["CPUExecutionProvider"])
        self.policy = json.loads((art / "policy.json").read_text())
        self.classes = self.policy["classes"]
        b = np.load(art / "triage_bank.npz")
        self.bank_z, self.bank_y = b["z"], b["y"]
        self.bank_zn = self.bank_z / np.linalg.norm(self.bank_z, axis=1, keepdims=True)
        self.mu, self.prec = b["mu"], b["prec"]
        self.pca_mean, self.pca_comp = b["pca_mean"], b["pca_comp"]
        self.ref, self.gamma, self.ref_pred = b["ref"], float(b["gamma"]), b["ref_pred"]
        self.feed = feed

    # ---------- core ----------
    def _forward(self, chips: np.ndarray):
        logits, z = self.sess.run(None, {"chips": D.normalize(chips)})
        return logits, z

    def _novelty(self, z):
        return np.stack([np.einsum("ij,jk,ik->i", z - m, self.prec, z - m) for m in self.mu], 1).min(1)

    def classify(self, chips: np.ndarray, ids: list[str] | None = None) -> list[dict]:
        logits, z = self._forward(chips)
        p = _softmax(logits / self.policy["temperature"])
        nov = self._novelty(z)
        out = []
        for i in range(len(chips)):
            top = np.argsort(-p[i])[:3]
            novel = bool(nov[i] > self.policy["novelty_threshold"])
            auto = (not novel) and bool(p[i, top[0]] >= self.policy["conf_threshold"])
            out.append({
                "chip_id": ids[i] if ids else str(i),
                "label": self.classes[top[0]],
                "confidence": round(float(p[i, top[0]]), 4),
                "top3": [[self.classes[j], round(float(p[i, j]), 4)] for j in top],
                "novelty_score": round(float(nov[i]), 2),
                "novelty_threshold": round(self.policy["novelty_threshold"], 2),
                "route": "auto" if auto else "analyst",
                "reason": "novel" if novel else ("auto" if auto else "low_confidence"),
            })
        return out

    def similar(self, chip: np.ndarray, k: int = 5) -> dict:
        _, z = self._forward(chip[None])
        zn = z[0] / np.linalg.norm(z[0])
        sim = self.bank_zn @ zn
        idx = np.argsort(-sim)[:k]
        labels = [self.classes[self.bank_y[j]] for j in idx]
        return {"neighbors": [{"label": lab, "cosine": round(float(sim[j]), 3)} for lab, j in zip(labels, idx)],
                "max_cosine": round(float(sim[idx[0]]), 3),
                "neighbor_vote": max(set(labels), key=labels.count)}

    def drift(self, chips: np.ndarray) -> dict:
        logits, z = self._forward(chips)
        zp = (z - self.pca_mean) @ self.pca_comp.T
        t = mmd_test(self.ref, zp, n_perm=200, gamma=self.gamma)
        s = psi(self.ref_pred, logits.argmax(1), len(self.classes))
        return {"n_chips": int(len(chips)), "mmd2": round(t["mmd2"], 5), "p_value": round(t["p_value"], 4),
                "label_psi": round(s, 3), "drift_alarm": bool(t["p_value"] < 0.01),
                "note": "MMD on embeddings vs. validation reference; alarm at p<0.01"}

    # ---------- feed helpers (collection passes on disk) ----------
    def passes(self) -> list[dict]:
        if not self.feed.exists():
            return []
        out = []
        for d in sorted(p for p in self.feed.iterdir() if p.is_dir()):
            meta = d / "meta.json"
            info = json.loads(meta.read_text()) if meta.exists() else {}
            out.append({"pass_id": d.name, "n_chips": len(list(d.glob("*.png"))), **{k: v for k, v in info.items() if k == "collected"}})
        return out

    def _pass_dir(self, pass_id: str) -> Path:
        d = (self.feed / pass_id).resolve()
        if self.feed.resolve() not in d.parents or not d.is_dir():
            raise ValueError(f"unknown pass_id {pass_id!r}; call list_passes")
        return d

    def load_pass(self, pass_id: str):
        files = sorted(self._pass_dir(pass_id).glob("*.png"))
        return [f.stem for f in files], np.stack([decode_chip(f.read_bytes()) for f in files])

    def load_chip(self, chip_id: str) -> np.ndarray:
        pass_id = chip_id.split("-")[0]
        f = self._pass_dir(pass_id) / f"{chip_id}.png"
        if not f.exists():
            raise ValueError(f"unknown chip_id {chip_id!r}")
        return decode_chip(f.read_bytes())

    def triage_pass(self, pass_id: str) -> dict:
        ids, chips = self.load_pass(pass_id)
        rows = self.classify(chips, ids)
        counts: dict[str, int] = {}
        for r in rows:
            if r["route"] == "auto":
                counts[r["label"]] = counts.get(r["label"], 0) + 1
        queue = [{"chip_id": r["chip_id"], "reason": r["reason"], "best_guess": r["label"], "confidence": r["confidence"]}
                 for r in rows if r["route"] == "analyst"]
        return {"pass_id": pass_id, "n_chips": len(rows),
                "n_auto": len(rows) - len(queue), "n_analyst": len(queue),
                "n_novel": sum(r["reason"] == "novel" for r in rows),
                "n_low_confidence": sum(r["reason"] == "low_confidence" for r in rows),
                "auto_label_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                "analyst_queue": queue}
