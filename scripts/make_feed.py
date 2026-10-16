"""Build a small demo feed of 'collection passes' from the EuroSAT test split.

passA  routine: 60 known-class chips, clean
passB  weather: 60 known-class chips under patchy haze (severity 3)
passC  new activity: 48 known chips + 12 Highway/Industrial chips (classes the model never trained on)
passD  degraded link: 60 known-class chips, heavy JPEG compression (severity 4)

Ground truth goes to samples/feed_truth.json, which the tools never read.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoint import data as D  # noqa: E402
from geoint import degrade  # noqa: E402

FEED = D.ROOT / "samples" / "feed"


def main():
    d = D.load()
    rng = np.random.default_rng(42)
    known = np.where(D.known_mask(d["y_test"]))[0]
    novel = np.where(~D.known_mask(d["y_test"]))[0]
    picks = rng.choice(known, 228, replace=False)
    plan = {
        "passA": (picks[:60], None, "2026-10-01T14:05Z"),
        "passB": (picks[60:120], ("haze", 3), "2026-10-02T14:11Z"),
        "passC": (np.r_[picks[120:168], rng.choice(novel, 12, replace=False)], None, "2026-10-03T13:58Z"),
        "passD": (np.r_[picks[168:228]], ("jpeg", 4), "2026-10-04T14:02Z"),
    }
    if FEED.exists():
        shutil.rmtree(FEED)
    truth = {}
    for pid, (idx, deg, when) in plan.items():
        idx = rng.permutation(idx)
        x = d["x_test"][idx]
        if deg:
            x = degrade.apply(x, deg[0], deg[1], seed=7)
        out = FEED / pid
        out.mkdir(parents=True)
        (out / "meta.json").write_text(json.dumps({"collected": when}))
        for j, (im, i) in enumerate(zip(x, idx)):
            cid = f"{pid}-{j:03d}"
            Image.fromarray(im).save(out / f"{cid}.png", optimize=True)
            truth[cid] = D.ALL_CLASSES[d["y_test"][i]]
    (D.ROOT / "samples" / "feed_truth.json").write_text(json.dumps(truth, indent=0))
    print({p: len(v[0]) for p, v in plan.items()}, "chips ->", FEED)


if __name__ == "__main__":
    main()
