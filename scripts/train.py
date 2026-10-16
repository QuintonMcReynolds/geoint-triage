"""Train the clean and degradation-augmented classifiers on the 8 known classes.

    python scripts/train.py --variant clean
    python scripts/train.py --variant robust
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoint import data as D  # noqa: E402
from geoint.degrade import FlipRot, RandomDegrade  # noqa: E402
from geoint.torchdata import ChipDataset, seed_everything  # noqa: E402
from geoint.model import ChipNet, device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "robust"], default="clean")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_everything(a.seed)
    d = D.load()
    ktr, kva = D.known_mask(d["y_train"]), D.known_mask(d["y_validation"])
    xtr, ytr = d["x_train"][ktr], D.to_known_index(d["y_train"][ktr])
    xva, yva = d["x_validation"][kva], D.to_known_index(d["y_validation"][kva])

    tf = RandomDegrade(p=0.5, seed=a.seed) if a.variant == "robust" else FlipRot(seed=a.seed)
    dl = DataLoader(ChipDataset(xtr, ytr, tf), batch_size=a.bs, shuffle=True, num_workers=0, drop_last=True)
    xva_t = torch.from_numpy(D.normalize(xva))

    dev = device()
    net = ChipNet(len(D.KNOWN_CLASSES)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.15)

    log, best, out = [], -1.0, D.ART / f"chipnet_{a.variant}.pt"
    D.ART.mkdir(exist_ok=True)
    print(f"device={dev} train={len(xtr)} val={len(xva)} variant={a.variant}", flush=True)
    for ep in range(a.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(net(xb), yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(xva_t[i:i + 512].to(dev)).argmax(1).cpu() for i in range(0, len(xva_t), 512)])
        acc = (pred.numpy() == yva).mean()
        log.append({"epoch": ep + 1, "loss": tot / (len(dl) * a.bs), "val_acc": float(acc), "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if acc > best:
            best = acc
            torch.save(net.state_dict(), out)
    (D.ART / f"train_log_{a.variant}.json").write_text(json.dumps({"args": vars(a), "device": str(dev), "log": log}, indent=2))
    print(f"best val acc {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
