"""Batched inference returning logits and embeddings as numpy."""
from __future__ import annotations

import numpy as np
import torch

from . import data as D
from .model import ChipNet, device


def load_net(variant: str, dev=None) -> ChipNet:
    dev = dev or device()
    net = ChipNet(len(D.KNOWN_CLASSES), pretrained=False)
    net.load_state_dict(torch.load(D.ART / f"chipnet_{variant}.pt", map_location="cpu"))
    return net.to(dev).eval()


@torch.no_grad()
def run(net: ChipNet, x_uint8: np.ndarray, bs: int = 512) -> tuple[np.ndarray, np.ndarray]:
    dev = next(net.parameters()).device
    L, Z = [], []
    for i in range(0, len(x_uint8), bs):
        xb = torch.from_numpy(D.normalize(x_uint8[i:i + bs])).to(dev)
        z = net.embed(xb)
        L.append(net.head(z).cpu().numpy())
        Z.append(z.cpu().numpy())
    return np.concatenate(L), np.concatenate(Z)
