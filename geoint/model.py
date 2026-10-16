"""ResNet-18 chip classifier that also exposes its 512-d embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ChipNet(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # 64x64 chips: drop the stride-2 max-pool so the last feature map is 4x4, not 2x2
        m.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.head = nn.Linear(512, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class ExportWrapper(nn.Module):
    """ONNX graph returns both logits and embedding so the service can score novelty without a second model."""

    def __init__(self, net: ChipNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        z = self.net.embed(x)
        return self.net.head(z), z


def device() -> torch.device:
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps") + 1
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")
