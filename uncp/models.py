"""Model factory for UNCP experiments."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import resnet18


def resnet18_small(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 28×28 input (Colored MNIST)."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet18_imagenet(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Standard ResNet-18 for 224×224 input (Waterbirds, ImageNet-resolution)."""
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class MLP(nn.Module):
    """Vanilla MLP for tabular benchmarks."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32),
                 num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(name: str, **kwargs: Any) -> nn.Module:
    if name == "resnet18_small":
        return resnet18_small(num_classes=kwargs.get("num_classes", 10))
    if name == "resnet18_imagenet":
        return resnet18_imagenet(num_classes=kwargs.get("num_classes", 2),
                                 pretrained=kwargs.get("pretrained", True))
    if name == "mlp":
        return MLP(input_dim=kwargs["input_dim"],
                   hidden_dims=tuple(kwargs.get("hidden_dims", (128, 64, 32))),
                   num_classes=kwargs.get("num_classes", 2),
                   dropout=kwargs.get("dropout", 0.2))
    raise ValueError(f"Unknown model name: {name}")
