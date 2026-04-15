from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    """Simple configurable dense MLP for tabular regression."""

    def __init__(self, input_dim: int, hidden_layers: Iterable[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = units
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
