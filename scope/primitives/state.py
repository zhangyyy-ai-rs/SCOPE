"""Latent-state bottleneck SCOPE route."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .base import PrimitiveConfig, ScopePrimitive


class StateBottleneck(ScopePrimitive):
    route_name = 'state_bottleneck'

    def __init__(self, cfg: PrimitiveConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim)
        self.fc1 = nn.Linear(cfg.dim, cfg.rank)
        self.fc2 = nn.Linear(cfg.rank, cfg.dim)
        self.scale = nn.Parameter(torch.tensor(cfg.init_scale))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.scale.to(tokens.dtype) * self.fc2(F.gelu(self.fc1(self.norm(tokens))))
