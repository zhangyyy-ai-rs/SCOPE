"""Coordinate rewrite SCOPE routes."""
from __future__ import annotations

import torch
from torch import nn

from .base import PrimitiveConfig, ScopePrimitive


class CoordinateRewrite(ScopePrimitive):
    route_name = 'coordinate_rewrite'

    def __init__(self, cfg: PrimitiveConfig, flat: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim)
        self.mix = nn.Sequential(nn.Linear(cfg.dim, cfg.rank), nn.GELU(), nn.Linear(cfg.rank, cfg.dim))
        self.token_gate = nn.Linear(cfg.dim, 1)
        self.flat = flat
        self.scale = nn.Parameter(torch.tensor(cfg.init_scale))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.norm(tokens)
        update = torch.sigmoid(self.token_gate(z)) * self.mix(z)
        if self.flat:
            update = update / update.norm(dim=-1, keepdim=True).detach().clamp_min(1.0)
        return self.scale.to(tokens.dtype) * update


class FlatCoordinateRewrite(CoordinateRewrite):
    route_name = 'flat_coordinate_rewrite'

    def __init__(self, cfg: PrimitiveConfig):
        super().__init__(cfg, flat=True)
