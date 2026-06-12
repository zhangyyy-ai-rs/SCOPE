"""Orthogonal tangent SCOPE routes."""
from __future__ import annotations

import torch
from torch import nn

from .base import PrimitiveConfig, ScopePrimitive


class OrthogonalTangent(ScopePrimitive):
    route_name = 'orthogonal_tangent'

    def __init__(self, cfg: PrimitiveConfig):
        super().__init__()
        self.left = nn.Parameter(torch.randn(cfg.dim, cfg.rank) * cfg.init_scale)
        self.right = nn.Parameter(torch.randn(cfg.dim, cfg.rank) * cfg.init_scale)
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.dropout(tokens)
        xa = x @ self.left
        xb = x @ self.right
        update = xa @ self.right.t() - xb @ self.left.t()
        return torch.tanh(self.log_scale).to(update.dtype) * update


class FlatOrthogonalTangent(OrthogonalTangent):
    route_name = 'flat_orthogonal_tangent'

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        update = super().forward(tokens)
        return 0.5 * update / update.norm(dim=-1, keepdim=True).detach().clamp_min(1.0)
