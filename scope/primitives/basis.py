"""Shared diagnostic basis route."""
from __future__ import annotations

import torch
from torch import nn

from .base import PrimitiveConfig, ScopePrimitive


class SharedDiagnosticBasis(ScopePrimitive):
    route_name = 'shared_diagnostic_basis'

    def __init__(self, cfg: PrimitiveConfig):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(cfg.rank, cfg.dim) * cfg.init_scale)
        self.coeff = nn.Linear(cfg.dim, cfg.rank, bias=False)
        self.gate = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1))
        self.scale = nn.Parameter(torch.tensor(cfg.init_scale))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        coeff = self.coeff(tokens)
        gate = torch.sigmoid(self.gate(tokens[:, 0])).view(tokens.shape[0], 1, 1)
        return self.scale.to(tokens.dtype) * gate * (coeff @ self.basis)
