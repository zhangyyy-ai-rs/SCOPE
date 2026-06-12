"""Relation rewrite SCOPE route."""
from __future__ import annotations

import torch
from torch import nn

from .base import PrimitiveConfig, ScopePrimitive


class RelationRewrite(ScopePrimitive):
    route_name = 'relation_rewrite'

    def __init__(self, cfg: PrimitiveConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim)
        self.q = nn.Linear(cfg.dim, cfg.rank, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.rank, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.rank, bias=False)
        self.out = nn.Linear(cfg.rank, cfg.dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(cfg.init_scale))
        nn.init.zeros_(self.out.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.norm(tokens)
        q, k, v = self.q(z), self.k(z), self.v(z)
        attn = torch.softmax((q @ k.transpose(-1, -2)) / max(1.0, q.shape[-1] ** 0.5), dim=-1)
        return self.scale.to(tokens.dtype) * self.out(attn @ v)
