"""Evidence-writing SCOPE routes."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .base import PrimitiveConfig, ScopePrimitive


class EvidenceGeometry(ScopePrimitive):
    route_name = 'evidence_geometry'

    def __init__(self, cfg: PrimitiveConfig, redundancy_aware: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim)
        self.down = nn.Linear(cfg.dim, cfg.rank, bias=False)
        self.up = nn.Linear(cfg.rank, cfg.dim, bias=False)
        self.cls_gate = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.rank), nn.GELU(), nn.Linear(cfg.rank, 1))
        self.redundancy_aware = redundancy_aware
        if redundancy_aware:
            self.redundancy_gate = nn.Linear(cfg.rank, cfg.rank, bias=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = nn.Parameter(torch.tensor(cfg.init_scale))
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.down(self.norm(tokens))
        if self.redundancy_aware:
            energy = z.pow(2).mean(dim=1, keepdim=True).detach()
            z = z * torch.sigmoid(self.redundancy_gate(-energy))
        cls_gate = torch.sigmoid(self.cls_gate(tokens[:, 0])).view(tokens.shape[0], 1, 1)
        return self.scale.to(tokens.dtype) * cls_gate * self.dropout(self.up(F.gelu(z)))


class RedundancyAwareEvidence(EvidenceGeometry):
    route_name = 'redundancy_aware_evidence'

    def __init__(self, cfg: PrimitiveConfig):
        super().__init__(cfg, redundancy_aware=True)
