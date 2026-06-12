"""Base classes for SCOPE adaptation primitives."""
from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass
class PrimitiveConfig:
    dim: int
    rank: int = 8
    init_scale: float = 1e-3
    dropout: float = 0.0


class ScopePrimitive(nn.Module):
    route_name = 'base'

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
