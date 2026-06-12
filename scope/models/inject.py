"""Inject SCOPE primitives into ViT-like models."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scope.primitives import create_primitive


@dataclass
class InjectionConfig:
    route: str
    rank: int = 8
    init_scale: float = 1e-3
    dropout: float = 0.0
    layers: str = 'all'
    freeze_backbone: bool = True
    train_head: bool = True


class ScopeBlock(nn.Module):
    """Wrap a transformer block and add one SCOPE residual primitive."""

    def __init__(self, block: nn.Module, route: str, dim: int, rank: int, init_scale: float, dropout: float):
        super().__init__()
        self.block = block
        self.scope = create_primitive(route, dim=dim, rank=rank, init_scale=init_scale, dropout=dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.block(x, *args, **kwargs)
        return x + self.scope(x)


def select_layers(num_layers: int, spec: str) -> list[int]:
    spec = str(spec).strip().lower()
    if spec == 'all':
        return list(range(num_layers))
    if spec == 'last':
        return [num_layers - 1]
    if spec == 'even':
        return [i for i in range(num_layers) if i % 2 == 0]
    if spec == 'odd':
        return [i for i in range(num_layers) if i % 2 == 1]
    return [int(x) if int(x) >= 0 else num_layers + int(x) for x in spec.split(',') if x.strip()]


def infer_embed_dim(model: nn.Module) -> int:
    if hasattr(model, 'embed_dim'):
        return int(model.embed_dim)
    if hasattr(model, 'num_features'):
        return int(model.num_features)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            return int(module.normalized_shape[0])
    raise ValueError('Could not infer ViT embedding dimension.')


def inject_scope(model: nn.Module, cfg: InjectionConfig) -> nn.Module:
    """Inject SCOPE blocks into a timm-style ViT with ``model.blocks``."""
    if not hasattr(model, 'blocks'):
        raise ValueError('Expected a ViT-like model with a .blocks ModuleList.')
    dim = infer_embed_dim(model)
    selected = set(select_layers(len(model.blocks), cfg.layers))
    if cfg.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    for i, block in enumerate(model.blocks):
        if i in selected:
            model.blocks[i] = ScopeBlock(block, cfg.route, dim, cfg.rank, cfg.init_scale, cfg.dropout)
    if cfg.train_head:
        for name, param in model.named_parameters():
            if name.startswith('head') or '.head.' in name:
                param.requires_grad = True
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
