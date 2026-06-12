"""Frozen feature extraction helpers for timm-style ViTs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FeatureBatch:
    cls: np.ndarray
    patch_mean: np.ndarray
    patch_var: np.ndarray
    y: np.ndarray
    images: np.ndarray | None = None


def vit_tokens(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return token sequence for common timm ViT models."""
    if hasattr(model, 'forward_features'):
        out = model.forward_features(images)
        if isinstance(out, dict):
            if 'x' in out:
                out = out['x']
            elif 'tokens' in out:
                out = out['tokens']
            else:
                out = next(v for v in out.values() if torch.is_tensor(v))
        if out.ndim == 2:
            return out[:, None, :]
        return out
    raise ValueError('Model does not expose forward_features; provide features directly instead.')


@torch.no_grad()
def collect_frozen_features(
    model: nn.Module,
    loader: Iterable,
    device: torch.device | str,
    max_batches: int | None = None,
    keep_images: bool = False,
) -> FeatureBatch:
    model.eval().to(device)
    cls_list, patch_mean_list, patch_var_list, y_list, image_list = [], [], [], [], []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        tokens = vit_tokens(model, images)
        cls = F.normalize(tokens[:, 0].float(), dim=-1)
        if tokens.shape[1] > 1:
            patches = F.normalize(tokens[:, 1:].float(), dim=-1)
            patch_mean = F.normalize(patches.mean(dim=1), dim=-1)
            patch_var = patches.var(dim=1).mean(dim=-1)
        else:
            patch_mean = cls
            patch_var = torch.zeros(cls.shape[0], device=cls.device)
        cls_list.append(cls.cpu().numpy())
        patch_mean_list.append(patch_mean.cpu().numpy())
        patch_var_list.append(patch_var.cpu().numpy())
        y_list.append(labels.cpu().numpy())
        if keep_images:
            image_list.append(images.detach().cpu().numpy())
    return FeatureBatch(
        cls=np.concatenate(cls_list, axis=0),
        patch_mean=np.concatenate(patch_mean_list, axis=0),
        patch_var=np.concatenate(patch_var_list, axis=0),
        y=np.concatenate(y_list, axis=0),
        images=np.concatenate(image_list, axis=0) if image_list else None,
    )
