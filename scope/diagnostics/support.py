"""High-level support diagnostic summarizer."""
from __future__ import annotations

import numpy as np

from .metrics import effective_rank, image_high_frequency, normalize
from .prototypes import prototype_stats
from .schema import SupportDiagnostics


def summarize_support(
    dataset: str,
    train_cls: np.ndarray,
    train_y: np.ndarray,
    probe_cls: np.ndarray | None = None,
    probe_y: np.ndarray | None = None,
    train_patch_mean: np.ndarray | None = None,
    train_patch_var: np.ndarray | None = None,
    train_patch_cls_align: np.ndarray | None = None,
    train_patch_spread: np.ndarray | None = None,
    train_images_nchw: np.ndarray | None = None,
) -> SupportDiagnostics:
    """Summarize a task from frozen-backbone support features only."""
    train_cls = normalize(np.asarray(train_cls, dtype=np.float64))
    cls_er, cls_top1, cls_top10 = effective_rank(train_cls)
    if train_patch_mean is None:
        patch_er, patch_top1, patch_top10 = cls_er, cls_top1, cls_top10
    else:
        patch_er, patch_top1, patch_top10 = effective_rank(normalize(np.asarray(train_patch_mean)))
    if train_images_nchw is not None:
        hf_grad, hf_fft = image_high_frequency(train_images_nchw)
    else:
        hf_grad = np.asarray([0.0])
        hf_fft = np.asarray([0.0])
    pstats = prototype_stats(train_cls, train_y, probe_cls, probe_y)
    return SupportDiagnostics(
        dataset=dataset,
        cls_effective_rank=cls_er,
        cls_top1_energy=cls_top1,
        cls_top10_energy=cls_top10,
        patchmean_effective_rank=patch_er,
        patchmean_top1_energy=patch_top1,
        patchmean_top10_energy=patch_top10,
        patch_var_mean=float(np.mean(train_patch_var)) if train_patch_var is not None else 0.0,
        patch_cls_align_mean=float(np.mean(train_patch_cls_align)) if train_patch_cls_align is not None else 0.0,
        patch_spread_mean=float(np.mean(train_patch_spread)) if train_patch_spread is not None else 0.0,
        hf_grad_mean=float(np.mean(hf_grad)),
        hf_fft_ratio_mean=float(np.mean(hf_fft)),
        hf_fft_ratio_std=float(np.std(hf_fft)),
        **pstats,
    )
