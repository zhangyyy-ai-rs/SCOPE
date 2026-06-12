"""Diagnostic dataclasses."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class SupportDiagnostics:
    dataset: str
    cls_effective_rank: float
    cls_top1_energy: float
    cls_top10_energy: float
    patchmean_effective_rank: float
    patchmean_top1_energy: float
    patchmean_top10_energy: float
    patch_var_mean: float
    patch_cls_align_mean: float
    patch_spread_mean: float
    hf_grad_mean: float
    hf_fft_ratio_mean: float
    hf_fft_ratio_std: float
    num_classes_seen: int
    prototype_inter_cos_mean: float
    prototype_inter_cos_max: float
    within_cos_dist: float
    prototype_risk_index: float
    nearest_centroid_support_acc: float
    nearest_centroid_margin: float
    class_imbalance_cv: float

    def to_row(self) -> Dict[str, object]:
        return asdict(self)
