"""Support-only SCOPE routing rules."""
from __future__ import annotations


def as_float(row: dict, key: str, default: float = float('nan')) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def select_route(row: dict) -> tuple[str, str, str]:
    """Return ``(scope_route, rule_id, support_reason)``.

    The rule reads support diagnostics only. It intentionally avoids any result,
    validation, test, or oracle information.
    """
    nc = as_float(row, 'nearest_centroid_support_acc', as_float(row, 'nearest_centroid_val_acc'))
    erank = as_float(row, 'cls_effective_rank')
    hf = as_float(row, 'hf_fft_ratio_mean')
    proto = as_float(row, 'prototype_inter_cos_mean')
    risk = as_float(row, 'prototype_risk_index')
    classes = int(round(as_float(row, 'num_classes_seen', 0)))

    if classes >= 300:
        return 'orthogonal_tangent', 'R00_many_class_head_dominated', 'very many classes; head-dominated support geometry'
    if nc >= 97.0 and classes >= 50 and risk < 1.0:
        return 'shared_diagnostic_basis', 'R01_saturated_shared_basis', 'nearest-centroid support already saturated'
    if hf >= 0.040 and classes > 20 and erank > 180.0:
        return 'redundancy_aware_evidence', 'R02_texture_redundancy', 'high-frequency texture support with high effective rank'
    if classes >= 90 and erank >= 250.0 and nc < 82.0:
        return 'evidence_geometry', 'R03_semantic_fragment_evidence', 'high-rank but class-confusable support'
    if classes >= 90 and nc >= 88.0:
        return 'state_bottleneck', 'R04_many_class_state_bottleneck', 'many-class support already separable'
    if 20 < classes < 60 and nc >= 88.0 and proto < 0.30 and hf <= 0.020:
        return 'shared_diagnostic_basis', 'R05_midclass_saturated_basis', 'mid-class separable low-interference support'
    if erank < 20.0 and proto > 0.990:
        return 'state_bottleneck', 'R06_extreme_factor_state', 'extreme factor collapse'
    if erank < 20.0 and proto > 0.970:
        return 'shared_diagnostic_basis', 'R07_lowrank_factor_basis', 'low-rank factor support with prototype coupling'
    if classes <= 2:
        return 'state_bottleneck', 'R16_binary_state', 'binary low-rank state mismatch'
    if classes <= 5 and hf >= 0.030:
        return 'orthogonal_tangent', 'R17_small_highfreq_conservative', 'small-class high-frequency overfit risk'
    if classes <= 5 and proto < 0.800:
        return 'orthogonal_tangent', 'R18_small_geometric_conservative', 'small-class geometric support'
    if classes <= 6 and erank < 90.0 and proto > 0.980 and nc >= 25.0:
        return 'relation_rewrite', 'R08_relation_collapse', 'relational geometry collapse'
    if classes <= 8 and erank < 90.0 and proto > 0.970:
        return 'state_bottleneck', 'R09_count_state_collapse', 'count/state collapse'
    if classes <= 6 and erank >= 90.0 and proto > 0.900:
        return 'state_bottleneck', 'R10_state_rich_small_class', 'state-rich small-class support'
    if classes == 10 and proto > 0.940 and nc < 35.0:
        return 'flat_coordinate_rewrite', 'R11_sharp_digit_coordinate', 'digit-like prototype collapse'
    if classes <= 10 and risk > 1.40:
        return 'flat_orthogonal_tangent', 'R12_pose_flat_orthogonal', 'sharp pose/factor support'
    if classes <= 15 and nc >= 65.0:
        return 'coordinate_rewrite', 'R13_lowclass_style_coordinate', 'coordinate/style shift'
    if 10 < classes < 25 and nc < 10.0:
        return 'coordinate_rewrite', 'R14_pose_weak_coordinate', 'weak pose separability'
    if 40 <= classes <= 60 and 0.010 <= hf <= 0.030 and proto > 0.40:
        return 'coordinate_rewrite', 'R15_scene_style_coordinate', 'scene/spatial-style risk'
    return 'orthogonal_tangent', 'R99_safe_fallback', 'safe orthogonal fallback'
