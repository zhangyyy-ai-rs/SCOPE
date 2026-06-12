#!/usr/bin/env python3
"""Smoke tests for the SCOPE public package."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from scope.routing.rules import select_route
from scope.routing.selector import load_diagnostics


def test_selector_rules() -> None:
    route, rule, _ = select_route({
        'nearest_centroid_support_acc': 55,
        'cls_effective_rank': 260,
        'hf_fft_ratio_mean': 0.05,
        'prototype_inter_cos_mean': 0.2,
        'prototype_risk_index': 1.2,
        'num_classes_seen': 47,
    })
    assert route == 'redundancy_aware_evidence', route
    assert rule == 'R02_texture_redundancy', rule


def test_result_columns_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'bad.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset', 'oracle_route', 'cls_effective_rank'])
            writer.writeheader()
            writer.writerow({'dataset': 'x', 'oracle_route': 'bad', 'cls_effective_rank': 1})
        try:
            load_diagnostics(path)
        except ValueError:
            return
        raise AssertionError('oracle/result columns must be rejected')


def test_vit_injection_forward() -> None:
    try:
        import timm
        import torch
    except ModuleNotFoundError as exc:
        print(f'skip ViT injection smoke test because dependency is missing: {exc.name}')
        return
    from scope.models import InjectionConfig, inject_scope, trainable_parameter_count

    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    model = inject_scope(model, InjectionConfig(route='evidence_geometry', rank=4, layers='last'))
    y = model(torch.randn(2, 3, 224, 224))
    assert tuple(y.shape) == (2, 2), tuple(y.shape)
    assert trainable_parameter_count(model) > 0


def main() -> None:
    test_selector_rules()
    test_result_columns_rejected()
    test_vit_injection_forward()
    print('SCOPE smoke test passed')


if __name__ == '__main__':
    main()
