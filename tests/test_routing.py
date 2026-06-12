from scope.routing.rules import select_route


def test_texture_route():
    route, rule, _ = select_route({
        'nearest_centroid_support_acc': 55,
        'cls_effective_rank': 260,
        'hf_fft_ratio_mean': 0.05,
        'prototype_inter_cos_mean': 0.2,
        'prototype_risk_index': 1.2,
        'num_classes_seen': 47,
    })
    assert route == 'redundancy_aware_evidence'
    assert rule == 'R02_texture_redundancy'


def test_many_class_route():
    route, _, _ = select_route({
        'nearest_centroid_support_acc': 60,
        'cls_effective_rank': 280,
        'hf_fft_ratio_mean': 0.02,
        'prototype_inter_cos_mean': 0.05,
        'prototype_risk_index': 1.0,
        'num_classes_seen': 397,
    })
    assert route == 'orthogonal_tangent'
