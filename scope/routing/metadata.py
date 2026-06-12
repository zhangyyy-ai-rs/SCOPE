"""Paper-facing SCOPE route metadata."""
from __future__ import annotations

DISPLAY_BLOCK = {
    'cifar': ('Natural', 'CIFAR-100'),
    'caltech101': ('Natural', 'Caltech101'),
    'dtd': ('Natural', 'DTD'),
    'oxford_flowers102': ('Natural', 'Flowers102'),
    'oxford_iiit_pet': ('Natural', 'Oxford-IIIT Pet'),
    'svhn': ('Natural', 'SVHN'),
    'sun397': ('Natural', 'SUN397'),
    'patch_camelyon': ('Specialized', 'PatchCamelyon'),
    'eurosat': ('Specialized', 'EuroSAT'),
    'resisc45': ('Specialized', 'RESISC45'),
    'diabetic_retinopathy': ('Specialized', 'Diabetic Retinopathy'),
    'clevr_count': ('Structured', 'CLEVR-Count'),
    'clevr_dist': ('Structured', 'CLEVR-Distance'),
    'dmlab': ('Structured', 'DMLab'),
    'kitti': ('Structured', 'KITTI-Distance'),
    'dsprites_loc': ('Structured', 'dSprites-Loc'),
    'dsprites_ori': ('Structured', 'dSprites-Ori'),
    'smallnorb_azi': ('Structured', 'SmallNORB-Azimuth'),
    'smallnorb_ele': ('Structured', 'SmallNORB-Elevation'),
}

DATASET_ORDER = list(DISPLAY_BLOCK)

ROUTE_INFO = {
    'orthogonal_tangent': {
        'operator': 'Orthogonal Tangent Route',
        'role': 'safe tangent-space rotation when support diagnostics show no reliable extra write locus',
    },
    'flat_orthogonal_tangent': {
        'operator': 'Flat Orthogonal Tangent Route',
        'role': 'orthogonal tangent repair with flatness bias for sharp pose/factor support',
    },
    'evidence_geometry': {
        'operator': 'Evidence Geometry Route',
        'role': 'write discriminative evidence for high-rank but class-confusable support',
    },
    'redundancy_aware_evidence': {
        'operator': 'Redundancy-Aware Evidence Route',
        'role': 'suppress repeated high-frequency tangent directions in texture-like support',
    },
    'shared_diagnostic_basis': {
        'operator': 'Shared Diagnostic Basis Route',
        'role': 'use a tiny shared basis for saturated or low-rank-collapsed support',
    },
    'coordinate_rewrite': {
        'operator': 'Coordinate Rewrite Route',
        'role': 'repair local coordinate/style shifts',
    },
    'flat_coordinate_rewrite': {
        'operator': 'Flat Coordinate Rewrite Route',
        'role': 'coordinate rewrite with flatness for prototype-collapse tasks',
    },
    'relation_rewrite': {
        'operator': 'Relation Rewrite Route',
        'role': 'repair collapsed attention-relation geometry',
    },
    'state_bottleneck': {
        'operator': 'State Bottleneck Route',
        'role': 'repair low/mid-rank latent-state mismatch',
    },
}

FORBIDDEN_DIAGNOSTIC_HINTS = (
    'oracle',
    'test_acc',
    'val_acc_result',
    'best_acc',
    'accuracy',
    'result',
)
