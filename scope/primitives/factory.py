"""Factory for SCOPE primitives."""
from __future__ import annotations

from typing import Dict, Type

from .base import PrimitiveConfig, ScopePrimitive
from .basis import SharedDiagnosticBasis
from .coordinate import CoordinateRewrite, FlatCoordinateRewrite
from .evidence import EvidenceGeometry, RedundancyAwareEvidence
from .relation import RelationRewrite
from .state import StateBottleneck
from .tangent import FlatOrthogonalTangent, OrthogonalTangent

PRIMITIVE_REGISTRY: Dict[str, Type[ScopePrimitive]] = {
    'orthogonal_tangent': OrthogonalTangent,
    'flat_orthogonal_tangent': FlatOrthogonalTangent,
    'evidence_geometry': EvidenceGeometry,
    'redundancy_aware_evidence': RedundancyAwareEvidence,
    'shared_diagnostic_basis': SharedDiagnosticBasis,
    'coordinate_rewrite': CoordinateRewrite,
    'flat_coordinate_rewrite': FlatCoordinateRewrite,
    'relation_rewrite': RelationRewrite,
    'state_bottleneck': StateBottleneck,
}


def create_primitive(route: str, dim: int, rank: int = 8, init_scale: float = 1e-3, dropout: float = 0.0) -> ScopePrimitive:
    if route not in PRIMITIVE_REGISTRY:
        raise KeyError(f'Unknown SCOPE route {route!r}. Available: {sorted(PRIMITIVE_REGISTRY)}')
    return PRIMITIVE_REGISTRY[route](PrimitiveConfig(dim=dim, rank=rank, init_scale=init_scale, dropout=dropout))
