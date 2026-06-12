"""SCOPE adaptation primitives."""
from .base import PrimitiveConfig, ScopePrimitive
from .factory import PRIMITIVE_REGISTRY, create_primitive

__all__ = ['PrimitiveConfig', 'ScopePrimitive', 'PRIMITIVE_REGISTRY', 'create_primitive']
