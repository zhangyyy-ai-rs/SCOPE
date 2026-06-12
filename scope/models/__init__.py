"""Model integration utilities."""
from .inject import InjectionConfig, ScopeBlock, inject_scope, trainable_parameter_count

__all__ = ['InjectionConfig', 'ScopeBlock', 'inject_scope', 'trainable_parameter_count']
