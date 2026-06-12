"""SCOPE support-diagnostic routing."""
from .rules import select_route
from .selector import build_route_rows, export_routes, load_diagnostics

__all__ = ['select_route', 'build_route_rows', 'export_routes', 'load_diagnostics']
