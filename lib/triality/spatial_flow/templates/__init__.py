"""
Spatial Flow Templates

High-level APIs for common routing and distribution patterns.

Available templates:
    - cable_routing: Power and signal cable layout
    - (more templates to be added: pipe_routing, hvac_routing, etc.)
"""

from . import cable_routing

__all__ = ['cable_routing']
