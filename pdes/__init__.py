"""
PDE Definitions Package

Provides abstract base class and concrete implementations for various 1D PDEs.

Available PDEs:
- HeatEquation1D: u_t = α·u_xx (parabolic)
- WaveEquation1D: u_tt = c²·u_xx (hyperbolic)
- AdvectionEquation1D: u_t + c·u_x = 0 (hyperbolic)
- BurgersEquation1D: u_t + u·u_x = ν·u_xx (nonlinear parabolic)
"""

from pdes.base import BasePDE
from pdes.heat import HeatEquation1D
from pdes.wave import WaveEquation1D
from pdes.advection import AdvectionEquation1D
from pdes.burgers import BurgersEquation1D

__all__ = [
    'BasePDE',
    'HeatEquation1D',
    'WaveEquation1D',
    'AdvectionEquation1D',
    'BurgersEquation1D',
]
