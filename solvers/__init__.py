"""
Numerical Solvers Package

Provides reference numerical solutions for PDEs.

Available Solvers:
- FiniteDifferenceSolver: FD schemes for heat, wave, advection, Burgers
"""

from solvers.finite_difference import FiniteDifferenceSolver

__all__ = ['FiniteDifferenceSolver']
