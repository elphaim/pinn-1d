"""
1D Wave Equation

PDE: u_tt = c²·u_xx

Standard setup:
- Domain: x ∈ [0, L], t ∈ [0, T]
- IC: u(x, 0) = sin(πx/L), u_t(x, 0) = 0
- BC: u(0, t) = u(L, t) = 0 (Dirichlet)
- Analytical: u(x, t) = sin(πx/L)·cos(c·π·t/L)

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
import numpy as np
from typing import Optional

from pdes.base import BasePDE


class WaveEquation1D(BasePDE):
    """
    1D Wave Equation: u_tt = c²·u_xx

    This is a hyperbolic PDE modeling wave propagation.

    Parameters:
        c: Wave speed (default: 1.0)

    Default problem setup:
        IC: u(x, 0) = sin(πx/L), u_t(x, 0) = 0
        BC: u(0, t) = u(L, t) = 0
        Solution: u(x, t) = sin(πx/L)·cos(c·π·t/L)
    """

    name = "WaveEquation1D"
    spatial_order = 2
    temporal_order = 2

    @classmethod
    def param_names(cls) -> list[str]:
        return ['c']

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {'c': 1.0}

    def residual(
        self,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        u_tt: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        params: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute wave equation residual: u_tt - c²·u_xx

        Args:
            u_tt: Second time derivative (required for wave equation)
            u_xx: Spatial second derivative
            params: Override parameters (for inverse problems)

        Returns:
            residual: u_tt - c²·u_xx (should be 0 for exact solution)
        """
        if u_tt is None:
            raise ValueError("Wave equation requires u_tt (second time derivative)")

        if params is not None and 'c' in params:
            c = params['c']
        else:
            c = self.params['c']

        return u_tt - c**2 * u_xx

    def weak_form_integrand(
        self,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        phi: torch.Tensor,
        phi_x: torch.Tensor,
        phi_t: torch.Tensor,
        params: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Weak form integrand for wave equation.

        Integration by parts: ∫∫ (u_tt - c²·u_xx)·φ dx dt = 0
        becomes: ∫∫ (u_t·φ_t - c²·u_x·φ_x) dx dt + boundary terms = 0

        With compact support test functions (φ=0 on boundary), boundary terms vanish.
        Further IBP would require φ_{t,x}=0 or u=0 on boundary, not guaranteed
        so stop here for weak form.

        Returns: u_t·φ_t - c²·u_x·φ_x
        """
        if params is not None and 'c' in params:
            c = params['c']
        else:
            c = self.params['c']

        return u_t * phi_t - c**2 * u_x * phi_x

    def analytical_solution(
        self,
        x: torch.Tensor | np.ndarray,
        t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """
        Analytical solution for wave equation with sin IC and zero velocity.

        u(x, t) = sin(πx/L)·cos(c·π·t/L)
        """
        c = self.params['c']
        L = self.L
        k = np.pi / L

        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return torch.sin(k * x) * torch.cos(c * k * t)
        else:
            return np.sin(k * x) * np.cos(c * k * t)

    def default_ic(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Default initial condition: u(x, 0) = sin(πx/L)
        """
        L = self.L
        if isinstance(x, torch.Tensor):
            return torch.sin(np.pi * x / L)
        else:
            return np.sin(np.pi * x / L)

    def default_ic_t(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Default initial velocity: u_t(x, 0) = 0

        Required for wave equation (second order in time).
        """
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        else:
            return np.zeros_like(x)

    def default_bc(
        self,
        t: torch.Tensor | np.ndarray,
        boundary: str
    ) -> torch.Tensor | np.ndarray:
        """
        Default boundary conditions: u(0, t) = u(L, t) = 0 (Dirichlet)
        """
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t)
        else:
            return np.zeros_like(t)


# ============================================================================
# Example usage and testing
# ============================================================================


if __name__ == "__main__":
    print("Testing WaveEquation1D...")

    # Create PDE
    pde = WaveEquation1D(params={'c': 1.0})
    print(f"Created: {pde}")
    print(f"Parameters: {pde.params}")
    print(f"Temporal order: {pde.temporal_order}")

    # Test analytical solution
    x = np.linspace(0, 1, 5)
    t = np.array([0.0, 0.25, 0.5])

    for t_val in t:
        u = pde.analytical_solution(x, np.full_like(x, t_val))
        print(f"t={t_val}: u_max={u.max():.4f}")

    # Test IC/BC
    x_ic = torch.linspace(0, 1, 5).reshape(-1, 1)
    u_ic = pde.default_ic(x_ic)
    u_t_ic = pde.default_ic_t(x_ic)
    assert isinstance(u_ic, torch.Tensor) and isinstance(u_t_ic, torch.Tensor)
    print(f"\nIC: u(x, 0) = {u_ic.squeeze().numpy()}")
    print(f"IC_t: u_t(x, 0) = {u_t_ic.squeeze().numpy()}")

    # Test residual with dummy data
    N = 10
    u = torch.rand(N, 1)
    u_t = torch.rand(N, 1)
    u_x = torch.rand(N, 1)
    u_xx = torch.rand(N, 1)
    u_tt = torch.rand(N, 1)

    res = pde.residual(u, u_t, u_x, u_xx, u_tt=u_tt)
    print(f"\nResidual shape: {res.shape}")

    # Test weak form
    phi = torch.rand(N, 1)
    phi_x = torch.rand(N, 1)
    phi_t = torch.rand(N, 1)

    weak = pde.weak_form_integrand(u, u_t, u_x, phi, phi_x, phi_t)
    print(f"Weak form integrand shape: {weak.shape}")

    print("\nWaveEquation1D tests passed.")
