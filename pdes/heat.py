"""
1D Heat Equation

PDE: u_t = α·u_xx

Standard setup:
- Domain: x ∈ [0, L], t ∈ [0, T]
- IC: u(x, 0) = sin(πx/L)
- BC: u(0, t) = u(L, t) = 0 (Dirichlet)
- Analytical: u(x, t) = sin(πx/L)·exp(-α·(π/L)²·t)

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
import numpy as np
from typing import Optional

from pdes.base import BasePDE


class HeatEquation1D(BasePDE):
    """
    1D Heat (Diffusion) Equation: u_t = α·u_xx

    This is a parabolic PDE modeling heat conduction or diffusion.

    Parameters:
        alpha: Thermal diffusivity (default: 0.01)

    Default problem setup:
        IC: u(x, 0) = sin(πx/L)
        BC: u(0, t) = u(L, t) = 0
        Solution: u(x, t) = sin(πx/L)·exp(-α·(π/L)²·t)
    """

    name = "HeatEquation1D"
    spatial_order = 2
    temporal_order = 1

    @classmethod
    def param_names(cls) -> list[str]:
        return ['alpha']

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {'alpha': 0.01}

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
        Compute heat equation residual: u_t - α·u_xx

        Args:
            u: Solution values (unused for linear heat equation)
            u_t: Time derivative
            u_x: Spatial first derivative (unused)
            u_xx: Spatial second derivative
            params: Override parameters (for inverse problems)

        Returns:
            residual: u_t - α·u_xx (should be 0 for exact solution)
        """
        if params is not None and 'alpha' in params:
            alpha = params['alpha']
        else:
            alpha = self.params['alpha']

        return u_t - alpha * u_xx

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
        Weak form integrand for heat equation.

        Integration by parts: ∫∫ (u_t - α·u_xx)·φ dx dt = 0
        becomes: ∫∫ (u·φ_t - α·u_x·φ_x) dx dt + boundary terms = 0

        With compact support test functions (φ=0 on boundary), boundary terms vanish.

        Returns: u·φ_t - α·u_x·φ_x
        """
        if params is not None and 'alpha' in params:
            alpha = params['alpha']
        else:
            alpha = self.params['alpha']

        return u * phi_t - alpha * u_x * phi_x

    def analytical_solution(
        self,
        x: torch.Tensor | np.ndarray,
        t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """
        Analytical solution for heat equation with sin IC and zero BCs.

        u(x, t) = sin(πx/L)·exp(-α·(π/L)²·t)
        """
        alpha = self.params['alpha']
        L = self.L
        k = np.pi / L

        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return torch.sin(k * x) * torch.exp(-alpha * k**2 * t)
        else:
            return np.sin(k * x) * np.exp(-alpha * k**2 * t)

    def default_ic(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Default initial condition: u(x, 0) = sin(πx/L)
        """
        L = self.L
        if isinstance(x, torch.Tensor):
            return torch.sin(np.pi * x / L)
        else:
            return np.sin(np.pi * x / L)

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
    print("Testing HeatEquation1D...")

    # Create PDE
    pde = HeatEquation1D(params={'alpha': 0.01})
    print(f"Created: {pde}")
    print(f"Parameters: {pde.params}")

    # Test analytical solution
    x = np.linspace(0, 1, 5)
    t = np.array([0.0, 0.5, 1.0])

    for t_val in t:
        u = pde.analytical_solution(x, np.full_like(x, t_val))
        print(f"t={t_val}: u_max={u.max():.4f}")

    # Test IC/BC
    x_ic = torch.linspace(0, 1, 5).reshape(-1, 1)
    u_ic = pde.default_ic(x_ic)
    assert isinstance(u_ic, torch.Tensor)
    print(f"\nIC: u(x, 0) = {u_ic.squeeze().numpy()}")

    t_bc = torch.linspace(0, 1, 5).reshape(-1, 1)
    u_bc_left = pde.default_bc(t_bc, 'left')
    u_bc_right = pde.default_bc(t_bc, 'right')
    assert isinstance(u_bc_left, torch.Tensor) and isinstance(u_bc_right, torch.Tensor)
    print(f"BC left: u(0, t) = {u_bc_left.squeeze().numpy()}")
    print(f"BC right: u(L, t) = {u_bc_right.squeeze().numpy()}")

    # Test residual with dummy data
    N = 10
    u = torch.rand(N, 1)
    u_t = torch.rand(N, 1)
    u_x = torch.rand(N, 1)
    u_xx = torch.rand(N, 1)

    res = pde.residual(u, u_t, u_x, u_xx)
    print(f"\nResidual shape: {res.shape}")

    # Test weak form
    phi = torch.rand(N, 1)
    phi_x = torch.rand(N, 1)
    phi_t = torch.rand(N, 1)

    weak = pde.weak_form_integrand(u, u_t, u_x, phi, phi_x, phi_t)
    print(f"Weak form integrand shape: {weak.shape}")

    print("\nHeatEquation1D tests passed.")