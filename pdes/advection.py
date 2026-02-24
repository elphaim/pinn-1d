"""
1D Advection Equation

PDE: u_t + c·u_x = 0

Standard setup:
- Domain: x ∈ [0, L], t ∈ [0, T]
- IC: u(x, 0) = sin(2πx/L) (periodic-friendly)
- BC: u(0, t) = -sin(2π·c·t/L) (incoming wave for c > 0)
- Analytical: u(x, t) = sin(2π(x - c·t)/L)

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
import numpy as np
from typing import Optional

from pdes.base import BasePDE


class AdvectionEquation1D(BasePDE):
    """
    1D Advection (Transport) Equation: u_t + c·u_x = 0

    This is a first-order hyperbolic PDE modeling wave propagation
    without dispersion or diffusion.

    Parameters:
        c: Advection velocity (default: 1.0)
            c > 0: rightward propagation
            c < 0: leftward propagation

    Default problem setup:
        IC: u(x, 0) = sin(2πx/L)
        BC: u(0, t) = -sin(2π·c·t/L) for c > 0 (inflow BC)
        Solution: u(x, t) = sin(2π(x - c·t)/L)

    Note: This is a first-order PDE (spatial_order=1), so u_xx is not needed.
    """

    name = "AdvectionEquation1D"
    spatial_order = 1
    temporal_order = 1

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
        Compute advection equation residual: u_t + c·u_x

        Args:
            u_t: Time derivative
            u_x: Spatial first derivative
            params: Override parameters (for inverse problems)

        Returns:
            residual: u_t + c·u_x (should be 0 for exact solution)
        """
        if params is not None and 'c' in params:
            c = params['c']
        else:
            c = self.params['c']

        return u_t + c * u_x

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
        Weak form integrand for advection equation.

        Integration by parts: ∫∫ (u_t + c·u_x)·φ dx dt = 0
        becomes: ∫∫ (u·φ_t + c·u·φ_x) dx dt + boundary terms = 0

        With compact support test functions, boundary terms vanish.

        Returns: u·φ_t + c·u·φ_x
        """
        if params is not None and 'c' in params:
            c = params['c']
        else:
            c = self.params['c']

        return u * phi_t + c * u * phi_x

    def analytical_solution(
        self,
        x: torch.Tensor | np.ndarray,
        t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """
        Analytical solution for advection equation.

        u(x, t) = sin(2π(x - c·t)/L)

        The solution is the initial condition shifted by c·t.
        """
        c = self.params['c']
        L = self.L
        k = 2 * np.pi / L

        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return torch.sin(k * (x - c * t))
        else:
            return np.sin(k * (x - c * t))

    def default_ic(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Default initial condition: u(x, 0) = sin(2πx/L)

        This IC is periodic-friendly and works well with periodic or
        characteristic boundary conditions.
        """
        L = self.L
        k = 2 * np.pi / L

        if isinstance(x, torch.Tensor):
            return torch.sin(k * x)
        else:
            return np.sin(k * x)

    def default_bc(
        self,
        t: torch.Tensor | np.ndarray,
        boundary: str
    ) -> torch.Tensor | np.ndarray:
        """
        Default boundary conditions for advection.

        For c > 0 (rightward flow):
            Left BC (x=0): u(0, t) = -sin(2π·c·t/L) (inflow)
            Right BC (x=L): Not needed (outflow)

        For c < 0 (leftward flow):
            Right BC (x=L): u(L, t) = sin(2π(L - c·t)/L) (inflow)
            Left BC (x=0): Not needed (outflow)

        We provide both BCs for compatibility with PINN training.
        """
        c = self.params['c']
        L = self.L
        k = 2 * np.pi / L

        if boundary == 'left':
            # u(0, t) from analytical solution
            x = 0.0
        else:
            # u(L, t) from analytical solution
            x = L

        if isinstance(t, torch.Tensor):
            return torch.sin(k * (x - c * t))
        else:
            return np.sin(k * (x - c * t))


# ============================================================================
# Example usage and testing
# ============================================================================


if __name__ == "__main__":
    print("Testing AdvectionEquation1D...")

    # Create PDE
    pde = AdvectionEquation1D(params={'c': 1.0})
    print(f"Created: {pde}")
    print(f"Parameters: {pde.params}")
    print(f"Spatial order: {pde.spatial_order}")

    # Test analytical solution
    x = np.linspace(0, 1, 5)
    t = np.array([0.0, 0.25, 0.5])

    print("\nAnalytical solution (wave propagates right):")
    for t_val in t:
        u = pde.analytical_solution(x, np.full_like(x, t_val))
        assert isinstance(u, np.ndarray)
        print(f"t={t_val}: u = {u.round(3)}")

    # Test IC/BC
    x_ic = torch.linspace(0, 1, 5).reshape(-1, 1)
    u_ic = pde.default_ic(x_ic)
    assert isinstance(u_ic, torch.Tensor)
    print(f"\nIC: u(x, 0) = {u_ic.squeeze().numpy().round(3)}")

    t_bc = torch.tensor([[0.0], [0.25], [0.5]])
    u_bc_left = pde.default_bc(t_bc, 'left')
    u_bc_right = pde.default_bc(t_bc, 'right')
    assert isinstance(u_bc_left, torch.Tensor) and isinstance(u_bc_right, torch.Tensor)
    print(f"BC left: u(0, t) = {u_bc_left.squeeze().numpy().round(3)}")
    print(f"BC right: u(L, t) = {u_bc_right.squeeze().numpy().round(3)}")

    # Test residual
    N = 10
    u = torch.rand(N, 1)
    u_t = torch.rand(N, 1)
    u_x = torch.rand(N, 1)
    u_xx = torch.rand(N, 1)  # Not used for advection

    res = pde.residual(u, u_t, u_x, u_xx)
    print(f"\nResidual shape: {res.shape}")

    # Test weak form
    phi = torch.rand(N, 1)
    phi_x = torch.rand(N, 1)
    phi_t = torch.rand(N, 1)

    weak = pde.weak_form_integrand(u, u_t, u_x, phi, phi_x, phi_t)
    print(f"Weak form integrand shape: {weak.shape}")

    print("\nAdvectionEquation1D tests passed.")
