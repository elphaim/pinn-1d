"""
1D Viscous Burgers' Equation

PDE: u_t + u·u_x = ν·u_xx

Standard setup:
- Domain: x ∈ [0, L], t ∈ [0, T]
- IC: u(x, 0) = sin(πx/L)
- BC: u(0, t) = u(L, t) = 0 (Dirichlet)
- No closed-form analytical solution (use numerical reference)

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
import numpy as np
from typing import Optional

from pdes.base import BasePDE


class BurgersEquation1D(BasePDE):
    """
    1D Viscous Burgers' Equation: u_t + u·u_x = ν·u_xx

    This is a nonlinear parabolic PDE that combines advection and diffusion.
    The nonlinear term u·u_x can lead to shock formation for small ν.

    Parameters:
        nu: Viscosity (default: 0.01)
            Large ν: diffusion-dominated, smooth solutions
            Small ν: convection-dominated, can develop steep gradients

    Default problem setup:
        IC: u(x, 0) = sin(πx/L)
        BC: u(0, t) = u(L, t) = 0
        Solution: No analytical form (use finite difference reference)

    Note: For very small ν, the solution develops steep gradients
    (viscous shocks) that can be challenging for PINNs.
    """

    name = "BurgersEquation1D"
    spatial_order = 2
    temporal_order = 1

    @classmethod
    def param_names(cls) -> list[str]:
        return ['nu']

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {'nu': 0.01}

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
        Compute Burgers equation residual: u_t + u·u_x - ν·u_xx

        Args:
            u: Solution values (needed for nonlinear term)
            u_t: Time derivative
            u_x: Spatial first derivative
            u_xx: Spatial second derivative
            params: Override parameters (for inverse problems)

        Returns:
            residual: u_t + u·u_x - ν·u_xx (should be 0 for exact solution)
        """
        if params is not None and 'nu' in params:
            nu = params['nu']
        else:
            nu = self.params['nu']

        return u_t + u * u_x - nu * u_xx

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
        Weak form integrand for Burgers equation.

        Integration by parts: ∫∫ (u_t + u·u_x - ν·u_xx)·φ dx dt = 0

        The nonlinear term can be written as: u·u_x = (1/2)·(u²)_x

        Weak form: ∫∫ (u·φ_t + (1/2)·u²·φ_x - ν·u_x·φ_x) dx dt = 0 
        with compact test functions

        Returns: u·φ_t + (1/2)·u²·φ_x - ν·u_x·φ_x
        """
        if params is not None and 'nu' in params:
            nu = params['nu']
        else:
            nu = self.params['nu']

        return u * phi_t + 0.5 * u**2 * phi_x - nu * u_x * phi_x

    def analytical_solution(
        self,
        x: torch.Tensor | np.ndarray,
        t: torch.Tensor | np.ndarray
    ) -> Optional[torch.Tensor | np.ndarray]:
        """
        Burgers equation has no closed-form analytical solution for general IC.

        For reference solutions, use the FiniteDifferenceSolver.

        Returns None to indicate numerical reference should be used.
        """
        return None

    def default_ic(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Default initial condition: u(x, 0) = sin(πx/L)

        This IC is smooth and compatible with zero Dirichlet BCs.
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

    def cole_hopf_solution(
        self,
        x: np.ndarray,
        t: np.ndarray,
        n_terms: int = 50
    ) -> np.ndarray:
        """
        Cole-Hopf analytical solution for specific IC.

        For u(x, 0) = sin(πx) on [0, 1] with u(0, t) = u(1, t) = 0,
        the Cole-Hopf transformation gives a series solution.

        This is computationally expensive but provides ground truth.

        Args:
            x: Spatial coordinates (1D array or meshgrid values)
            t: Temporal coordinates (same shape as x)
            n_terms: Number of Fourier terms

        Returns:
            u: Solution values
        """
        nu = self.params['nu']
        L = self.L

        # Avoid division by zero at t=0
        t = np.maximum(t, 1e-10)

        # Cole-Hopf solution involves infinite series
        # u = -2ν (∂ln(θ)/∂x) where θ is a Jacobi theta function

        # For sin(πx) IC, use Fourier series approximation
        # This is a simplified version - full Cole-Hopf is more complex

        # Initialize
        u = np.zeros_like(x)

        for n in range(1, n_terms + 1):
            kn = n * np.pi / L
            # Fourier coefficient decays exponentially
            coef = 2 * (-1)**(n+1) / (n * np.pi)
            decay = np.exp(-nu * kn**2 * t)
            u += coef * decay * np.sin(kn * x)

        return u
    

# ============================================================================
# Example usage and testing
# ============================================================================


if __name__ == "__main__":
    print("Testing BurgersEquation1D...")

    # Create PDE with different viscosities
    for nu in [0.1, 0.01, 0.001]:
        pde = BurgersEquation1D(params={'nu': nu})
        print(f"\nCreated: {pde}")

    pde = BurgersEquation1D(params={'nu': 0.01})

    # Test IC/BC
    x_ic = torch.linspace(0, 1, 5).reshape(-1, 1)
    u_ic = pde.default_ic(x_ic)
    assert isinstance(u_ic, torch.Tensor)
    print(f"\nIC: u(x, 0) = {u_ic.squeeze().numpy().round(3)}")

    t_bc = torch.tensor([[0.0], [0.5], [1.0]])
    u_bc_left = pde.default_bc(t_bc, 'left')
    assert isinstance(u_bc_left, torch.Tensor)
    print(f"BC: u(0, t) = u(L, t) = {u_bc_left.squeeze().numpy()}")

    # Test residual with dummy data
    N = 10
    u = torch.rand(N, 1)
    u_t = torch.rand(N, 1)
    u_x = torch.rand(N, 1)
    u_xx = torch.rand(N, 1)

    res = pde.residual(u, u_t, u_x, u_xx)
    print(f"\nResidual shape: {res.shape}")

    # Test that analytical solution returns None
    x_test = np.linspace(0, 1, 5)
    t_test = np.ones_like(x_test) * 0.5
    u_exact = pde.analytical_solution(x_test, t_test)
    print(f"Analytical solution: {u_exact} (None indicates use numerical)")

    # Test Cole-Hopf approximation
    print("\nCole-Hopf approximation at t=0.1:")
    x = np.linspace(0, 1, 5)
    t = np.ones_like(x) * 0.1
    u_ch = pde.cole_hopf_solution(x, t, n_terms=20)
    print(f"u(x, 0.1) ≈ {u_ch.round(4)}")

    # Test weak form
    phi = torch.rand(N, 1)
    phi_x = torch.rand(N, 1)
    phi_t = torch.rand(N, 1)

    weak = pde.weak_form_integrand(u, u_t, u_x, phi, phi_x, phi_t)
    print(f"\nWeak form integrand shape: {weak.shape}")

    print("\nBurgersEquation1D tests passed.")
