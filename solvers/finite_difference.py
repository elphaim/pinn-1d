"""
Finite Difference Solvers for 1D PDEs

Provides reference numerical solutions for validation and multi-fidelity training.

Supported PDEs:
- Heat equation (FTCS explicit scheme)
- Wave equation (Central difference)
- Advection equation (Upwind scheme)
- Burgers equation (FTCS + upwind for nonlinear term)

Author: elphaim with Claude Code
Date: February 2026
"""

import numpy as np
from typing import Optional

from pdes.base import BasePDE
from pdes.heat import HeatEquation1D
from pdes.wave import WaveEquation1D
from pdes.advection import AdvectionEquation1D
from pdes.burgers import BurgersEquation1D


class FiniteDifferenceSolver:
    """
    Finite difference solver for 1D time-dependent PDEs.

    Automatically selects appropriate scheme based on PDE type:
    - Heat: FTCS (Forward Time Central Space) explicit scheme
    - Wave: Central difference in time and space
    - Advection: Upwind scheme (direction based on velocity sign)
    - Burgers: FTCS for diffusion + upwind for convection

    Args:
        pde: BasePDE instance defining the PDE and its parameters
        nx: Number of spatial grid points (including boundaries)
        nt: Number of time steps

    Example:
        pde = HeatEquation1D(params={'alpha': 0.01})
        solver = FiniteDifferenceSolver(pde, nx=101, nt=1001)
        x, t, u = solver.solve()
    """

    def __init__(
        self,
        pde: BasePDE,
        nx: int = 101,
        nt: int = 1001
    ):
        self.pde = pde
        self.nx = nx
        self.nt = nt

        # Domain from PDE
        self.L = pde.L
        self.T = pde.T

        # Grid setup
        self.x = np.linspace(0, self.L, nx)
        self.t = np.linspace(0, self.T, nt)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

        # Select solver based on PDE type
        self._solver_map = {
            'HeatEquation1D': self._solve_heat,
            'WaveEquation1D': self._solve_wave,
            'AdvectionEquation1D': self._solve_advection,
            'BurgersEquation1D': self._solve_burgers,
        }

    def solve(
        self,
        params: Optional[dict[str, float]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the PDE using finite differences.

        Args:
            params: Override PDE parameters (e.g., different alpha for model error)
                    If None, uses pde.params

        Returns:
            x: Spatial grid, shape (nx,)
            t: Temporal grid, shape (nt,)
            u: Solution field, shape (nx, nt)
        """
        pde_name = self.pde.name

        if pde_name not in self._solver_map:
            raise NotImplementedError(
                f"No FD solver implemented for {pde_name}. "
                f"Available: {list(self._solver_map.keys())}"
            )

        # Merge override params with PDE params
        solve_params = self.pde.params.copy()
        if params is not None:
            solve_params.update(params)

        return self._solver_map[pde_name](solve_params)

    def _solve_heat(
        self,
        params: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve heat equation using FTCS (Forward Time Central Space).

        u_t = α·u_xx

        Discretization:
        u[i, n+1] = u[i, n] + r·(u[i+1, n] - 2·u[i, n] + u[i-1, n])
        where r = α·dt/dx²

        Stability: r ≤ 0.5 (CFL condition)
        """
        alpha = params['alpha']

        # Stability parameter
        r = alpha * self.dt / self.dx**2
        if r > 0.5:
            print(f"Warning: Heat FTCS unstable (r={r:.3f} > 0.5). "
                  f"Consider increasing nx or nt.")

        # Initialize solution
        u = np.zeros((self.nx, self.nt))

        # Initial condition
        u[:, 0] = self.pde.default_ic(self.x)

        # Time stepping
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                u[i, n+1] = u[i, n] + r * (u[i+1, n] - 2*u[i, n] + u[i-1, n])

            # Boundary conditions (Dirichlet, already zero from initialization)
            u[0, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'left')[0]
            u[-1, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'right')[0]

        return self.x, self.t, u

    def _solve_wave(
        self,
        params: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve wave equation using central differences.

        u_tt = c²·u_xx

        Discretization:
        u[i, n+1] = 2·u[i, n] - u[i, n-1] + r²·(u[i+1, n] - 2·u[i, n] + u[i-1, n])
        where r = c·dt/dx

        Stability: r ≤ 1 (CFL condition)
        """
        c = params['c']

        # Courant number
        r = c * self.dt / self.dx
        if r > 1:
            print(f"Warning: Wave equation unstable (CFL={r:.3f} > 1). "
                  f"Consider increasing nx or nt.")

        # Initialize solution
        u = np.zeros((self.nx, self.nt))

        # Initial conditions
        u[:, 0] = self.pde.default_ic(self.x)

        # Initial velocity condition u_t(x, 0) = 0
        # Use: u[i, 1] = u[i, 0] + dt·u_t[i, 0] + (1/2)·(c·dt/dx)²·(u[i+1, 0] - 2·u[i, 0] + u[i-1, 0])
        u_t_0 = self.pde.default_ic_t(self.x)
        if u_t_0 is None:
            u_t_0 = np.zeros_like(self.x)

        for i in range(1, self.nx - 1):
            u[i, 1] = (u[i, 0] + self.dt * u_t_0[i] +
                      0.5 * r**2 * (u[i+1, 0] - 2*u[i, 0] + u[i-1, 0]))

        # Time stepping (n >= 1)
        for n in range(1, self.nt - 1):
            for i in range(1, self.nx - 1):
                u[i, n+1] = (2*u[i, n] - u[i, n-1] +
                           r**2 * (u[i+1, n] - 2*u[i, n] + u[i-1, n]))

            # Boundary conditions
            u[0, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'left')[0]
            u[-1, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'right')[0]

        return self.x, self.t, u

    def _solve_advection(
        self,
        params: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve advection equation using upwind scheme.

        u_t + c·u_x = 0

        Upwind discretization (c > 0):
        u[i, n+1] = u[i, n] - c·dt/dx·(u[i, n] - u[i-1, n])

        Upwind discretization (c < 0):
        u[i, n+1] = u[i, n] - c·dt/dx·(u[i+1, n] - u[i, n])

        Stability: |c|·dt/dx ≤ 1 (CFL condition)
        """
        c = params['c']

        # CFL number
        cfl = abs(c) * self.dt / self.dx
        if cfl > 1:
            print(f"Warning: Advection upwind unstable (CFL={cfl:.3f} > 1). "
                  f"Consider increasing nx or nt.")

        # Initialize solution
        u = np.zeros((self.nx, self.nt))

        # Initial condition
        u[:, 0] = self.pde.default_ic(self.x)

        # Time stepping
        r = c * self.dt / self.dx

        for n in range(self.nt - 1):
            if c >= 0:
                # Upwind (use left point)
                for i in range(1, self.nx):
                    u[i, n+1] = u[i, n] - r * (u[i, n] - u[i-1, n])
                # Left BC (inflow)
                u[0, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'left')[0]
            else:
                # Downwind (use right point)
                for i in range(self.nx - 1):
                    u[i, n+1] = u[i, n] - r * (u[i+1, n] - u[i, n])
                # Right BC (inflow)
                u[-1, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'right')[0]

        return self.x, self.t, u

    def _solve_burgers(
        self,
        params: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Burgers equation using FTCS + upwind.

        u_t + u·u_x = ν·u_xx

        Discretization:
        - Diffusion term: central difference (FTCS)
        - Convection term: upwind based on local velocity u

        u[i, n+1] = u[i, n]
                    - dt·u[i, n]·(upwind derivative)
                    + ν·dt/dx²·(u[i+1, n] - 2·u[i, n] + u[i-1, n])

        Stability: Requires both CFL and diffusion number constraints.
        """
        nu = params['nu']

        # Stability parameters
        r_diff = nu * self.dt / self.dx**2

        # Initialize solution
        u = np.zeros((self.nx, self.nt))

        # Initial condition
        u[:, 0] = self.pde.default_ic(self.x)

        # Time stepping
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                # Diffusion term (central difference)
                diff = r_diff * (u[i+1, n] - 2*u[i, n] + u[i-1, n])

                # Convection term (upwind based on local velocity)
                if u[i, n] >= 0:
                    # Use backward difference
                    conv = u[i, n] * self.dt / self.dx * (u[i, n] - u[i-1, n])
                else:
                    # Use forward difference
                    conv = u[i, n] * self.dt / self.dx * (u[i+1, n] - u[i, n])

                u[i, n+1] = u[i, n] - conv + diff

            # Boundary conditions
            u[0, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'left')[0]
            u[-1, n+1] = self.pde.default_bc(np.array([self.t[n+1]]), 'right')[0]

        # Stability check (approximate)
        u_max = np.abs(u).max()
        cfl_conv = u_max * self.dt / self.dx
        if r_diff > 0.5 or cfl_conv > 1:
            print(f"Warning: Burgers FD potentially unstable "
                  f"(r_diff={r_diff:.3f}, cfl_conv≈{cfl_conv:.3f})")

        return self.x, self.t, u

    def get_solution_at_points(
        self,
        x_query: np.ndarray,
        t_query: np.ndarray,
        params: Optional[dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get interpolated solution at arbitrary (x, t) points.

        Useful for generating training data from FD solution.

        Args:
            x_query: Query x coordinates, shape (N,) or (N, 1)
            t_query: Query t coordinates, shape (N,) or (N, 1)
            params: Override PDE parameters

        Returns:
            u_query: Interpolated solution values, shape (N,)
        """
        from scipy.interpolate import RegularGridInterpolator

        # Solve if not already done or params changed
        x_grid, t_grid, u_grid = self.solve(params)

        # Create interpolator
        interp = RegularGridInterpolator(
            (x_grid, t_grid),
            u_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Flatten query points
        x_flat = np.asarray(x_query).flatten()
        t_flat = np.asarray(t_query).flatten()
        points = np.column_stack([x_flat, t_flat])

        return interp(points)


# ============================================================================
# Example usage and testing
# ============================================================================


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing FiniteDifferenceSolver...")

    # Test each PDE type
    pdes = [
        HeatEquation1D(params={'alpha': 0.01}),
        WaveEquation1D(params={'c': 1.0}),
        AdvectionEquation1D(params={'c': 0.5}),
        BurgersEquation1D(params={'nu': 0.01}),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, pde in zip(axes, pdes):
        print(f"\nSolving {pde.name}...")

        # Create solver with appropriate resolution
        if pde.name == 'BurgersEquation1D':
            solver = FiniteDifferenceSolver(pde, nx=201, nt=5001)
        else:
            solver = FiniteDifferenceSolver(pde, nx=101, nt=1001)

        # Solve
        x, t, u = solver.solve()

        # Plot
        X, T = np.meshgrid(x, t, indexing='ij')
        im = ax.contourf(X, T, u, levels=20, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(pde.name)
        plt.colorbar(im, ax=ax)

        # Compare to analytical if available
        u_exact = pde.analytical_solution(X, T)
        if u_exact is not None:
            error = np.sqrt(np.mean((u - u_exact)**2))
            print(f"  RMSE vs analytical: {error:.6e}")
        else:
            print(f"  No analytical solution (numerical only)")

    plt.tight_layout()
    plt.savefig('results/fd_solver_test.png', dpi=150)
    print("\nPlot saved to results/fd_solver_test.png")
    plt.show()

    # Test interpolation
    print("\nTesting interpolation...")
    pde = HeatEquation1D(params={'alpha': 0.01})
    solver = FiniteDifferenceSolver(pde, nx=51, nt=501)

    x_query = np.array([0.25, 0.5, 0.75])
    t_query = np.array([0.1, 0.1, 0.1])

    u_interp = solver.get_solution_at_points(x_query, t_query)
    u_exact = pde.analytical_solution(x_query, t_query)
    assert isinstance(u_exact, np.ndarray)

    print(f"Query points: x={x_query}, t={t_query}")
    print(f"Interpolated: {u_interp.round(6)}")
    print(f"Analytical:   {u_exact.round(6)}")

    print("\nFiniteDifferenceSolver tests passed.")
