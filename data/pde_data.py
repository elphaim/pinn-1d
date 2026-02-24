"""
Generalized Data Generation for PDEs

Generates training data for any PDE implementing the BasePDE interface:
- Collocation points for PDE residual
- Boundary and initial condition points
- Synthetic measurements with noise
- Multi-fidelity data from numerical solvers

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from pdes.base import BasePDE


class PDEData:
    """
    Data generator for any 1D time-dependent PDE.

    Uses the PDE's interface methods to generate:
    - Collocation points for PDE residual
    - Boundary conditions via pde.default_bc()
    - Initial conditions via pde.default_ic() and pde.default_ic_t()
    - Measurements via pde.analytical_solution() or numerical solver

    Args:
        pde: BasePDE instance defining the PDE
        N_f: Number of collocation points for residual
        N_bc: Number of boundary condition points
        N_ic: Number of initial condition points
        N_sensors: Number of spatial sensor locations for measurements
        N_time_measurements: Number of time measurements per sensor
        noise_level: Measurement noise std as fraction of signal
        device: PyTorch device
        random_seed: Random seed for reproducibility

    Example:
        pde = HeatEquation1D(params={'alpha': 0.01})
        data_gen = PDEData(pde, N_f=10000, N_bc=100, N_ic=100)
        data = data_gen.generate_full_dataset()
    """

    def __init__(
        self,
        pde: BasePDE,
        N_f: int = 10000,
        N_bc: int = 100,
        N_ic: int = 100,
        N_sensors: int = 10,
        N_time_measurements: int = 10,
        noise_level: float = 0.01,
        device: str = 'cpu',
        random_seed: int = 42
    ):
        self.pde = pde
        self.N_f = N_f
        self.N_bc = N_bc
        self.N_ic = N_ic
        self.N_sensors = N_sensors
        self.N_time_measurements = N_time_measurements
        self.noise_level = noise_level
        self.device = device

        # Domain from PDE
        self.L = pde.L
        self.T = pde.T

        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Set dtype
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
            self.dtype = torch.float64
        else:
            torch.set_default_dtype(torch.float32)
            self.dtype = torch.float32

        print(f"PDEData initialized for {pde.name}:")
        print(f"  Domain: x in [0, {self.L}], t in [0, {self.T}]")
        print(f"  Collocation: {N_f}, BC: {N_bc}, IC: {N_ic}")
        print(f"  Measurements: {N_sensors} sensors x {N_time_measurements} times")

    def generate_collocation_points(
        self,
        method: str = 'uniform'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate collocation points for PDE residual.

        Args:
            method: 'uniform' or 'lhs' (Latin Hypercube Sampling)

        Returns:
            x_f: Spatial coordinates, shape (N_f, 1)
            t_f: Temporal coordinates, shape (N_f, 1)
        """
        if method == 'uniform':
            x_f = torch.rand(self.N_f, 1, dtype=self.dtype) * self.L
            t_f = torch.rand(self.N_f, 1, dtype=self.dtype) * self.T
        elif method == 'lhs':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=2, rng=42)
            sample = sampler.random(n=self.N_f)
            x_f = torch.tensor(sample[:, 0:1] * self.L, dtype=self.dtype)
            t_f = torch.tensor(sample[:, 1:2] * self.T, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown method: {method}")

        return x_f.to(self.device), t_f.to(self.device)

    def generate_boundary_conditions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate boundary condition points using pde.default_bc().

        Returns:
            x_bc: Spatial coordinates (at x=0 and x=L)
            t_bc: Temporal coordinates
            u_bc: Boundary values
        """
        # Left boundary (x=0)
        t_left = torch.rand(self.N_bc, 1, dtype=self.dtype) * self.T
        x_left = torch.zeros(self.N_bc, 1, dtype=self.dtype)
        u_left = torch.tensor(
            self.pde.default_bc(t_left.numpy(), 'left'),
            dtype=self.dtype
        )

        # Right boundary (x=L)
        t_right = torch.rand(self.N_bc, 1, dtype=self.dtype) * self.T
        x_right = torch.ones(self.N_bc, 1, dtype=self.dtype) * self.L
        u_right = torch.tensor(
            self.pde.default_bc(t_right.numpy(), 'right'),
            dtype=self.dtype
        )

        # Combine
        x_bc = torch.cat([x_left, x_right], dim=0)
        t_bc = torch.cat([t_left, t_right], dim=0)
        u_bc = torch.cat([u_left.reshape(-1, 1), u_right.reshape(-1, 1)], dim=0)

        return x_bc.to(self.device), t_bc.to(self.device), u_bc.to(self.device)

    def generate_initial_conditions(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate initial condition points using pde.default_ic() and pde.default_ic_t().

        Returns:
            x_ic: Spatial coordinates
            t_ic: Temporal coordinates (all zeros)
            u_ic: Initial condition values
            u_ic_t: Initial velocity (for wave equation, else None)
        """
        x_ic = torch.rand(self.N_ic, 1, dtype=self.dtype) * self.L
        t_ic = torch.zeros(self.N_ic, 1, dtype=self.dtype)

        # Initial condition u(x, 0)
        u_ic = self.pde.default_ic(x_ic)
        if isinstance(u_ic, np.ndarray):
            u_ic = torch.tensor(u_ic, dtype=self.dtype)

        # Initial velocity u_t(x, 0) for hyperbolic PDEs
        u_ic_t = self.pde.default_ic_t(x_ic)
        if u_ic_t is not None:
            if isinstance(u_ic_t, np.ndarray):
                u_ic_t = torch.tensor(u_ic_t, dtype=self.dtype)
            u_ic_t = u_ic_t.to(self.device)

        return (
            x_ic.to(self.device),
            t_ic.to(self.device),
            u_ic.to(self.device),
            u_ic_t
        )

    def generate_measurements(
        self,
        add_noise: bool = True,
        use_numerical: bool = False,
        solver_nx: int = 101,
        solver_nt: int = 1001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Generate synthetic sensor measurements.

        Uses analytical solution if available, otherwise numerical solver.

        Args:
            add_noise: Whether to add Gaussian noise
            use_numerical: Force use of numerical solver even if analytical exists
            solver_nx, solver_nt: Resolution for numerical solver

        Returns:
            x_m: Measurement spatial coordinates
            t_m: Measurement temporal coordinates
            u_m: Measured values (with noise if add_noise=True)
            info: Metadata dictionary
        """
        # Sensor locations (evenly spaced, excluding boundaries)
        x_sensors = np.linspace(
            self.L / (self.N_sensors + 1),
            self.L * self.N_sensors / (self.N_sensors + 1),
            self.N_sensors
        )

        # Measurement times
        t_measurements = np.linspace(0, self.T, self.N_time_measurements)

        # Create meshgrid
        X_mesh, T_mesh = np.meshgrid(x_sensors, t_measurements)
        x_m_np = X_mesh.flatten()
        t_m_np = T_mesh.flatten()

        # Get true values
        u_true = self.pde.analytical_solution(x_m_np, t_m_np)

        if u_true is None or use_numerical:
            # Use numerical solver
            from solvers.finite_difference import FiniteDifferenceSolver

            solver = FiniteDifferenceSolver(self.pde, nx=solver_nx, nt=solver_nt)
            u_true = solver.get_solution_at_points(x_m_np, t_m_np)
            solution_type = 'numerical'
        else:
            solution_type = 'analytical'

        # Add noise
        if add_noise:
            noise = np.random.normal(0, self.noise_level * np.abs(u_true).mean(), size=u_true.shape)
            u_measured = u_true + noise

            signal_power = np.mean(u_true ** 2)
            noise_power = np.mean(noise ** 2)
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        else:
            u_measured = u_true
            snr_db = np.inf

        # Convert to tensors
        x_m = torch.tensor(x_m_np.reshape(-1, 1), dtype=self.dtype)
        t_m = torch.tensor(t_m_np.reshape(-1, 1), dtype=self.dtype)
        u_m = torch.tensor(u_measured.reshape(-1, 1), dtype=self.dtype)

        info = {
            'x_sensors': x_sensors,
            't_measurements': t_measurements,
            'u_true': u_true,
            'u_measured': u_measured,
            'snr_db': snr_db,
            'noise_std': self.noise_level * np.abs(u_true).mean() if add_noise else 0,
            'solution_type': solution_type
        }

        print(f"Measurements generated ({solution_type}):")
        print(f"  Total: {len(x_m)} points")
        if add_noise:
            print(f"  SNR: {snr_db:.1f} dB")

        return x_m.to(self.device), t_m.to(self.device), u_m.to(self.device), info

    def generate_full_dataset(
        self,
        collocation_method: str = 'uniform',
        include_measurements: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Generate complete dataset for training.

        Returns:
            data: Dictionary containing:
                - x_f, t_f: Collocation points
                - x_bc, t_bc, u_bc: Boundary conditions
                - x_ic, t_ic, u_ic: Initial conditions
                - u_ic_t: Initial velocity (for wave equation)
                - x_m, t_m, u_m: Measurements (if include_measurements)
                - measurement_info: Metadata
        """
        print(f"\n{'='*60}")
        print(f"Generating dataset for {self.pde.name}...")
        print(f"{'='*60}")

        # Generate all data
        x_f, t_f = self.generate_collocation_points(method=collocation_method)
        x_bc, t_bc, u_bc = self.generate_boundary_conditions()
        x_ic, t_ic, u_ic, u_ic_t = self.generate_initial_conditions()

        data = {
            'x_f': x_f, 't_f': t_f,
            'x_bc': x_bc, 't_bc': t_bc, 'u_bc': u_bc,
            'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
            'u_ic_t': u_ic_t,  # None for parabolic PDEs
        }

        if include_measurements:
            x_m, t_m, u_m, info = self.generate_measurements()
            data.update({
                'x_m': x_m, 't_m': t_m, 'u_m': u_m,
                'measurement_info': info
            })

        print("Dataset generation complete.")
        return data

    def generate_multi_fidelity_data(
        self,
        hf_sensors: int = 15,
        hf_times: int = 15,
        lf_nx: int = 21,
        lf_nt: int = 101,
        lf_param_error: Optional[dict[str, float]] = None,
        lf_noise: float = 0.05,
        collocation_method: str = 'uniform'
    ) -> dict:
        """
        Generate multi-fidelity dataset.

        High-fidelity: From analytical solution (or fine numerical)
        Low-fidelity: From coarse numerical solver with parameter error

        Args:
            hf_sensors: Number of high-fidelity sensor locations
            hf_times: Number of high-fidelity time measurements
            lf_nx, lf_nt: Low-fidelity solver resolution
            lf_param_error: Parameter perturbations for LF (e.g., {'alpha': 1.2 * true_alpha})
            lf_noise: Noise level for low-fidelity data
            collocation_method: 'uniform' or 'lhs'

        Returns:
            data: Dictionary with all training data
        """
        from solvers.finite_difference import FiniteDifferenceSolver

        print(f"\n{'='*60}")
        print(f"Generating multi-fidelity data for {self.pde.name}...")
        print(f"{'='*60}")

        # Collocation points
        x_f, t_f = self.generate_collocation_points(method=collocation_method)

        # BC and IC
        x_bc, t_bc, u_bc = self.generate_boundary_conditions()
        x_ic, t_ic, u_ic, u_ic_t = self.generate_initial_conditions()

        # High-fidelity measurements
        orig_sensors = self.N_sensors
        orig_times = self.N_time_measurements
        self.N_sensors = hf_sensors
        self.N_time_measurements = hf_times

        x_hf, t_hf, u_hf, hf_info = self.generate_measurements(add_noise=True)
        sigma_hf = hf_info['noise_std']

        self.N_sensors = orig_sensors
        self.N_time_measurements = orig_times

        # Low-fidelity measurements from numerical solver
        lf_params = self.pde.params.copy()
        if lf_param_error is not None:
            lf_params.update(lf_param_error)

        solver = FiniteDifferenceSolver(self.pde, nx=lf_nx, nt=lf_nt)
        x_grid, t_grid, u_grid = solver.solve(params=lf_params)

        # Sample from interior
        x_interior = x_grid[1:-1]
        t_interior = t_grid[1:]

        X_lf, T_lf = np.meshgrid(x_interior, t_interior, indexing='ij')
        x_lf_np = X_lf.flatten()
        t_lf_np = T_lf.flatten()
        u_lf_np = u_grid[1:-1, 1:].flatten()

        # Add noise to LF data
        noise_lf = np.random.normal(0, lf_noise * np.abs(u_lf_np).mean(), size=u_lf_np.shape)
        u_lf_noisy = u_lf_np + noise_lf

        # Compute LF uncertainty (noise + model error)
        u_exact = self.pde.analytical_solution(x_lf_np, t_lf_np)
        if u_exact is not None:
            model_error = np.sqrt(np.mean((u_lf_np - u_exact)**2))
        else:
            model_error = 0.1 * np.abs(u_lf_np).mean()  # Estimate

        sigma_lf = np.sqrt((lf_noise * np.abs(u_lf_np).mean())**2 + model_error**2)

        # Convert to tensors
        x_lf = torch.tensor(x_lf_np.reshape(-1, 1), dtype=self.dtype)
        t_lf = torch.tensor(t_lf_np.reshape(-1, 1), dtype=self.dtype)
        u_lf = torch.tensor(u_lf_noisy.reshape(-1, 1), dtype=self.dtype)

        data = {
            'x_f': x_f, 't_f': t_f,
            'x_bc': x_bc, 't_bc': t_bc, 'u_bc': u_bc,
            'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
            'u_ic_t': u_ic_t,
            'x_hf': x_hf, 't_hf': t_hf, 'u_hf': u_hf,
            'x_lf': x_lf.to(self.device),
            't_lf': t_lf.to(self.device),
            'u_lf': u_lf.to(self.device),
            'sigma_hf': sigma_hf,
            'sigma_lf': sigma_lf,
            'hf_info': hf_info,
            'lf_info': {
                'nx': lf_nx, 'nt': lf_nt,
                'params': lf_params,
                'model_error': model_error,
                'n_measurements': len(x_lf_np)
            }
        }

        print(f"High-fidelity: {x_hf.shape[0]} points, σ={sigma_hf:.4f}")
        print(f"Low-fidelity: {x_lf.shape[0]} points, σ={sigma_lf:.4f}")
        print(f"Effective weight ratio (HF/LF): {(sigma_lf/sigma_hf)**2:.1f}x")

        return data
    
    def visualize_data(
        self, 
        data: dict[str, torch.Tensor], 
        save_path: Optional[str] = None
        ):
        """
        Visualize the generated dataset.
        
        Args:
            data: Dictionary from generate_full_dataset()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Convert to numpy for plotting
        x_f = data['x_f'].cpu().numpy()
        t_f = data['t_f'].cpu().numpy()
        x_bc = data['x_bc'].cpu().numpy()
        t_bc = data['t_bc'].cpu().numpy()
        x_ic = data['x_ic'].cpu().numpy()
        t_ic = data['t_ic'].cpu().numpy()
        x_m = data['x_m'].cpu().numpy()
        t_m = data['t_m'].cpu().numpy()
        u_m = data['u_m'].cpu().numpy()
        
        # Plot 1: Collocation points
        axes[0, 0].scatter(x_f, t_f, s=1, alpha=0.3, c='blue')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('t')
        axes[0, 0].set_title(f'Collocation Points (N={len(x_f)})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Boundary and Initial Conditions
        axes[0, 1].scatter(x_bc, t_bc, s=10, c='red', label='BC', alpha=0.5)
        axes[0, 1].scatter(x_ic, t_ic, s=10, c='green', label='IC', alpha=0.5)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('t')
        axes[0, 1].set_title(f'Boundary (N={len(x_bc)}) & Initial Conditions (N={len(x_ic)})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Measurement locations
        axes[1, 0].scatter(x_m, t_m, s=50, c='orange', marker='x', linewidths=2)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('t')
        axes[1, 0].set_title(f'Measurement Locations (N={len(x_m)})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Measured temperature values
        scatter = axes[1, 1].scatter(x_m, t_m, s=100, c=u_m, cmap='coolwarm', marker='o')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('t')
        axes[1, 1].set_title('Measured Temperature')
        plt.colorbar(scatter, ax=axes[1, 1], label='u (temperature)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


# ============================================================================
# Example usage and testing
# ============================================================================


if __name__ == "__main__":
    print("Testing PDEData...")

    from pdes.heat import HeatEquation1D
    from pdes.wave import WaveEquation1D
    from pdes.burgers import BurgersEquation1D

    # Test 1: Heat equation
    print("\n" + "="*60)
    print("Test 1: Heat Equation Data")
    print("="*60)

    pde = HeatEquation1D(params={'alpha': 0.01})
    data_gen = PDEData(pde, N_f=1000, N_bc=50, N_ic=50)
    data = data_gen.generate_full_dataset()

    print("\nDataset summary:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif key == 'measurement_info':
            print(f"  {key}: {list(value.keys())}")

    # Test 2: Wave equation (with initial velocity)
    print("\n" + "="*60)
    print("Test 2: Wave Equation Data")
    print("="*60)

    pde_wave = WaveEquation1D(params={'c': 1.0})
    data_gen_wave = PDEData(pde_wave, N_f=1000, N_bc=50, N_ic=50)
    data_wave = data_gen_wave.generate_full_dataset()

    print(f"\nInitial velocity data: {data_wave['u_ic_t'].shape if data_wave['u_ic_t'] is not None else 'None'}")

    # Test 3: Burgers equation (numerical measurements)
    print("\n" + "="*60)
    print("Test 3: Burgers Equation Data (numerical)")
    print("="*60)

    pde_burg = BurgersEquation1D(params={'nu': 0.01})
    data_gen_burg = PDEData(pde_burg, N_f=1000, N_bc=50, N_ic=50,
                            N_sensors=5, N_time_measurements=5)
    data_burg = data_gen_burg.generate_full_dataset()

    print(f"Measurement info: {data_burg['measurement_info']}")

    # Test 4: Multi-fidelity data
    print("\n" + "="*60)
    print("Test 4: Multi-Fidelity Data")
    print("="*60)

    pde = HeatEquation1D(params={'alpha': 0.01})
    data_gen = PDEData(pde, N_f=1000, N_bc=50, N_ic=50)

    mf_data = data_gen.generate_multi_fidelity_data(
        hf_sensors=10, hf_times=10,
        lf_nx=11, lf_nt=51,
        lf_param_error={'alpha': 0.012}  # 20% error
    )

    print(f"\nMulti-fidelity dataset:")
    print(f"  HF points: {mf_data['x_hf'].shape[0]}")
    print(f"  LF points: {mf_data['x_lf'].shape[0]}")
    print(f"  σ_hf: {mf_data['sigma_hf']:.4f}")
    print(f"  σ_lf: {mf_data['sigma_lf']:.4f}")

    print("\nPDEData tests passed.")
