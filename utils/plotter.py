"""
Plotting module to visualize PINN solutions for any 1D time-dependent PDE

Supports forward and inverse problems with automatic reference solution
computation (analytical or finite difference fallback).

Author: elphaim with Claude Code
Date: February 2026
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional

from models.pinn import GeneralizedPINN
from pdes.base import BasePDE
from solvers.finite_difference import FiniteDifferenceSolver


def _get_reference_solution(
    pde: BasePDE,
    x_flat: np.ndarray,
    t_flat: np.ndarray,
    shape: tuple[int, int],
    nx_fd: int = 201,
    nt_fd: int = 2001
) -> Optional[np.ndarray]:
    """
    Compute reference solution: analytical if available, otherwise finite difference.

    Args:
        pde: BasePDE instance
        x_flat: Flattened x coordinates, shape (N,) or (N, 1)
        t_flat: Flattened t coordinates, shape (N,) or (N, 1)
        shape: Target reshape dimensions (nx, nt)
        nx_fd: FD spatial resolution (used only if no analytical solution)
        nt_fd: FD temporal resolution (used only if no analytical solution)

    Returns:
        Reference solution reshaped to `shape`, or None if unavailable
    """
    u_ref = pde.analytical_solution(x_flat, t_flat)
    if u_ref is not None:
        if isinstance(u_ref, torch.Tensor):
            u_ref = u_ref.numpy()
        return u_ref.reshape(shape)

    # Fallback to finite difference
    solver = FiniteDifferenceSolver(pde, nx=nx_fd, nt=nt_fd)
    u_ref_flat = solver.get_solution_at_points(x_flat, t_flat)
    return u_ref_flat.reshape(shape)


def _reference_label(pde: BasePDE) -> str:
    """Return label for reference solution based on availability."""
    # Check if analytical solution exists by testing a single point
    x_test = np.array([pde.L / 2])
    t_test = np.array([pde.T / 2])
    if pde.analytical_solution(x_test, t_test) is not None:
        return "Analytical Solution"
    return "FD Reference Solution"


def plot_solution(
    model: GeneralizedPINN,
    data: dict,
    u_ref: Optional[np.ndarray] = None,
    ref_label: Optional[str] = None,
    n_grid: int = 100,
    cmap: str = 'hot',
    save_path: Optional[str] = None
):
    """
    Plot PINN solution for any PDE. Handles both forward and inverse problems.

    Forward problem: predicted solution, reference solution, pointwise error.
    Inverse problem: reconstruction with measurement locations, measurement fit.

    Args:
        model: GeneralizedPINN or StrategicGeneralizedPINN instance
        data: Training data dictionary (needs 'x_m', 't_m', 'u_m' for inverse)
        u_ref: Optional reference solution on the evaluation grid, shape (n_grid, n_grid).
               If None, computed automatically from PDE analytical solution or FD solver.
        ref_label: Label for reference solution plot. Auto-detected if None.
        n_grid: Number of grid points per axis for evaluation
        cmap: Colormap for contour plots
        save_path: Path to save figure (optional)
    """
    pde = model.pde

    # Generate evaluation mesh using PDE domain
    x_eval = torch.linspace(0, pde.L, n_grid).reshape(-1, 1)
    t_eval = torch.linspace(0, pde.T, n_grid).reshape(-1, 1)
    X, T = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij')
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)

    X_np = X.numpy()
    T_np = T.numpy()

    if model.is_inverse:
        _plot_inverse(model, data, pde, x_flat, t_flat, X_np, T_np,
                      n_grid, cmap, save_path)
    else:
        _plot_forward(model, pde, x_flat, t_flat, X_np, T_np,
                      u_ref, ref_label, n_grid, cmap, save_path)


def _plot_forward(
    model: GeneralizedPINN,
    pde: BasePDE,
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    X_np: np.ndarray,
    T_np: np.ndarray,
    u_ref: Optional[np.ndarray],
    ref_label: Optional[str],
    n_grid: int,
    cmap: str,
    save_path: Optional[str]
):
    """Plot forward problem: predicted, reference, error."""
    u_pred = model.predict(x_flat, t_flat).reshape(n_grid, n_grid)

    # Get reference solution
    if u_ref is None:
        u_ref = _get_reference_solution(
            pde, x_flat.numpy(), t_flat.numpy(), (n_grid, n_grid)
        )
    assert u_ref is not None

    if ref_label is None:
        ref_label = _reference_label(pde)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{pde.name} — {pde}', fontsize=13)

    # Plot 1: Predicted solution
    im1 = axes[0].contourf(X_np, T_np, u_pred, levels=20, cmap=cmap)
    axes[0].set_title('PINN Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: Reference solution
    im2 = axes[1].contourf(X_np, T_np, u_ref, levels=20, cmap=cmap)
    axes[1].set_title(ref_label)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    plt.colorbar(im2, ax=axes[1])

    # Plot 3: Pointwise error
    error = np.abs(u_pred - u_ref)
    u_ref_norm = np.sqrt(np.sum(u_ref**2))
    if u_ref_norm > 0:
        rel_l2 = np.sqrt(np.sum((u_pred - u_ref)**2)) / u_ref_norm * 100
    else:
        rel_l2 = 0.0

    im3 = axes[2].contourf(X_np, T_np, error, levels=20, cmap='viridis')
    axes[2].set_title(f'|Error| (Rel. L2={rel_l2:.4f}%)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _plot_inverse(
    model: GeneralizedPINN,
    data: dict,
    pde: BasePDE,
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    X_np: np.ndarray,
    T_np: np.ndarray,
    n_grid: int,
    cmap: str,
    save_path: Optional[str]
):
    """Plot inverse problem: reconstruction with measurements, measurement fit."""
    u_pred = model.predict(x_flat, t_flat).reshape(n_grid, n_grid)
    learned = model.get_learned_params()

    # Build title showing learned parameters
    param_strs = []
    for name, value in learned.items():
        true_val = pde.params.get(name)
        if true_val is not None:
            error_pct = abs(value - true_val) / abs(true_val) * 100
            param_strs.append(f'{name}={value:.6f} (err={error_pct:.3f}%)')
        else:
            param_strs.append(f'{name}={value:.6f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{pde.name} — Inverse Problem', fontsize=13)

    # Plot 1: Reconstructed solution with measurements
    im1 = axes[0].contourf(X_np, T_np, u_pred, levels=20, cmap=cmap)
    if 'x_m' in data and data['x_m'] is not None:
        x_m = data['x_m'].cpu().numpy()
        t_m = data['t_m'].cpu().numpy()
        axes[0].scatter(x_m, t_m, c='cyan', s=50, marker='x',
                        linewidths=2, label='Measurements')
        axes[0].legend()
    title_params = ', '.join(f'{n}={v:.6f}' for n, v in learned.items())
    axes[0].set_title(f'PINN Reconstruction ({title_params})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: Measurement fit
    if 'x_m' in data and data['x_m'] is not None:
        u_m = data['u_m'].cpu().numpy()
        u_m_pred = model.predict(data['x_m'], data['t_m'])
        axes[1].scatter(u_m, u_m_pred, alpha=0.6, s=50)
        axes[1].plot([u_m.min(), u_m.max()], [u_m.min(), u_m.max()],
                     'r--', linewidth=2)
        axes[1].set_xlabel('Measured u')
        axes[1].set_ylabel('Predicted u')
        axes[1].set_title('Measurement Fit')
        axes[1].grid(True, alpha=0.3)

        r2 = 1 - np.sum((u_m - u_m_pred)**2) / np.sum((u_m - u_m.mean())**2)
        info_lines = [f'R\u00b2 = {r2:.4f}']
        for line in param_strs:
            info_lines.append(line)
        axes[1].text(
            0.05, 0.95, '\n'.join(info_lines),
            transform=axes[1].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )
    else:
        axes[1].text(0.5, 0.5, 'No measurement data',
                     transform=axes[1].transAxes, ha='center', va='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_time_slices(
    model: GeneralizedPINN,
    times: Optional[list[float]] = None,
    u_ref: Optional[np.ndarray] = None,
    ref_label: Optional[str] = None,
    n_x: int = 200,
    n_grid_fd: int = 201,
    save_path: Optional[str] = None
):
    """
    Plot u(x) at specific time snapshots comparing PINN vs reference.

    Args:
        model: GeneralizedPINN or StrategicGeneralizedPINN instance
        times: List of time values to plot. Default: 5 evenly spaced in [0, T].
        u_ref: Optional pre-computed reference on (n_x, len(times)) grid.
               If None, computed automatically from analytical or FD solver.
        ref_label: Label for reference curves. Auto-detected if None.
        n_x: Number of spatial evaluation points
        n_grid_fd: FD resolution if fallback is needed
        save_path: Path to save figure (optional)
    """
    pde = model.pde

    if times is None:
        times = np.linspace(0, pde.T, 5).tolist()
    assert times is not None

    if ref_label is None:
        ref_label = _reference_label(pde)

    x = torch.linspace(0, pde.L, n_x).reshape(-1, 1)
    x_np = x.numpy().flatten()

    # Pre-compute FD solution once if needed (avoid solving per slice)
    has_analytical = pde.analytical_solution(
        np.array([pde.L / 2]), np.array([pde.T / 2])
    ) is not None

    fd_solver = None
    if not has_analytical and u_ref is None:
        fd_solver = FiniteDifferenceSolver(pde, nx=n_grid_fd, nt=n_grid_fd * 10)

    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(4.5 * n_times, 4), squeeze=False)
    axes = axes[0]
    fig.suptitle(f'{pde.name} — Time Slices', fontsize=13)

    for i, t_val in enumerate(times):
        t_tensor = torch.full((n_x, 1), t_val)

        # PINN prediction
        u_pred = model.predict(x, t_tensor).flatten()

        # Reference solution
        if u_ref is not None:
            u_exact = u_ref[:, i]
        elif has_analytical:
            u_exact = pde.analytical_solution(x_np, np.full(n_x, t_val))
            if isinstance(u_exact, torch.Tensor):
                u_exact = u_exact.numpy()
            assert u_exact is not None
            u_exact = u_exact.flatten()
        else:
            assert fd_solver is not None
            t_query = np.full(n_x, t_val)
            u_exact = fd_solver.get_solution_at_points(x_np, t_query)

        axes[i].plot(x_np, u_exact, 'k-', linewidth=2, label=ref_label)
        axes[i].plot(x_np, u_pred, 'r--', linewidth=2, label='PINN')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('u')
        axes[i].set_title(f't = {t_val:.3f}')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
