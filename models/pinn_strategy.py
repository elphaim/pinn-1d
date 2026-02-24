"""
Generalized Strategy Pattern for PINN Loss Computation

Loss strategies that work with any PDE through the BasePDE interface.

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable

from models.pinn import GeneralizedPINN
from utils.integrator import IntegratorFactory


class GeneralizedLossStrategy(ABC):
    """
    Abstract strategy for computing PINN loss with any PDE.

    Subclasses implement different formulations:
    - Strong form (point-wise PDE residual)
    - Weak form (integrated residual)
    """

    @abstractmethod
    def compute_loss(
        self,
        model: GeneralizedPINN,
        data: dict,
        lambdas: dict[str, float]
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given model and data.

        Args:
            model: GeneralizedPINN instance
            data: Dictionary with required data (format depends on strategy)
            lambdas: Loss weights {'f': ..., 'bc': ..., 'ic': ..., 'm': ...}

        Returns:
            total_loss: Scalar loss tensor
            losses: Dictionary with loss components and tensors
        """
        pass


class GeneralizedStrongFormLoss(GeneralizedLossStrategy):
    """
    Strong-form loss: point-wise PDE residual evaluation.

    Works with any PDE by calling model.residual() which delegates to pde.residual().

    Required data keys:
    - x_f, t_f: Collocation points for PDE residual
    - x_bc, t_bc, u_bc: Boundary conditions
    - x_ic, t_ic, u_ic: Initial conditions
    - x_m, t_m, u_m: Measurements (for inverse problem, optional)
    - x_ic, t_ic, u_ic_t: Initial velocity (for wave equation, optional)
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

    def compute_loss(
        self,
        model: GeneralizedPINN,
        data: dict,
        lambdas: dict[str, float]
    ) -> tuple[torch.Tensor, dict]:
        """Strong-form loss computation."""

        # 1. PDE Residual loss
        x_f = data['x_f'].requires_grad_(True)
        t_f = data['t_f'].requires_grad_(True)

        residual = model.residual(x_f, t_f)
        loss_f = torch.mean(residual ** 2)

        # 2. Boundary loss
        u_bc_pred = model.forward(data['x_bc'], data['t_bc'])
        loss_bc = torch.mean((u_bc_pred - data['u_bc']) ** 2)

        # 3. Initial condition loss
        u_ic_pred = model.forward(data['x_ic'], data['t_ic'])
        loss_ic = torch.mean((u_ic_pred - data['u_ic']) ** 2)

        # 4. Initial velocity loss (for wave equation)
        loss_ic_t = torch.tensor(0.0, device=self.device)
        if 'u_ic_t' in data and data['u_ic_t'] is not None:
            # Compute u_t at t=0
            x_ic = data['x_ic'].requires_grad_(True)
            t_ic = data['t_ic'].requires_grad_(True)
            derivs = model.compute_derivatives(x_ic, t_ic)
            u_t_pred = derivs['u_t']
            loss_ic_t = torch.mean((u_t_pred - data['u_ic_t']) ** 2)

        # 5. Measurement loss (for inverse problem)
        if 'x_m' in data and data['x_m'] is not None:
            u_m_pred = model.forward(data['x_m'], data['t_m'])
            loss_m = torch.mean((u_m_pred - data['u_m']) ** 2)
        else:
            loss_m = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = (
            lambdas.get('f', 1.0) * loss_f +
            lambdas.get('bc', 1.0) * loss_bc +
            lambdas.get('ic', 1.0) * loss_ic +
            lambdas.get('ic_t', 1.0) * loss_ic_t +
            lambdas.get('m', 1.0) * loss_m
        )

        losses = {
            'total': total_loss.item(),
            'residual': loss_f.item(),
            'boundary': loss_bc.item(),
            'initial': loss_ic.item(),
            'initial_velocity': loss_ic_t.item(),
            'measurement': loss_m.item() if torch.is_tensor(loss_m) else 0.0,
            # Tensors for gradient tracking
            'total_t': total_loss,
            'residual_t': loss_f,
            'boundary_t': loss_bc,
            'initial_t': loss_ic,
            'initial_velocity_t': loss_ic_t,
            'measurement_t': loss_m
        }

        return total_loss, losses


class GeneralizedWeakFormLoss(GeneralizedLossStrategy):
    """
    Weak-form loss: integrated PDE residual.

    Uses the PDE's weak_form_integrand() method if available.

    Required data keys:
    - test_funcs: List of test functions φ(x, t)
    - test_doms: List of integration domains [[x_min, x_max], [t_min, t_max]]
    - x_bc, t_bc, u_bc: Boundary conditions
    - x_ic, t_ic, u_ic: Initial conditions
    - x_m, t_m, u_m: Measurements (optional)
    """

    def __init__(
        self,
        integration_method: str = 'gauss_legendre',
        n_integration_points: int = 15,
        device: str = 'cpu'
    ):
        self.device = device

        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.integrator = IntegratorFactory.create(
            method=integration_method,
            n_points=n_integration_points,
            device=device
        )

        print(f"GeneralizedWeakFormLoss initialized:")
        print(f"  Method: {integration_method}")
        print(f"  Points: {n_integration_points}")

    def _compute_weak_residual(
        self,
        model: GeneralizedPINN,
        test_func: Callable,
        domain: list[list[float]]
    ) -> torch.Tensor:
        """Compute weak residual for one test function."""

        def integrand(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Weak form integrand using PDE's method."""
            if not x.requires_grad:
                x = x.requires_grad_(True)
            if not t.requires_grad:
                t = t.requires_grad_(True)

            # Test function and its derivatives
            phi = test_func(x, t)

            phi_x = torch.autograd.grad(
                phi, x, grad_outputs=torch.ones_like(phi),
                create_graph=True, retain_graph=True
            )[0]

            phi_t = torch.autograd.grad(
                phi, t, grad_outputs=torch.ones_like(phi),
                create_graph=True, retain_graph=True
            )[0]

            # Solution and its derivatives
            u = model.forward(x, t)

            u_t = torch.autograd.grad(
                u, t, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]

            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]

            # Get params for PDE
            params = model.get_params()

            # Call PDE's weak form integrand
            return model.pde.weak_form_integrand(u, u_t, u_x, phi, phi_x, phi_t, params)

        return self.integrator.integrate(integrand, domain)

    def compute_loss(
        self,
        model: GeneralizedPINN,
        data: dict,
        lambdas: dict[str, float]
    ) -> tuple[torch.Tensor, dict]:
        """Weak-form loss computation."""

        test_funcs = data['test_funcs']
        test_doms = data['test_doms']

        # 1. Weak-form residuals
        weak_residuals = []
        for phi_func, domain in zip(test_funcs, test_doms):
            try:
                weak_res = self._compute_weak_residual(model, phi_func, domain)
                weak_residuals.append(weak_res)
            except Exception as e:
                print(f"Warning: Integration failed: {e}")
                weak_residuals.append(torch.tensor(0.0, device=self.device))

        weak_residuals_tensor = torch.stack(weak_residuals)
        loss_f = torch.mean(weak_residuals_tensor ** 2)

        # 2. Boundary loss
        u_bc_pred = model.forward(data['x_bc'], data['t_bc'])
        loss_bc = torch.mean((u_bc_pred - data['u_bc']) ** 2)

        # 3. Initial condition loss
        u_ic_pred = model.forward(data['x_ic'], data['t_ic'])
        loss_ic = torch.mean((u_ic_pred - data['u_ic']) ** 2)

        # 4. Initial velocity loss (for wave equation)
        loss_ic_t = torch.tensor(0.0, device=self.device)
        if 'u_ic_t' in data and data['u_ic_t'] is not None:
            x_ic = data['x_ic'].requires_grad_(True)
            t_ic = data['t_ic'].requires_grad_(True)
            derivs = model.compute_derivatives(x_ic, t_ic)
            u_t_pred = derivs['u_t']
            loss_ic_t = torch.mean((u_t_pred - data['u_ic_t']) ** 2)

        # 5. Measurement loss
        if 'x_m' in data and data['x_m'] is not None:
            u_m_pred = model.forward(data['x_m'], data['t_m'])
            loss_m = torch.mean((u_m_pred - data['u_m']) ** 2)
        else:
            loss_m = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = (
            lambdas.get('f', 1.0) * loss_f +
            lambdas.get('bc', 1.0) * loss_bc +
            lambdas.get('ic', 1.0) * loss_ic +
            lambdas.get('ic_t', 1.0) * loss_ic_t +
            lambdas.get('m', 1.0) * loss_m
        )

        # Diagnostics
        nonzero = (weak_residuals_tensor.abs() > 1e-8).sum().item()

        losses = {
            'total': total_loss.item(),
            'residual': loss_f.item(),
            'boundary': loss_bc.item(),
            'initial': loss_ic.item(),
            'initial_velocity': loss_ic_t.item(),
            'measurement': loss_m.item() if torch.is_tensor(loss_m) else 0.0,
            'weak_res_nonzero': nonzero,
            'weak_res_mean': weak_residuals_tensor.mean().item(),
            'weak_res_std': weak_residuals_tensor.std().item(),
            # Tensors for gradient tracking
            'total_t': total_loss,
            'residual_t': loss_f,
            'boundary_t': loss_bc,
            'initial_t': loss_ic,
            'initial_velocity_t': loss_ic_t,
            'measurement_t': loss_m
        }

        return total_loss, losses


class GeneralizedMultiFidelityLoss(GeneralizedLossStrategy):
    """
    Multi-fidelity loss: combines high and low fidelity measurements.

    Required data keys:
    - x_f, t_f: Collocation points
    - x_bc, t_bc, u_bc: Boundary conditions
    - x_ic, t_ic, u_ic: Initial conditions
    - x_hf, t_hf, u_hf, sigma_hf: High-fidelity measurements
    - x_lf, t_lf, u_lf, sigma_lf: Low-fidelity measurements
    """

    def __init__(
        self,
        weighting: str = 'uncertainty',
        lambda_hf: float = 1.0,
        lambda_lf: float = 0.1,
        device: str = 'cpu'
    ):
        self.weighting = weighting
        self.lambda_hf = lambda_hf
        self.lambda_lf = lambda_lf
        self.device = device

        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        print(f"GeneralizedMultiFidelityLoss initialized:")
        print(f"  Weighting: {weighting}")

    def compute_loss(
        self,
        model: GeneralizedPINN,
        data: dict,
        lambdas: dict[str, float]
    ) -> tuple[torch.Tensor, dict]:
        """Multi-fidelity loss computation."""

        # 1. PDE residual loss
        x_f = data['x_f'].requires_grad_(True)
        t_f = data['t_f'].requires_grad_(True)

        residual = model.residual(x_f, t_f)
        loss_f = torch.mean(residual ** 2)

        # 2. Boundary loss
        u_bc_pred = model.forward(data['x_bc'], data['t_bc'])
        loss_bc = torch.mean((u_bc_pred - data['u_bc']) ** 2)

        # 3. Initial condition loss
        u_ic_pred = model.forward(data['x_ic'], data['t_ic'])
        loss_ic = torch.mean((u_ic_pred - data['u_ic']) ** 2)

        # 4. High-fidelity measurement loss
        u_hf_pred = model.forward(data['x_hf'], data['t_hf'])
        residuals_hf = u_hf_pred - data['u_hf']

        # 5. Low-fidelity measurement loss
        u_lf_pred = model.forward(data['x_lf'], data['t_lf'])
        residuals_lf = u_lf_pred - data['u_lf']

        # Compute weighted measurement losses
        if self.weighting == 'uncertainty':
            sigma_hf = data.get('sigma_hf', 0.01)
            sigma_lf = data.get('sigma_lf', 0.1)

            loss_hf = torch.mean(residuals_hf ** 2) / (sigma_hf ** 2)
            loss_lf = torch.mean(residuals_lf ** 2) / (sigma_lf ** 2)

            weight_sum = 1.0 / sigma_hf**2 + 1.0 / sigma_lf**2
            loss_hf = loss_hf / weight_sum
            loss_lf = loss_lf / weight_sum

            effective_lambda_hf = (1.0 / sigma_hf**2) / weight_sum
            effective_lambda_lf = (1.0 / sigma_lf**2) / weight_sum
        else:
            loss_hf = torch.mean(residuals_hf ** 2)
            loss_lf = torch.mean(residuals_lf ** 2)
            effective_lambda_hf = self.lambda_hf
            effective_lambda_lf = self.lambda_lf

        # Combined measurement loss
        if self.weighting == 'fixed':
            loss_m = self.lambda_hf * loss_hf + self.lambda_lf * loss_lf
        else:
            loss_m = loss_hf + loss_lf

        # Total loss
        total_loss = (
            lambdas.get('f', 1.0) * loss_f +
            lambdas.get('bc', 1.0) * loss_bc +
            lambdas.get('ic', 1.0) * loss_ic +
            lambdas.get('m', 1.0) * loss_m
        )

        losses = {
            'total': total_loss.item(),
            'residual': loss_f.item(),
            'boundary': loss_bc.item(),
            'initial': loss_ic.item(),
            'measurement': loss_m.item(),
            'measurement_hf': loss_hf.item(),
            'measurement_lf': loss_lf.item(),
            'effective_lambda_hf': effective_lambda_hf,
            'effective_lambda_lf': effective_lambda_lf,
            'n_hf': data['x_hf'].shape[0],
            'n_lf': data['x_lf'].shape[0],
            # Tensors for gradient tracking
            'total_t': total_loss,
            'residual_t': loss_f,
            'boundary_t': loss_bc,
            'initial_t': loss_ic,
            'measurement_t': loss_m
        }

        return total_loss, losses


class StrategicGeneralizedPINN(GeneralizedPINN):
    """
    GeneralizedPINN with pluggable loss computation strategy.

    Usage:
        pde = HeatEquation1D(params={'alpha': 0.01})
        model = StrategicGeneralizedPINN(pde=pde)
        model.set_loss_strategy(GeneralizedStrongFormLoss())
        loss, losses = model.compute_loss(data)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Default to strong form
        self.loss_strategy = GeneralizedStrongFormLoss(device=self.device)
        print(f"StrategicGeneralizedPINN initialized with {self.loss_strategy.__class__.__name__}")

    def set_loss_strategy(self, strategy: GeneralizedLossStrategy):
        """Set the loss computation strategy."""
        self.loss_strategy = strategy
        print(f"Loss strategy changed to {strategy.__class__.__name__}")

    def compute_loss(
        self,
        data: dict,
        lambda_f: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_ic_t: float = 1.0,
        lambda_m: float = 1.0
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss using current strategy.

        Args:
            data: Dictionary with all required data (format depends on strategy)
            lambda_*: Loss weights

        Returns:
            total_loss: Scalar loss
            losses: Dictionary with components
        """
        lambdas = {
            'f': lambda_f,
            'bc': lambda_bc,
            'ic': lambda_ic,
            'ic_t': lambda_ic_t,
            'm': lambda_m
        }

        return self.loss_strategy.compute_loss(self, data, lambdas)


# ============================================================================
# Example usage and testing
# ============================================================================


if __name__ == "__main__":
    print("Testing Generalized Loss Strategies...")

    from pdes.heat import HeatEquation1D
    from pdes.wave import WaveEquation1D
    from data.pde_data import PDEData

    # Test 1: Strong form with heat equation
    print("\n" + "="*60)
    print("Test 1: Strong Form - Heat Equation")
    print("="*60)

    pde = HeatEquation1D(params={'alpha': 0.01})
    model = StrategicGeneralizedPINN(pde=pde)

    data_gen = PDEData(pde, N_f=1000, N_bc=50, N_ic=50)
    data = data_gen.generate_full_dataset()

    loss, losses = model.compute_loss(data)
    print(f"Total loss: {loss.item():.6e}")
    print(f"Residual: {losses['residual']:.6e}")
    print(f"Boundary: {losses['boundary']:.6e}")
    print(f"Initial: {losses['initial']:.6e}")

    # Test backprop
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Gradient norm: {grad_norm:.4f}")

    # Test 2: Wave equation with IC velocity
    print("\n" + "="*60)
    print("Test 2: Strong Form - Wave Equation")
    print("="*60)

    pde_wave = WaveEquation1D(params={'c': 1.0})
    model_wave = StrategicGeneralizedPINN(pde=pde_wave)

    data_gen_wave = PDEData(pde_wave, N_f=1000, N_bc=50, N_ic=50)
    data_wave = data_gen_wave.generate_full_dataset()

    loss_wave, losses_wave = model_wave.compute_loss(data_wave)
    print(f"Total loss: {loss_wave.item():.6e}")
    print(f"Initial velocity: {losses_wave['initial_velocity']:.6e}")

    # Test 3: Weak form with heat equation
    print("\n" + "="*60)
    print("Test 3: Weak Form - Heat Equation")
    print("="*60)

    from utils.test_functions import generate_compact_gaussians

    pde = HeatEquation1D(params={'alpha': 0.01})
    model = StrategicGeneralizedPINN(pde=pde)
    model.set_loss_strategy(
        GeneralizedWeakFormLoss(
            integration_method='gauss_legendre',
            n_integration_points=10
        )
    )

    test_funcs, test_doms = generate_compact_gaussians(n_funcs=5, support_radius=0.3)

    weak_data = {
        'test_funcs': test_funcs,
        'test_doms': test_doms,
        'x_bc': data['x_bc'],
        't_bc': data['t_bc'],
        'u_bc': data['u_bc'],
        'x_ic': data['x_ic'],
        't_ic': data['t_ic'],
        'u_ic': data['u_ic']
    }

    loss_weak, losses_weak = model.compute_loss(weak_data)
    print(f"Weak form loss: {loss_weak.item():.6e}")
    print(f"Non-zero residuals: {losses_weak['weak_res_nonzero']}/5")

    print("\nGeneralized Loss Strategies tests passed.")
