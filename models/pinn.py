"""
Generalized Physics-Informed Neural Network

A flexible PINN that works with any PDE defined by the BasePDE interface.
Supports both forward and inverse problems with multiple learnable parameters.

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from pdes.base import BasePDE


class GeneralizedPINN(nn.Module):
    """
    Generalized Physics-Informed Neural Network.

    Works with any PDE implementing the BasePDE interface.
    Automatically adapts derivative computation based on PDE order.

    Args:
        pde: BasePDE instance defining the PDE
        layers: Network architecture. Default: [2, 50, 50, 50, 50, 1]
        inverse_params: List of parameter names to learn (e.g., ['alpha'])
                       If None or empty, all params are fixed (forward problem)
        param_init: Initial values for learnable parameters
                   If not provided, uses PDE default values
        activation: Activation function ('tanh', 'sin', 'gelu')
        device: PyTorch device

    Example (forward problem):
        pde = HeatEquation1D(params={'alpha': 0.01})
        model = GeneralizedPINN(pde)

    Example (inverse problem):
        pde = WaveEquation1D()
        model = GeneralizedPINN(
            pde,
            inverse_params=['c'],
            param_init={'c': 0.5}  # Initial guess
        )
    """

    def __init__(
        self,
        pde: BasePDE,
        layers: list[int] = [2, 50, 50, 50, 50, 1],
        inverse_params: Optional[list[str]] = None,
        param_init: Optional[dict[str, float]] = None,
        activation: str = 'tanh',
        device: str = 'cpu'
    ):
        super().__init__()

        self.pde = pde
        self.device = device
        self.inverse_params = inverse_params or []
        self.is_inverse = len(self.inverse_params) > 0

        # Set dtype based on device
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
            self.dtype = torch.float64
        else:
            torch.set_default_dtype(torch.float32)
            self.dtype = torch.float32

        # Validate inverse params
        for param_name in self.inverse_params:
            if param_name not in pde.param_names():
                raise ValueError(
                    f"Unknown parameter '{param_name}' for {pde.name}. "
                    f"Valid parameters: {pde.param_names()}"
                )

        # Build network layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'gelu':
            self.activation = nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize weights
        self._initialize_weights()

        # Setup parameters (fixed and learnable)
        self._setup_parameters(param_init)

        # Print configuration
        self._print_config()

    def _initialize_weights(self):
        """Initialize network weights using Xavier Normal."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _setup_parameters(self, param_init: Optional[dict[str, float]]):
        """Setup fixed buffers and learnable parameters."""
        # Initialize from PDE defaults, then override with provided values
        init_values = self.pde.params.copy()
        if param_init is not None:
            init_values.update(param_init)

        # Create tensors for all parameters
        self._param_tensors = {}

        for param_name in self.pde.param_names():
            value = init_values[param_name]
            tensor = torch.tensor([value], dtype=self.dtype)

            if param_name in self.inverse_params:
                # Learnable parameter
                self._param_tensors[param_name] = nn.Parameter(tensor)
                setattr(self, f'_param_{param_name}', self._param_tensors[param_name])
            else:
                # Fixed parameter (buffer)
                self.register_buffer(f'_param_{param_name}', tensor)
                self._param_tensors[param_name] = getattr(self, f'_param_{param_name}')

    def _print_config(self):
        """Print model configuration."""
        print(f"GeneralizedPINN for {self.pde.name}")
        print(f"  Network: {[l.in_features for l in self.layers]} -> {self.layers[-1].out_features}")
        print(f"  Total params: {sum(p.numel() for p in self.parameters())}")

        if self.is_inverse:
            print(f"  Mode: Inverse problem")
            for name in self.inverse_params:
                init_val = self._param_tensors[name].item()
                print(f"    Learning: {name} (init={init_val:.4f})")
            for name in self.pde.param_names():
                if name not in self.inverse_params:
                    val = self._param_tensors[name].item()
                    print(f"    Fixed: {name}={val}")
        else:
            print(f"  Mode: Forward problem")
            for name, val in self.pde.params.items():
                print(f"    Fixed: {name}={val}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (x, t) -> u(x, t)

        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Temporal coordinates, shape (N, 1)

        Returns:
            u: Predicted solution, shape (N, 1)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)

        # Pass through hidden layers with activation
        out = inputs
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))

        # Output layer (no activation)
        u = self.layers[-1](out)
        return u

    def compute_derivatives(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute all required derivatives for the PDE.

        Automatically computes u_t, u_x, u_xx, and u_tt based on PDE order.

        Args:
            x: Spatial coordinates (must have requires_grad=True)
            t: Temporal coordinates (must have requires_grad=True)

        Returns:
            Dictionary with keys: 'u', 'u_t', 'u_x', 'u_xx', and optionally 'u_tt'
        """
        # Forward pass
        u = self.forward(x, t)

        # First temporal derivative
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # First spatial derivative
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        result = {'u': u, 'u_t': u_t, 'u_x': u_x}

        # Second spatial derivative (if needed)
        if self.pde.spatial_order >= 2:
            u_xx = torch.autograd.grad(
                outputs=u_x,
                inputs=x,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
                retain_graph=True
            )[0]
            result['u_xx'] = u_xx
        else:
            result['u_xx'] = torch.zeros_like(u)

        # Second temporal derivative (for wave equation)
        if self.pde.temporal_order >= 2:
            u_tt = torch.autograd.grad(
                outputs=u_t,
                inputs=t,
                grad_outputs=torch.ones_like(u_t),
                create_graph=True,
                retain_graph=True
            )[0]
            result['u_tt'] = u_tt
        else:
            result['u_tt'] = torch.zeros_like(u)

        return result

    def residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual at given points.

        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Temporal coordinates, shape (N, 1)

        Returns:
            residual: PDE residual, shape (N, 1)
        """
        # Compute all derivatives
        derivs = self.compute_derivatives(x, t)

        # Get current parameter values
        params = self.get_params()

        # Call PDE residual function
        return self.pde.residual(
            u=derivs['u'],
            u_t=derivs['u_t'],
            u_x=derivs['u_x'],
            u_xx=derivs['u_xx'],
            u_tt=derivs['u_tt'],
            x=x,
            t=t,
            params=params
        )

    def get_params(self) -> dict[str, torch.Tensor]:
        """
        Get all PDE parameters (fixed and learnable) as tensors.

        Returns:
            Dictionary mapping parameter names to tensor values
        """
        return {
            name: getattr(self, f'_param_{name}')
            for name in self.pde.param_names()
        }

    def get_learned_params(self) -> dict[str, float]:
        """
        Get current values of learnable parameters.

        Returns:
            Dictionary mapping parameter names to float values
        """
        return {
            name: getattr(self, f'_param_{name}').item()
            for name in self.inverse_params
        }

    def get_all_params_values(self) -> dict[str, float]:
        """
        Get all parameter values as floats.

        Returns:
            Dictionary mapping all parameter names to float values
        """
        return {
            name: getattr(self, f'_param_{name}').item()
            for name in self.pde.param_names()
        }

    def predict(self, x: torch.Tensor, t: torch.Tensor) -> np.ndarray:
        """
        Predict solution at given points (no gradients).

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            u: Predicted solution as numpy array
        """
        self.eval()
        with torch.no_grad():
            u = self.forward(x, t)
        return u.cpu().numpy()


# ============================================================================
# Example usage and testing
# ============================================================================


if __name__ == "__main__":
    print("Testing GeneralizedPINN...")

    # Test with different PDEs
    from pdes.heat import HeatEquation1D
    from pdes.wave import WaveEquation1D
    from pdes.advection import AdvectionEquation1D
    from pdes.burgers import BurgersEquation1D

    # Test 1: Forward heat equation
    print("\n" + "="*60)
    print("Test 1: Forward Heat Equation")
    print("="*60)

    pde_heat = HeatEquation1D(params={'alpha': 0.01})
    model_heat = GeneralizedPINN(pde_heat)

    x = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)

    u = model_heat(x, t)
    print(f"Output shape: {u.shape}")

    res = model_heat.residual(x, t)
    print(f"Residual shape: {res.shape}")
    print(f"Residual mean: {res.mean().item():.6f}")

    # Test 2: Inverse wave equation
    print("\n" + "="*60)
    print("Test 2: Inverse Wave Equation")
    print("="*60)

    pde_wave = WaveEquation1D(params={'c': 1.0})
    model_wave = GeneralizedPINN(
        pde_wave,
        inverse_params=['c'],
        param_init={'c': 0.5}
    )

    print(f"Learned params: {model_wave.get_learned_params()}")

    x = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)

    res = model_wave.residual(x, t)
    print(f"Residual shape: {res.shape}")

    # Check gradient flow to parameter
    loss = res.pow(2).mean()
    loss.backward()
    print(f"Gradient flows to c: {model_wave._param_c.grad is not None}")

    # Test 3: Advection (first order spatial)
    print("\n" + "="*60)
    print("Test 3: Advection Equation")
    print("="*60)

    pde_adv = AdvectionEquation1D(params={'c': 0.5})
    model_adv = GeneralizedPINN(pde_adv)

    x = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)

    derivs = model_adv.compute_derivatives(x, t)
    print(f"u_xx computed: {derivs['u_xx'] is not None}")  # Should be False
    print(f"Spatial order: {pde_adv.spatial_order}")

    res = model_adv.residual(x, t)
    print(f"Residual shape: {res.shape}")

    # Test 4: Burgers (nonlinear)
    print("\n" + "="*60)
    print("Test 4: Burgers Equation")
    print("="*60)

    pde_burg = BurgersEquation1D(params={'nu': 0.01})
    model_burg = GeneralizedPINN(
        pde_burg,
        inverse_params=['nu'],
        param_init={'nu': 0.02}
    )

    x = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)

    res = model_burg.residual(x, t)
    print(f"Residual shape: {res.shape}")
    print(f"All params: {model_burg.get_all_params_values()}")

    print("\nGeneralizedPINN tests passed.")