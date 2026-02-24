"""
Abstract Base Class for 1D PDEs

Defines the interface that all PDE classes must implement for use
with GeneralizedPINN.

Author: elphaim with Claude Code
Date: February 2026
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BasePDE(ABC):
    """
    Abstract base class for 1D time-dependent PDEs.

    All PDEs operate on domain x ∈ [0, L], t ∈ [0, T] and define:
    - PDE residual for strong form
    - Weak form integrand (optional)
    - Initial and boundary conditions
    - Analytical solution (if available)

    Subclasses must implement:
    - param_names(): List of parameter names (e.g., ['alpha'] for heat)
    - default_params(): Default parameter values
    - residual(): PDE residual computation
    - default_ic(): Initial condition u(x, 0)
    - default_bc(): Boundary conditions u(0, t), u(L, t)

    Attributes:
        name: Human-readable PDE name
        spatial_order: Highest spatial derivative order
        temporal_order: Highest temporal derivative order
        params: Dictionary of PDE parameters
        domain: Dictionary with 'L' (spatial length) and 'T' (final time)
        device: PyTorch device ('cpu' or 'cuda')
    """

    name: str = "BasePDE"
    spatial_order: int = 2
    temporal_order: int = 1

    def __init__(
        self,
        params: Optional[dict[str, float]] = None,
        domain: Optional[dict[str, float]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize PDE with parameters and domain.

        Args:
            params: Dictionary of PDE parameters. Missing keys filled with defaults.
            domain: Dictionary with 'L' (spatial) and 'T' (temporal) extent.
                    Default: {'L': 1.0, 'T': 1.0}
            device: PyTorch device
        """
        self.device = device

        # Set default dtype
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
            self.dtype = torch.float64
        else:
            torch.set_default_dtype(torch.float32)
            self.dtype = torch.float32

        # Initialize parameters with defaults, then update with provided values
        self.params = self.default_params().copy()
        if params is not None:
            for key in params:
                if key not in self.param_names():
                    raise ValueError(f"Unknown parameter '{key}' for {self.name}. "
                                   f"Valid parameters: {self.param_names()}")
            self.params.update(params)

        # Initialize domain
        self.domain = {'L': 1.0, 'T': 1.0}
        if domain is not None:
            self.domain.update(domain)

        self.L = self.domain['L']
        self.T = self.domain['T']

    @classmethod
    @abstractmethod
    def param_names(cls) -> list[str]:
        """
        Return list of parameter names for this PDE.

        Example: ['alpha'] for heat equation, ['c'] for wave equation.
        """
        pass

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict[str, float]:
        """
        Return default parameter values.

        Example: {'alpha': 0.01} for heat equation.
        """
        pass

    @abstractmethod
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
        Compute PDE residual from derivatives.

        For a PDE F(u, u_t, u_x, u_xx, ...) = 0, returns F.

        Args:
            u: Solution values, shape (N, 1)
            u_t: Time derivative, shape (N, 1)
            u_x: Spatial first derivative, shape (N, 1)
            u_xx: Spatial second derivative, shape (N, 1)
            u_tt: Temporal second derivative (for wave equation), shape (N, 1)
            x: Spatial coordinates (for spatially varying coefficients)
            t: Temporal coordinates
            params: Override parameters (useful for learnable params in inverse problems)

        Returns:
            residual: PDE residual, shape (N, 1). Should be zero for exact solution.
        """
        pass

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
        Compute weak form integrand for variational formulation.

        Weak form: ∫∫ [weak_form_integrand] dx dt = 0

        Default implementation raises NotImplementedError.
        Override in subclasses that support weak form.

        Args:
            u: Solution values
            u_x: Spatial derivative of solution
            phi: Test function values
            phi_x: Spatial derivative of test function
            phi_t: Temporal derivative of test function
            params: Override parameters

        Returns:
            integrand: Weak form integrand values
        """
        raise NotImplementedError(
            f"Weak form not implemented for {self.name}. "
            "Use strong form loss strategy instead."
        )

    def analytical_solution(
        self,
        x: torch.Tensor | np.ndarray,
        t: torch.Tensor | np.ndarray
    ) -> Optional[torch.Tensor | np.ndarray]:
        """
        Compute analytical solution if available.

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            u: Solution values, or None if no analytical solution exists
        """
        return None

    @abstractmethod
    def default_ic(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Compute default initial condition u(x, 0).

        Args:
            x: Spatial coordinates

        Returns:
            u_ic: Initial condition values
        """
        pass

    def default_ic_t(self, x: torch.Tensor | np.ndarray) -> Optional[torch.Tensor | np.ndarray]:
        """
        Compute initial velocity u_t(x, 0) for hyperbolic PDEs.

        Only needed for PDEs with temporal_order >= 2 (e.g., wave equation).

        Args:
            x: Spatial coordinates

        Returns:
            u_t_ic: Initial velocity values, or None if not applicable
        """
        return None

    @abstractmethod
    def default_bc(
        self,
        t: torch.Tensor | np.ndarray,
        boundary: str
    ) -> torch.Tensor | np.ndarray:
        """
        Compute default boundary condition.

        Args:
            t: Temporal coordinates
            boundary: 'left' (x=0) or 'right' (x=L)

        Returns:
            u_bc: Boundary condition values
        """
        pass

    def get_param(self, name: str) -> float:
        """Get parameter value by name."""
        if name not in self.params:
            raise ValueError(f"Unknown parameter '{name}'. Valid: {list(self.params.keys())}")
        return self.params[name]

    def set_param(self, name: str, value: float):
        """Set parameter value by name."""
        if name not in self.param_names():
            raise ValueError(f"Unknown parameter '{name}'. Valid: {self.param_names()}")
        self.params[name] = value

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str}, L={self.L}, T={self.T})"
