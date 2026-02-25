# pinn-1d

A generalized Physics-Informed Neural Network framework for benchmarking PINN optimization techniques on 1D time-dependent PDEs. Supports heat, wave, advection, and Burgers equations through a unified `BasePDE` abstraction, with pluggable loss formulations (strong form, weak form, multi-fidelity) and adaptive training infrastructure.

Evolved from [pinn-heat-1d](https://github.com/elphaim/pinn-heat-1d) and generalized with the assistance of Claude Code.

## Motivation

PINN performance is sensitive to the choice of loss formulation, optimizer schedule, and loss weighting — and these sensitivities vary across PDE types. A framework that supports multiple equations under a common interface makes it possible to benchmark these choices systematically, without rewriting the model or training loop for each PDE.

## Supported PDEs

| PDE | Equation | Parameters | Analytical Solution |
|-----|----------|------------|---------------------|
| Heat | u_t = α·u_xx | α (diffusivity) | ✓ |
| Wave | u_tt = c²·u_xx | c (wave speed) | ✓ |
| Advection | u_t + c·u_x = 0 | c (velocity) | ✓ |
| Burgers | u_t + u·u_x = ν·u_xx | ν (viscosity) | ✗ (use FD solver) |

Each PDE implements `BasePDE`, providing: residual computation, weak-form integrand (via integration by parts), default IC/BC, and analytical solution where available. Adding a new PDE requires implementing a single class.

## Architecture

```
├── pdes/                        # PDE definitions (BasePDE interface)
│   ├── base.py                  # Abstract base: residual, IC, BC, analytical solution
│   ├── heat.py                  # u_t = α·u_xx
│   ├── wave.py                  # u_tt = c²·u_xx (temporal_order=2)
│   ├── advection.py             # u_t + c·u_x = 0 (spatial_order=1)
│   └── burgers.py               # u_t + u·u_x = ν·u_xx (nonlinear)
├── models/
│   ├── pinn.py                  # GeneralizedPINN: auto-derivatives based on PDE order
│   └── pinn_strategy.py         # Strategy pattern: Strong / Weak / MultiFidelity loss
├── solvers/
│   └── finite_difference.py     # FTCS, central diff, upwind — auto-selected per PDE
├── data/
│   └── pde_data.py              # Collocation, BC/IC, measurements, multi-fidelity data
├── training/
│   └── trainer_strategy.py      # Adam→L-BFGS switching, adaptive loss weighting
├── utils/
│   ├── integrator.py            # Gauss-Legendre, Simpson, Monte Carlo quadrature
│   └── test_functions.py        # Compact Gaussians for weak form
├── notebooks/
│   └── PINN_benchmarks.ipynb    # Benchmark notebook
├── results/                     # Saved plots (heat, advection, burgers)
├── CLAUDE.md                    # Claude Code project context
├── pyproject.toml
└── requirements.txt
```

## Key Design Decisions

**PDE abstraction.** `BasePDE` declares `spatial_order` and `temporal_order`; `GeneralizedPINN.compute_derivatives()` uses these to decide which derivatives to compute via autograd. This means the wave equation (order 2 in time) automatically gets `u_tt`, while advection (order 1 in space) skips `u_xx`. Adding a PDE does not require touching the model or trainer.

**Named parameter dicts for inverse problems.** Rather than a single scalar `alpha`, parameters are stored as a `dict[str, Tensor]`. For inverse problems, any subset can be declared learnable: `inverse_params=['nu']`. This extends naturally to multi-parameter PDEs.

**Automatic FD solver selection.** `FiniteDifferenceSolver` dispatches to FTCS (heat), central-difference (wave), upwind (advection), or FTCS+upwind (Burgers) based on the PDE name. CFL stability warnings are emitted when the discretization is under-resolved.

**Loss strategy pattern.** Three loss formulations share a common interface and can be swapped at runtime: `GeneralizedStrongFormLoss`, `GeneralizedWeakFormLoss`, `GeneralizedMultiFidelityLoss`. Each handles wave-equation initial velocity (`u_ic_t`) transparently.

## Getting Started

```bash
pip install -e .

# Quick smoke test — each PDE module is self-testing
python pdes/heat.py
python pdes/burgers.py
python models/pinn.py
python solvers/finite_difference.py

# Train a heat PINN (strong vs weak form comparison)
python training/trainer_strategy.py

# Explore the benchmark notebook
jupyter notebook notebooks/PINN_benchmarks.ipynb
```

## Usage

```python
from pdes.heat import HeatEquation1D
from models.pinn_strategy import StrategicGeneralizedPINN, GeneralizedStrongFormLoss
from data.pde_data import PDEData
from training.trainer_strategy import StrategicPINNTrainer

pde = HeatEquation1D(params={'alpha': 0.01})
model = StrategicGeneralizedPINN(pde=pde)
model.set_loss_strategy(GeneralizedStrongFormLoss())

data_gen = PDEData(pde, N_f=10000, N_bc=100, N_ic=100)
data = data_gen.generate_full_dataset()

trainer = StrategicPINNTrainer(model, data)
trainer.train(epochs=5000)
```

For an inverse problem (e.g., learning viscosity in Burgers):

```python
from pdes.burgers import BurgersEquation1D

pde = BurgersEquation1D(params={'nu': 0.01})
model = StrategicGeneralizedPINN(
    pde=pde,
    inverse_params=['nu'],
    param_init={'nu': 0.02}
)
```

## Tech Stack

PyTorch · NumPy · SciPy · Matplotlib

## References

- Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks* (2019)
- Wang, Teng & Perdikaris, *Understanding and mitigating gradient flow pathologies in PINNs* (2021)

## License

MIT