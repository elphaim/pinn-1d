# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics-Informed Neural Networks (PINNs) for 1D time-dependent PDEs. The project supports multiple PDE types (Heat, Wave, Advection, Burgers) with a unified architecture and explores PINN optimization techniques including strong/weak form losses, multi-fidelity learning, adaptive loss weighting, and Adam-to-L-BFGS optimizer switching.

## Commands

```bash
# Install dependencies
pip install -e .

# Run example scripts directly
python models/heat_pinn_strategy.py    # Test heat-specific loss strategies
python models/pinn_strategy.py         # Test generalized loss strategies
python training/trainer_strategy.py    # Train comparison (strong vs weak)
python data/heat_data.py               # Test heat data generation
python data/pde_data.py                # Test generalized data generation

# Test PDE definitions
python pdes/heat.py
python pdes/wave.py
python pdes/advection.py
python pdes/burgers.py

# Test FD solvers
python solvers/finite_difference.py

# Experiments
python experiments/multi_fidelity_comparison.py
```

## Architecture

### PDE Definitions (`pdes/`)

**pdes/base.py** - `BasePDE` abstract class:
- Defines interface for all PDEs: `residual()`, `default_ic()`, `default_bc()`, `analytical_solution()`
- Named parameters via `param_names()` and `default_params()`
- `spatial_order` and `temporal_order` for derivative computation

**pdes/heat.py** - `HeatEquation1D`: u_t = α·u_xx
**pdes/wave.py** - `WaveEquation1D`: u_tt = c²·u_xx
**pdes/advection.py** - `AdvectionEquation1D`: u_t + c·u_x = 0
**pdes/burgers.py** - `BurgersEquation1D`: u_t + u·u_x = ν·u_xx

### Numerical Solvers (`solvers/`)

**solvers/finite_difference.py** - `FiniteDifferenceSolver`:
- FTCS for heat, central difference for wave, upwind for advection/Burgers
- Provides reference solutions for PDEs without analytical solutions
- Interpolation for arbitrary query points

### Generalized PINN (`models/`)

**models/pinn.py** - `GeneralizedPINN`:
- Works with any `BasePDE` instance
- Automatic derivative computation based on PDE order
- Named parameter dict for inverse problems (e.g., `inverse_params=['alpha', 'nu']`)

**models/pinn_strategy.py** - Generalized loss strategies:
- `GeneralizedStrongFormLoss`, `GeneralizedWeakFormLoss`, `GeneralizedMultiFidelityLoss`
- `StrategicGeneralizedPINN` wraps GeneralizedPINN with pluggable loss strategy
- Supports initial velocity loss (`u_ic_t`) for wave equation

### Legacy Heat-Specific (`models/`)

**models/heat_pinn.py** - Base `HeatPINN` class (backward compatible)
**models/heat_pinn_strategy.py** - `StrategicPINN` for heat equation only

### Training (`training/`)

**training/trainer_strategy.py** - `StrategicPINNTrainer`:
- Works with both legacy and generalized PINNs
- Two-phase optimization: Adam → L-BFGS
- Tracks multiple learned parameters for inverse problems
- Adaptive loss weighting using gradient norms

### Data Generation (`data/`)

**data/pde_data.py** - `PDEData`:
- Generates data for any PDE using its `default_ic()`, `default_bc()`, `analytical_solution()`
- Automatic fallback to numerical solver when no analytical solution
- `generate_multi_fidelity_data()` for HF/LF datasets

**data/heat_data.py** - `HeatEquationData` (legacy, backward compatible)

### Utilities (`utils/`)

- `utils/integrator.py` - Numerical integration (Gauss-Legendre, Monte Carlo, Simpson)
- `utils/test_functions.py` - Compact Gaussian test functions for weak form

## Supported PDEs

| PDE | Equation | Parameters | Has Analytical |
|-----|----------|------------|----------------|
| `HeatEquation1D` | u_t = α·u_xx | alpha | Yes |
| `WaveEquation1D` | u_tt = c²·u_xx | c | Yes |
| `AdvectionEquation1D` | u_t + c·u_x = 0 | c | Yes |
| `BurgersEquation1D` | u_t + u·u_x = ν·u_xx | nu | No (use FD) |

## Usage Patterns

### Forward Problem (Generalized)

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

### Inverse Problem (Multiple Parameters)

```python
from pdes.burgers import BurgersEquation1D
from models.pinn_strategy import StrategicGeneralizedPINN

pde = BurgersEquation1D(params={'nu': 0.01})
model = StrategicGeneralizedPINN(
    pde=pde,
    inverse_params=['nu'],
    param_init={'nu': 0.02}  # Initial guess
)
```

### Wave Equation (with Initial Velocity)

```python
from pdes.wave import WaveEquation1D
from data.pde_data import PDEData

pde = WaveEquation1D(params={'c': 1.0})
data_gen = PDEData(pde)
data = data_gen.generate_full_dataset()

# data['u_ic_t'] contains initial velocity u_t(x, 0)
```

### Using FD Solver for Reference

```python
from pdes.burgers import BurgersEquation1D
from solvers.finite_difference import FiniteDifferenceSolver

pde = BurgersEquation1D(params={'nu': 0.01})
solver = FiniteDifferenceSolver(pde, nx=201, nt=2001)
x, t, u = solver.solve()
```

### Legacy Heat Equation (Backward Compatible)

```python
from data.heat_data import HeatEquationData
from models.heat_pinn_strategy import StrategicPINN, StrongFormLoss
from training.trainer_strategy import StrategicPINNTrainer

data_gen = HeatEquationData(alpha=0.01, N_f=10000)
data = data_gen.generate_full_dataset()

model = StrategicPINN(alpha_true=0.01, inverse=False)
model.set_loss_strategy(StrongFormLoss())

trainer = StrategicPINNTrainer(model, data, adaptive_weights=True)
trainer.train(epochs=5000)
```

## Data Dictionary Keys

### Strong Form (Generalized)
`x_f`, `t_f`, `x_bc`, `t_bc`, `u_bc`, `x_ic`, `t_ic`, `u_ic`, `u_ic_t` (wave only), `x_m`, `t_m`, `u_m`

### Weak Form
`test_funcs`, `test_doms` (plus BC/IC/measurement)

### Multi-Fidelity
`x_hf`, `t_hf`, `u_hf`, `sigma_hf`, `x_lf`, `t_lf`, `u_lf`, `sigma_lf` (plus collocation, BC/IC)

## Conventions

- Tensor dtype: float64 on CPU, float32 on CUDA
- All coordinate tensors have shape (N, 1)
- Loss components: `loss_f` (PDE), `loss_bc` (boundary), `loss_ic` (initial), `loss_ic_t` (initial velocity), `loss_m` (measurement)
- Type hints use Python 3.10+ syntax (`dict[str, float]` not `Dict`)
- PDE parameters are named dicts (e.g., `{'alpha': 0.01}` not positional)
