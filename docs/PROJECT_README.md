# Optimization Algorithms Comparison

Comprehensive comparison of optimization algorithms (GA, ES, PABBO) for different problems.

## Overview

This repository contains three independent projects:

1. **`lda_hyperopt/`** - LDA hyperparameter optimization
2. **`test_function_opt/`** - Test function optimization benchmark
3. **`pabbo_method/`** - Full PABBO with Transformer (for GitHub)

All projects use **OOP design** with base `Optimizer` class and share the same optimization algorithms for fair comparison.

## Project Structure

```
Llabs/
├── README.md                      # This file
├── requirements.txt               # Global dependencies
│
├── lda_hyperopt/                  # LDA hyperparameter optimization
│   ├── README.md
│   ├── run.py
│   ├── optimizers/                # GA, ES, PABBO
│   └── lda_init_population.json   # Fixed initialization
│
├── test_function_opt/             # Test function benchmark
│   ├── README.md
│   ├── run.py
│   ├── test_function.py
│   ├── optimizers/                # GA, ES, PABBO
│   └── test_init_population.json  # Fixed initialization
│
└── pabbo_method/                  # Full PABBO (Transformer-based)
    ├── README.md
    ├── train.py
    ├── evaluate_continuous.py
    ├── configs/
    ├── policies/
    └── ...
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Llabs.git
cd Llabs

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

#### 1. LDA Hyperparameter Optimization

```bash
cd lda_hyperopt
python run.py --data /path/to/val_bow.npz --iterations 50 --outdir results
```

See `lda_hyperopt/README.md` for details.

#### 2. Test Function Optimization

```bash
cd test_function_opt
python run.py --iterations 50 --outdir results
```

See `test_function_opt/README.md` for details.

#### 3. PABBO Training

```bash
cd pabbo_method
python train.py --config-name train_rastrigin1d_test
```

See `pabbo_method/README.md` for details.

## Algorithms

All three optimization algorithms use **identical initial population** for fair comparison:

### 1. GA (Genetic Algorithm)

**Features:**
- Binary crossover at bit level
- Integer mutation with bounded random walk
- Tournament selection
- Elitism (best individuals survive)
- Early stopping

**Parameters:**
- `cxpb=0.9` (crossover probability)
- `mutpb=0.2` (mutation probability)
- `elite=3-5` (elite individuals)

### 2. ES (Evolution Strategy)

**Features:**
- (μ, λ)-ES strategy
- Self-adaptive mutation
- Parent selection from best μ individuals
- Gaussian perturbation
- Early stopping

**Parameters:**
- `mu=5` (number of parents)
- `lambda=10` (number of offspring)
- `dT=5` (mutation step size)

### 3. PABBO (Preference-Augmented BBO)

**Two versions:**

#### Simplified PABBO (used in lda_hyperopt and test_function_opt)
- Exploration/exploitation balance
- Temperature-based selection
- Simulated annealing
- Early stopping

**Parameters:**
- `exploration_rate=0.3`
- `temperature_decay=0.95`

#### Full PABBO (pabbo_method/)
- Transformer-based policy
- Preference learning
- Meta-learning across functions
- Requires pretraining

## Key Features

### OOP Design

All optimizers inherit from `BaseOptimizer`:

```python
class BaseOptimizer(ABC):
    def __init__(self, obj, eval_func, T_bounds, seed, ...):
        ...

    @abstractmethod
    def run(self, iterations, writer, outdir, initial_population):
        ...

    def decode(self, individual):
        ...
```

### Fixed Initial Population

Each experiment has a **fixed initial population** saved in JSON:
- `lda_hyperopt/lda_init_population.json`
- `test_function_opt/test_init_population.json`

This ensures:
- **Fair comparison** - all algorithms start from same point
- **Reproducibility** - same results across runs
- **Transparency** - initial conditions are visible

Example:
```json
{
  "T_range": [1, 1000],
  "pop_size": 20,
  "seed": 42,
  "initial_population": [732, 811, 32, 354, ...]
}
```

### Comprehensive Logging

All experiments log to:
- **Console**: Real-time progress
- **Files**: Complete history (`.log` files)
- **CSV**: Iteration-by-iteration data
- **JSON**: Summary and configuration
- **Plots**: Convergence visualization

### Early Stopping

All algorithms implement early stopping:
- Stop if relative improvement < 1% for 3 consecutive iterations
- Prevents wasting time on converged solutions
- Reported in summary (`stopped_early: true/false`)

## Requirements

### Core Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
deap>=1.4.0
tensorboardX>=2.6
```

### For PABBO (full version)

```
torch>=2.0.0
botorch>=0.9.0
gpytorch>=1.11
hydra-core>=1.3.0
```

Install all:

```bash
pip install -r requirements.txt
```

Or install per-project:

```bash
# LDA optimization
pip install numpy scipy scikit-learn matplotlib tensorboardX deap

# Test function optimization
pip install numpy scipy matplotlib deap

# Full PABBO
pip install torch botorch gpytorch hydra-core
```

## Results

### Typical Performance

#### LDA Hyperparameter Optimization (T ∈ [2, 1000])

```
Algorithm    Best T    Best Perplexity    Time (s)
--------------------------------------------------
GA           511       0.9305            120.45
ES           500       0.6022             95.23
PABBO        502       0.7823            110.87
```

**Winner:** ES (lowest perplexity)

#### Test Function Optimization (x ∈ [-5, 5])

```
Algorithm    Best x       Best y       Error      Time (s)
------------------------------------------------------------
GA           -0.095095    0.854995     0.252830   0.12
ES           -0.005005    0.602165     0.000000   0.08
PABBO        0.105105     0.930474     0.328309   0.10
```

**Winner:** ES (lowest error)

### Visualization

Each experiment generates convergence plots with **4 subplots**:
1. Best Value vs Iteration
2. **Best Value vs Time (wall-clock)** ⏱️
3. Best x/T vs Iteration
4. **Best x/T vs Time (wall-clock)** ⏱️

All algorithms start from the same point (same initial population).

## Documentation

Each project has detailed README:

- **`lda_hyperopt/README.md`** - LDA optimization guide
- **`test_function_opt/README.md`** - Test function benchmark guide
- **`pabbo_method/README.md`** - Full PABBO implementation guide

## Development

### Project Principles

1. **OOP Design** - Base class with inheritors
2. **Code Reuse** - Same optimizers for different problems
3. **Fair Comparison** - Identical initialization and conditions
4. **Reproducibility** - Fixed seeds and logged configurations
5. **Transparency** - Complete logging and visualization

### Adding New Optimizer

1. Create optimizer class inheriting from `BaseOptimizer`
2. Implement `run()` method
3. Add to `optimizers/__init__.py`
4. Update `run.py` to include new optimizer
5. Test with fixed initial population

### Adding New Problem

1. Create new directory (e.g., `new_problem/`)
2. Copy optimizer structure from existing project
3. Implement problem-specific objective function
4. Generate fixed initial population
5. Create `README.md` with usage instructions

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llabs2024,
  title={Optimization Algorithms Comparison},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/Llabs}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or issues:
- Open a GitHub issue
- Check individual project READMEs
- Email: your.email@example.com

## Acknowledgments

This project uses:
- **DEAP** - Distributed Evolutionary Algorithms in Python
- **PyTorch** - Deep learning framework
- **BoTorch** - Bayesian Optimization in PyTorch
- **scikit-learn** - Machine learning library
- **Hydra** - Configuration management

## TODO

- [ ] Add more baseline algorithms (CMA-ES, Bayesian Optimization)
- [ ] Add multi-dimensional test functions
- [ ] Integrate full PABBO into lda_hyperopt
- [ ] Add GPU support for large-scale experiments
- [ ] Create Docker container for easy deployment
- [ ] Add more comprehensive benchmarks

## Version History

- **v1.0.0** (2024-11-02) - Initial release
  - Three independent projects
  - OOP design with BaseOptimizer
  - Fixed initial populations
  - Comprehensive logging and visualization

---

**Last Updated:** 2024-11-02