# Test Function Optimization

Benchmark comparison of optimization algorithms (GA, ES, PABBO) on a complex non-differentiable test function.

## Overview

This project tests three optimization algorithms on a challenging mathematical function with:
- **Non-differentiable** points (absolute values)
- **Discontinuities** (step function)
- **Multiple local minima** (>5)
- **One global minimum** around x ≈ 0

**Algorithms:**
- **GA** (Genetic Algorithm): Binary crossover, tournament selection, elitism
- **ES** (Evolution Strategy): (μ, λ)-ES with self-adaptive mutation
- **PABBO** (Preference-Augmented BBO): Simplified version with exploration/exploitation balance

All algorithms use **identical initial population** for fair comparison.

## Project Structure

```
test_function_opt/
├── README.md                      # This file
├── run.py                         # Main script
├── test_function.py               # Test function definition
├── utils.py                       # Utilities (logging, plotting)
├── generate_init.py               # Generate fixed initial population
├── test_init_population.json      # Fixed initial population (shared)
├── optimizers/                    # Optimizer implementations
│   ├── __init__.py
│   ├── ga.py                      # Genetic Algorithm
│   ├── es.py                      # Evolution Strategy
│   └── pabbo.py                   # PABBO (simplified)
└── logs/                          # Logs (created automatically)
```

## Installation

### Requirements

- Python 3.7+
- numpy
- scipy
- matplotlib
- deap

### Install Dependencies

```bash
pip install numpy scipy matplotlib deap
```

## Test Function

The test function is defined as:

```python
y(x) = |sin(2πx)| + 0.5·|cos(5πx)| + 0.2·|sin(10πx)|
       + 0.3·x² + 2·|x| + step(x) + 0.1·|sin(20πx)|
```

Where `step(x)` is a discontinuous step function:
- `x < -3`: +5
- `-3 ≤ x < -1`: +3
- `-1 ≤ x < 1`: +0 (best region)
- `1 ≤ x < 3`: +2
- `x ≥ 3`: +4

**Domain:** x ∈ [-5, 5]

**Global minimum:** x ≈ 0, y ≈ 0.6

### Visualize the Function

```bash
python test_function.py
```

This generates `test_function.png` showing the function landscape and approximate minimum.

## Usage

### 1. Generate Initial Population (Optional)

The repository includes a pre-generated `test_init_population.json`. To regenerate:

```bash
python generate_init.py
```

### 2. Run Optimization

```bash
python run.py \
  --x-range -5.0 5.0 \
  --T-range 1 1000 \
  --iterations 50 \
  --seed 42 \
  --outdir results \
  --algorithms GA ES PABBO
```

**Arguments:**
- `--x-range`: Range for continuous x domain (default: -5.0 5.0)
- `--T-range`: Range for integer T domain (default: 1 1000)
- `--init`: Path to initial population JSON (default: `test_init_population.json`)
- `--iterations`: Number of optimization iterations (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--outdir`: Output directory for results (default: `results`)
- `--algorithms`: Algorithms to run (choices: GA, ES, PABBO; default: all)

### 3. View Results

Results are saved in `outdir/`:
```
results/
├── main.log                       # Main log file
├── overall_summary.json           # Summary of all algorithms
├── test_function.png              # Function visualization
├── convergence.png                # 4-plot convergence comparison
├── GA/
│   ├── GA_test.log                # GA log
│   ├── summary.json               # GA summary
│   └── history.csv                # Iteration history
├── ES/
│   └── ...                        # Same structure
└── PABBO/
    └── ...                        # Same structure
```

## Example

```bash
# Run all algorithms
python run.py --iterations 50 --outdir results

# Run only GA and ES with more iterations
python run.py --algorithms GA ES --iterations 100 --outdir results_ga_es

# Custom domain and seed
python run.py --x-range -3.0 3.0 --seed 123 --outdir results_custom
```

## Output Explanation

### Summary Table

```
OPTIMIZATION SUMMARY
================================================================================
True minimum: x=-0.005005, y=0.602165
--------------------------------------------------------------------------------
Algorithm    Best x       Best y       Error        Time (s)
--------------------------------------------------------------------------------
GA           -0.095095    0.854995     0.252830     0.12
ES           -0.005005    0.602165     0.000000     0.08
PABBO        0.105105     0.930474     0.328309     0.10
--------------------------------------------------------------------------------

Best algorithm: ES
   Found: x=-0.005005, y=0.602165
   Error: 0.000000
```

**Interpretation:**
- **Best x/y**: Found solution
- **Error**: |found_y - true_min_y|
- **Time**: Wall-clock time in seconds
- **Best algorithm**: Lowest error

### Convergence Plots

`convergence.png` contains 4 subplots:
1. **Best Value vs Iteration**: Shows optimization progress
2. **Best Value vs Time**: Shows wall-clock efficiency
3. **Best x vs Iteration**: Shows parameter evolution
4. **Best x vs Time**: Shows parameter evolution over time

All three algorithms should start from the same point (same initial population).

## Fixed Initial Population

All algorithms start from the **same initial population** to ensure fair comparison:

```json
{
  "T_range": [1, 1000],
  "pop_size": 20,
  "seed": 42,
  "initial_population": [732, 811, 32, 354, ...]
}
```

This ensures:
- Fair comparison between algorithms
- Reproducible results
- Same starting conditions

**Integer-to-Real Mapping:**

The optimizers work with integers `T ∈ [T_min, T_max]`, which are linearly mapped to continuous `x ∈ [x_min, x_max]`:

```
x = x_min + (T - T_min) · (x_max - x_min) / (T_max - T_min)
```

## Customization

### Modify Test Function

Edit `test_function.py`:

```python
def test_function(x: float) -> float:
    # Your custom function here
    return y
```

### Change Optimization Parameters

```bash
# More iterations
python run.py --iterations 100

# Narrower search range
python run.py --x-range -2.0 2.0

# Different population size (regenerate init first)
# Edit generate_init.py, change pop_size
python generate_init.py
python run.py --init test_init_population.json
```

### Add New Optimizer

1. Create new optimizer in `optimizers/new_optimizer.py`
2. Inherit from `BaseOptimizer` in `utils.py`
3. Implement `run()` method
4. Import in `optimizers/__init__.py`
5. Add to `run.py` algorithm choices

## Technical Details

### Integer-to-Real Adapter

Since the optimizers are designed for LDA (integer T parameter), we use an adapter:

```python
class IntegerToRealAdapter:
    def T_to_x(self, T: int) -> float:
        # Map T → x

    def x_to_T(self, x: float) -> int:
        # Map x → T
```

This allows using the same optimizer code for both LDA and continuous optimization.

### Early Stopping

All algorithms support early stopping:
- Stop if relative improvement < 1% for 3 consecutive iterations
- Prevents wasting time on converged solutions

### Logging

All optimization steps are logged to:
- **Console**: Real-time progress
- **Files**: Complete history (`{algorithm}_test.log`)

Each log contains:
- Configuration
- Initial population (verifiable)
- Iteration progress
- Final results

### OOP Design

```python
BaseOptimizer (abstract base class)
    │
    ├── GAOptimizer
    ├── ESOptimizer
    └── PABBOOptimizer
```

All optimizers implement:
- `__init__()`: Setup parameters
- `run()`: Execute optimization
- `decode()`: Convert individual to (T, alpha, eta)

## Verification

### Check Same Initial Population

```bash
grep "Initial population" results/GA/GA_test.log
grep "Initial population" results/ES/ES_test.log
grep "Initial population" results/PABBO/PABBO_test.log
```

All three should show identical populations.

### Check Convergence

Open `results/convergence.png`:
- All three lines should start from approximately the same y value
- Lines diverge as algorithms optimize differently
- Best algorithm should converge closest to true minimum

## References

- Genetic Algorithms: Holland, "Adaptation in Natural and Artificial Systems" (1975)
- Evolution Strategies: Rechenberg, "Evolutionsstrategie" (1973)
- PABBO: Full implementation available in `pabbo_method/`

## License

MIT

## Contact

For questions or issues, please open a GitHub issue.