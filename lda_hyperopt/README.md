# LDA Hyperparameter Optimization

Comparison of optimization algorithms (GA, ES, PABBO Simple, PABBO Full) for LDA hyperparameter tuning.

## Overview

This project optimizes the `T` parameter (number of topics) for Latent Dirichlet Allocation (LDA) by minimizing corpus perplexity on validation data. The `alpha` and `eta` parameters are set to `1/T`.

**Algorithms:**
- **GA** (Genetic Algorithm): Binary crossover, tournament selection, elitism
- **ES** (Evolution Strategy): (μ, λ)-ES with self-adaptive mutation
- **PABBO Simple**: Adaptive random search with exploration/exploitation balance, no Transformer
- **PABBO Full**: Full PABBO implementation with Transformer-based policy learning (requires trained model)

All algorithms use **identical initial population** for fair comparison.

## Project Structure

```
lda_hyperopt/
├── README.md                      # This file
├── run.py                         # Main script
├── utils.py                       # Utilities (logging, LDA training, plotting)
├── generate_init.py               # Generate fixed initial population
├── lda_init_population.json       # Fixed initial population (shared)
├── optimizers/                    # Optimizer implementations
│   ├── __init__.py
│   ├── ga.py                      # Genetic Algorithm
│   ├── es.py                      # Evolution Strategy
│   ├── pabbo_simple.py            # PABBO Simple (no Transformer)
│   └── pabbo_full.py              # PABBO Full (with Transformer)
└── logs/                          # Logs (created automatically)
```

## Installation

### Requirements

- Python 3.7+
- numpy
- scipy
- scikit-learn
- matplotlib
- tensorboardX
- deap

### Install Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib tensorboardX deap
```

## Usage

### 1. Generate Initial Population (Optional)

The repository includes a pre-generated `lda_init_population.json`. To regenerate:

```bash
python generate_init.py
```

### 2. Run Optimization

```bash
python run.py \
  --data /path/to/validation_data.npz \
  --iterations 50 \
  --seed 42 \
  --outdir results \
  --algorithms GA ES PABBO_Simple
```

**Arguments:**
- `--data`: Path to validation data (`.npz` file with sparse BoW matrix)
- `--init`: Path to initial population JSON (default: `lda_init_population.json`)
- `--iterations`: Number of optimization iterations (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--outdir`: Output directory for results (default: `results`)
- `--max-iter`: Max LDA iterations (default: 60)
- `--batch-size`: LDA batch size (default: 2048)
- `--algorithms`: Algorithms to run (choices: GA, ES, PABBO, PABBO_Simple, PABBO_Full; default: GA ES PABBO_Simple)
- `--pabbo-model`: Path to trained Transformer model for PABBO_Full (optional)

### 3. View Results

Results are saved in `outdir/`:
```
results/
├── main.log                       # Main log file
├── overall_summary.json           # Summary of all algorithms
├── comparison.png                 # Comparison plots
├── GA/
│   ├── GA_optimization.log        # GA log
│   ├── summary.json               # GA summary
│   ├── history.csv                # Iteration history
│   ├── optimization_plots.png     # GA plots
│   └── tensorboard/               # TensorBoard logs
├── ES/
│   └── ...                        # Same structure
├── PABBO_Simple/
│   └── ...                        # Same structure
└── PABBO_Full/
    └── ...                        # Same structure (if run)
```

### 4. View TensorBoard (Optional)

```bash
tensorboard --logdir results/GA/tensorboard
```

## Example

```bash
# Run all default algorithms (GA, ES, PABBO_Simple)
python run.py --data data/val_bow.npz --iterations 50 --outdir results

# Run only GA and ES
python run.py --data data/val_bow.npz --algorithms GA ES --outdir results_ga_es

# Run with PABBO Full (requires trained Transformer model)
python run.py --data data/val_bow.npz --algorithms PABBO_Full \
  --pabbo-model ../pabbo_method/checkpoints/best_model.pt --outdir results_pabbo_full

# Compare all PABBO variants
python run.py --data data/val_bow.npz --algorithms PABBO_Simple PABBO_Full \
  --pabbo-model ../pabbo_method/checkpoints/best_model.pt --outdir results_pabbo_compare

# Custom iterations and seed
python run.py --data data/val_bow.npz --iterations 100 --seed 123 --outdir results_custom
```

## Output Explanation

### Summary Table

```
OPTIMIZATION SUMMARY
================================================================================
True minimum: x=-0.005005, y=0.602165
--------------------------------------------------------------------------------
Algorithm    Best T       Best Perplexity      Time (s)
--------------------------------------------------------------------------------
GA           511          0.9305               120.45
ES           500          0.6022               95.23
PABBO        502          0.7823               110.87
--------------------------------------------------------------------------------
Best algorithm: ES (perplexity: 0.6022)
```

### Plots

1. **Perplexity vs Iteration**: Shows convergence speed
2. **Perplexity vs Time**: Shows wall-clock efficiency
3. **T vs Iteration**: Shows parameter evolution
4. **T vs Time**: Shows parameter evolution over time

## Fixed Initial Population

All algorithms start from the **same initial population** to ensure fair comparison:

```json
{
  "T_range": [2, 1000],
  "pop_size": 20,
  "seed": 42,
  "initial_population": [733, 811, 33, 355, ...]
}
```

This is loaded from `lda_init_population.json` and ensures:
- All algorithms start from the same point
- Results are directly comparable
- Reproducibility across runs

## Customization

### Modify Optimization Parameters

Edit `run.py` or pass arguments:

```bash
# More aggressive GA
python run.py --data data/val_bow.npz --iterations 100

# Different LDA settings
python run.py --data data/val_bow.npz --max-iter 200 --batch-size 4096
```

### Add New Optimizer

1. Create new optimizer in `optimizers/new_optimizer.py`
2. Inherit from `BaseOptimizer` in `utils.py`
3. Implement `run()` method
4. Import in `optimizers/__init__.py`
5. Add to `run.py` algorithm choices

## Technical Details

### LDA Training

- **Training**: LDA is trained on **validation set** (for hyperparameter optimization)
- **Evaluation**: Perplexity measured on same validation set
- **Caching**: Evaluations are cached to avoid recomputation
- **Method**: Online learning with mini-batches

### Early Stopping

All algorithms support early stopping:
- Stop if relative improvement < 1% for 5 consecutive iterations
- Prevents wasting time on converged solutions

### OOP Design

```python
BaseOptimizer (abstract base class)
    │
    ├── GAOptimizer
    ├── ESOptimizer
    ├── PABBOSimpleOptimizer
    └── PABBOFullOptimizer
```

All optimizers implement:
- `__init__()`: Setup parameters
- `run()`: Execute optimization
- `decode()`: Convert individual to (T, alpha, eta)

## References

- LDA: Blei et al., "Latent Dirichlet Allocation" (2003)
- Genetic Algorithms: Holland, "Adaptation in Natural and Artificial Systems" (1975)
- Evolution Strategies: Rechenberg, "Evolutionsstrategie" (1973)
- PABBO: Full implementation available in `pabbo_method/`

## License

MIT

## Contact

For questions or issues, please open a GitHub issue.