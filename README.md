# LDA Optimization with GA and ES

Optimization of LDA (Latent Dirichlet Allocation) hyperparameters using Genetic Algorithm (GA) and Evolution Strategy (ES).

## Features

- Genetic Algorithm (GA) and Evolution Strategy (ES) optimizers
- TensorBoard integration for real-time metrics visualization
- Comprehensive logging (console + files)
- Train and validation perplexity tracking
- Intelligent parameter initialization (alpha, eta as 1/T)

## Usage

Run the Jupyter notebook `Untitled1 (3).ipynb`

## TensorBoard

View training progress:
```bash
tensorboard --logdir=runs/agnews/ga/tensorboard
```

## Dependencies

Managed via Poetry (see pyproject.toml)