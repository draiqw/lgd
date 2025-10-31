"""
LDA Hyperparameter Optimization Experiments

This package contains implementations of three optimization algorithms
for finding optimal LDA hyperparameters:
- Genetic Algorithm (GA)
- Evolution Strategy (ES)
- PABBO-inspired Random Search

All algorithms optimize T (number of topics) with alpha=1/T and eta=1/T,
training on validation data to minimize perplexity.
"""

__version__ = "1.0.0"
__author__ = "LDA Optimization Team"

from .exp_ga import GAOptimizer
from .exp_es import ESOptimizer
from .exp_pabbo import PABBOOptimizer
from .utils import (
    load_bow_data,
    make_objective,
    make_eval_func,
    setup_logger,
    plot_optimization_results
)

__all__ = [
    "GAOptimizer",
    "ESOptimizer",
    "PABBOOptimizer",
    "load_bow_data",
    "make_objective",
    "make_eval_func",
    "setup_logger",
    "plot_optimization_results"
]