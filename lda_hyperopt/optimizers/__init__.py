"""
Optimizers package for LDA hyperparameter optimization.

Contains:
- ga: Genetic Algorithm
- es: Evolution Strategy
- pabbo_simple: PABBO Simple (adaptive sampling, no Transformer)
- pabbo_full: PABBO Full (with Transformer, requires trained model)
- sabo: SABO (Stochastic Adaptive Bayesian Optimization)
"""

from .ga import GAOptimizer
from .es import ESOptimizer
from .pabbo_simple import PABBOSimpleOptimizer
from .pabbo_full import PABBOFullOptimizer
from .sabo import SABOOptimizer

# For backward compatibility
PABBOOptimizer = PABBOSimpleOptimizer

__all__ = [
    'GAOptimizer',
    'ESOptimizer',
    'PABBOSimpleOptimizer',
    'PABBOFullOptimizer',
    'SABOOptimizer',
    'PABBOOptimizer'  # Alias for backward compatibility
]