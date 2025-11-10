"""
Quick test for SABO optimizer with dummy objective function.

This test verifies that SABO can optimize a simple quadratic function
without requiring actual LDA evaluation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lda_hyperopt.optimizers import SABOOptimizer
from lda_hyperopt.utils import setup_logger


def dummy_objective(T, alpha, eta):
    """
    Simple quadratic objective with minimum at T=50.
    
    Returns:
        Fitness value (perplexity-like)
    """
    return (T - 50) ** 2 + 100


def dummy_eval_func(T, alpha, eta):
    """
    Evaluation function that returns full metrics.
    
    Returns:
        Dictionary with metrics
    """
    perplexity = dummy_objective(T, alpha, eta)
    return {
        "T": T,
        "alpha": alpha,
        "eta": eta,
        "perplexity": perplexity,
        "fit_time": 0.01,
        "eval_time": 0.001,
        "n_iter_lda": 10
    }


def test_sabo_basic():
    """Test basic SABO functionality."""
    print("=" * 80)
    print("Testing SABO with dummy quadratic objective: f(T) = (T-50)² + 100")
    print("Expected minimum: T=50, f(50)=100")
    print("=" * 80)
    
    logger = setup_logger('sabo_test')
    
    # Create optimizer
    optimizer = SABOOptimizer(
        obj=dummy_objective,
        eval_func=dummy_eval_func,
        T_bounds=(2, 100),
        seed=42,
        rho=0.15,
        beta_t=0.05,
        delta_Sigma=0.5,
        delta_mu=0.5,
        N_batch=8,
        early_stop_eps_pct=0.001,
        max_no_improvement=10,
        logger=logger
    )
    
    # Run optimization
    result = optimizer.run(
        iterations=30,
        n_initial=5
    )
    
    # Check results
    best_T = result['best']['T']
    best_perplexity = result['best']['perplexity']
    
    print("\n" + "=" * 80)
    print("SABO Test Results:")
    print("=" * 80)
    print(f"Best T found: {best_T} (expected: 50)")
    print(f"Best perplexity: {best_perplexity:.4f} (expected: 100.0)")
    print(f"Final μ: {result['final_mu']:.4f}")
    print(f"Final Σ: {result['final_Sigma']:.4f}")
    print(f"Total evaluations: {result['total_evaluations']}")
    print(f"Total time: {result['total_time']:.2f}s")
    print("=" * 80)
    
    # Verify convergence
    error = abs(best_T - 50)
    if error <= 5:
        print(f"✓ TEST PASSED: Found T={best_T}, error={error} (within tolerance)")
        return True
    else:
        print(f"✗ TEST FAILED: Found T={best_T}, error={error} (expected ≤5)")
        return False


def test_sabo_convergence():
    """Test SABO convergence behavior."""
    print("\n" + "=" * 80)
    print("Testing SABO convergence over iterations")
    print("=" * 80)
    
    logger = setup_logger('sabo_convergence_test')
    
    optimizer = SABOOptimizer(
        obj=dummy_objective,
        eval_func=dummy_eval_func,
        T_bounds=(2, 100),
        seed=123,
        rho=0.1,
        beta_t=0.03,
        N_batch=10,
        logger=logger
    )
    
    result = optimizer.run(iterations=20, n_initial=5)
    
    # Check if perplexity decreased
    history = result['history']
    initial_perp = history[0]['best_perplexity']
    final_perp = history[-1]['best_perplexity']
    
    print(f"Initial perplexity: {initial_perp:.4f}")
    print(f"Final perplexity: {final_perp:.4f}")
    print(f"Improvement: {initial_perp - final_perp:.4f}")
    
    if final_perp < initial_perp:
        print("✓ TEST PASSED: Perplexity decreased")
        return True
    else:
        print("✗ TEST FAILED: Perplexity did not decrease")
        return False


def test_sabo_parameters():
    """Test SABO parameter adaptation."""
    print("\n" + "=" * 80)
    print("Testing SABO parameter adaptation (μ, Σ)")
    print("=" * 80)
    
    logger = setup_logger('sabo_params_test')
    
    optimizer = SABOOptimizer(
        obj=dummy_objective,
        eval_func=dummy_eval_func,
        T_bounds=(2, 100),
        seed=456,
        N_batch=8,
        logger=logger
    )
    
    result = optimizer.run(iterations=15, n_initial=5)
    
    # Check parameter evolution
    history = result['history']
    mu_values = [h['mu'] for h in history if h['iter'] >= 0]
    sigma_values = [h['Sigma'] for h in history if h['iter'] >= 0]
    
    print(f"μ evolution: {mu_values[0]:.4f} → {mu_values[-1]:.4f}")
    print(f"Σ evolution: {sigma_values[0]:.4f} → {sigma_values[-1]:.4f}")
    
    # Verify μ moved towards optimal region
    # Optimal T=50 in [2,100] → normalized ≈ 0.49
    optimal_mu_normalized = (50 - 2) / (100 - 2)
    final_mu = mu_values[-1]
    mu_error = abs(final_mu - optimal_mu_normalized)
    
    print(f"Final μ: {final_mu:.4f}, optimal: {optimal_mu_normalized:.4f}, error: {mu_error:.4f}")
    
    if mu_error < 0.2:
        print("✓ TEST PASSED: μ converged towards optimal region")
        return True
    else:
        print("✗ TEST FAILED: μ did not converge")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("SABO Optimizer Tests")
    print("=" * 80 + "\n")
    
    tests = [
        ("Basic functionality", test_sabo_basic),
        ("Convergence", test_sabo_convergence),
        ("Parameter adaptation", test_sabo_parameters)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ TEST ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print("=" * 80)
    print(f"Total: {passed_count}/{total} tests passed")
    print("=" * 80)
    
    # Exit code
    sys.exit(0 if passed_count == total else 1)
