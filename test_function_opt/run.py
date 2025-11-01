"""
Test Function Optimization using GA, ES, and PABBO.

This script optimizes a complex non-differentiable test function using three
optimization algorithms with the same initial population for fair comparison.

The test function has:
- Multiple local minima (>5)
- Non-differentiable points (absolute values)
- Discontinuities (step function)
- Global minimum around x ≈ 0
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Import test function
from test_function import test_function, test_function_vectorized, get_approximate_minimum

# Import utilities
from utils import setup_logger, ensure_dir, write_history_csv, save_json

# Import optimizers
from optimizers import GAOptimizer, ESOptimizer, PABBOOptimizer


class IntegerToRealAdapter:
    """
    Adapter to convert integer optimization to continuous real-valued optimization.

    Maps integer T in [T_min, T_max] to real x in [x_min, x_max].
    """

    def __init__(self, x_range: Tuple[float, float], T_range: Tuple[int, int]):
        """
        Initialize adapter.

        Args:
            x_range: (x_min, x_max) for continuous domain
            T_range: (T_min, T_max) for integer domain
        """
        self.x_min, self.x_max = x_range
        self.T_min, self.T_max = T_range

    def T_to_x(self, T: int) -> float:
        """Convert integer T to real x."""
        return self.x_min + (T - self.T_min) * (self.x_max - self.x_min) / (self.T_max - self.T_min)

    def x_to_T(self, x: float) -> int:
        """Convert real x to integer T."""
        T = int(round(self.T_min + (x - self.x_min) * (self.T_max - self.T_min) / (self.x_max - self.x_min)))
        return max(self.T_min, min(self.T_max, T))


def load_initial_population(json_path: str) -> List[int]:
    """Load fixed initial population from JSON file."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config['initial_population']


def make_objective_and_eval(adapter: IntegerToRealAdapter):
    """
    Create objective and evaluation functions.

    Args:
        adapter: Integer-to-real adapter

    Returns:
        Tuple of (objective_func, eval_func)
    """

    def objective(T: int, alpha: float, eta: float) -> float:
        """Objective function: returns function value (to minimize)."""
        x = adapter.T_to_x(T)
        return test_function(x)

    def eval_func(T: int, alpha: float, eta: float) -> Dict:
        """Evaluation function: returns detailed metrics."""
        x = adapter.T_to_x(T)
        y = test_function(x)
        return {
            'T': T,
            'x': x,
            'y': y,
            'alpha': alpha,
            'eta': eta,
            'perplexity': y  # Use same name as LDA for compatibility
        }

    return objective, eval_func


def run_single_optimizer(
    algorithm_name: str,
    optimizer_class,
    obj,
    eval_func,
    adapter: IntegerToRealAdapter,
    initial_population: List[int],
    iterations: int,
    outdir: str,
    seed: int = 42,
    **optimizer_kwargs
) -> Dict:
    """
    Run single optimization algorithm on test function.

    Args:
        algorithm_name: Name of algorithm (GA, ES, PABBO)
        optimizer_class: Optimizer class
        obj: Objective function
        eval_func: Evaluation function
        adapter: Integer-to-real adapter
        initial_population: Fixed initial population
        iterations: Number of iterations
        outdir: Output directory
        seed: Random seed
        **optimizer_kwargs: Additional optimizer-specific arguments

    Returns:
        Dictionary with optimization results
    """
    ensure_dir(outdir)
    log_file = os.path.join(outdir, f"{algorithm_name}_test.log")
    logger = setup_logger(f"test_{algorithm_name.lower()}", log_file=log_file)

    logger.info("=" * 80)
    logger.info(f"{algorithm_name} Optimization Test")
    logger.info("=" * 80)
    logger.info(f"Domain: x ∈ [{adapter.x_min}, {adapter.x_max}]")
    logger.info(f"T range: [{adapter.T_min}, {adapter.T_max}]")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Population size: {len(initial_population)}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Initial population: {initial_population}")
    logger.info("=" * 80)

    # Create optimizer
    optimizer = optimizer_class(
        obj=obj,
        eval_func=eval_func,
        T_bounds=(adapter.T_min, adapter.T_max),
        seed=seed,
        logger=logger,
        **optimizer_kwargs
    )

    # Run optimization
    start_time = time.perf_counter()
    result = optimizer.run(
        iterations=iterations,
        writer=None,  # No TensorBoard for test function
        outdir=outdir,
        initial_population=initial_population
    )
    end_time = time.perf_counter()

    # Convert results back to x domain
    best_x = adapter.T_to_x(result['best']['T'])
    result['best']['x'] = best_x

    # Add x values to history
    for h in result['history']:
        h['best_x'] = adapter.T_to_x(h['T_best'])

    # Save results
    write_history_csv(result['history'], os.path.join(outdir, "history.csv"))

    summary = {
        "algorithm": algorithm_name,
        "best_T": result['best']['T'],
        "best_x": best_x,
        "best_y": result['best']['perplexity'],
        "total_time": result['total_time'],
        "avg_step_time": result.get('avg_step_time', 0),
        "stopped_early": result.get('stopped_early', False),
        "num_iterations": len(result['history']),
        "initial_population": initial_population
    }
    save_json(summary, os.path.join(outdir, "summary.json"))

    logger.info("=" * 80)
    logger.info(f"{algorithm_name} Optimization Complete!")
    logger.info(f"Best x: {best_x:.6f}, y: {result['best']['perplexity']:.6f}")
    logger.info(f"Total time: {result['total_time']:.2f}s")
    logger.info("=" * 80)

    return result


def plot_test_function(adapter: IntegerToRealAdapter, outdir: str):
    """Plot the test function."""
    x = np.linspace(adapter.x_min, adapter.x_max, 1000)
    y = test_function_vectorized(x)

    # Find approximate minimum
    x_min, y_min = get_approximate_minimum()

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, linewidth=2, label='Test Function')
    plt.plot(x_min, y_min, 'ro', markersize=10,
            label=f'Approx. minimum: ({x_min:.3f}, {y_min:.3f})')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title('Test Function: Non-differentiable with discontinuities',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, 'test_function.png'), dpi=150)
    plt.close()


def plot_convergence(results: List[Dict], adapter: IntegerToRealAdapter, outdir: str):
    """Plot convergence comparison of all algorithms."""
    ensure_dir(outdir)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # Plot 1: Best y vs Iteration
    for result in results:
        history = result['history']
        iterations = [h['iter'] for h in history]
        best_y = [h['best_perplexity'] for h in history]
        axes[0, 0].plot(iterations, best_y, marker='o', linewidth=2,
                       label=result['algorithm'])

    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Best y', fontsize=12)
    axes[0, 0].set_title('Best Value vs Iteration', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Best y vs Time
    for result in results:
        history = result['history']
        times = [h['cum_time'] for h in history]
        best_y = [h['best_perplexity'] for h in history]
        axes[0, 1].plot(times, best_y, marker='o', linewidth=2,
                       label=result['algorithm'])

    axes[0, 1].set_xlabel('Time (s)', fontsize=12)
    axes[0, 1].set_ylabel('Best y', fontsize=12)
    axes[0, 1].set_title('Best Value vs Time (wall-clock)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Best x vs Iteration
    for result in results:
        history = result['history']
        iterations = [h['iter'] for h in history]
        best_x = [h['best_x'] for h in history]
        axes[1, 0].plot(iterations, best_x, marker='o', linewidth=2,
                       label=result['algorithm'])

    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Best x', fontsize=12)
    axes[1, 0].set_title('Best x vs Iteration', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Best x vs Time
    for result in results:
        history = result['history']
        times = [h['cum_time'] for h in history]
        best_x = [h['best_x'] for h in history]
        axes[1, 1].plot(times, best_x, marker='o', linewidth=2,
                       label=result['algorithm'])

    axes[1, 1].set_xlabel('Time (s)', fontsize=12)
    axes[1, 1].set_ylabel('Best x', fontsize=12)
    axes[1, 1].set_title('Best x vs Time (wall-clock)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'convergence.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test Function Optimization')
    parser.add_argument('--x-range', type=float, nargs=2, default=[-5.0, 5.0],
                       help='Range for continuous x domain')
    parser.add_argument('--T-range', type=int, nargs=2, default=[1, 1000],
                       help='Range for integer T domain')
    parser.add_argument('--init', type=str, default='test_init_population.json',
                       help='Path to initial population JSON file')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of optimization iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--outdir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['GA', 'ES', 'PABBO'],
                       choices=['GA', 'ES', 'PABBO'],
                       help='Algorithms to run')

    args = parser.parse_args()

    # Setup main logger
    ensure_dir(args.outdir)
    logger = setup_logger('main', log_file=os.path.join(args.outdir, 'main.log'))

    logger.info("=" * 80)
    logger.info("Test Function Optimization")
    logger.info("=" * 80)
    logger.info(f"x range: {args.x_range}")
    logger.info(f"T range: {args.T_range}")
    logger.info(f"Initial population file: {args.init}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {args.outdir}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info("=" * 80)

    # Create adapter
    adapter = IntegerToRealAdapter(
        x_range=tuple(args.x_range),
        T_range=tuple(args.T_range)
    )

    # Plot test function
    logger.info("Plotting test function...")
    plot_test_function(adapter, args.outdir)
    x_min, y_min = get_approximate_minimum()
    logger.info(f"Approximate minimum: x = {x_min:.6f}, y = {y_min:.6f}")

    # Load initial population
    logger.info(f"Loading initial population from {args.init}...")
    initial_population = load_initial_population(args.init)
    logger.info(f"Initial population: {initial_population}")

    # Create objective and eval functions
    obj, eval_func = make_objective_and_eval(adapter)

    # Run algorithms
    results = []

    if 'GA' in args.algorithms:
        logger.info("\n" + "=" * 80)
        logger.info("Running Genetic Algorithm (GA)")
        logger.info("=" * 80)
        ga_result = run_single_optimizer(
            algorithm_name='GA',
            optimizer_class=GAOptimizer,
            obj=obj,
            eval_func=eval_func,
            adapter=adapter,
            initial_population=initial_population.copy(),
            iterations=args.iterations,
            outdir=os.path.join(args.outdir, 'GA'),
            seed=args.seed,
            cxpb=0.9,
            mutpb=0.2,
            elite=3
        )
        ga_result['algorithm'] = 'GA'
        results.append(ga_result)

    if 'ES' in args.algorithms:
        logger.info("\n" + "=" * 80)
        logger.info("Running Evolution Strategy (ES)")
        logger.info("=" * 80)
        es_result = run_single_optimizer(
            algorithm_name='ES',
            optimizer_class=ESOptimizer,
            obj=obj,
            eval_func=eval_func,
            adapter=adapter,
            initial_population=initial_population.copy(),
            iterations=args.iterations,
            outdir=os.path.join(args.outdir, 'ES'),
            seed=args.seed,
            mu=5,
            lmbda=10
        )
        es_result['algorithm'] = 'ES'
        results.append(es_result)

    if 'PABBO' in args.algorithms:
        logger.info("\n" + "=" * 80)
        logger.info("Running PABBO")
        logger.info("=" * 80)
        pabbo_result = run_single_optimizer(
            algorithm_name='PABBO',
            optimizer_class=PABBOOptimizer,
            obj=obj,
            eval_func=eval_func,
            adapter=adapter,
            initial_population=initial_population.copy(),
            iterations=args.iterations,
            outdir=os.path.join(args.outdir, 'PABBO'),
            seed=args.seed,
            exploration_rate=0.3
        )
        pabbo_result['algorithm'] = 'PABBO'
        results.append(pabbo_result)

    # Plot comparison
    if len(results) > 0:
        logger.info("\nGenerating convergence plots...")
        plot_convergence(results, adapter, args.outdir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"True minimum: x={x_min:.6f}, y={y_min:.6f}")
    logger.info("-" * 80)
    logger.info(f"{'Algorithm':<12} {'Best x':<12} {'Best y':<12} {'Error':<12} {'Time (s)':<15}")
    logger.info("-" * 80)

    best_algorithm = None
    best_error = float('inf')

    for result in results:
        algo = result['algorithm']
        best_x = result['best']['x']
        best_y = result['best']['perplexity']
        error = abs(best_y - y_min)
        total_time = result['total_time']

        logger.info(f"{algo:<12} {best_x:<12.6f} {best_y:<12.6f} {error:<12.6f} {total_time:<15.2f}")

        if error < best_error:
            best_error = error
            best_algorithm = algo

    logger.info("-" * 80)
    logger.info(f"\nBest algorithm: {best_algorithm}")
    logger.info(f"   Found: x={results[[r['algorithm'] for r in results].index(best_algorithm)]['best']['x']:.6f}, "
               f"y={results[[r['algorithm'] for r in results].index(best_algorithm)]['best']['perplexity']:.6f}")
    logger.info(f"   Error: {best_error:.6f}")
    logger.info("=" * 80)

    # Save overall summary
    overall_summary = {
        "true_minimum": {"x": x_min, "y": y_min},
        "results": [
            {
                "algorithm": r['algorithm'],
                "best_x": r['best']['x'],
                "best_y": r['best']['perplexity'],
                "error": abs(r['best']['perplexity'] - y_min),
                "total_time": r['total_time']
            }
            for r in results
        ],
        "best_algorithm": best_algorithm
    }
    save_json(overall_summary, os.path.join(args.outdir, 'overall_summary.json'))

    logger.info(f"\nAll results saved to: {args.outdir}")


if __name__ == '__main__':
    main()
