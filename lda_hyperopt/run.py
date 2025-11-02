"""
LDA Hyperparameter Optimization using GA, ES, and PABBO.

This script optimizes the T parameter (number of topics) for LDA by minimizing
corpus perplexity on the validation set. Alpha and eta are set to 1/T.

All three algorithms (GA, ES, PABBO) use the same initial population for fair comparison.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# Import base utilities
from utils import (
    load_bow_data,
    make_objective,
    make_eval_func,
    setup_logger,
    ensure_dir,
    write_history_csv,
    save_json
)

# Import optimizers
from optimizers import GAOptimizer, ESOptimizer, PABBOSimpleOptimizer, PABBOFullOptimizer, PABBOOptimizer


def load_initial_population(json_path: str) -> List[int]:
    """Load fixed initial population from JSON file."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config['initial_population']


def run_single_optimizer(
    algorithm_name: str,
    optimizer_class,
    obj,
    eval_func,
    initial_population: List[int],
    iterations: int,
    outdir: str,
    seed: int = 42,
    **optimizer_kwargs
) -> Dict:
    """
    Run single optimization algorithm.

    Args:
        algorithm_name: Name of algorithm (GA, ES, PABBO)
        optimizer_class: Optimizer class
        obj: Objective function
        eval_func: Evaluation function
        initial_population: Fixed initial population
        iterations: Number of iterations
        outdir: Output directory
        seed: Random seed
        **optimizer_kwargs: Additional optimizer-specific arguments

    Returns:
        Dictionary with optimization results
    """
    ensure_dir(outdir)
    log_file = os.path.join(outdir, f"{algorithm_name}_optimization.log")
    logger = setup_logger(f"{algorithm_name}_opt", log_file=log_file)

    logger.info("=" * 80)
    logger.info(f"{algorithm_name} Optimization Starting")
    logger.info("=" * 80)
    logger.info(f"Initial population: {initial_population}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Output directory: {outdir}")
    logger.info("=" * 80)

    # Create optimizer
    optimizer = optimizer_class(
        obj=obj,
        eval_func=eval_func,
        T_bounds=(2, 1000),
        seed=seed,
        logger=logger,
        **optimizer_kwargs
    )

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(outdir, "tensorboard"))

    # Run optimization
    start_time = time.perf_counter()
    result = optimizer.run(
        iterations=iterations,
        writer=writer,
        outdir=outdir,
        initial_population=initial_population
    )
    end_time = time.perf_counter()

    writer.close()

    # Save results
    write_history_csv(result['history'], os.path.join(outdir, "history.csv"))

    summary = {
        "algorithm": algorithm_name,
        "best_T": result['best']['T'],
        "best_alpha": result['best']['alpha'],
        "best_eta": result['best']['eta'],
        "best_perplexity": result['best']['perplexity'],
        "total_time": result['total_time'],
        "avg_step_time": result.get('avg_step_time', 0),
        "stopped_early": result.get('stopped_early', False),
        "num_iterations": len(result['history']),
        "initial_population": initial_population
    }
    save_json(summary, os.path.join(outdir, "summary.json"))

    # Plot results
    if result['history']:
        plot_optimization_history(
            result['history'],
            algorithm_name,
            outdir
        )

    logger.info("=" * 80)
    logger.info(f"{algorithm_name} Optimization Complete!")
    logger.info(f"Best T: {result['best']['T']}, alpha: {result['best']['alpha']:.6f}, eta: {result['best']['eta']:.6f}")
    logger.info(f"Best perplexity: {result['best']['perplexity']:.4f}")
    logger.info(f"Total time: {result['total_time']:.2f}s")
    logger.info("=" * 80)

    return result


def plot_optimization_history(history: List[Dict], algorithm_name: str, outdir: str):
    """Plot optimization history."""
    ensure_dir(outdir)

    iterations = [h['iter'] for h in history]
    perplexities = [h['best_perplexity'] for h in history]
    T_values = [h['T_best'] for h in history]
    times = [h['cum_time'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Perplexity vs Iteration
    axes[0, 0].plot(iterations, perplexities, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Perplexity')
    axes[0, 0].set_title(f'{algorithm_name}: Perplexity vs Iteration')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Perplexity vs Time
    axes[0, 1].plot(times, perplexities, marker='o', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title(f'{algorithm_name}: Perplexity vs Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: T vs Iteration
    axes[1, 0].plot(iterations, T_values, marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('T (topics)')
    axes[1, 0].set_title(f'{algorithm_name}: T vs Iteration')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: T vs Time
    axes[1, 1].plot(times, T_values, marker='o', linewidth=2, color='green')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('T (topics)')
    axes[1, 1].set_title(f'{algorithm_name}: T vs Time')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'optimization_plots.png'), dpi=150)
    plt.close()


def plot_comparison(results: List[Dict], outdir: str):
    """Plot comparison of all algorithms."""
    ensure_dir(outdir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Perplexity comparison
    for result in results:
        history = result['history']
        iterations = [h['iter'] for h in history]
        perplexities = [h['best_perplexity'] for h in history]
        axes[0].plot(iterations, perplexities, marker='o', linewidth=2,
                    label=result['algorithm'])

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Perplexity')
    axes[0].set_title('Algorithm Comparison: Perplexity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Time comparison
    for result in results:
        history = result['history']
        times = [h['cum_time'] for h in history]
        perplexities = [h['best_perplexity'] for h in history]
        axes[1].plot(times, perplexities, marker='o', linewidth=2,
                    label=result['algorithm'])

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Algorithm Comparison: Time Efficiency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'comparison.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='LDA Hyperparameter Optimization')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to validation data (.npz file)')
    parser.add_argument('--init', type=str, default='lda_init_population.json',
                       help='Path to initial population JSON file')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of optimization iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--outdir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Max LDA iterations')
    parser.add_argument('--batch-size', type=int, default=2048,
                       help='LDA batch size')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['GA', 'ES', 'PABBO_Simple'],
                       choices=['GA', 'ES', 'PABBO', 'PABBO_Simple', 'PABBO_Full'],
                       help='Algorithms to run (PABBO defaults to PABBO_Simple)')
    parser.add_argument('--pabbo-model', type=str, default=None,
                       help='Path to trained PABBO Transformer model (for PABBO_Full)')

    args = parser.parse_args()

    # Setup main logger
    ensure_dir(args.outdir)
    logger = setup_logger('main', log_file=os.path.join(args.outdir, 'main.log'))

    logger.info("=" * 80)
    logger.info("LDA Hyperparameter Optimization")
    logger.info("=" * 80)
    logger.info(f"Data: {args.data}")
    logger.info(f"Initial population file: {args.init}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {args.outdir}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info("=" * 80)

    # Load data
    logger.info("Loading validation data...")
    Xval = load_bow_data(args.data)
    logger.info(f"Loaded: {Xval.shape[0]} documents, {Xval.shape[1]} vocabulary")

    # Load initial population
    logger.info(f"Loading initial population from {args.init}...")
    initial_population = load_initial_population(args.init)
    logger.info(f"Initial population: {initial_population}")

    # Create objective and eval functions
    obj = make_objective(
        Xval,
        seed=args.seed,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        learning_method="online"
    )
    eval_func = make_eval_func(
        Xval,
        seed=args.seed,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        learning_method="online"
    )

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
            initial_population=initial_population.copy(),
            iterations=args.iterations,
            outdir=os.path.join(args.outdir, 'GA'),
            seed=args.seed,
            cxpb=0.9,
            mutpb=0.2,
            elite=5
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
            initial_population=initial_population.copy(),
            iterations=args.iterations,
            outdir=os.path.join(args.outdir, 'ES'),
            seed=args.seed,
            mu=5,
            lmbda=10
        )
        es_result['algorithm'] = 'ES'
        results.append(es_result)

    # Handle PABBO variants
    if 'PABBO' in args.algorithms or 'PABBO_Simple' in args.algorithms:
        logger.info("\n" + "=" * 80)
        logger.info("Running PABBO Simple")
        logger.info("=" * 80)
        pabbo_result = run_single_optimizer(
            algorithm_name='PABBO_Simple',
            optimizer_class=PABBOSimpleOptimizer,
            obj=obj,
            eval_func=eval_func,
            initial_population=initial_population.copy(),
            iterations=args.iterations,
            outdir=os.path.join(args.outdir, 'PABBO_Simple'),
            seed=args.seed,
            exploration_rate=0.3
        )
        pabbo_result['algorithm'] = 'PABBO_Simple'
        results.append(pabbo_result)

    if 'PABBO_Full' in args.algorithms:
        logger.info("\n" + "=" * 80)
        logger.info("Running PABBO Full (with Transformer)")
        logger.info("=" * 80)
        pabbo_full_result = run_single_optimizer(
            algorithm_name='PABBO_Full',
            optimizer_class=PABBOFullOptimizer,
            obj=obj,
            eval_func=eval_func,
            initial_population=initial_population.copy(),
            iterations=args.iterations,
            outdir=os.path.join(args.outdir, 'PABBO_Full'),
            seed=args.seed,
            model_path=args.pabbo_model,
            exploration_rate=0.3
        )
        pabbo_full_result['algorithm'] = 'PABBO_Full'
        results.append(pabbo_full_result)

    # Plot comparison
    if len(results) > 1:
        logger.info("\nGenerating comparison plots...")
        plot_comparison(results, args.outdir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Algorithm':<10} {'Best T':<10} {'Best Perplexity':<20} {'Time (s)':<15}")
    logger.info("-" * 80)

    best_algorithm = None
    best_perplexity = float('inf')

    for result in results:
        algo = result['algorithm']
        best_T = result['best']['T']
        best_ppl = result['best']['perplexity']
        total_time = result['total_time']

        logger.info(f"{algo:<10} {best_T:<10} {best_ppl:<20.4f} {total_time:<15.2f}")

        if best_ppl < best_perplexity:
            best_perplexity = best_ppl
            best_algorithm = algo

    logger.info("-" * 80)
    logger.info(f"Best algorithm: {best_algorithm} (perplexity: {best_perplexity:.4f})")
    logger.info("=" * 80)

    # Save overall summary
    overall_summary = {
        "results": [
            {
                "algorithm": r['algorithm'],
                "best_T": r['best']['T'],
                "best_perplexity": r['best']['perplexity'],
                "total_time": r['total_time']
            }
            for r in results
        ],
        "best_algorithm": best_algorithm,
        "best_perplexity": best_perplexity
    }
    save_json(overall_summary, os.path.join(args.outdir, 'overall_summary.json'))

    logger.info(f"\nAll results saved to: {args.outdir}")


if __name__ == '__main__':
    main()
