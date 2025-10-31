"""
Main script for running LDA hyperparameter optimization experiments.

This script:
1. Loads validation datasets
2. Creates initial population (shared across algorithms for fair comparison)
3. Runs GA, ES, and PABBO optimizers
4. Saves results and generates visualizations
5. Supports parallel execution

Usage:
    python main.py --sequential                    # Run all algorithms sequentially
    python main.py --parallel                      # Run algorithms in parallel
    python main.py --algorithm ga                  # Run only GA
    python main.py --datasets 20news agnews        # Run on specific datasets
"""

import os
import sys
import argparse
import time
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np
from tensorboardX import SummaryWriter

# Import our modules
from utils import (
    load_bow_data,
    make_objective,
    make_eval_func,
    setup_logger,
    ensure_dir,
    write_history_csv,
    save_json,
    plot_optimization_results,
    timing_decorator
)
from exp_ga import GAOptimizer
from exp_es import ESOptimizer
from exp_pabbo import PABBOOptimizer


# ==================== CONFIGURATION ====================

DATASETS = {
    "20news": "../data/X_20news_val_bow.npz",
    "agnews": "../data/X_agnews_val_bow.npz",
    "val_out": "../data/X_val_out_val_bow.npz",
    "yelp": "../data/X_yelp_val_bow.npz",
}

# Default hyperparameters
DEFAULT_CONFIG = {
    "iterations": 200,
    "pop_size": 10,
    "seed": 42,
    "max_iter_lda": 60,
    "batch_size": 2048,
    "learning_method": "online",
    "T_bounds": (2, 1000),
    "early_stop_eps_pct": 0.01,
    "max_no_improvement": 3,

    # GA specific
    "ga_cxpb": 0.9,
    "ga_mutpb": 0.2,
    "ga_tournsize": 3,
    "ga_elite": 5,
    "ga_dT": 5,

    # ES specific
    "es_mu": 5,
    "es_lmbda": 10,
    "es_dT": 5,

    # PABBO specific
    "pabbo_exploration_rate": 0.3,
    "pabbo_temperature_decay": 0.95,
    "pabbo_min_temperature": 0.1,
}


# ==================== INITIAL POPULATION GENERATION ====================

def generate_shared_initial_population(
    pop_size: int,
    T_bounds: Tuple[int, int],
    seed: int
) -> List[int]:
    """
    Generate shared initial population for all algorithms.

    This ensures fair comparison by starting all algorithms from the same points.

    Args:
        pop_size: Population size
        T_bounds: Bounds for T parameter
        seed: Random seed

    Returns:
        List of T values
    """
    random.seed(seed)
    np.random.seed(seed)

    # Use Latin Hypercube Sampling for good coverage
    population = []
    for i in range(pop_size):
        # Divide range into bins and sample from each
        bin_size = (T_bounds[1] - T_bounds[0]) / pop_size
        T = int(T_bounds[0] + (i + random.random()) * bin_size)
        T = max(T_bounds[0], min(T_bounds[1], T))
        population.append(T)

    # Shuffle
    random.shuffle(population)

    return population


# ==================== SINGLE EXPERIMENT RUNNER ====================

@timing_decorator
def run_single_experiment(
    algorithm_name: str,
    optimizer_class,
    dataset_name: str,
    Xval,
    outdir: str,
    config: Dict,
    initial_population: Optional[List[int]] = None,
    **optimizer_kwargs
) -> Dict:
    """
    Run a single optimization experiment.

    Args:
        algorithm_name: Name of algorithm (GA/ES/PABBO)
        optimizer_class: Optimizer class
        dataset_name: Dataset name
        Xval: Validation data
        outdir: Output directory
        config: Configuration dictionary
        initial_population: Shared initial population
        **optimizer_kwargs: Additional optimizer-specific arguments

    Returns:
        Results dictionary
    """
    ensure_dir(outdir)

    # Setup logger
    log_file = os.path.join(outdir, "experiment.log")
    logger = setup_logger(
        name=f"{algorithm_name}_{dataset_name}",
        log_file=log_file,
        level="INFO"
    )

    logger.info("="*80)
    logger.info(f"EXPERIMENT: {algorithm_name} on {dataset_name}")
    logger.info(f"Output directory: {outdir}")
    logger.info("="*80)

    # Create objective and eval functions
    obj = make_objective(
        Xval,
        seed=config["seed"],
        max_iter=config["max_iter_lda"],
        batch_size=config["batch_size"],
        learning_method=config["learning_method"],
        logger=logger
    )

    eval_func = make_eval_func(
        Xval,
        seed=config["seed"],
        max_iter=config["max_iter_lda"],
        batch_size=config["batch_size"],
        learning_method=config["learning_method"],
        logger=logger
    )

    # Create optimizer
    optimizer = optimizer_class(
        obj=obj,
        eval_func=eval_func,
        T_bounds=config["T_bounds"],
        seed=config["seed"],
        early_stop_eps_pct=config["early_stop_eps_pct"],
        max_no_improvement=config["max_no_improvement"],
        logger=logger,
        **optimizer_kwargs
    )

    # Setup TensorBoard
    tb_dir = os.path.join(outdir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"TensorBoard logs: {tb_dir}")

    # Run optimization
    if algorithm_name == "GA":
        results = optimizer.run(
            iterations=config["iterations"],
            pop_size=config["pop_size"],
            writer=writer,
            outdir=outdir,
            initial_population=initial_population
        )
    elif algorithm_name == "ES":
        results = optimizer.run(
            iterations=config["iterations"],
            writer=writer,
            outdir=outdir,
            initial_population=initial_population
        )
    elif algorithm_name == "PABBO":
        results = optimizer.run(
            iterations=config["iterations"],
            n_initial=config["pop_size"],
            writer=writer,
            outdir=outdir,
            initial_population=initial_population
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    writer.close()

    # Save results
    write_history_csv(results["history"], os.path.join(outdir, "history.csv"))
    logger.info(f"History saved to {os.path.join(outdir, 'history.csv')}")

    # Generate plots
    plot_optimization_results(
        results["history"],
        algorithm_name,
        outdir
    )
    logger.info(f"Plots saved to {outdir}")

    # Save summary
    summary = {
        "algorithm": algorithm_name,
        "dataset": dataset_name,
        "best": results["best"],
        "total_time": results["total_time"],
        "avg_step_time": results["avg_step_time"],
        "stopped_early": results["stopped_early"],
        "num_iterations": len(results["history"]),
        "config": config
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    logger.info(f"Summary saved to {os.path.join(outdir, 'summary.json')}")

    logger.info("="*80)
    logger.info(f"EXPERIMENT COMPLETE: {algorithm_name} on {dataset_name}")
    logger.info(f"Best T: {results['best']['T']}")
    logger.info(f"Best perplexity: {results['best']['perplexity']:.4f}")
    logger.info(f"Total time: {results['total_time']:.2f}s")
    logger.info("="*80)

    return summary


# ==================== PARALLEL RUNNER ====================

def run_task_wrapper(args):
    """Wrapper for parallel execution."""
    return run_single_experiment(*args[0], **args[1])


def run_all_experiments_parallel(
    tasks: List,
    max_workers: int = 4
) -> Dict:
    """
    Run multiple experiments in parallel.

    Args:
        tasks: List of task tuples (args, kwargs)
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary of results organized by dataset and algorithm
    """
    results = {}

    print("="*80)
    print(f"PARALLEL EXECUTION: {len(tasks)} tasks with {max_workers} workers")
    print("="*80)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task_wrapper, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                summary = future.result()
                dataset_name = summary["dataset"]
                algorithm = summary["algorithm"]

                if dataset_name not in results:
                    results[dataset_name] = {}
                results[dataset_name][algorithm] = summary

                print(f"\n[COMPLETED] {algorithm} on {dataset_name}")
                print(f"  Best T: {summary['best']['T']}")
                print(f"  Best perplexity: {summary['best']['perplexity']:.4f}")
                print(f"  Total time: {summary['total_time']:.2f}s\n")

            except Exception as e:
                print(f"Task failed with error: {e}")

    return results


# ==================== MAIN FUNCTION ====================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LDA Hyperparameter Optimization with GA, ES, and PABBO"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially (default)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ga", "es", "pabbo", "all"],
        default="all",
        help="Which algorithm(s) to run"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Which datasets to use"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_CONFIG["iterations"],
        help="Number of iterations/generations"
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=DEFAULT_CONFIG["pop_size"],
        help="Population size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )

    args = parser.parse_args()

    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = args.iterations
    config["pop_size"] = args.pop_size
    config["seed"] = args.seed

    # Determine which algorithms to run
    if args.algorithm == "all":
        algorithms = ["GA", "ES", "PABBO"]
    else:
        algorithms = [args.algorithm.upper()]

    # Main logger
    main_logger = setup_logger("main", log_file=None, level="INFO")

    main_logger.info("="*80)
    main_logger.info("LDA HYPERPARAMETER OPTIMIZATION EXPERIMENTS")
    main_logger.info(f"Algorithms: {', '.join(algorithms)}")
    main_logger.info(f"Datasets: {', '.join(args.datasets)}")
    main_logger.info(f"Mode: {'PARALLEL' if args.parallel else 'SEQUENTIAL'}")
    main_logger.info(f"Output directory: {args.output_dir}")
    main_logger.info("="*80)

    # Generate shared initial population
    main_logger.info("Generating shared initial population for fair comparison...")
    shared_population = generate_shared_initial_population(
        pop_size=config["pop_size"],
        T_bounds=config["T_bounds"],
        seed=config["seed"]
    )
    main_logger.info(
        f"Initial population: {len(shared_population)} individuals, "
        f"T range: [{min(shared_population)}, {max(shared_population)}]"
    )

    # Prepare tasks
    tasks = []
    all_results = {}

    for dataset_name in args.datasets:
        dataset_path = DATASETS[dataset_name]

        # Check if file exists
        if not os.path.exists(dataset_path):
            main_logger.error(f"Dataset not found: {dataset_path}")
            continue

        main_logger.info(f"\nPreparing experiments for dataset: {dataset_name}")

        # Load data (only once per dataset)
        Xval = load_bow_data(dataset_path)
        main_logger.info(
            f"Loaded {dataset_name}: {Xval.shape[0]} documents, "
            f"{Xval.shape[1]} vocabulary"
        )

        all_results[dataset_name] = {}

        # GA
        if "GA" in algorithms:
            outdir = os.path.join(args.output_dir, dataset_name, "ga")
            task_args = (
                "GA",
                GAOptimizer,
                dataset_name,
                Xval,
                outdir,
                config
            )
            task_kwargs = {
                "initial_population": shared_population,
                "cxpb": config["ga_cxpb"],
                "mutpb": config["ga_mutpb"],
                "tournsize": config["ga_tournsize"],
                "elite": config["ga_elite"],
                "dT": config["ga_dT"]
            }

            if args.parallel:
                tasks.append((task_args, task_kwargs))
            else:
                summary = run_single_experiment(*task_args, **task_kwargs)
                all_results[dataset_name]["GA"] = summary

        # ES
        if "ES" in algorithms:
            outdir = os.path.join(args.output_dir, dataset_name, "es")
            task_args = (
                "ES",
                ESOptimizer,
                dataset_name,
                Xval,
                outdir,
                config
            )
            task_kwargs = {
                "initial_population": shared_population,
                "mu": config["es_mu"],
                "lmbda": config["es_lmbda"],
                "dT": config["es_dT"]
            }

            if args.parallel:
                tasks.append((task_args, task_kwargs))
            else:
                summary = run_single_experiment(*task_args, **task_kwargs)
                all_results[dataset_name]["ES"] = summary

        # PABBO
        if "PABBO" in algorithms:
            outdir = os.path.join(args.output_dir, dataset_name, "pabbo")
            task_args = (
                "PABBO",
                PABBOOptimizer,
                dataset_name,
                Xval,
                outdir,
                config
            )
            task_kwargs = {
                "initial_population": shared_population,
                "exploration_rate": config["pabbo_exploration_rate"],
                "temperature_decay": config["pabbo_temperature_decay"],
                "min_temperature": config["pabbo_min_temperature"]
            }

            if args.parallel:
                tasks.append((task_args, task_kwargs))
            else:
                summary = run_single_experiment(*task_args, **task_kwargs)
                all_results[dataset_name]["PABBO"] = summary

    # Run parallel tasks if applicable
    if args.parallel and tasks:
        parallel_results = run_all_experiments_parallel(tasks, args.max_workers)
        all_results.update(parallel_results)

    # Save overall results
    main_logger.info("\n" + "="*80)
    main_logger.info("ALL EXPERIMENTS COMPLETED!")
    main_logger.info("="*80)

    results_file = os.path.join(args.output_dir, "all_results.json")
    save_json(all_results, results_file)
    main_logger.info(f"All results saved to: {results_file}")

    # Print summary table
    main_logger.info("\nRESULTS SUMMARY:")
    main_logger.info("-"*80)
    main_logger.info(f"{'Dataset':<12} {'Algorithm':<8} {'Best T':<8} {'Perplexity':<12} {'Time (s)':<10}")
    main_logger.info("-"*80)

    for dataset in args.datasets:
        if dataset in all_results:
            for algo in algorithms:
                if algo in all_results[dataset]:
                    r = all_results[dataset][algo]
                    main_logger.info(
                        f"{dataset:<12} {algo:<8} {r['best']['T']:<8} "
                        f"{r['best']['perplexity']:<12.4f} {r['total_time']:<10.2f}"
                    )

    main_logger.info("-"*80)
    main_logger.info("\nExperiments finished successfully!")


if __name__ == "__main__":
    main()