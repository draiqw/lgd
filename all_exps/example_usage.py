"""
Example usage of the LDA optimization framework.

This script demonstrates how to use the optimization algorithms
on a single dataset with a small number of iterations.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_bow_data,
    make_objective,
    make_eval_func,
    setup_logger,
    plot_optimization_results
)
from exp_ga import GAOptimizer
from exp_es import ESOptimizer
from exp_pabbo import PABBOOptimizer


def main():
    """Run a quick example on one dataset."""

    # Setup
    dataset_name = "20news"
    dataset_path = f"../data/X_{dataset_name}_val_bow.npz"

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure datasets are in the data/ directory")
        return

    print("="*80)
    print("LDA OPTIMIZATION EXAMPLE")
    print("="*80)

    # Load data
    print(f"\nLoading dataset: {dataset_name}")
    Xval = load_bow_data(dataset_path)
    print(f"Loaded: {Xval.shape[0]} documents, {Xval.shape[1]} vocabulary")

    # Configuration
    config = {
        "seed": 42,
        "max_iter_lda": 30,  # Small for quick demo
        "batch_size": 2048,
        "learning_method": "online",
        "T_bounds": (2, 1000),
        "iterations": 10,  # Small for quick demo
        "pop_size": 5,
    }

    print(f"\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # Setup logger
    logger = setup_logger("example", log_file=None)

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

    # ========== Run GA ==========
    print("\n" + "="*80)
    print("RUNNING GENETIC ALGORITHM (GA)")
    print("="*80)

    ga = GAOptimizer(
        obj=obj,
        eval_func=eval_func,
        T_bounds=config["T_bounds"],
        seed=config["seed"],
        elite=2,
        logger=logger
    )

    ga_results = ga.run(
        iterations=config["iterations"],
        pop_size=config["pop_size"]
    )

    print("\nGA Results:")
    print(f"  Best T: {ga_results['best']['T']}")
    print(f"  Best perplexity: {ga_results['best']['perplexity']:.4f}")
    print(f"  Total time: {ga_results['total_time']:.2f}s")

    # ========== Run ES ==========
    print("\n" + "="*80)
    print("RUNNING EVOLUTION STRATEGY (ES)")
    print("="*80)

    es = ESOptimizer(
        obj=obj,
        eval_func=eval_func,
        T_bounds=config["T_bounds"],
        seed=config["seed"],
        mu=3,
        lmbda=6,
        logger=logger
    )

    es_results = es.run(
        iterations=config["iterations"]
    )

    print("\nES Results:")
    print(f"  Best T: {es_results['best']['T']}")
    print(f"  Best perplexity: {es_results['best']['perplexity']:.4f}")
    print(f"  Total time: {es_results['total_time']:.2f}s")

    # ========== Run PABBO ==========
    print("\n" + "="*80)
    print("RUNNING PABBO")
    print("="*80)

    pabbo = PABBOOptimizer(
        obj=obj,
        eval_func=eval_func,
        T_bounds=config["T_bounds"],
        seed=config["seed"],
        logger=logger
    )

    pabbo_results = pabbo.run(
        iterations=config["iterations"],
        n_initial=config["pop_size"]
    )

    print("\nPABBO Results:")
    print(f"  Best T: {pabbo_results['best']['T']}")
    print(f"  Best perplexity: {pabbo_results['best']['perplexity']:.4f}")
    print(f"  Total time: {pabbo_results['total_time']:.2f}s")

    # ========== Summary ==========
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<12} {'Best T':<8} {'Perplexity':<12} {'Time (s)':<10}")
    print("-"*50)
    print(f"{'GA':<12} {ga_results['best']['T']:<8} {ga_results['best']['perplexity']:<12.4f} {ga_results['total_time']:<10.2f}")
    print(f"{'ES':<12} {es_results['best']['T']:<8} {es_results['best']['perplexity']:<12.4f} {es_results['total_time']:<10.2f}")
    print(f"{'PABBO':<12} {pabbo_results['best']['T']:<8} {pabbo_results['best']['perplexity']:<12.4f} {pabbo_results['total_time']:<10.2f}")
    print("-"*50)

    print("\n✓ Example completed successfully!")
    print("\nTo run full experiments, use:")
    print("  python main.py --sequential")
    print("  python main.py --parallel --max-workers 4")


if __name__ == "__main__":
    main()
