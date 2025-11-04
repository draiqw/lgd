"""
Utils module for LDA hyperparameter optimization experiments.

This module contains:
- Data loading utilities
- LDA training and evaluation functions
- Logging utilities
- Visualization functions
- Base optimizer class
"""

import time
import os
import csv
import json
import logging
from functools import wraps
from typing import Dict, List, Tuple, Callable, Optional
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from tensorboardX import SummaryWriter


# ==================== LOGGING SETUP ====================

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMING] {func.__name__} completed in {end - start:.2f}s")
        return result
    return wrapper


# ==================== DATA LOADING ====================

def load_bow_data(val_path: str) -> sp.csr_matrix:
    """
    Load validation data (bag-of-words sparse matrix).

    Args:
        val_path: Path to validation data .npz file

    Returns:
        Sparse CSR matrix with shape (n_documents, n_vocabulary)
    """
    Xval = sp.load_npz(val_path).tocsr(copy=False)
    return Xval


# ==================== LDA TRAINING & EVALUATION ====================

# Global cache for LDA evaluations to avoid recomputing same configurations
EVAL_CACHE = {}


def _fit_eval_on_val(
    T: int,
    alpha: float,
    eta: float,
    Xval: sp.csr_matrix,
    seed: int = 42,
    max_iter: int = 100,
    batch_size: int = 2048,
    learning_method: str = "online",
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Train LDA on validation set and evaluate perplexity.

    This is the correct approach for hyperparameter optimization:
    train and evaluate on the same validation set.

    Args:
        T: Number of topics
        alpha: Document-topic prior (doc_topic_prior)
        eta: Topic-word prior (topic_word_prior)
        Xval: Validation data
        seed: Random seed
        max_iter: Maximum number of iterations for LDA
        batch_size: Batch size for online learning
        learning_method: Learning method ('online' or 'batch')
        logger: Optional logger instance

    Returns:
        Dictionary with evaluation metrics:
            - T, alpha, eta: hyperparameters
            - perplexity: perplexity on validation set
            - fit_time: training time
            - eval_time: evaluation time
            - n_iter_lda: actual number of LDA iterations
    """
    # Check cache
    key = (int(T), float(alpha), float(eta), int(seed), int(max_iter),
           int(batch_size), learning_method, id(Xval))
    if key in EVAL_CACHE:
        if logger:
            logger.debug(f"Using cached result for T={T}, alpha={alpha:.6f}, eta={eta:.6f}")
        return EVAL_CACHE[key]

    log_msg = f"Training LDA on val: T={T}, alpha={alpha:.6f}, eta={eta:.6f}"
    if logger:
        logger.info(log_msg)
    else:
        print(f"[LDA] {log_msg}")

    # Train LDA
    lda = LatentDirichletAllocation(
        n_components=int(T),
        doc_topic_prior=float(alpha),
        topic_word_prior=float(eta),
        learning_method=learning_method,
        max_iter=int(max_iter),
        batch_size=int(batch_size),
        random_state=int(seed),
        evaluate_every=-1,
        n_jobs=-1
    )

    t0 = time.perf_counter()
    lda.fit(Xval)
    fit_time = time.perf_counter() - t0

    n_iter = getattr(lda, 'n_iter_', 'N/A')
    log_msg = f"Training completed in {fit_time:.2f}s (n_iter={n_iter})"
    if logger:
        logger.info(log_msg)
    else:
        print(f"[LDA] {log_msg}")

    # Evaluate perplexity
    t1 = time.perf_counter()
    perplexity = float(lda.perplexity(Xval))
    eval_time = time.perf_counter() - t1

    log_msg = f"Evaluation completed in {eval_time:.2f}s | Perplexity: {perplexity:.4f}"
    if logger:
        logger.info(log_msg)
    else:
        print(f"[LDA] {log_msg}")

    # Store result
    res = {
        "T": int(T),
        "alpha": float(alpha),
        "eta": float(eta),
        "perplexity": perplexity,
        "fit_time": fit_time,
        "eval_time": eval_time,
        "n_iter_lda": n_iter
    }

    EVAL_CACHE[key] = res
    return res


def make_objective(
    Xval: sp.csr_matrix,
    seed: int = 42,
    max_iter: int = 100,
    batch_size: int = 2048,
    learning_method: str = "online",
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Create objective function that returns perplexity.

    Args:
        Xval: Validation data
        seed: Random seed
        max_iter: Max LDA iterations
        batch_size: Batch size
        learning_method: Learning method
        logger: Optional logger

    Returns:
        Objective function that takes (T, alpha, eta) and returns perplexity
    """
    def objective(T: int, alpha: float, eta: float) -> float:
        r = _fit_eval_on_val(
            T, alpha, eta, Xval,
            seed=seed,
            max_iter=max_iter,
            batch_size=batch_size,
            learning_method=learning_method,
            logger=logger
        )
        return r["perplexity"]
    return objective


def make_eval_func(
    Xval: sp.csr_matrix,
    seed: int = 42,
    max_iter: int = 100,
    batch_size: int = 2048,
    learning_method: str = "online",
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Create evaluation function that returns full metrics.

    Args:
        Xval: Validation data
        seed: Random seed
        max_iter: Max LDA iterations
        batch_size: Batch size
        learning_method: Learning method
        logger: Optional logger

    Returns:
        Eval function that takes (T, alpha, eta) and returns metrics dict
    """
    def eval_func(T: int, alpha: float, eta: float) -> Dict:
        return _fit_eval_on_val(
            T, alpha, eta, Xval,
            seed=seed,
            max_iter=max_iter,
            batch_size=batch_size,
            learning_method=learning_method,
            logger=logger
        )
    return eval_func


# ==================== FILE I/O ====================

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def write_history_csv(history: List[Dict], path: str):
    """Write optimization history to CSV file."""
    if not history:
        return

    ensure_dir(os.path.dirname(path))
    fields = list(history[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ==================== VISUALIZATION ====================

def plot_series(
    xs: List,
    ys: List,
    xlabel: str,
    ylabel: str,
    title: str,
    path: str
):
    """
    Plot and save a line series.

    Args:
        xs: X values
        ys: Y values
        xlabel: X axis label
        ylabel: Y axis label
        title: Plot title
        path: Output path
    """
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ensure_dir(os.path.dirname(path))

    # Save in both PNG and SVG formats
    base_path = os.path.splitext(path)[0]  # Remove extension
    plt.savefig(f"{base_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight')
    plt.close()


def plot_optimization_results(
    history: List[Dict],
    algorithm_name: str,
    outdir: str
):
    """
    Generate standard optimization plots.

    Args:
        history: Optimization history
        algorithm_name: Algorithm name (for titles)
        outdir: Output directory
    """
    if not history:
        return

    xs = [h["iter"] for h in history]

    # Perplexity plot
    ys_perplexity = [h["best_perplexity"] for h in history]
    plot_series(
        xs, ys_perplexity,
        "Iteration", "Perplexity",
        f"{algorithm_name}: Best Perplexity vs Iteration",
        os.path.join(outdir, "perplexity.png")
    )

    # T plot
    ys_T = [h["T_best"] for h in history]
    plot_series(
        xs, ys_T,
        "Iteration", "T (Topics)",
        f"{algorithm_name}: Best T vs Iteration",
        os.path.join(outdir, "T_over_time.png")
    )

    # Step time plot
    ys_step = [h["step_time"] for h in history]
    plot_series(
        xs, ys_step,
        "Iteration", "Time (seconds)",
        f"{algorithm_name}: Step Time vs Iteration",
        os.path.join(outdir, "step_time.png")
    )

    # Population statistics (if available)
    if "pop_mean" in history[0]:
        ys_mean = [h["pop_mean"] for h in history]
        ys_std = [h["pop_std"] for h in history]

        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys_mean, 'b-', marker="o", label="Mean", linewidth=1.5)
        plt.fill_between(xs,
                         np.array(ys_mean) - np.array(ys_std),
                         np.array(ys_mean) + np.array(ys_std),
                         alpha=0.3)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title(f"{algorithm_name}: Population Statistics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save in both PNG and SVG formats
        plt.savefig(os.path.join(outdir, "population_stats.png"), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(outdir, "population_stats.svg"), format='svg', bbox_inches='tight')
        plt.close()


# ==================== UTILITY FUNCTIONS ====================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return lo if x < lo else hi if x > hi else x


# ==================== BASE OPTIMIZER CLASS ====================

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.

    All optimizers should:
    1. Accept same initialization interface
    2. Implement run() method
    3. Support logging and TensorBoard
    4. Return standardized results
    """

    def __init__(
        self,
        obj: Callable,
        eval_func: Callable,
        T_bounds: Tuple[int, int] = (2, 1000),
        seed: int = 42,
        early_stop_eps_pct: float = 0.01,
        max_no_improvement: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize base optimizer.

        Args:
            obj: Objective function (T, alpha, eta) -> perplexity
            eval_func: Evaluation function (T, alpha, eta) -> metrics dict
            T_bounds: Bounds for T parameter (min, max)
            seed: Random seed
            early_stop_eps_pct: Early stopping threshold (relative change %)
            max_no_improvement: Number of iterations without improvement before stopping
            logger: Logger instance
        """
        self.obj = obj
        self.eval_func = eval_func
        self.Tb = T_bounds
        self.seed = int(seed)
        self.early_stop_eps_pct = float(early_stop_eps_pct)
        self.max_no_improvement = int(max_no_improvement)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Set seeds
        np.random.seed(self.seed)

    @abstractmethod
    def run(
        self,
        iterations: int,
        writer: Optional[SummaryWriter] = None,
        outdir: Optional[str] = None,
        initial_population: Optional[List] = None
    ) -> Dict:
        """
        Run optimization.

        Args:
            iterations: Number of iterations/generations/steps
            writer: TensorBoard writer
            outdir: Output directory for saving populations
            initial_population: Optional initial population (for fair comparison)

        Returns:
            Dictionary with:
                - best: best hyperparameters and metrics
                - history: iteration-by-iteration history
                - total_time: total optimization time
                - avg_step_time: average time per iteration
                - stopped_early: whether early stopping was triggered
        """
        pass

    def decode(self, individual) -> Tuple[int, float, float]:
        """
        Decode individual to (T, alpha, eta).

        Alpha and eta are set to 1/T as per problem formulation.

        Args:
            individual: Individual representation (depends on algorithm)

        Returns:
            Tuple of (T, alpha, eta)
        """
        T = int(round(individual[0]))
        T = clamp(T, self.Tb[0], self.Tb[1])
        alpha = 1.0 / T
        eta = 1.0 / T
        return T, float(alpha), float(eta)

    def log_iteration(self, iteration: int, metrics: Dict):
        """
        Log iteration information.

        Args:
            iteration: Current iteration number
            metrics: Metrics dictionary
        """
        msg = (
            f"Iter {iteration} | "
            f"Best: T={metrics.get('T_best', 'N/A')}, "
            f"perplexity={metrics.get('best_perplexity', 'N/A'):.4f}"
        )
        self.logger.info(msg)