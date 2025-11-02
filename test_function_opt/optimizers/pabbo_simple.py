"""
PABBO-inspired optimizer for LDA hyperparameters.

This module implements a simplified PABBO-inspired approach using:
- Random search with adaptive sampling
- Exploration-exploitation balance
- Searches for optimal T (number of topics)
- Sets alpha and eta to 1/T
- Trains on validation set
- Optimizes for minimal perplexity

Note: This is a simplified version. Full PABBO requires preference-based
optimization which is not directly applicable to perplexity minimization.
"""

import time
import os
import csv
import random
from typing import Optional, List, Dict, Tuple

import numpy as np
from tensorboardX import SummaryWriter

from utils import BaseOptimizer, clamp, ensure_dir


class PABBOSimpleOptimizer(BaseOptimizer):
    """
    PABBO-inspired optimizer for LDA hyperparameters.

    Uses random search with adaptive sampling:
    - Initial random exploration
    - Focused sampling around best regions
    - Gradually reduces search radius
    - Maintains diversity through temperature parameter
    """

    def __init__(
        self,
        obj,
        eval_func,
        T_bounds=(2, 1000),
        seed=42,
        exploration_rate=0.3,
        temperature_decay=0.95,
        min_temperature=0.1,
        early_stop_eps_pct=0.01,
        max_no_improvement=5,
        logger=None
    ):
        """
        Initialize PABBO optimizer.

        Args:
            obj: Objective function (T, alpha, eta) -> perplexity
            eval_func: Evaluation function (T, alpha, eta) -> metrics dict
            T_bounds: Bounds for T parameter (min, max)
            seed: Random seed
            exploration_rate: Probability of random exploration vs exploitation
            temperature_decay: Decay rate for search temperature
            min_temperature: Minimum search temperature
            early_stop_eps_pct: Early stopping threshold (relative change %)
            max_no_improvement: Iterations without improvement before stopping
            logger: Logger instance
        """
        super().__init__(
            obj=obj,
            eval_func=eval_func,
            T_bounds=T_bounds,
            seed=seed,
            early_stop_eps_pct=early_stop_eps_pct,
            max_no_improvement=max_no_improvement,
            logger=logger
        )

        self.exploration_rate = exploration_rate
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Track evaluated points
        self.evaluated_points = []  # List of (T, fitness)

        self.logger.info(
            f"PABBO initialized: exploration_rate={exploration_rate}, "
            f"temperature_decay={temperature_decay}"
        )

    def _evaluate_point(self, T: int) -> float:
        """
        Evaluate a single point.

        Args:
            T: Number of topics

        Returns:
            Fitness (perplexity)
        """
        T, a, e = self.decode([T])
        try:
            v = float(self.obj(T, a, e))
        except Exception as ex:
            self.logger.warning(f"Evaluation failed for T={T}: {ex}")
            v = float("inf")

        self.evaluated_points.append((T, v))
        return v

    def _sample_exploration(self) -> int:
        """
        Sample random point from entire search space.

        Returns:
            Random T value
        """
        return random.randint(self.Tb[0], self.Tb[1])

    def _sample_exploitation(self, temperature: float) -> int:
        """
        Sample point around best regions using temperature-controlled distribution.

        Args:
            temperature: Controls width of sampling distribution

        Returns:
            T value sampled around best regions
        """
        if not self.evaluated_points:
            return self._sample_exploration()

        # Get top k best points
        k = min(5, len(self.evaluated_points))
        sorted_points = sorted(self.evaluated_points, key=lambda x: x[1])
        top_k = sorted_points[:k]

        # Choose one of top k points
        center_T, _ = random.choice(top_k)

        # Sample with Gaussian noise scaled by temperature
        search_radius = (self.Tb[1] - self.Tb[0]) * temperature
        T_new = int(round(np.random.normal(center_T, search_radius)))

        # Clamp to bounds
        T_new = clamp(T_new, self.Tb[0], self.Tb[1])

        return T_new

    def _save_evaluated_points(self, iteration: int, outdir: str):
        """
        Save all evaluated points to CSV.

        Args:
            iteration: Current iteration
            outdir: Output directory
        """
        points_file = os.path.join(outdir, f"evaluated_points_iter_{iteration}.csv")
        with open(points_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['T', 'fitness'])
            for T, fitness in self.evaluated_points:
                writer.writerow([T, fitness])

        self.logger.debug(f"Evaluated points saved to {points_file}")

    def create_initial_population(self, n_initial: int) -> List[int]:
        """
        Create initial set of points using random sampling.

        Args:
            n_initial: Number of initial points

        Returns:
            List of T values
        """
        self.logger.info(f"Creating initial sample of size {n_initial}")

        # Use Latin Hypercube Sampling for better coverage
        points = []
        for i in range(n_initial):
            # Divide range into n_initial bins and sample from each
            bin_size = (self.Tb[1] - self.Tb[0]) / n_initial
            T = int(self.Tb[0] + (i + random.random()) * bin_size)
            T = clamp(T, self.Tb[0], self.Tb[1])
            points.append(T)

        # Shuffle to randomize evaluation order
        random.shuffle(points)

        self.logger.info(
            f"Initial T values: min={min(points)}, max={max(points)}, "
            f"mean={np.mean(points):.1f}, median={np.median(points):.1f}"
        )

        return points

    def run(
        self,
        iterations: int = 200,
        n_initial: int = 10,
        writer: Optional[SummaryWriter] = None,
        outdir: Optional[str] = None,
        initial_population: Optional[List] = None
    ) -> Dict:
        """
        Run PABBO optimization.

        Args:
            iterations: Number of iterations
            n_initial: Number of initial random samples
            writer: TensorBoard writer
            outdir: Output directory
            initial_population: Optional initial population (list of T values)

        Returns:
            Optimization results dictionary
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting PABBO optimization")
        self.logger.info(f"Iterations: {iterations}, Initial samples: {n_initial}")
        self.logger.info(f"T bounds: {self.Tb}, alpha=1/T, eta=1/T")
        self.logger.info("="*80)

        # Create or use provided initial points
        if initial_population is not None:
            self.logger.info("Using provided initial population")
            # Use ALL provided points (for fair comparison)
            initial_points = initial_population
        else:
            initial_points = self.create_initial_population(n_initial)

        # Initialize tracking
        history = []
        t0 = time.perf_counter()
        no_improvement_count = 0
        prev_perplexity = float('inf')
        temperature = 1.0
        best_T = None
        best_fitness = float('inf')

        # Evaluate initial points
        self.logger.info("Evaluating initial samples...")
        for T in initial_points:
            fitness = self._evaluate_point(T)
            if fitness < best_fitness:
                best_fitness = fitness
                best_T = T
                self.logger.info(f"Initial: New best found! T={T}, fitness={fitness:.4f}")

        # Get initial metrics
        Tb, ab, eb = self.decode([best_T])
        best_metrics = self.eval_func(Tb, ab, eb)
        prev_perplexity = best_metrics['perplexity']

        self.logger.info(
            f"Initial best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}, "
            f"perplexity={best_metrics['perplexity']:.4f}"
        )

        if outdir:
            ensure_dir(outdir)
            self._save_evaluated_points(0, outdir)

        # Add initial point to history (iteration -1)
        all_fitness = [f for _, f in self.evaluated_points]
        history.append({
            "iter": -1,
            "best_perplexity": best_metrics['perplexity'],
            "current_T": best_T,
            "current_fitness": best_fitness,
            "sample_type": "initial",
            "pop_mean": float(np.mean(all_fitness)),
            "pop_std": float(np.std(all_fitness)),
            "pop_min": float(np.min(all_fitness)),
            "pop_max": float(np.max(all_fitness)),
            "T_best": Tb,
            "alpha_best": ab,
            "eta_best": eb,
            "temperature": 1.0,
            "step_time": 0.0,
            "cum_time": 0.0,
            "no_improvement_count": 0,
            "relative_change_pct": 0.0
        })

        # Main PABBO loop
        for iteration in range(iterations):
            iter_start = time.perf_counter()

            # Decide: exploration or exploitation
            if random.random() < self.exploration_rate:
                # Exploration: random sample
                T_new = self._sample_exploration()
                sample_type = "exploration"
            else:
                # Exploitation: sample around best regions
                T_new = self._sample_exploitation(temperature)
                sample_type = "exploitation"

            # Evaluate new point
            fitness = self._evaluate_point(T_new)

            # Update best
            if fitness < best_fitness:
                best_fitness = fitness
                best_T = T_new
                self.logger.info(
                    f"Iter {iteration+1}: New best found! T={T_new}, "
                    f"fitness={fitness:.4f} ({sample_type})"
                )

            # Decay temperature
            temperature = max(self.min_temperature, temperature * self.temperature_decay)

            # Get metrics
            Tb, ab, eb = self.decode([best_T])
            best_metrics = self.eval_func(Tb, ab, eb)

            # Timing
            iter_time = time.perf_counter() - iter_start
            cum_time = time.perf_counter() - t0

            # Statistics over all evaluated points
            all_fitness = [f for _, f in self.evaluated_points]
            pop_mean = float(np.mean(all_fitness))
            pop_std = float(np.std(all_fitness))
            pop_min = float(np.min(all_fitness))
            pop_max = float(np.max(all_fitness))

            # Early stopping check
            current_perplexity = best_metrics['perplexity']
            relative_change = abs(current_perplexity - prev_perplexity) / prev_perplexity if prev_perplexity > 0 else float('inf')

            if relative_change <= self.early_stop_eps_pct:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            prev_perplexity = current_perplexity

            # Logging
            self.logger.info(
                f"Iter {iteration+1}/{iterations} | Best: T={Tb}, perplexity={best_metrics['perplexity']:.4f} | "
                f"Current: T={T_new}, fitness={fitness:.4f} ({sample_type}) | "
                f"Temp: {temperature:.3f} | No improvement: {no_improvement_count}/{self.max_no_improvement}"
            )

            # TensorBoard logging
            if writer:
                iter_num = iteration + 1
                writer.add_scalar("PABBO/Best/perplexity", best_metrics['perplexity'], iter_num)
                writer.add_scalar("PABBO/Best/T", Tb, iter_num)
                writer.add_scalar("PABBO/Best/alpha", ab, iter_num)
                writer.add_scalar("PABBO/Best/eta", eb, iter_num)
                writer.add_scalar("PABBO/Current/fitness", fitness, iter_num)
                writer.add_scalar("PABBO/Current/T", T_new, iter_num)
                writer.add_scalar("PABBO/Statistics/mean_fitness", pop_mean, iter_num)
                writer.add_scalar("PABBO/Statistics/std_fitness", pop_std, iter_num)
                writer.add_scalar("PABBO/Statistics/min_fitness", pop_min, iter_num)
                writer.add_scalar("PABBO/Statistics/max_fitness", pop_max, iter_num)
                writer.add_scalar("PABBO/Parameters/temperature", temperature, iter_num)
                writer.add_scalar("PABBO/Time/step_time", iter_time, iter_num)
                writer.add_scalar("PABBO/Time/cumulative", cum_time, iter_num)
                writer.add_scalar("PABBO/EarlyStopping/no_improvement_count", no_improvement_count, iter_num)
                writer.add_scalar("PABBO/EarlyStopping/relative_change_pct", relative_change * 100, iter_num)

            # History
            history.append({
                "iter": iteration,
                "best_perplexity": best_metrics['perplexity'],
                "current_T": T_new,
                "current_fitness": fitness,
                "sample_type": sample_type,
                "pop_mean": pop_mean,
                "pop_std": pop_std,
                "pop_min": pop_min,
                "pop_max": pop_max,
                "T_best": Tb,
                "alpha_best": ab,
                "eta_best": eb,
                "temperature": temperature,
                "step_time": iter_time,
                "cum_time": cum_time,
                "no_improvement_count": no_improvement_count,
                "relative_change_pct": relative_change * 100
            })

            # Save evaluated points periodically
            if outdir and (iteration + 1) % 10 == 0:
                self._save_evaluated_points(iteration + 1, outdir)

            # Early stopping
            if no_improvement_count >= self.max_no_improvement:
                self.logger.info(
                    f"Early stopping: |Δ perplexity|/prev ≤ {self.early_stop_eps_pct*100:.2f}% "
                    f"for {self.max_no_improvement} iterations"
                )
                break

        # Final save
        if outdir:
            self._save_evaluated_points(iterations, outdir)

        # Final results
        final_cum_time = time.perf_counter() - t0
        self.logger.info("="*80)
        self.logger.info("PABBO Optimization Complete!")
        self.logger.info(f"Total time: {final_cum_time:.2f}s")
        self.logger.info(f"Total evaluations: {len(self.evaluated_points)}")
        self.logger.info(
            f"Final best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}, "
            f"perplexity={best_metrics['perplexity']:.4f}"
        )
        self.logger.info("="*80)

        return {
            "best": {
                "T": Tb,
                "alpha": ab,
                "eta": eb,
                **best_metrics
            },
            "history": history,
            "total_time": final_cum_time,
            "avg_step_time": float(np.mean([h["step_time"] for h in history])) if history else 0.0,
            "stopped_early": no_improvement_count >= self.max_no_improvement,
            "total_evaluations": len(self.evaluated_points),
            "algorithm": "PABBO_Simple"
        }