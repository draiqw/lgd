"""
Evolution Strategy (ES) optimizer for LDA hyperparameters.

This module implements an evolution strategy that:
- Uses (μ + λ) selection strategy
- Searches for optimal T (number of topics)
- Sets alpha and eta to 1/T
- Trains on validation set
- Optimizes for minimal perplexity
"""

import time
import os
import csv
import random
from typing import Optional, List, Dict

import numpy as np
from tensorboardX import SummaryWriter
from deap import base, creator, tools

from test_function_opt.utils import BaseOptimizer, clamp, ensure_dir


# Create DEAP fitness and individual classes (only once)
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
except RuntimeError:
    pass  # Already created

try:
    creator.create("Individual", list, fitness=creator.FitnessMin)
except RuntimeError:
    pass  # Already created


class ESOptimizer(BaseOptimizer):
    """
    Evolution Strategy (ES) optimizer for LDA hyperparameters.

    The ES uses:
    - (μ + λ) selection: keep best μ from μ parents + λ offspring
    - Integer mutation with bounded random walk
    - No crossover (standard ES approach)
    - Early stopping based on relative improvement
    """

    def __init__(
        self,
        obj,
        eval_func,
        T_bounds=(2, 1000),
        seed=42,
        mu=5,
        lmbda=10,
        dT=5,
        early_stop_eps_pct=0.01,
        max_no_improvement=5,
        logger=None
    ):
        """
        Initialize ES optimizer.

        Args:
            obj: Objective function (T, alpha, eta) -> perplexity
            eval_func: Evaluation function (T, alpha, eta) -> metrics dict
            T_bounds: Bounds for T parameter (min, max)
            seed: Random seed
            mu: Number of parents (μ)
            lmbda: Number of offspring (λ)
            dT: Maximum mutation step for T
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

        self.mu = int(mu)
        self.lmbda = int(lmbda)
        self.dT = int(dT)

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Setup DEAP toolbox
        self.toolbox = base.Toolbox()

        def create_individual():
            T = random.randint(self.Tb[0], self.Tb[1])
            return creator.Individual([T])

        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)

        self.logger.info(f"ES initialized: mu={mu}, lambda={lmbda}, dT={dT}")

    def _evaluate(self, ind):
        """
        Evaluate individual fitness.

        Args:
            ind: Individual (list with single element T)

        Returns:
            Tuple with fitness value (perplexity)
        """
        T, a, e = self.decode(ind)
        try:
            v = float(self.obj(T, a, e))
        except Exception as ex:
            self.logger.warning(f"Evaluation failed for T={T}: {ex}")
            v = float("inf")
        return (v,)

    def _mutate(self, parent):
        """
        Create offspring via mutation.

        Args:
            parent: Parent individual

        Returns:
            Mutated child individual
        """
        child = creator.Individual(parent[:])
        child[0] = int(round(child[0] + random.randint(-self.dT, self.dT)))
        child[0] = clamp(child[0], self.Tb[0], self.Tb[1])
        return child

    def _save_population(self, pop: List, step: int, outdir: str):
        """
        Save population to CSV file.

        Args:
            pop: Population (parents)
            step: Step number
            outdir: Output directory
        """
        pop_file = os.path.join(outdir, f"population_step_{step}.csv")
        with open(pop_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['individual_id', 'T', 'alpha', 'eta', 'fitness'])
            for i, ind in enumerate(pop):
                T, a, e = self.decode(ind)
                writer.writerow([i, T, a, e, ind.fitness.values[0]])

        self.logger.debug(f"Population saved to {pop_file}")

    def create_initial_population(self, mu: int) -> List:
        """
        Create initial parent population.

        Args:
            mu: Number of parents

        Returns:
            List of individuals
        """
        self.logger.info(f"Creating initial population of size {mu}")
        pop = self.toolbox.population(n=mu)

        # Log initial population
        T_values = [ind[0] for ind in pop]
        self.logger.info(
            f"Initial T values: min={min(T_values)}, max={max(T_values)}, "
            f"mean={np.mean(T_values):.1f}, median={np.median(T_values):.1f}"
        )

        return pop

    def run(
        self,
        iterations: int = 200,
        writer: Optional[SummaryWriter] = None,
        outdir: Optional[str] = None,
        initial_population: Optional[List] = None
    ) -> Dict:
        """
        Run ES optimization.

        Args:
            iterations: Number of steps/generations
            writer: TensorBoard writer
            outdir: Output directory
            initial_population: Optional initial population (list of T values)

        Returns:
            Optimization results dictionary
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting ES optimization")
        self.logger.info(f"Steps: {iterations}, mu={self.mu}, lambda={self.lmbda}")
        self.logger.info(f"T bounds: {self.Tb}, alpha=1/T, eta=1/T")
        self.logger.info("="*80)

        # Create or use provided initial population
        if initial_population is not None:
            self.logger.info("Using provided initial population")
            # Use ALL provided individuals (for fair comparison)
            parents = []
            for T_val in initial_population:
                ind = creator.Individual([T_val])
                parents.append(ind)
            # If we have more than mu, keep only mu best after evaluation
            # If we have less than mu, this is fine - we'll still have at least some
        else:
            parents = self.create_initial_population(self.mu)

        # Initialize tracking
        history = []
        t0 = time.perf_counter()
        no_improvement_count = 0
        prev_perplexity = float('inf')

        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        for ind in parents:
            ind.fitness.values = self.toolbox.evaluate(ind)

        # If we have more than mu parents, select best mu
        if len(parents) > self.mu:
            initial_count = len(parents)
            parents.sort(key=lambda x: x.fitness.values[0])
            selected_parents = parents[:self.mu]
            parents = [creator.Individual(ind[:]) for ind in selected_parents]
            for i in range(self.mu):
                parents[i].fitness.values = selected_parents[i].fitness.values
            self.logger.info(f"Selected best {self.mu} parents from {initial_count} initial individuals")

        best_sofar = min(parents, key=lambda x: x.fitness.values[0])

        # Get initial metrics
        Tb, ab, eb = self.decode(best_sofar)
        best_metrics = self.eval_func(Tb, ab, eb)
        prev_perplexity = best_metrics['perplexity']

        self.logger.info(
            f"Initial best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}, "
            f"perplexity={best_metrics['perplexity']:.4f}"
        )

        # Save initial population
        if outdir:
            ensure_dir(outdir)
            self._save_population(parents, 0, outdir)

        # Add initial point to history (iteration -1)
        vals = [ind.fitness.values[0] for ind in parents]
        history.append({
            "iter": -1,
            "best_perplexity": best_metrics['perplexity'],
            "pop_mean": float(np.mean(vals)),
            "pop_std": float(np.std(vals)),
            "pop_min": float(np.min(vals)),
            "pop_max": float(np.max(vals)),
            "T_best": Tb,
            "alpha_best": ab,
            "eta_best": eb,
            "step_time": 0.0,
            "cum_time": 0.0,
            "no_improvement_count": 0,
            "relative_change_pct": 0.0
        })

        # Main ES loop
        for s in range(iterations):
            step_start = time.perf_counter()

            # Generate offspring
            offspring = []
            for _ in range(self.lmbda):
                # Select random parent
                p = random.choice(parents)
                # Mutate to create child
                c = self._mutate(p)
                # Evaluate child
                c.fitness.values = self.toolbox.evaluate(c)
                offspring.append(c)

            self.logger.debug(f"Step {s+1}: Generated {len(offspring)} offspring")

            # (μ + λ) selection: combine and select best μ
            pool = parents + offspring
            pool.sort(key=lambda x: x.fitness.values[0])

            # Create new parent population
            parents = [creator.Individual(ind[:]) for ind in pool[:self.mu]]
            for i in range(self.mu):
                parents[i].fitness.values = pool[i].fitness.values

            # Update best
            cur_best = parents[0]
            if cur_best.fitness.values[0] < best_sofar.fitness.values[0]:
                best_sofar = cur_best
                self.logger.info(f"Step {s+1}: New best found! Fitness={cur_best.fitness.values[0]:.4f}")

            # Get metrics
            Tb, ab, eb = self.decode(best_sofar)
            best_metrics = self.eval_func(Tb, ab, eb)

            # Timing
            step_time = time.perf_counter() - step_start
            cum_time = time.perf_counter() - t0

            # Population statistics (parents only)
            vals = [ind.fitness.values[0] for ind in parents]
            pop_mean = float(np.mean(vals))
            pop_std = float(np.std(vals))
            pop_min = float(np.min(vals))
            pop_max = float(np.max(vals))

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
                f"Step {s+1}/{iterations} | Best: T={Tb}, perplexity={best_metrics['perplexity']:.4f} | "
                f"Parents: mean={pop_mean:.4f}, std={pop_std:.4f} | "
                f"Time: {step_time:.2f}s | No improvement: {no_improvement_count}/{self.max_no_improvement}"
            )

            # TensorBoard logging
            if writer:
                iter_num = s + 1
                writer.add_scalar("ES/Best/perplexity", best_metrics['perplexity'], iter_num)
                writer.add_scalar("ES/Best/T", Tb, iter_num)
                writer.add_scalar("ES/Best/alpha", ab, iter_num)
                writer.add_scalar("ES/Best/eta", eb, iter_num)
                writer.add_scalar("ES/Population/mean_fitness", pop_mean, iter_num)
                writer.add_scalar("ES/Population/std_fitness", pop_std, iter_num)
                writer.add_scalar("ES/Population/min_fitness", pop_min, iter_num)
                writer.add_scalar("ES/Population/max_fitness", pop_max, iter_num)
                writer.add_scalar("ES/Time/step_time", step_time, iter_num)
                writer.add_scalar("ES/Time/cumulative", cum_time, iter_num)
                writer.add_scalar("ES/EarlyStopping/no_improvement_count", no_improvement_count, iter_num)
                writer.add_scalar("ES/EarlyStopping/relative_change_pct", relative_change * 100, iter_num)

            # History
            history.append({
                "iter": s,
                "best_perplexity": best_metrics['perplexity'],
                "pop_mean": pop_mean,
                "pop_std": pop_std,
                "pop_min": pop_min,
                "pop_max": pop_max,
                "T_best": Tb,
                "alpha_best": ab,
                "eta_best": eb,
                "step_time": step_time,
                "cum_time": cum_time,
                "no_improvement_count": no_improvement_count,
                "relative_change_pct": relative_change * 100
            })

            # Save population
            if outdir:
                self._save_population(parents, s + 1, outdir)

            # Early stopping
            if no_improvement_count >= self.max_no_improvement:
                self.logger.info(
                    f"Early stopping: |delta perplexity|/prev ≤ {self.early_stop_eps_pct*100:.2f}% "
                    f"for {self.max_no_improvement} steps"
                )
                break

        # Final results
        final_cum_time = time.perf_counter() - t0
        self.logger.info("="*80)
        self.logger.info("ES Optimization Complete!")
        self.logger.info(f"Total time: {final_cum_time:.2f}s")
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
            "algorithm": "ES"
        }
