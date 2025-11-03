"""
Genetic Algorithm (GA) optimizer for LDA hyperparameters.

This module implements a genetic algorithm that:
- Searches for optimal T (number of topics)
- Sets alpha and eta to 1/T
- Trains on validation set
- Optimizes for minimal perplexity
"""

import time
import os
import csv
import copy
import random
from typing import Optional, List, Dict, Tuple

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


class GAOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for LDA hyperparameters.

    The GA uses:
    - Binary crossover at bit level for T parameter
    - Integer mutation with bounded random walk
    - Tournament selection
    - Elitism (best individuals always survive)
    - Early stopping based on relative improvement
    """

    def __init__(
        self,
        obj,
        eval_func,
        T_bounds=(2, 1000),
        seed=42,
        cxpb=0.9,
        mutpb=0.2,
        tournsize=3,
        elite=5,
        dT=5,
        early_stop_eps_pct=0.01,
        max_no_improvement=5,
        logger=None
    ):
        """
        Initialize GA optimizer.

        Args:
            obj: Objective function (T, alpha, eta) -> perplexity
            eval_func: Evaluation function (T, alpha, eta) -> metrics dict
            T_bounds: Bounds for T parameter (min, max)
            seed: Random seed
            cxpb: Crossover probability
            mutpb: Mutation probability
            tournsize: Tournament size for selection
            elite: Number of elite individuals to preserve
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

        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tournsize = tournsize
        self.elite = elite
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
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        self.toolbox.register("clone", copy.deepcopy)

        self.logger.info(f"GA initialized: cxpb={cxpb}, mutpb={mutpb}, elite={elite}, dT={dT}")

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

    def _crossover(self, ind1, ind2):
        """
        Binary crossover - split T in half at bit level.

        Args:
            ind1: First parent
            ind2: Second parent

        Returns:
            Modified ind1, ind2
        """
        T1 = int(ind1[0])
        T2 = int(ind2[0])

        num_bits = 10
        crossover_point = num_bits // 2

        # Create mask for crossover
        mask = ((1 << crossover_point) - 1) << (num_bits - crossover_point)

        # Perform crossover
        child1 = (T1 & mask) | (T2 & ~mask)
        child2 = (T2 & mask) | (T1 & ~mask)

        # Clamp to bounds
        ind1[0] = clamp(child1, self.Tb[0], self.Tb[1])
        ind2[0] = clamp(child2, self.Tb[0], self.Tb[1])

        return ind1, ind2

    def _mutate(self, ind):
        """
        Mutation for T: bounded random walk.

        Args:
            ind: Individual to mutate

        Returns:
            Tuple with mutated individual
        """
        ind[0] = clamp(
            int(round(ind[0] + random.randint(-self.dT, self.dT))),
            self.Tb[0],
            self.Tb[1]
        )
        return (ind,)

    def _save_population(self, pop: List, gen: int, outdir: str):
        """
        Save population to CSV file.

        Args:
            pop: Population
            gen: Generation number
            outdir: Output directory
        """
        pop_file = os.path.join(outdir, f"population_gen_{gen}.csv")
        with open(pop_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['individual_id', 'T', 'alpha', 'eta', 'fitness'])
            for i, ind in enumerate(pop):
                T, a, e = self.decode(ind)
                writer.writerow([i, T, a, e, ind.fitness.values[0]])

        self.logger.debug(f"Population saved to {pop_file}")

    def create_initial_population(self, pop_size: int) -> List:
        """
        Create initial population.

        Args:
            pop_size: Population size

        Returns:
            List of individuals
        """
        self.logger.info(f"Creating initial population of size {pop_size}")
        pop = self.toolbox.population(n=pop_size)

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
        pop_size: int = 10,
        writer: Optional[SummaryWriter] = None,
        outdir: Optional[str] = None,
        initial_population: Optional[List] = None
    ) -> Dict:
        """
        Run GA optimization.

        Args:
            iterations: Number of generations
            pop_size: Population size
            writer: TensorBoard writer
            outdir: Output directory
            initial_population: Optional initial population (list of T values)

        Returns:
            Optimization results dictionary
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting GA optimization")
        self.logger.info(f"Generations: {iterations}, Population size: {pop_size}")
        self.logger.info(f"T bounds: {self.Tb}, alpha=1/T, eta=1/T")
        self.logger.info("="*80)

        # Create or use provided initial population
        if initial_population is not None:
            self.logger.info("Using provided initial population")
            pop = []
            for T_val in initial_population:
                ind = creator.Individual([T_val])
                pop.append(ind)
        else:
            pop = self.create_initial_population(pop_size)

        # Initialize tracking
        hall = tools.HallOfFame(maxsize=self.elite)
        history = []
        t0 = time.perf_counter()
        no_improvement_count = 0
        prev_perplexity = float('inf')

        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)
        hall.update(pop)
        best_sofar = min(pop, key=lambda x: x.fitness.values[0])

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
            self._save_population(pop, 0, outdir)

        # Add initial point to history (iteration -1)
        vals = [ind.fitness.values[0] for ind in pop]
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

        # Main GA loop
        for g in range(iterations):
            gs = time.perf_counter()

            # Selection: keep elite + select offspring
            elites = tools.selBest(pop, self.elite)
            offspring = self.toolbox.select(pop, len(pop) - self.elite)
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.cxpb:
                    self._crossover(offspring[i], offspring[i + 1])

            # Mutation
            for i in range(len(offspring)):
                if random.random() < self.mutpb:
                    self._mutate(offspring[i])
                del offspring[i].fitness.values

            # Evaluate invalid individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            self.logger.debug(f"Gen {g+1}: Evaluating {len(invalid)} new individuals")

            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Create new population
            pop = elites + offspring
            hall.update(pop)

            # Update best
            cur_best = min(pop, key=lambda x: x.fitness.values[0])
            if cur_best.fitness.values[0] < best_sofar.fitness.values[0]:
                best_sofar = cur_best
                self.logger.info(f"Gen {g+1}: New best found! Fitness={cur_best.fitness.values[0]:.4f}")

            # Get metrics
            Tb, ab, eb = self.decode(best_sofar)
            best_metrics = self.eval_func(Tb, ab, eb)

            # Timing
            gen_time = time.perf_counter() - gs
            cum_time = time.perf_counter() - t0

            # Population statistics
            vals = [ind.fitness.values[0] for ind in pop]
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
                f"Gen {g+1}/{iterations} | Best: T={Tb}, perplexity={best_metrics['perplexity']:.4f} | "
                f"Pop: mean={pop_mean:.4f}, std={pop_std:.4f} | "
                f"Time: {gen_time:.2f}s | No improvement: {no_improvement_count}/{self.max_no_improvement}"
            )

            # TensorBoard logging
            if writer:
                iter_num = g + 1
                writer.add_scalar("GA/Best/perplexity", best_metrics['perplexity'], iter_num)
                writer.add_scalar("GA/Best/T", Tb, iter_num)
                writer.add_scalar("GA/Best/alpha", ab, iter_num)
                writer.add_scalar("GA/Best/eta", eb, iter_num)
                writer.add_scalar("GA/Population/mean_fitness", pop_mean, iter_num)
                writer.add_scalar("GA/Population/std_fitness", pop_std, iter_num)
                writer.add_scalar("GA/Population/min_fitness", pop_min, iter_num)
                writer.add_scalar("GA/Population/max_fitness", pop_max, iter_num)
                writer.add_scalar("GA/Time/step_time", gen_time, iter_num)
                writer.add_scalar("GA/Time/cumulative", cum_time, iter_num)
                writer.add_scalar("GA/EarlyStopping/no_improvement_count", no_improvement_count, iter_num)
                writer.add_scalar("GA/EarlyStopping/relative_change_pct", relative_change * 100, iter_num)

            # History
            history.append({
                "iter": g,
                "best_perplexity": best_metrics['perplexity'],
                "pop_mean": pop_mean,
                "pop_std": pop_std,
                "pop_min": pop_min,
                "pop_max": pop_max,
                "T_best": Tb,
                "alpha_best": ab,
                "eta_best": eb,
                "step_time": gen_time,
                "cum_time": cum_time,
                "no_improvement_count": no_improvement_count,
                "relative_change_pct": relative_change * 100
            })

            # Save population
            if outdir:
                self._save_population(pop, g + 1, outdir)

            # Early stopping
            if no_improvement_count >= self.max_no_improvement:
                self.logger.info(
                    f"Early stopping: |Δ perplexity|/prev ≤ {self.early_stop_eps_pct*100:.2f}% "
                    f"for {self.max_no_improvement} iterations"
                )
                break

        # Final results
        final_cum_time = time.perf_counter() - t0
        self.logger.info("="*80)
        self.logger.info("GA Optimization Complete!")
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
            "algorithm": "GA"
        }