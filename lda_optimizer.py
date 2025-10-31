import time
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
import random
from deap import base, creator, tools
import os
import csv
import json
from tensorboardX import SummaryWriter
from functools import wraps
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMING] {func.__name__} completed in {end - start:.2f}s")
        return result
    return wrapper


def load_bow_data(val_path: str):
    """Load validation data only"""
    Xval = sp.load_npz(val_path).tocsr(copy=False)
    return Xval


try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
except Exception:
    pass
try:
    creator.create("Individual", list, fitness=creator.FitnessMin)
except Exception:
    pass


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class GAOptimizer:
    """
    Genetic Algorithm optimizer for LDA hyperparameters.
    Searches only for T parameter, alpha and eta are set to 1/T.
    Trains on validation set and optimizes for minimal perplexity.
    """
    def __init__(
        self,
        obj,
        eval_func,
        T_bounds=(2, 1000),
        *,
        seed=42,
        cxpb=0.9,
        mutpb=0.2,
        tournsize=3,
        elite=5,
        dT=5,
        early_stop_eps_pct=0.01,
        max_no_improvement=3
    ):
        self.obj = obj
        self.eval_func = eval_func
        self.Tb = T_bounds
        self.seed = int(seed)
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tournsize = tournsize
        self.elite = elite
        self.dT = int(dT)
        self.early_stop_eps_pct = float(early_stop_eps_pct)
        self.max_no_improvement = int(max_no_improvement)
        self.toolbox = base.Toolbox()

        random.seed(self.seed)
        np.random.seed(self.seed)

        def create_individual():
            T = random.randint(self.Tb[0], self.Tb[1])
            return creator.Individual([T])

        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        self.toolbox.register("clone", copy.deepcopy)

    def _decode(self, ind):
        """Decode individual: T -> (T, alpha=1/T, eta=1/T)"""
        T = int(round(ind[0]))
        T = _clamp(T, self.Tb[0], self.Tb[1])
        alpha = 1.0 / T
        eta = 1.0 / T
        return T, float(alpha), float(eta)

    def _evaluate(self, ind):
        T, a, e = self._decode(ind)
        try:
            v = float(self.obj(T, a, e))
        except Exception:
            v = float("inf")
        return (v,)

    def _cx(self, ind1, ind2):
        """Binary crossover - split T in half at bit level"""
        T1 = int(ind1[0])
        T2 = int(ind2[0])

        num_bits = 10
        crossover_point = num_bits // 2

        mask = ((1 << crossover_point) - 1) << (num_bits - crossover_point)

        child1 = (T1 & mask) | (T2 & ~mask)

        child2 = (T2 & mask) | (T1 & ~mask)

        ind1[0] = _clamp(child1, self.Tb[0], self.Tb[1])
        ind2[0] = _clamp(child2, self.Tb[0], self.Tb[1])

        return ind1, ind2

    def _mut(self, ind):
        """Mutation for T only"""
        ind[0] = _clamp(
            int(round(ind[0] + random.randint(-self.dT, self.dT))),
            self.Tb[0],
            self.Tb[1]
        )
        return (ind,)

    def _save_population(self, pop, gen, outdir):
        pop_file = os.path.join(outdir, f"population_gen_{gen}.csv")
        with open(pop_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['individual_id', 'T', 'alpha', 'eta', 'fitness'])
            for i, ind in enumerate(pop):
                T, a, e = self._decode(ind)
                writer.writerow([i, T, a, e, ind.fitness.values[0]])

    def run(self, gens=200, pop_size=10, writer=None, outdir=None):
        print(f"[GA] Starting optimization: {gens} generations, population size {pop_size}")
        print(f"[GA] T bounds: {self.Tb}, alpha=1/T, eta=1/T")
        pop = self.toolbox.population(n=pop_size)
        hall = tools.HallOfFame(maxsize=self.elite)
        history = []
        t0 = time.perf_counter()
        no_improvement_count = 0
        prev_perplexity = float('inf')

        print(f"[GA] Evaluating initial population...")
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)
        hall.update(pop)
        best_sofar = min(pop, key=lambda x: x.fitness.values[0])

        Tb, ab, eb = self._decode(best_sofar)
        best_metrics = self.eval_func(Tb, ab, eb)
        prev_perplexity = best_metrics['perplexity']

        print(f"[GA] Initial best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}")
        print(f"[GA]   perplexity={best_metrics['perplexity']:.4f}")

        if outdir:
            self._save_population(pop, 0, outdir)

        for g in range(gens):
            gs = time.perf_counter()
            elites = tools.selBest(pop, self.elite)
            offspring = self.toolbox.select(pop, len(pop) - self.elite)
            offspring = list(map(self.toolbox.clone, offspring))
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.cxpb:
                    self._cx(offspring[i], offspring[i + 1])
            for i in range(len(offspring)):
                if random.random() < self.mutpb:
                    self._mut(offspring[i])
                del offspring[i].fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)
            pop = elites + offspring
            hall.update(pop)

            cur_best = min(pop, key=lambda x: x.fitness.values[0])
            if cur_best.fitness.values[0] < best_sofar.fitness.values[0]:
                best_sofar = cur_best

            Tb, ab, eb = self._decode(best_sofar)
            best_metrics = self.eval_func(Tb, ab, eb)

            gen_time = time.perf_counter() - gs
            cum_time = time.perf_counter() - t0

            vals = [ind.fitness.values[0] for ind in pop]
            pop_mean = float(np.mean(vals))
            pop_std = float(np.std(vals))
            pop_min = float(np.min(vals))
            pop_max = float(np.max(vals))

            current_perplexity = best_metrics['perplexity']
            relative_change = abs(current_perplexity - prev_perplexity) / prev_perplexity if prev_perplexity > 0 else float('inf')
            if relative_change <= self.early_stop_eps_pct:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            prev_perplexity = current_perplexity

            print(f"[GA] Gen {g+1}/{gens} | Best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}")
            print(f"     perplexity={best_metrics['perplexity']:.4f}")
            print(f"     Pop: mean={pop_mean:.4f}, std={pop_std:.4f}, min={pop_min:.4f}, max={pop_max:.4f}")
            print(f"     Time: {gen_time:.2f}s (total: {cum_time:.2f}s) | No improvement: {no_improvement_count}/{self.max_no_improvement}")

            if writer:
                iter_num = g + 1
                writer.add_scalar("Best/perplexity", best_metrics['perplexity'], iter_num)
                writer.add_scalar("Best/T", Tb, iter_num)
                writer.add_scalar("Best/alpha", ab, iter_num)
                writer.add_scalar("Best/eta", eb, iter_num)
                writer.add_scalar("Population/mean_fitness", pop_mean, iter_num)
                writer.add_scalar("Population/std_fitness", pop_std, iter_num)
                writer.add_scalar("Population/min_fitness", pop_min, iter_num)
                writer.add_scalar("Population/max_fitness", pop_max, iter_num)
                writer.add_scalar("Time/step_time", gen_time, iter_num)
                writer.add_scalar("Time/cumulative", cum_time, iter_num)

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

            if outdir:
                self._save_population(pop, g + 1, outdir)

            if no_improvement_count >= self.max_no_improvement:
                print(f"[GA] Early stopping: |Δ perplexity|/prev ≤ {self.early_stop_eps_pct*100:.2f}% for {self.max_no_improvement} iterations")
                break

        final_cum_time = time.perf_counter() - t0
        print(f"[GA] Optimization complete! Total time: {final_cum_time:.2f}s")
        print(f"[GA] Final best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}")
        print(f"[GA]   perplexity={best_metrics['perplexity']:.4f}")

        best = self._decode(best_sofar)
        return {
            "best": {
                "T": best[0],
                "alpha": best[1],
                "eta": best[2],
                **best_metrics
            },
            "history": history,
            "total_time": final_cum_time,
            "avg_step_time": float(np.mean([h["step_time"] for h in history])) if history else 0.0,
            "stopped_early": no_improvement_count >= self.max_no_improvement
        }


class ESOptimizer:
    """
    Evolution Strategy optimizer for LDA hyperparameters.
    Searches only for T parameter, alpha and eta are set to 1/T.
    Trains on validation set and optimizes for minimal perplexity.
    """
    def __init__(
        self,
        obj,
        eval_func,
        T_bounds=(2, 1000),
        *,
        seed=42,
        mu=5,
        lmbda=10,
        dT=5,
        early_stop_eps_pct=0.01,
        max_no_improvement=3
    ):
        self.obj = obj
        self.eval_func = eval_func
        self.Tb = T_bounds
        self.seed = int(seed)
        self.mu = int(mu)
        self.lmbda = int(lmbda)
        self.dT = int(dT)
        self.early_stop_eps_pct = float(early_stop_eps_pct)
        self.max_no_improvement = int(max_no_improvement)
        self.toolbox = base.Toolbox()

        random.seed(self.seed)
        np.random.seed(self.seed)

        def create_individual():
            T = random.randint(self.Tb[0], self.Tb[1])
            return creator.Individual([T])

        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)

    def _clamp(self, ind):
        """Clamp T to bounds"""
        ind[0] = _clamp(int(round(ind[0])), self.Tb[0], self.Tb[1])

    def _decode(self, ind):
        """Decode individual: T -> (T, alpha=1/T, eta=1/T)"""
        T = int(round(ind[0]))
        T = _clamp(T, self.Tb[0], self.Tb[1])
        alpha = 1.0 / T
        eta = 1.0 / T
        return T, float(alpha), float(eta)

    def _evaluate(self, ind):
        T, a, e = self._decode(ind)
        try:
            v = float(self.obj(T, a, e))
        except Exception:
            v = float("inf")
        return (v,)

    def _mut(self, parent):
        """Mutation for T only"""
        child = creator.Individual(parent[:])
        child[0] = int(round(child[0] + random.randint(-self.dT, self.dT)))
        self._clamp(child)
        return child

    def _save_population(self, pop, step, outdir):
        pop_file = os.path.join(outdir, f"population_step_{step}.csv")
        with open(pop_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['individual_id', 'T', 'alpha', 'eta', 'fitness'])
            for i, ind in enumerate(pop):
                T, a, e = self._decode(ind)
                writer.writerow([i, T, a, e, ind.fitness.values[0]])

    def run(self, steps=200, writer=None, outdir=None):
        print(f"[ES] Starting optimization: {steps} steps, mu={self.mu}, lambda={self.lmbda}")
        print(f"[ES] T bounds: {self.Tb}, alpha=1/T, eta=1/T")
        parents = self.toolbox.population(n=self.mu)
        history = []
        t0 = time.perf_counter()
        no_improvement_count = 0
        prev_perplexity = float('inf')

        print(f"[ES] Evaluating initial population...")
        for ind in parents:
            ind.fitness.values = self.toolbox.evaluate(ind)
        best_sofar = min(parents, key=lambda x: x.fitness.values[0])

        Tb, ab, eb = self._decode(best_sofar)
        best_metrics = self.eval_func(Tb, ab, eb)
        prev_perplexity = best_metrics['perplexity']

        print(f"[ES] Initial best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}")
        print(f"[ES]   perplexity={best_metrics['perplexity']:.4f}")

        if outdir:
            self._save_population(parents, 0, outdir)

        for s in range(steps):
            step_start = time.perf_counter()
            off = []
            for _ in range(self.lmbda):
                p = random.choice(parents)
                c = self._mut(p)
                c.fitness.values = self.toolbox.evaluate(c)
                off.append(c)

            pool = parents + off
            pool.sort(key=lambda x: x.fitness.values[0])
            parents = [creator.Individual(ind[:]) for ind in pool[:self.mu]]
            for i in range(self.mu):
                parents[i].fitness.values = pool[i].fitness.values

            cur_best = parents[0]
            if cur_best.fitness.values[0] < best_sofar.fitness.values[0]:
                best_sofar = cur_best

            Tb, ab, eb = self._decode(best_sofar)
            best_metrics = self.eval_func(Tb, ab, eb)

            step_time = time.perf_counter() - step_start
            cum_time = time.perf_counter() - t0

            vals = [ind.fitness.values[0] for ind in parents]
            pop_mean = float(np.mean(vals))
            pop_std = float(np.std(vals))
            pop_min = float(np.min(vals))
            pop_max = float(np.max(vals))

            current_perplexity = best_metrics['perplexity']
            relative_change = abs(current_perplexity - prev_perplexity) / prev_perplexity if prev_perplexity > 0 else float('inf')
            if relative_change <= self.early_stop_eps_pct:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            prev_perplexity = current_perplexity

            print(f"[ES] Step {s+1}/{steps} | Best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}")
            print(f"     perplexity={best_metrics['perplexity']:.4f}")
            print(f"     Parents: mean={pop_mean:.4f}, std={pop_std:.4f}, min={pop_min:.4f}, max={pop_max:.4f}")
            print(f"     Time: {step_time:.2f}s (total: {cum_time:.2f}s) | No improvement: {no_improvement_count}/{self.max_no_improvement}")

            if writer:
                iter_num = s + 1
                writer.add_scalar("Best/perplexity", best_metrics['perplexity'], iter_num)
                writer.add_scalar("Best/T", Tb, iter_num)
                writer.add_scalar("Best/alpha", ab, iter_num)
                writer.add_scalar("Best/eta", eb, iter_num)
                writer.add_scalar("Population/mean_fitness", pop_mean, iter_num)
                writer.add_scalar("Population/std_fitness", pop_std, iter_num)
                writer.add_scalar("Population/min_fitness", pop_min, iter_num)
                writer.add_scalar("Population/max_fitness", pop_max, iter_num)
                writer.add_scalar("Time/step_time", step_time, iter_num)
                writer.add_scalar("Time/cumulative", cum_time, iter_num)

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

            if outdir:
                self._save_population(parents, s + 1, outdir)

            if no_improvement_count >= self.max_no_improvement:
                print(f"[ES] Early stopping: |Δ perplexity|/prev ≤ {self.early_stop_eps_pct*100:.2f}% for {self.max_no_improvement} steps")
                break

        final_cum_time = time.perf_counter() - t0
        print(f"[ES] Optimization complete! Total time: {final_cum_time:.2f}s")
        print(f"[ES] Final best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}")
        print(f"[ES]   perplexity={best_metrics['perplexity']:.4f}")

        best = self._decode(best_sofar)
        return {
            "best": {
                "T": best[0],
                "alpha": best[1],
                "eta": best[2],
                **best_metrics
            },
            "history": history,
            "total_time": final_cum_time,
            "avg_step_time": float(np.mean([h["step_time"] for h in history])) if history else 0.0,
            "stopped_early": no_improvement_count >= self.max_no_improvement
        }


EVAL_CACHE = {}

def _fit_eval_on_val(
        T,
        alpha,
        eta,
        Xval,
        seed=42,
        max_iter=100,
        batch_size=2048,
        learning_method="online"
):
    """
    Train LDA on validation set and evaluate perplexity on the same set.
    This is the correct approach for hyperparameter optimization.
    """
    key = (int(T), float(alpha), float(eta), int(seed), int(max_iter), int(batch_size), learning_method, id(Xval))
    if key in EVAL_CACHE:
        return EVAL_CACHE[key]

    print(f"[LDA] Training on validation set: T={T}, alpha={alpha:.6f}, eta={eta:.6f}")
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
    print(f"[LDA] Training completed in {fit_time:.2f}s (n_iter={getattr(lda, 'n_iter_', 'N/A')})")

    t1 = time.perf_counter()
    perplexity = float(lda.perplexity(Xval))
    eval_time = time.perf_counter() - t1
    print(f"[LDA] Evaluation completed in {eval_time:.2f}s | Perplexity: {perplexity:.4f}")

    res = {
        "T": int(T),
        "alpha": float(alpha),
        "eta": float(eta),
        "perplexity": perplexity,
        "fit_time": fit_time,
        "eval_time": eval_time,
        "n_iter_lda": getattr(lda, "n_iter_", None)
    }
    EVAL_CACHE[key] = res
    return res


def make_objective(
        Xval,
        seed=42,
        max_iter=100,
        batch_size=2048,
        learning_method="online"
):
    def objective(T, a, e):
        r = _fit_eval_on_val(
            T,
            a,
            e,
            Xval,
            seed=seed,
            max_iter=max_iter,
            batch_size=batch_size,
            learning_method=learning_method
        )
        return r["perplexity"]
    return objective


def make_eval_func(Xval, seed=42, max_iter=100, batch_size=2048, learning_method="online"):
    def eval_func(T, a, e):
        return _fit_eval_on_val(T, a, e, Xval, seed=seed, max_iter=max_iter, batch_size=batch_size, learning_method=learning_method)
    return eval_func


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _write_history_csv(history_rows, path):
    if not history_rows:
        return
    fields = list(history_rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in history_rows:
            w.writerow(row)


def _plot_series(xs, ys, xlabel, ylabel, title, path):
    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@timing_decorator
def run_single_optimization(
    algorithm_name,
    optimizer_class,
    Xval,
    outdir,
    dataset_name,
    gens_or_steps=200,
    pop_size=10,
    seed=42,
    max_iter=100,
    batch_size=2048,
    learning_method="online",
    **optimizer_kwargs
):
    """Run single optimization (GA or ES) on validation set"""
    _ensure_dir(outdir)
    writer = SummaryWriter(log_dir=os.path.join(outdir, "tensorboard"))
    print(f"\n{'='*80}")
    print(f"[OPTIMIZATION] Running {algorithm_name} on {dataset_name}")
    print(f"[OPTIMIZATION] Training on VALIDATION set for hyperparameter tuning")
    print(f"[OPTIMIZATION] Output directory: {outdir}")
    print(f"{'='*80}\n")

    obj = make_objective(
        Xval,
        seed=seed,
        max_iter=max_iter,
        batch_size=batch_size,
        learning_method=learning_method
    )
    eval_func = make_eval_func(
        Xval,
        seed=seed,
        max_iter=max_iter,
        batch_size=batch_size,
        learning_method=learning_method
    )
    optimizer = optimizer_class(
        obj,
        eval_func,
        seed=seed,
        **optimizer_kwargs
    )

    if algorithm_name == "GA":
        res = optimizer.run(
            gens=gens_or_steps,
            pop_size=pop_size,
            writer=writer,
            outdir=outdir
        )
    else:
        res = optimizer.run(
            steps=gens_or_steps,
            writer=writer,
            outdir=outdir
        )

    writer.close()
    print(f"\n[TensorBoard] Logs saved to: {os.path.join(outdir, 'tensorboard')}")

    _write_history_csv(res["history"], os.path.join(outdir, "history.csv"))

    if res["history"]:
        xs = [h["iter"] for h in res["history"]]
        ys_perplexity = [h["best_perplexity"] for h in res["history"]]
        ys_T = [h["T_best"] for h in res["history"]]
        ys_step = [h["step_time"] for h in res["history"]]

        _plot_series(
            xs,
            ys_perplexity,
            "Iteration",
            "Perplexity",
            f"{algorithm_name}: Perplexity vs Iteration",
            os.path.join(outdir, "perplexity.png")
        )
        _plot_series(
            xs,
            ys_T,
            "Iteration",
            "T",
            f"{algorithm_name}: Topics (T) vs Iteration",
            os.path.join(outdir, "T_over_time.png")
        )
        _plot_series(
            xs,
            ys_step,
            "Iteration",
            "seconds",
            f"{algorithm_name}: Step Time vs Iteration",
            os.path.join(outdir, "step_time.png")
        )

    summary = {
        "algorithm": algorithm_name,
        "dataset": dataset_name,
        "best": res["best"],
        "avg_step_time": res["avg_step_time"],
        "total_time": res["total_time"],
        "stopped_early": res.get("stopped_early", False),
        "num_iterations": len(res["history"])
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OPTIMIZATION] {algorithm_name} Completed!")
    print(f"[OPTIMIZATION] Best T: {res['best']['T']}, alpha: {res['best']['alpha']:.6f}, eta: {res['best']['eta']:.6f}")
    print(f"[OPTIMIZATION] Best perplexity: {res['best']['perplexity']:.4f}")
    print(f"[OPTIMIZATION] Total time: {res['total_time']:.2f}s\n")

    return summary


class ParallelRunner:
    """
    Class for running multiple optimizations in parallel using multiprocessing.
    """
    def __init__(self, max_workers=2):
        self.max_workers = max_workers

    def _run_task(self, task_args):
        """Worker function for parallel execution"""
        dataset_name, val_path, algorithm, params = task_args

        print(f"\n[WORKER] Starting {algorithm} on {dataset_name}")

        Xval = load_bow_data(val_path)

        # Run optimization
        if algorithm == "GA":
            summary = run_single_optimization(
                "GA",
                GAOptimizer,
                Xval,
                **params
            )
        else:  # ES
            summary = run_single_optimization(
                "ES",
                ESOptimizer,
                Xval,
                **params
            )

        print(f"\n[WORKER] Completed {algorithm} on {dataset_name}")
        return (dataset_name, algorithm, summary)

    def run_all(self, tasks):
        """
        Run all tasks in parallel.

        Args:
            tasks: List of (dataset_name, val_path, algorithm, params) tuples

        Returns:
            dict: Results organized by dataset and algorithm
        """
        results = {}

        print("="*80)
        print(f"PARALLEL EXECUTION: Running {len(tasks)} tasks with {self.max_workers} workers")
        print("="*80)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._run_task, task): task for task in tasks}

            for future in as_completed(futures):
                dataset_name, algorithm, summary = future.result()

                if dataset_name not in results:
                    results[dataset_name] = {}
                results[dataset_name][algorithm] = summary

                print(f"\n{'='*80}")
                print(f"[COMPLETED] {algorithm} on {dataset_name}")
                print(f"  Best T: {summary['best']['T']}")
                print(f"  Best perplexity: {summary['best']['perplexity']:.4f}")
                print(f"  Total time: {summary['total_time']:.2f}s")
                print(f"{'='*80}")

        return results


def main(parallel=False, algorithm_filter=None):
    """
    Main function to run optimization on all datasets

    Args:
        parallel: If True, run GA and ES in parallel using multiprocessing
        algorithm_filter: If specified, run only "GA" or "ES"
    """
    DATASETS = {
        "20news": "data/X_20news_val_bow.npz",
        "agnews": "data/X_agnews_val_bow.npz",
        "val_out": "data/X_val_out_val_bow.npz",
        "yelp": "data/X_yelp_val_bow.npz",
    }

    BASE_DIR = "logs"
    gens_ga = 200
    steps_es = 200
    pop_size = 10
    elite = 5
    seed = 42
    batch_size = 2048
    learning_method = "online"
    max_iter = 60
    early_stop_eps_pct = 0.01
    max_no_improvement = 3

    ga_params = {
        "early_stop_eps_pct": early_stop_eps_pct,
        "max_no_improvement": max_no_improvement,
        "elite": elite
    }

    es_params = {
        "early_stop_eps_pct": early_stop_eps_pct,
        "max_no_improvement": max_no_improvement,
        "mu": 5,
        "lmbda": 10
    }

    print("="*80)
    print("LDA HYPERPARAMETER OPTIMIZATION")
    print("Training on VALIDATION set to find optimal hyperparameters")
    print(f"Mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    if algorithm_filter:
        print(f"Running only: {algorithm_filter}")
    print("="*80)

    if parallel:
        # Parallel execution using ParallelRunner
        tasks = []
        for name, val_path in DATASETS.items():
            # GA task
            if algorithm_filter is None or algorithm_filter.upper() == "GA":
                ga_task_params = {
                    "outdir": os.path.join(BASE_DIR, name, "ga"),
                    "dataset_name": name,
                    "gens_or_steps": gens_ga,
                    "pop_size": pop_size,
                    "seed": seed,
                    "max_iter": max_iter,
                    "batch_size": batch_size,
                    "learning_method": learning_method,
                    **ga_params
                }
                tasks.append((name, val_path, "GA", ga_task_params))

            # ES task
            if algorithm_filter is None or algorithm_filter.upper() == "ES":
                es_task_params = {
                    "outdir": os.path.join(BASE_DIR, name, "es"),
                    "dataset_name": name,
                    "gens_or_steps": steps_es,
                    "pop_size": pop_size,
                    "seed": seed,
                    "max_iter": max_iter,
                    "batch_size": batch_size,
                    "learning_method": learning_method,
                    **es_params
                }
                tasks.append((name, val_path, "ES", es_task_params))

        runner = ParallelRunner(max_workers=min(len(tasks), 8))
        results = runner.run_all(tasks)
    else:
        results = {}
        for name, val_path in DATASETS.items():
            print(f"\n[DATASET] Loading {name} from {val_path}")
            Xval = load_bow_data(val_path)
            print(f"[DATASET] Loaded: {Xval.shape[0]} documents, {Xval.shape[1]} vocabulary")

            results[name] = {}

            if algorithm_filter is None or algorithm_filter.upper() == "GA":
                outdir_ga = os.path.join(BASE_DIR, name, "ga")
                summary_ga = run_single_optimization(
                    "GA",
                    GAOptimizer,
                    Xval=Xval,
                    outdir=outdir_ga,
                    dataset_name=name,
                    gens_or_steps=gens_ga,
                    pop_size=pop_size,
                    seed=seed,
                    max_iter=max_iter,
                    batch_size=batch_size,
                    learning_method=learning_method,
                    **ga_params
                )
                results[name]["GA"] = summary_ga

            if algorithm_filter is None or algorithm_filter.upper() == "ES":
                outdir_es = os.path.join(BASE_DIR, name, "es")
                summary_es = run_single_optimization(
                    "ES",
                    ESOptimizer,
                    Xval=Xval,
                    outdir=outdir_es,
                    dataset_name=name,
                    gens_or_steps=steps_es,
                    pop_size=pop_size,
                    seed=seed,
                    max_iter=max_iter,
                    batch_size=batch_size,
                    learning_method=learning_method,
                    **es_params
                )
                results[name]["ES"] = summary_es

    print("\n" + "="*80)
    print("ALL OPTIMIZATIONS COMPLETED!")
    print("="*80)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    with open(os.path.join(BASE_DIR, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {os.path.join(BASE_DIR, 'all_results.json')}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LDA Hyperparameter Optimization"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run optimizations in parallel"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ga", "es", "GA", "ES"],
        default=None,
        help="Run only specific algorithm (GA or ES)"
    )
    args = parser.parse_args()
    main(
        parallel=args.parallel,
        algorithm_filter=args.algorithm
    )