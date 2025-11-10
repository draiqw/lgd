"""
SABO (Stochastic Adaptive Bayesian Optimization) for LDA hyperparameters.

This module implements SABO that:
- Uses adaptive sampling with Gaussian proposals
- Applies natural gradient updates
- Searches for optimal T (number of topics)
- Sets alpha and eta to 1/T
- Trains on validation set
- Optimizes for minimal perplexity

Based on Algorithm 1 from SABO paper.
"""

import time
import os
import csv
import random
from typing import Optional, List, Dict, Tuple

import numpy as np
from tensorboardX import SummaryWriter

from lda_hyperopt.utils import BaseOptimizer, clamp, ensure_dir


class SABOOptimizer(BaseOptimizer):
    """
    SABO optimizer for LDA hyperparameters.

    The SABO uses:
    - Gaussian proposal distribution N(μ, Σ)
    - Natural gradient updates with adaptive covariance
    - Batch observations for gradient estimation
    - Adaptive learning rate and momentum
    """

    def __init__(
        self,
        obj,
        eval_func,
        T_bounds=(2, 1000),
        seed=42,
        rho=0.1,
        beta_t=0.01,
        delta_Sigma=0.5,
        delta_mu=0.5,
        N_batch=10,
        early_stop_eps_pct=0.01,
        max_no_improvement=5,
        logger=None
    ):
        """
        Initialize SABO optimizer.

        Args:
            obj: Objective function (T, alpha, eta) -> perplexity
            eval_func: Evaluation function (T, alpha, eta) -> metrics dict
            T_bounds: Bounds for T parameter (min, max)
            seed: Random seed
            rho: Neighborhood size parameter (controls proposal std)
            beta_t: Learning rate
            delta_Sigma: Momentum for covariance update
            delta_mu: Momentum for mean update
            N_batch: Number of samples per iteration
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

        self.rho = float(rho)
        self.beta_t = float(beta_t)
        self.delta_Sigma = float(delta_Sigma)
        self.delta_mu = float(delta_mu)
        self.N_batch = int(N_batch)

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Track evaluated points
        self.evaluated_points = []  # List of (T, fitness)

        # SABO parameters (1D case)
        # μ - mean of proposal distribution (in normalized [0,1] space)
        # Σ - variance of proposal distribution
        self.mu = 0.5  # Start in middle of search space
        self.Sigma = 0.25  # Initial variance
        
        self.logger.info(
            f"SABO initialized: rho={rho}, beta_t={beta_t}, "
            f"N_batch={N_batch}, delta_Sigma={delta_Sigma}, delta_mu={delta_mu}"
        )

    def _normalize_T(self, T: int) -> float:
        """
        Normalize T to [0, 1] range.
        
        Args:
            T: Number of topics
            
        Returns:
            Normalized value in [0, 1]
        """
        return (T - self.Tb[0]) / (self.Tb[1] - self.Tb[0])
    
    def _denormalize_T(self, x: float) -> int:
        """
        Denormalize x from [0, 1] to T range.
        
        Args:
            x: Normalized value
            
        Returns:
            T value (clamped to bounds)
        """
        T = int(round(x * (self.Tb[1] - self.Tb[0]) + self.Tb[0]))
        return clamp(T, self.Tb[0], self.Tb[1])

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

    def _sample_batch(self, mu: float, Sigma: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample batch from Gaussian proposal N(μ, Σ).
        
        Args:
            mu: Mean of proposal
            Sigma: Variance of proposal
            N: Number of samples
            
        Returns:
            Tuple of (normalized_samples, T_samples)
        """
        # Sample from N(μ, Σ) in [0,1] space
        std = np.sqrt(Sigma)
        z_samples = np.random.normal(mu, std, size=N)
        
        # Clamp to [0, 1]
        z_samples = np.clip(z_samples, 0.0, 1.0)
        
        # Convert to T values
        T_samples = np.array([self._denormalize_T(z) for z in z_samples])
        
        return z_samples, T_samples

    def _compute_gradient(
        self, 
        z_samples: np.ndarray, 
        fitness_values: np.ndarray,
        mu: float,
        Sigma: float
    ) -> Tuple[float, float]:
        """
        Compute natural gradient using Eq. (16-20) from SABO.
        
        Args:
            z_samples: Normalized samples
            fitness_values: Corresponding fitness values
            mu: Current mean
            Sigma: Current variance
            
        Returns:
            Tuple of (grad_mu, grad_Sigma)
        """
        N = len(z_samples)
        
        # Compute standardized residuals (for natural gradient)
        # g_i = (z_i - μ) / Σ
        g = (z_samples - mu) / (Sigma + 1e-8)
        
        # Compute gradient weights (inverse fitness for minimization)
        # Use ranking or inverse for stability
        # Normalize fitness to avoid numerical issues
        f_norm = (fitness_values - np.min(fitness_values)) / (np.ptp(fitness_values) + 1e-8)
        weights = 1.0 / (f_norm + 0.1)  # Add small constant for stability
        weights = weights / np.sum(weights)  # Normalize
        
        # Gradient for μ (Eq. 16): ∇μ = Σ^{-1} * Σ_i w_i * g_i
        grad_mu = np.sum(weights * g)
        
        # Gradient for Σ (Eq. 17): simplified for 1D
        # ∇Σ = 0.5 * (Σ^{-1} - Σ^{-1} * Σ_i w_i * g_i^2 * Σ^{-1})
        g_squared = g ** 2
        grad_Sigma = 0.5 * (1.0 / Sigma - np.sum(weights * g_squared) / Sigma)
        
        return grad_mu, grad_Sigma

    def _update_parameters(
        self,
        mu: float,
        Sigma: float,
        grad_mu: float,
        grad_Sigma: float,
        lambda_val: float
    ) -> Tuple[float, float]:
        """
        Update parameters using natural gradient with momentum.
        
        Args:
            mu: Current mean
            Sigma: Current variance
            grad_mu: Gradient w.r.t. mean
            grad_Sigma: Gradient w.r.t. variance
            lambda_val: Step size
            
        Returns:
            Tuple of (new_mu, new_Sigma)
        """
        # Update with momentum (Eq. 9)
        mu_new = mu - lambda_val * self.delta_mu * grad_mu
        
        # Update Sigma with momentum and ensure positivity
        Sigma_new = Sigma - lambda_val * self.delta_Sigma * grad_Sigma
        
        # Clamp to reasonable bounds
        mu_new = clamp(mu_new, 0.0, 1.0)
        Sigma_new = clamp(Sigma_new, 0.001, 0.5)  # Keep variance in reasonable range
        
        return mu_new, Sigma_new

    def _compute_step_size(self, iteration: int, grad_mu: float, grad_Sigma: float) -> float:
        """
        Compute adaptive step size (Eq. 6 and 18 from SABO).
        
        Args:
            iteration: Current iteration
            grad_mu: Gradient magnitude for mu
            grad_Sigma: Gradient magnitude for Sigma
            
        Returns:
            Step size lambda
        """
        # Compute gradient norm (simplified for 1D)
        grad_norm = np.sqrt(grad_mu**2 + grad_Sigma**2)
        
        # Adaptive step size with decay
        lambda_val = self.beta_t / (1.0 + 0.1 * iteration) * (1.0 + grad_norm)
        
        return lambda_val

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
        Create initial set of points using Latin Hypercube sampling.

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
        Run SABO optimization.

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
        self.logger.info(f"Starting SABO optimization")
        self.logger.info(f"Iterations: {iterations}, Initial samples: {n_initial}, Batch size: {self.N_batch}")
        self.logger.info(f"T bounds: {self.Tb}, alpha=1/T, eta=1/T")
        self.logger.info("="*80)

        # Create or use provided initial points
        if initial_population is not None:
            self.logger.info("Using provided initial population")
            initial_points = initial_population
        else:
            initial_points = self.create_initial_population(n_initial)

        # Initialize tracking
        history = []
        t0 = time.perf_counter()
        no_improvement_count = 0
        prev_perplexity = float('inf')
        best_T = None
        best_fitness = float('inf')

        # Initialize SABO parameters from initial data
        self.logger.info("Evaluating initial samples...")
        for T in initial_points:
            fitness = self._evaluate_point(T)
            if fitness < best_fitness:
                best_fitness = fitness
                best_T = T
                self.logger.info(f"Initial: New best found! T={T}, fitness={fitness:.4f}")

        # Initialize μ to normalized best_T
        if best_T is not None:
            self.mu = self._normalize_T(best_T)
            self.logger.info(f"Initialized μ={self.mu:.4f} (T={best_T})")

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
            "pop_mean": float(np.mean(all_fitness)),
            "pop_std": float(np.std(all_fitness)),
            "pop_min": float(np.min(all_fitness)),
            "pop_max": float(np.max(all_fitness)),
            "T_best": Tb,
            "alpha_best": ab,
            "eta_best": eb,
            "mu": self.mu,
            "Sigma": self.Sigma,
            "step_time": 0.0,
            "cum_time": 0.0,
            "no_improvement_count": 0,
            "relative_change_pct": 0.0
        })

        # Main SABO loop
        for iteration in range(iterations):
            iter_start = time.perf_counter()

            # Step 3-4: Sample batch from N(μ, Σ)
            z_samples, T_samples = self._sample_batch(self.mu, self.Sigma, self.N_batch)
            
            # Step 4-5: Query batch observations (evaluate fitness)
            fitness_values = np.array([self._evaluate_point(T) for T in T_samples])
            
            # Update best
            batch_best_idx = np.argmin(fitness_values)
            if fitness_values[batch_best_idx] < best_fitness:
                best_fitness = fitness_values[batch_best_idx]
                best_T = T_samples[batch_best_idx]
                self.logger.info(
                    f"Iter {iteration+1}: New best found! T={best_T}, "
                    f"fitness={best_fitness:.4f}"
                )

            # Step 5-6: Compute gradients (natural gradient)
            grad_mu, grad_Sigma = self._compute_gradient(
                z_samples, fitness_values, self.mu, self.Sigma
            )
            
            # Step 6: Compute adaptive step size
            lambda_val = self._compute_step_size(iteration, grad_mu, grad_Sigma)
            
            # Step 8-9: Update parameters with momentum
            mu_prev = self.mu
            Sigma_prev = self.Sigma
            self.mu, self.Sigma = self._update_parameters(
                self.mu, self.Sigma, grad_mu, grad_Sigma, lambda_val
            )

            # Get decoded parameters (no re-evaluation needed)
            Tb, ab, eb = self.decode([best_T])

            # Timing
            iter_time = time.perf_counter() - iter_start
            cum_time = time.perf_counter() - t0

            # Statistics over all evaluated points
            all_fitness = [f for _, f in self.evaluated_points]
            pop_mean = float(np.mean(all_fitness))
            pop_std = float(np.std(all_fitness))
            pop_min = float(np.min(all_fitness))
            pop_max = float(np.max(all_fitness))

            # Early stopping check (use best_fitness directly, not re-evaluate)
            current_perplexity = best_fitness
            relative_change = abs(current_perplexity - prev_perplexity) / prev_perplexity if prev_perplexity > 0 else float('inf')

            if relative_change <= self.early_stop_eps_pct:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            prev_perplexity = current_perplexity

            # Logging
            self.logger.info(
                f"Iter {iteration+1}/{iterations} | Best: T={Tb}, perplexity={best_fitness:.4f} | "
                f"μ={self.mu:.4f}, Σ={self.Sigma:.4f} | "
                f"∇μ={grad_mu:.4f}, ∇Σ={grad_Sigma:.4f} | "
                f"λ={lambda_val:.4f} | No improvement: {no_improvement_count}/{self.max_no_improvement}"
            )

            # TensorBoard logging
            if writer:
                iter_num = iteration
                writer.add_scalar("SABO/Best/perplexity", best_fitness, iter_num)
                writer.add_scalar("SABO/Best/T", Tb, iter_num)
                writer.add_scalar("SABO/Best/alpha", ab, iter_num)
                writer.add_scalar("SABO/Best/eta", eb, iter_num)
                writer.add_scalar("SABO/Parameters/mu", self.mu, iter_num)
                writer.add_scalar("SABO/Parameters/Sigma", self.Sigma, iter_num)
                writer.add_scalar("SABO/Gradients/grad_mu", grad_mu, iter_num)
                writer.add_scalar("SABO/Gradients/grad_Sigma", grad_Sigma, iter_num)
                writer.add_scalar("SABO/StepSize/lambda", lambda_val, iter_num)
                writer.add_scalar("SABO/Statistics/mean_fitness", pop_mean, iter_num)
                writer.add_scalar("SABO/Statistics/std_fitness", pop_std, iter_num)
                writer.add_scalar("SABO/Time/step_time", iter_time, iter_num)
                writer.add_scalar("SABO/Time/cumulative", cum_time, iter_num)
                writer.add_scalar("SABO/EarlyStopping/no_improvement_count", no_improvement_count, iter_num)
                writer.add_scalar("SABO/EarlyStopping/relative_change_pct", relative_change * 100, iter_num)

            # History
            history.append({
                "iter": iteration,
                "best_perplexity": best_fitness,
                "pop_mean": pop_mean,
                "pop_std": pop_std,
                "pop_min": pop_min,
                "pop_max": pop_max,
                "T_best": Tb,
                "alpha_best": ab,
                "eta_best": eb,
                "mu": self.mu,
                "Sigma": self.Sigma,
                "grad_mu": grad_mu,
                "grad_Sigma": grad_Sigma,
                "lambda": lambda_val,
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
                    f"Early stopping: |delta perplexity|/prev ≤ {self.early_stop_eps_pct*100:.2f}% "
                    f"for {self.max_no_improvement} iterations"
                )
                break

        # Final save
        if outdir:
            self._save_evaluated_points(iterations, outdir)

        # Get final full metrics (only call eval_func once at the end)
        Tb, ab, eb = self.decode([best_T])
        final_metrics = self.eval_func(Tb, ab, eb)

        # Final results
        final_cum_time = time.perf_counter() - t0
        self.logger.info("="*80)
        self.logger.info("SABO Optimization Complete!")
        self.logger.info(f"Total time: {final_cum_time:.2f}s")
        self.logger.info(f"Total evaluations: {len(self.evaluated_points)}")
        self.logger.info(
            f"Final best: T={Tb}, alpha={ab:.6f}, eta={eb:.6f}, "
            f"perplexity={final_metrics['perplexity']:.4f}"
        )
        self.logger.info(f"Final parameters: μ={self.mu:.4f}, Σ={self.Sigma:.4f}")
        self.logger.info("="*80)

        return {
            "best": {
                "T": Tb,
                "alpha": ab,
                "eta": eb,
                **final_metrics
            },
            "history": history,
            "total_time": final_cum_time,
            "avg_step_time": float(np.mean([h["step_time"] for h in history])) if history else 0.0,
            "stopped_early": no_improvement_count >= self.max_no_improvement,
            "total_evaluations": len(self.evaluated_points),
            "algorithm": "SABO",
            "final_mu": self.mu,
            "final_Sigma": self.Sigma
        }
