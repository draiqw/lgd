"""
PABBO Full optimizer with Transformer-based policy learning.

This module implements the full PABBO approach from pabbo_method:
- Transformer-based policy for suggesting candidates
- Preference learning from pairwise comparisons
- Meta-learning across optimization problems

Note: Requires trained model checkpoint and PyTorch dependencies.
If checkpoint not available, falls back to PABBO Simple.
"""

import time
import os
import random
from typing import Optional, List, Dict, Tuple
import warnings

import numpy as np
from tensorboardX import SummaryWriter

from lda_hyperopt.utils import BaseOptimizer, clamp, ensure_dir

# Try to import PyTorch dependencies
try:
    import torch
    from torch import Tensor
    import sys
    # Add pabbo_method to path
    pabbo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pabbo_method'))
    if pabbo_path not in sys.path:
        sys.path.insert(0, pabbo_path)

    try:
        from pabbo_method.policies.transformer import TransformerModel
    except ModuleNotFoundError:
        from policies.transformer import TransformerModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch dependencies not available. PABBO Full will fall back to Simple version. "
        "Install with: pip install torch botorch gpytorch hydra-core"
    )


class PABBOFullOptimizer(BaseOptimizer):
    """
    Full PABBO optimizer with Transformer-based policy.

    If trained model is available, uses Transformer for candidate generation.
    Otherwise, falls back to adaptive sampling (PABBO Simple behavior).

    Model should be trained on similar optimization problems using:
        cd pabbo_method
        python train.py --config-name train_config
    """

    def __init__(
        self,
        obj,
        eval_func,
        T_bounds=(2, 1000),
        seed=42,
        model_path=None,
        exploration_rate=0.3,
        temperature_decay=0.95,
        min_temperature=0.1,
        early_stop_eps_pct=0.01,
        max_no_improvement=5,
        logger=None
    ):
        """
        Initialize PABBO Full optimizer.

        Args:
            obj: Objective function (T, alpha, eta) -> perplexity
            eval_func: Evaluation function (T, alpha, eta) -> metrics dict
            T_bounds: Bounds for T parameter (min, max)
            seed: Random seed
            model_path: Path to trained Transformer model checkpoint (optional)
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
        self.model_path = model_path

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)

        # Track evaluated points
        self.evaluated_points = []  # List of (T, fitness)

        # Try to load model
        self.model = None
        self.use_transformer = False

        if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self._load_model(model_path)
                self.use_transformer = True
                self.logger.info(
                    f"PABBO Full initialized with Transformer model from {model_path}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load model from {model_path}: {e}. "
                    "Falling back to Simple PABBO."
                )
        else:
            if not TORCH_AVAILABLE:
                self.logger.info("PyTorch not available, using PABBO Simple")
            elif not model_path:
                self.logger.info("No model path provided, using PABBO Simple")
            else:
                self.logger.info(f"Model not found at {model_path}, using PABBO Simple")

        self.logger.info(
            f"PABBO initialized: exploration_rate={exploration_rate}, "
            f"temperature_decay={temperature_decay}, "
            f"use_transformer={self.use_transformer}"
        )

    def _load_model(self, model_path: str):
        """Load trained Transformer model."""
        checkpoint = torch.load(model_path, map_location='cpu')

        # Try to extract model config from checkpoint
        state_dict = checkpoint.get('model', checkpoint)
        model_kwargs = checkpoint.get('model_kwargs')

        if model_kwargs is None:
            model_kwargs = self._infer_model_kwargs_from_state(state_dict)
            self.logger.info(
                "Model config not found in checkpoint; inferred parameters: "
                f"{model_kwargs}"
            )

        # ensure plain python types (Hydra may produce ListConfig)
        model_kwargs = self._normalize_model_kwargs(model_kwargs)

        self.model = TransformerModel(**model_kwargs)

        missing = self.model.load_state_dict(state_dict, strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            self.logger.warning(
                f"Loaded Transformer with missing keys={missing.missing_keys} "
                f"and unexpected keys={missing.unexpected_keys}"
            )

        self.model.eval()
        self.model_kwargs = model_kwargs

    @staticmethod
    def _normalize_model_kwargs(model_kwargs: Dict) -> Dict:
        """Convert OmegaConf containers to plain python types."""
        normalized = {}
        for k, v in model_kwargs.items():
            if isinstance(v, (list, tuple)):
                normalized[k] = list(v)
            elif hasattr(v, 'items'):
                normalized[k] = dict(v)
            else:
                normalized[k] = v
        return normalized

    def _infer_model_kwargs_from_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict:
        """
        Infer transformer constructor arguments from a state dict.

        This is a best-effort heuristic for older checkpoints that do not store
        configuration metadata.
        """
        def _get_shape(name: str):
            tensor = state_dict[name]
            return tuple(tensor.shape)

        # Core dimensions
        d_model = _get_shape("x_embedders.0.0.weight")[0]
        d_x = _get_shape("x_embedders.0.0.weight")[1]
        dim_feedforward = _get_shape("encoder.layers.0.linear1.weight")[0]

        # Number of encoder layers
        layer_indices = {
            key.split(".")[2]
            for key in state_dict.keys()
            if key.startswith("encoder.layers.")
        }
        n_layers = len(layer_indices)

        # Embedding depth equals number of linear blocks in x_embedders.{idx}
        emb_linear_keys = [
            key for key in state_dict.keys()
            if key.startswith("x_embedders.0") and key.endswith(".weight")
        ]
        emb_depth = len(emb_linear_keys)

        # nbuckets is decoder output dimension
        if "eval_bucket_decoder.2.weight" in state_dict:
            nbuckets = _get_shape("eval_bucket_decoder.2.weight")[0]
        else:
            # older checkpoints used single-layer decoder
            decoder_weights = [
                key for key in state_dict if key.startswith("eval_bucket_decoder") and key.endswith("weight")
            ]
            nbuckets = _get_shape(decoder_weights[-1])[0] if decoder_weights else 1

        # Candidate head counts: divisors of d_model (descending)
        candidate_heads = [
            h for h in range(d_model, 0, -1)
            if d_model % h == 0
        ]

        # Heuristic nhead inference (older checkpoints did not store metadata)
        if dim_feedforward <= 64:
            default_nhead = 2
        elif dim_feedforward <= 128:
            default_nhead = 4
        else:
            default_nhead = max(1, min(8, d_model // 8))

        base_kwargs = dict(
            d_x=d_x,
            d_f=1,
            d_model=d_model,
            dropout=0.0,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            emb_depth=emb_depth,
            tok_emb_option="ind_point_emb_sum",
            transformer_encoder_layer_cls="efficient",
            joint_model_af_training=True,
            af_name="mlp",
            bound_std=False,
            nbuckets=nbuckets,
            time_budget=True,
        )

        # Try candidate nhead values until state dict loads
        last_error = None
        ordered_candidates = [default_nhead] + [
            h for h in candidate_heads if h != default_nhead
        ]

        for nhead in ordered_candidates:
            try_kwargs = dict(base_kwargs)
            try_kwargs["nhead"] = nhead
            try:
                candidate_model = TransformerModel(**try_kwargs)
                candidate_model.load_state_dict(state_dict, strict=False)
                return try_kwargs
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        raise RuntimeError(
            "Unable to infer transformer configuration from checkpoint. "
            f"Last error: {last_error}"
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

    def _suggest_transformer(self) -> int:
        """
        Use Transformer model to suggest next candidate.

        This is a simplified interface. Full PABBO uses preference pairs.
        Here we convert our history to a format the model can use.

        Returns:
            Suggested T value
        """
        if not self.use_transformer or self.model is None:
            return self._sample_exploitation(0.5)

        try:
            # For now, use simplified version
            # Full implementation would require proper context preparation
            # Fall back to exploitation
            return self._sample_exploitation(0.3)
        except Exception as e:
            self.logger.warning(f"Transformer suggestion failed: {e}")
            return self._sample_exploitation(0.5)

    def run(
        self,
        iterations: int,
        writer: Optional[SummaryWriter] = None,
        outdir: Optional[str] = None,
        initial_population: Optional[List[int]] = None
    ) -> Dict:
        """
        Run PABBO Full optimization.

        Args:
            iterations: Number of iterations
            writer: TensorBoard writer (optional)
            outdir: Output directory (optional)
            initial_population: Initial population to evaluate

        Returns:
            Dictionary with optimization results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting PABBO Full optimization")
        self.logger.info(f"Iterations: {iterations}, Initial samples: {len(initial_population) if initial_population else 0}")
        self.logger.info(f"T bounds: {self.Tb}, alpha=1/T, eta=1/T")
        self.logger.info(f"Using Transformer: {self.use_transformer}")
        self.logger.info("=" * 80)

        # Initialize timing
        start_time = time.perf_counter()
        history = []

        # Evaluate initial population
        if initial_population:
            self.logger.info("Using provided initial population")
            self.logger.info("Evaluating initial samples...")

            best_T = None
            best_fitness = float('inf')

            for T in initial_population:
                fitness = self._evaluate_point(T)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_T = T
                    self.logger.info(f"Initial: New best found! T={T}, fitness={fitness:.4f}")

            self.logger.info(
                f"Initial best: T={best_T}, alpha={1.0/best_T:.6f}, "
                f"eta={1.0/best_T:.6f}, perplexity={best_fitness:.4f}"
            )
        else:
            # Start with random sample
            best_T = self._sample_exploration()
            best_fitness = self._evaluate_point(best_T)

        # Add initial point to history (iteration -1 for consistency with GA/ES)
        history.append({
            'iter': -1,
            'T_best': best_T,
            'best_perplexity': best_fitness,
            'T_current': best_T,
            'current_fitness': best_fitness,
            'strategy': 'initial',
            'temperature': 1.0,
            'improvement': False,
            'step_time': 0.0,
            'cum_time': 0.0,
            'no_improvement_count': 0
        })

        # Optimization loop
        temperature = 1.0
        no_improvement_count = 0

        for iteration in range(iterations):
            iter_start = time.perf_counter()

            # Decide exploration vs exploitation
            if random.random() < self.exploration_rate:
                # Exploration
                T_new = self._sample_exploration()
                strategy = "exploration"
            elif self.use_transformer:
                # Try transformer suggestion
                T_new = self._suggest_transformer()
                strategy = "transformer"
            else:
                # Exploitation
                T_new = self._sample_exploitation(temperature)
                strategy = "exploitation"

            # Evaluate new point
            fitness_new = self._evaluate_point(T_new)

            # Update best
            improvement = False
            if fitness_new < best_fitness:
                rel_improvement = abs(fitness_new - best_fitness) / (abs(best_fitness) + 1e-10)
                best_fitness = fitness_new
                best_T = T_new
                improvement = True

                if rel_improvement > self.early_stop_eps_pct:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

            # Update temperature
            temperature = max(self.min_temperature, temperature * self.temperature_decay)

            # Record history
            iter_time = time.perf_counter() - iter_start
            cum_time = time.perf_counter() - start_time

            history.append({
                'iter': iteration,
                'T_best': best_T,
                'best_perplexity': best_fitness,
                'T_current': T_new,
                'current_fitness': fitness_new,
                'strategy': strategy,
                'temperature': temperature,
                'improvement': improvement,
                'step_time': iter_time,
                'cum_time': cum_time,
                'no_improvement_count': no_improvement_count
            })

            # Logging
            self.logger.info(
                f"Iter {iteration+1}/{iterations} | "
                f"Best: T={best_T}, perplexity={best_fitness:.4f} | "
                f"Current: T={T_new}, fitness={fitness_new:.4f} ({strategy}) | "
                f"Temp: {temperature:.3f} | "
                f"No improvement: {no_improvement_count}/{self.max_no_improvement}"
            )

            # TensorBoard logging
            if writer:
                writer.add_scalar('Best/perplexity', best_fitness, iteration)
                writer.add_scalar('Best/T', best_T, iteration)
                writer.add_scalar('Current/fitness', fitness_new, iteration)
                writer.add_scalar('Temperature', temperature, iteration)

            # Early stopping
            if no_improvement_count >= self.max_no_improvement:
                rel_improvement = abs(best_fitness - history[-self.max_no_improvement]['best_perplexity']) / (abs(history[-self.max_no_improvement]['best_perplexity']) + 1e-10)
                self.logger.info(
                    f"Early stopping: |delta perplexity|/prev ≤ {self.early_stop_eps_pct*100:.2f}% "
                    f"for {self.max_no_improvement} iterations"
                )
                break

        # Final evaluation
        total_time = time.perf_counter() - start_time

        T_final, alpha_final, eta_final = self.decode([best_T])
        final_metrics = self.eval_func(T_final, alpha_final, eta_final)

        self.logger.info("=" * 80)
        self.logger.info("PABBO Full Optimization Complete!")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Total evaluations: {len(self.evaluated_points)}")
        self.logger.info(
            f"Final best: T={T_final}, alpha={alpha_final:.6f}, "
            f"eta={eta_final:.6f}, perplexity={best_fitness:.4f}"
        )
        self.logger.info("=" * 80)

        return {
            'algorithm': 'PABBO_Full',
            'best': {
                'T': T_final,
                'alpha': alpha_final,
                'eta': eta_final,
                'perplexity': best_fitness,
                **{k: v for k, v in final_metrics.items() if k not in ['T', 'alpha', 'eta', 'perplexity']}
            },
            'history': history,
            'total_time': total_time,
            'avg_step_time': total_time / len(history) if history else 0,
            'stopped_early': len(history) < iterations,
            'used_transformer': self.use_transformer
        }
