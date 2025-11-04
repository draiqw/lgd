#!/usr/bin/env python
"""
=============================================================================
LDA Hyperparameter Optimization - Full Pipeline
=============================================================================

Complete pipeline that:
1. Trains PABBO model from scratch (light version)
2. Evaluates the trained model
3. Runs LDA optimization experiments (GA, ES, PABBO_Full) on all datasets
4. Repeats experiments 10 times for statistical significance
5. Aggregates results and generates comprehensive visualizations

Author: Auto-generated
Date: November 2024
=============================================================================
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).parent
PABBO_METHOD_DIR = PROJECT_ROOT / "pabbo_method"
LDA_HYPEROPT_DIR = PROJECT_ROOT / "lda_hyperopt"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "lda_pipeline_results"

# Add paths
sys.path.insert(0, str(PABBO_METHOD_DIR))
sys.path.insert(0, str(LDA_HYPEROPT_DIR))


class PipelineLogger:
    """Centralized logging for the entire pipeline."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        self.main_log = log_dir / "pipeline_main.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.main_log),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("LDA_Pipeline")

        # Metrics tracking
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }

    def log_stage(self, stage_name: str, status: str, duration: float = None, **kwargs):
        """Log a pipeline stage."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Stage: {stage_name}")
        self.logger.info(f"Status: {status}")
        if duration:
            self.logger.info(f"Duration: {duration:.2f}s ({duration/60:.2f}min)")
        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info(f"{'='*80}")

        self.metrics['stages'][stage_name] = {
            'status': status,
            'duration': duration,
            **kwargs
        }

    def save_metrics(self):
        """Save all metrics to JSON."""
        self.metrics['end_time'] = datetime.now().isoformat()
        metrics_file = self.log_dir / "pipeline_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")


class PABBOTrainer:
    """Handles PABBO model training."""

    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.pabbo_dir = PABBO_METHOD_DIR
        self.model_path = None

    def train_light_model(self) -> Tuple[str, float]:
        """
        Train a light PABBO model for quick testing.

        Returns:
            (model_path, training_time)
        """
        self.logger.logger.info("\n" + "="*80)
        self.logger.logger.info("STAGE 1: Training PABBO Model (Light Version)")
        self.logger.logger.info("="*80)

        start_time = time.time()

        # Training command
        cmd = [
            sys.executable,
            str(self.pabbo_dir / "train.py"),
            "--config-name=train_rastrigin1d_test",
            "experiment.wandb=false",  # Disable wandb for cleaner output
        ]

        self.logger.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.logger.info("Training in progress...")

        try:
            # Run training
            result = subprocess.run(
                cmd,
                cwd=str(self.pabbo_dir),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            training_time = time.time() - start_time

            if result.returncode != 0:
                self.logger.logger.error("Training failed!")
                self.logger.logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError("PABBO training failed")

            # Find the trained model
            results_dir = self.pabbo_dir / "results" / "PABBO"
            if not results_dir.exists():
                raise FileNotFoundError("Results directory not found")

            # Get the latest model
            model_dirs = sorted(results_dir.glob("*"), key=os.path.getmtime, reverse=True)
            if not model_dirs:
                raise FileNotFoundError("No trained model found")

            self.model_path = str(model_dirs[0] / "ckpt.tar")

            self.logger.log_stage(
                "PABBO_Training",
                "SUCCESS",
                training_time,
                model_path=self.model_path,
                config="train_rastrigin1d_test"
            )

            return self.model_path, training_time

        except Exception as e:
            self.logger.log_stage("PABBO_Training", "FAILED", time.time() - start_time, error=str(e))
            raise

    def evaluate_model(self, model_path: str) -> Dict:
        """
        Evaluate the trained PABBO model.

        Returns:
            Evaluation metrics
        """
        self.logger.logger.info("\n" + "="*80)
        self.logger.logger.info("STAGE 2: Evaluating PABBO Model")
        self.logger.logger.info("="*80)

        start_time = time.time()

        # Extract expid from model path
        expid = Path(model_path).parent.name

        # Evaluation command
        cmd = [
            sys.executable,
            str(self.pabbo_dir / "evaluate_continuous.py"),
            "--config-name=evaluate",
            f"experiment.expid={expid}",
            f"experiment.model=PABBO",
            "experiment.device=cpu",
            "experiment.wandb=false",
            "data.name=rastrigin1D",
            "data.d_x=1",
            'data.x_range="[[-5.12,5.12]]"',
            'data.Xopt="[[0.0]]"',
            'data.yopt="[[0.0]]"',
            "eval.eval_max_T=30",
            "eval.eval_num_query_points=128",
        ]

        self.logger.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.pabbo_dir),
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )

            eval_time = time.time() - start_time

            if result.returncode != 0:
                self.logger.logger.warning("Evaluation failed (non-critical)")
                self.logger.logger.warning(f"STDERR: {result.stderr}")

            metrics = {
                'eval_time': eval_time,
                'status': 'SUCCESS' if result.returncode == 0 else 'WARNING'
            }

            self.logger.log_stage("PABBO_Evaluation", metrics['status'], eval_time)

            return metrics

        except Exception as e:
            self.logger.log_stage("PABBO_Evaluation", "FAILED", time.time() - start_time, error=str(e))
            # Non-critical, continue pipeline
            return {'status': 'FAILED', 'error': str(e)}


class LDAExperimentRunner:
    """Runs LDA optimization experiments."""

    def __init__(self, logger: PipelineLogger, model_path: str):
        self.logger = logger
        self.model_path = model_path
        self.lda_dir = LDA_HYPEROPT_DIR
        self.data_dir = DATA_DIR

        # Available datasets
        self.datasets = self._find_datasets()

    def _find_datasets(self) -> List[str]:
        """Find all available validation datasets."""
        datasets = []
        for file in self.data_dir.glob("X_*_val_bow.npz"):
            # Extract dataset name (e.g., "20news" from "X_20news_val_bow.npz")
            name = file.stem.replace("X_", "").replace("_val_bow", "")
            datasets.append(name)

        self.logger.logger.info(f"Found {len(datasets)} datasets: {datasets}")
        return sorted(datasets)

    def run_single_experiment(
        self,
        dataset_name: str,
        algorithms: List[str],
        iterations: int,
        run_id: int,
        output_dir: Path
    ) -> Dict:
        """
        Run a single LDA optimization experiment.

        Args:
            dataset_name: Name of dataset
            algorithms: List of algorithms to run
            iterations: Number of optimization iterations
            run_id: Run identifier (0-9)
            output_dir: Where to save results

        Returns:
            Results dictionary
        """
        self.logger.logger.info(f"\n{'='*60}")
        self.logger.logger.info(f"Dataset: {dataset_name} | Run: {run_id+1}/10")
        self.logger.logger.info(f"Algorithms: {', '.join(algorithms)}")
        self.logger.logger.info(f"{'='*60}")

        start_time = time.time()

        # Prepare data paths
        train_data = self.data_dir / f"X_{dataset_name}_train_bow.npz"
        val_data = self.data_dir / f"X_{dataset_name}_val_bow.npz"

        if not train_data.exists() or not val_data.exists():
            raise FileNotFoundError(f"Data not found for {dataset_name}")

        # Create output directory for this run
        run_dir = output_dir / dataset_name / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            str(self.lda_dir / "run.py"),
            "--train-data", str(train_data),
            "--val-data", str(val_data),
            "--algorithms", *algorithms,
            "--iterations", str(iterations),
            "--seed", str(42 + run_id),  # Different seed for each run
            "--output-dir", str(run_dir),
        ]

        # Add PABBO model path if needed
        if "PABBO_Full" in algorithms:
            cmd.extend(["--pabbo-model", self.model_path])

        self.logger.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.lda_dir),
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours timeout
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                self.logger.logger.error(f"Experiment failed for {dataset_name} run {run_id}")
                self.logger.logger.error(f"STDERR: {result.stderr}")
                return {
                    'status': 'FAILED',
                    'dataset': dataset_name,
                    'run_id': run_id,
                    'elapsed': elapsed,
                    'error': result.stderr
                }

            # Parse results
            results = self._parse_results(run_dir, algorithms)
            results.update({
                'status': 'SUCCESS',
                'dataset': dataset_name,
                'run_id': run_id,
                'elapsed': elapsed
            })

            self.logger.logger.info(f"✓ Completed in {elapsed:.2f}s")
            return results

        except subprocess.TimeoutExpired:
            self.logger.logger.error(f"Experiment timed out for {dataset_name} run {run_id}")
            return {
                'status': 'TIMEOUT',
                'dataset': dataset_name,
                'run_id': run_id,
                'error': 'Timeout'
            }
        except Exception as e:
            self.logger.logger.error(f"Experiment error: {e}")
            return {
                'status': 'ERROR',
                'dataset': dataset_name,
                'run_id': run_id,
                'error': str(e)
            }

    def _parse_results(self, run_dir: Path, algorithms: List[str]) -> Dict:
        """Parse results from a run."""
        results = {}

        for algo in algorithms:
            summary_file = run_dir / algo / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    results[algo] = json.load(f)
            else:
                results[algo] = {'status': 'NO_SUMMARY'}

        return results

    def run_all_experiments(
        self,
        algorithms: List[str],
        iterations: int,
        num_runs: int,
        output_dir: Path
    ) -> List[Dict]:
        """
        Run experiments on all datasets, repeated num_runs times.

        Args:
            algorithms: List of algorithms
            iterations: Optimization iterations
            num_runs: Number of repetitions (default: 10)
            output_dir: Where to save all results

        Returns:
            List of all results
        """
        self.logger.logger.info("\n" + "="*80)
        self.logger.logger.info(f"STAGE 3: Running LDA Experiments")
        self.logger.logger.info(f"Datasets: {len(self.datasets)}")
        self.logger.logger.info(f"Algorithms: {algorithms}")
        self.logger.logger.info(f"Iterations per run: {iterations}")
        self.logger.logger.info(f"Repetitions: {num_runs}")
        self.logger.logger.info(f"Total experiments: {len(self.datasets) * num_runs}")
        self.logger.logger.info("="*80)

        all_results = []
        start_time = time.time()

        total_experiments = len(self.datasets) * num_runs
        pbar = tqdm(total=total_experiments, desc="Running experiments")

        for dataset in self.datasets:
            for run_id in range(num_runs):
                result = self.run_single_experiment(
                    dataset_name=dataset,
                    algorithms=algorithms,
                    iterations=iterations,
                    run_id=run_id,
                    output_dir=output_dir
                )
                all_results.append(result)
                pbar.update(1)

        pbar.close()

        total_time = time.time() - start_time

        # Count successes/failures
        successes = sum(1 for r in all_results if r['status'] == 'SUCCESS')
        failures = len(all_results) - successes

        self.logger.log_stage(
            "LDA_Experiments",
            f"COMPLETED",
            total_time,
            total_experiments=total_experiments,
            successes=successes,
            failures=failures
        )

        return all_results


class ResultsAggregator:
    """Aggregates and analyzes results from multiple runs."""

    def __init__(self, logger: PipelineLogger):
        self.logger = logger

    def aggregate_results(self, all_results: List[Dict], output_dir: Path):
        """
        Aggregate results from all runs and create comprehensive analysis.

        Args:
            all_results: List of all experiment results
            output_dir: Directory for aggregated results
        """
        self.logger.logger.info("\n" + "="*80)
        self.logger.logger.info("STAGE 4: Aggregating Results")
        self.logger.logger.info("="*80)

        agg_dir = output_dir / "aggregated_results"
        agg_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = self._results_to_dataframe(all_results)
        df.to_csv(agg_dir / "all_results.csv", index=False)

        # Generate statistics
        stats = self._compute_statistics(df)
        with open(agg_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

        # Create visualizations
        self._create_visualizations(df, agg_dir)

        self.logger.logger.info(f"Aggregated results saved to {agg_dir}")
        self.logger.log_stage("Results_Aggregation", "SUCCESS", 0)

    def _results_to_dataframe(self, all_results: List[Dict]) -> pd.DataFrame:
        """Convert results list to pandas DataFrame."""
        rows = []

        for result in all_results:
            if result['status'] != 'SUCCESS':
                continue

            base = {
                'dataset': result['dataset'],
                'run_id': result['run_id'],
                'elapsed_time': result['elapsed']
            }

            # Extract algorithm results
            for algo in ['GA', 'ES', 'PABBO_Full']:
                if algo in result:
                    row = base.copy()
                    row['algorithm'] = algo
                    row['best_T'] = result[algo].get('best_T', np.nan)
                    row['best_perplexity'] = result[algo].get('best_perplexity', np.nan)
                    row['total_time'] = result[algo].get('total_time', np.nan)
                    row['num_iterations'] = result[algo].get('num_iterations', np.nan)
                    rows.append(row)

        return pd.DataFrame(rows)

    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics."""
        stats = {}

        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            stats[dataset] = {}

            for algo in df['algorithm'].unique():
                algo_df = dataset_df[dataset_df['algorithm'] == algo]

                if len(algo_df) > 0:
                    stats[dataset][algo] = {
                        'mean_perplexity': float(algo_df['best_perplexity'].mean()),
                        'std_perplexity': float(algo_df['best_perplexity'].std()),
                        'min_perplexity': float(algo_df['best_perplexity'].min()),
                        'max_perplexity': float(algo_df['best_perplexity'].max()),
                        'mean_time': float(algo_df['total_time'].mean()),
                        'std_time': float(algo_df['total_time'].std()),
                        'num_runs': len(algo_df)
                    }

        return stats

    def _create_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Create comprehensive visualizations."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. Perplexity comparison across datasets
        self._plot_perplexity_comparison(df, output_dir)

        # 2. Time comparison
        self._plot_time_comparison(df, output_dir)

        # 3. Box plots for each dataset
        self._plot_boxplots(df, output_dir)

        # 4. Convergence curves (if history available)
        # TODO: Implement if history data is accessible

        self.logger.logger.info(f"Visualizations saved to {output_dir}")

    def _plot_perplexity_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot perplexity comparison across algorithms and datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        datasets = df['dataset'].unique()

        for idx, dataset in enumerate(datasets):
            if idx >= 4:
                break

            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]

            # Bar plot with error bars
            means = dataset_df.groupby('algorithm')['best_perplexity'].mean()
            stds = dataset_df.groupby('algorithm')['best_perplexity'].std()

            means.plot(kind='bar', yerr=stds, ax=ax, capsize=5)
            ax.set_title(f'Dataset: {dataset}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
            ax.set_xlabel('Algorithm', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "perplexity_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_time_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot execution time comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        time_data = df.groupby(['dataset', 'algorithm'])['total_time'].mean().unstack()
        time_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title('Average Execution Time by Algorithm and Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.legend(title='Algorithm', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_boxplots(self, df: pd.DataFrame, output_dir: Path):
        """Plot box plots for perplexity distribution."""
        datasets = df['dataset'].unique()
        n_datasets = len(datasets)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            if idx >= 4:
                break

            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]

            sns.boxplot(
                data=dataset_df,
                x='algorithm',
                y='best_perplexity',
                ax=ax,
                palette='Set2'
            )

            ax.set_title(f'Perplexity Distribution: {dataset}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Perplexity', fontsize=12)
            ax.set_xlabel('Algorithm', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / "perplexity_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main pipeline execution."""
    print("="*80)
    print("LDA HYPERPARAMETER OPTIMIZATION - FULL PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    pipeline_logger = PipelineLogger(results_dir / "logs")
    pipeline_logger.logger.info("Pipeline started")

    try:
        # =====================================================================
        # STAGE 1: Train PABBO Model
        # =====================================================================
        trainer = PABBOTrainer(pipeline_logger)
        model_path, training_time = trainer.train_light_model()

        pipeline_logger.logger.info(f"✓ PABBO model trained: {model_path}")
        pipeline_logger.logger.info(f"✓ Training time: {training_time:.2f}s ({training_time/60:.2f}min)")

        # =====================================================================
        # STAGE 2: Evaluate PABBO Model
        # =====================================================================
        eval_metrics = trainer.evaluate_model(model_path)
        pipeline_logger.logger.info(f"✓ Evaluation completed: {eval_metrics['status']}")

        # =====================================================================
        # STAGE 3: Run LDA Experiments (10 runs on all datasets)
        # =====================================================================
        experiment_runner = LDAExperimentRunner(pipeline_logger, model_path)

        algorithms = ["GA", "ES", "PABBO_Full"]
        iterations = 200
        num_runs = 10

        all_results = experiment_runner.run_all_experiments(
            algorithms=algorithms,
            iterations=iterations,
            num_runs=num_runs,
            output_dir=results_dir / "experiments"
        )

        # Save raw results
        with open(results_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

        # =====================================================================
        # STAGE 4: Aggregate Results and Create Visualizations
        # =====================================================================
        aggregator = ResultsAggregator(pipeline_logger)
        aggregator.aggregate_results(all_results, results_dir)

        # =====================================================================
        # Pipeline Complete
        # =====================================================================
        pipeline_logger.logger.info("\n" + "="*80)
        pipeline_logger.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        pipeline_logger.logger.info("="*80)
        pipeline_logger.logger.info(f"Results saved to: {results_dir}")
        pipeline_logger.logger.info(f"Model path: {model_path}")
        pipeline_logger.logger.info(f"Total experiments: {len(all_results)}")

        successes = sum(1 for r in all_results if r['status'] == 'SUCCESS')
        pipeline_logger.logger.info(f"Successful runs: {successes}/{len(all_results)}")

        pipeline_logger.save_metrics()

        return 0

    except Exception as e:
        pipeline_logger.logger.error(f"\n{'='*80}")
        pipeline_logger.logger.error("PIPELINE FAILED!")
        pipeline_logger.logger.error(f"Error: {e}")
        pipeline_logger.logger.error("="*80)

        import traceback
        pipeline_logger.logger.error(traceback.format_exc())

        pipeline_logger.save_metrics()

        return 1


if __name__ == "__main__":
    sys.exit(main())