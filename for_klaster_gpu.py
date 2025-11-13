"""
=============================================================================
LDA Hyperparameter Optimization - GPU VERSION
=============================================================================

ОПИСАНИЕ ФАЙЛА:
Этот файл предназначен для запуска на кластере с GPU.

Отличия от for_klaster.py:
1. Автоматическое определение и использование GPU
2. Оптимизированные параметры для GPU (увеличенные batch_size и n_steps)
3. Правильная передача device='cuda' во все команды
4. Мониторинг использования GPU памяти
5. Освобождение GPU памяти между этапами

Предобучаются ДВА трансформера:
   - Маленький (small): d_model=32, nhead=2, n_layers=3, dim_feedforward=64
   - Большой (large): d_model=64, nhead=4, n_layers=6, dim_feedforward=128

Запускаются 4 процесса (4 ядра CPU для LDA экспериментов):
   - Process 1: GA
   - Process 2: ES
   - Process 3: PABBO_Full с маленьким трансформером
   - Process 4: PABBO_Full с большим трансформером

Этапы выполнения:
   - STAGE 0: Проверка GPU и инициализация
   - STAGE 1: Обучение маленького трансформера на GPU
   - STAGE 2: Оценка маленького трансформера на GPU
   - STAGE 3: Обучение большого трансформера на GPU
   - STAGE 4: Оценка большого трансформера на GPU
   - STAGE 5: Параллельный запуск LDA экспериментов (4 процесса, 20 итераций)
   - STAGE 6: Агрегация и визуализация результатов

GPU оптимизации:
   - Автоматическое увеличение batch_size
   - Использование mixed precision (если доступно)
   - Оптимизация числа workers для DataLoader
   - Мониторинг GPU памяти
   - Очистка кэша GPU между этапами

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
from typing import Dict, List, Tuple, Optional
from multiprocessing import Process, Queue, Manager
from threading import Thread, Lock
import queue

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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


def _subprocess_env() -> Dict[str, str]:
    """Return environment dict with project root on PYTHONPATH."""
    env = os.environ.copy()
    py_paths = env.get("PYTHONPATH", "")
    paths = [str(PROJECT_ROOT)]
    if py_paths:
        paths.append(py_paths)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def check_gpu_availability() -> Tuple[bool, Optional[str], Dict]:
    """
    Check if GPU is available and get GPU information.

    Returns:
        (gpu_available, device_name, gpu_info)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None, {}

        device_name = torch.cuda.get_device_name(0)
        gpu_info = {
            'device_name': device_name,
            'device_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

        return True, device_name, gpu_info

    except ImportError:
        return False, None, {'error': 'PyTorch not available'}
    except Exception as e:
        return False, None, {'error': str(e)}


def get_gpu_memory_info() -> Dict:
    """Get current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
                'reserved_gb': torch.cuda.memory_reserved(0) / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated(0) / 1e9,
            }
    except:
        pass
    return {}


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass


class ThreadSafePipelineLogger:
    """Thread-safe centralized logging for the parallel pipeline."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

        # Main log file
        self.main_log = log_dir / "pipeline_main.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [%(processName)s-%(threadName)s] %(message)s',
            handlers=[
                logging.FileHandler(self.main_log),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("LDA_GPU_Pipeline")

        # Metrics tracking
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }

    def log_stage(self, stage_name: str, status: str, duration: float = None, **kwargs):
        """Thread-safe log a pipeline stage."""
        with self.lock:
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

    def log_info(self, message: str):
        """Thread-safe info logging."""
        with self.lock:
            self.logger.info(message)

    def log_error(self, message: str):
        """Thread-safe error logging."""
        with self.lock:
            self.logger.error(message)

    def log_gpu_memory(self, prefix: str = ""):
        """Log current GPU memory usage."""
        mem_info = get_gpu_memory_info()
        if mem_info:
            with self.lock:
                self.logger.info(f"{prefix}GPU Memory - Allocated: {mem_info['allocated_gb']:.2f}GB, "
                               f"Reserved: {mem_info['reserved_gb']:.2f}GB, "
                               f"Max: {mem_info['max_allocated_gb']:.2f}GB")

    def save_metrics(self):
        """Save all metrics to JSON."""
        with self.lock:
            self.metrics['end_time'] = datetime.now().isoformat()
            metrics_file = self.log_dir / "pipeline_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.logger.info(f"Metrics saved to {metrics_file}")


class PABBOTrainerGPU:
    """Handles PABBO model training for both small and large transformers on GPU."""

    def __init__(self, logger: ThreadSafePipelineLogger, device: str = "cuda"):
        self.logger = logger
        self.pabbo_dir = PABBO_METHOD_DIR
        self.device = device

    def train_model(self, model_size: str) -> Tuple[str, float]:
        """
        Train a PABBO model (small or large) on GPU.

        Args:
            model_size: "small" or "large"

        Returns:
            (model_path, training_time)
        """
        if model_size == "small":
            config_name = "train_rastrigin1d_test"
            stage_name = "PABBO_Training_Small_GPU"
            description = "Small Transformer (d_model=32, nhead=2, n_layers=3) on GPU"
            # GPU-optimized parameters for small model
            n_steps = 4000  # Increased from 2000
            batch_size = 32  # Increased from 16
        elif model_size == "large":
            config_name = "train"
            stage_name = "PABBO_Training_Large_GPU"
            description = "Large Transformer (d_model=64, nhead=4, n_layers=6) on GPU"
            # GPU-optimized parameters for large model
            n_steps = 12000  # Increased from 8000
            batch_size = 64  # Increased from 32
        else:
            raise ValueError(f"Invalid model_size: {model_size}. Must be 'small' or 'large'.")

        self.logger.log_info("\n" + "="*80)
        self.logger.log_info(f"Training PABBO Model - {model_size.upper()} ON GPU")
        self.logger.log_info(f"{description}")
        self.logger.log_info(f"GPU-optimized parameters: n_steps={n_steps}, batch_size={batch_size}")
        self.logger.log_info("="*80)

        # Clear GPU cache before training
        clear_gpu_cache()
        self.logger.log_gpu_memory("Before training: ")

        start_time = time.time()

        # Training command with GPU parameters
        cmd = [
            sys.executable,
            str(self.pabbo_dir / "train.py"),
            f"--config-name={config_name}",
            "experiment.wandb=false",
            f"experiment.device={self.device}",
            f"train.n_steps={n_steps}",
            f"train.train_batch_size={batch_size}",
        ]

        # Dataset configuration
        cmd.extend([
            "data.name=GP1D",
            "data.d_x=1",
            "data.x_range=[[-1.0,1.0]]",
            "data.min_num_ctx=1",
            "data.max_num_ctx=50",  # Increased for better training
        ])

        self.logger.log_info(f"Command: {' '.join(cmd)}")
        self.logger.log_info("Training in progress on GPU...")

        try:
            # Run training
            result = subprocess.run(
                cmd,
                cwd=str(self.pabbo_dir),
                capture_output=True,
                text=True,
                env=_subprocess_env(),
            )

            training_time = time.time() - start_time

            # Log GPU memory after training
            self.logger.log_gpu_memory("After training: ")

            if result.returncode != 0:
                self.logger.log_error(f"Training failed for {model_size} model!")
                self.logger.log_error(f"STDOUT: {result.stdout}")
                self.logger.log_error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"PABBO {model_size} model training failed")

            # Find the trained model
            results_dir = self.pabbo_dir / "results" / "PABBO"
            if not results_dir.exists():
                raise FileNotFoundError("Results directory not found")

            # Get the latest model
            model_dirs = sorted(results_dir.glob("*"), key=os.path.getmtime, reverse=True)
            if not model_dirs:
                raise FileNotFoundError("No trained model found")

            model_path = str(model_dirs[0] / "ckpt.tar")

            # Get final GPU memory stats
            mem_info = get_gpu_memory_info()

            self.logger.log_stage(
                stage_name,
                "SUCCESS",
                training_time,
                model_path=model_path,
                config=config_name,
                description=description,
                n_steps=n_steps,
                batch_size=batch_size,
                device=self.device,
                gpu_memory_gb=mem_info.get('max_allocated_gb', 0)
            )

            # Clear GPU cache after training
            clear_gpu_cache()

            return model_path, training_time

        except Exception as e:
            self.logger.log_stage(stage_name, "FAILED", time.time() - start_time, error=str(e))
            clear_gpu_cache()
            raise

    def evaluate_model(self, model_path: str, model_size: str) -> Dict:
        """
        Evaluate the trained PABBO model on GPU.

        Args:
            model_path: Path to model checkpoint
            model_size: "small" or "large"

        Returns:
            Evaluation metrics
        """
        stage_name = f"PABBO_Evaluation_{model_size.capitalize()}_GPU"

        self.logger.log_info("\n" + "="*80)
        self.logger.log_info(f"Evaluating PABBO Model - {model_size.upper()} ON GPU")
        self.logger.log_info("="*80)

        # Clear GPU cache before evaluation
        clear_gpu_cache()
        self.logger.log_gpu_memory("Before evaluation: ")

        start_time = time.time()

        # Extract expid from model path
        expid = Path(model_path).parent.name

        # Model parameters based on size
        if model_size == "small":
            d_model, nhead, n_layers, dim_ff, emb_depth = 32, 2, 3, 64, 2
        else:
            d_model, nhead, n_layers, dim_ff, emb_depth = 64, 4, 6, 128, 3

        # Evaluation command with GPU
        cmd = [
            sys.executable,
            str(self.pabbo_dir / "evaluate_continuous.py"),
            "--config-name=evaluate",
            f"experiment.expid={expid}",
            "experiment.model=PABBO",
            f"experiment.device={self.device}",
            "experiment.wandb=false",
            "data.name=GP1D",
            "data.d_x=1",
            "data.x_range=[[-1.0,1.0]]",
            "data.min_num_ctx=5",
            "data.max_num_ctx=50",
            "eval.eval_num_query_points=512",  # Increased for GPU
            f"model.d_model={d_model}",
            f"model.n_layers={n_layers}",
            f"model.nhead={nhead}",
            f"model.dim_feedforward={dim_ff}",
            f"model.emb_depth={emb_depth}",
        ]

        self.logger.log_info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.pabbo_dir),
                capture_output=True,
                text=True,
                env=_subprocess_env(),
            )

            eval_time = time.time() - start_time

            # Log GPU memory after evaluation
            self.logger.log_gpu_memory("After evaluation: ")

            if result.returncode != 0:
                self.logger.log_info("Evaluation failed (non-critical)")
                self.logger.log_info(f"STDERR: {result.stderr}")

            mem_info = get_gpu_memory_info()

            metrics = {
                'eval_time': eval_time,
                'status': 'SUCCESS' if result.returncode == 0 else 'WARNING',
                'model_size': model_size,
                'device': self.device,
                'gpu_memory_gb': mem_info.get('max_allocated_gb', 0)
            }

            self.logger.log_stage(stage_name, metrics['status'], eval_time, **metrics)

            # Clear GPU cache after evaluation
            clear_gpu_cache()

            return metrics

        except Exception as e:
            self.logger.log_stage(stage_name, "FAILED", time.time() - start_time, error=str(e))
            clear_gpu_cache()
            # Non-critical, continue pipeline
            return {'status': 'FAILED', 'error': str(e), 'model_size': model_size}


class ClusterLDAExperimentRunner:
    """Runs LDA optimization experiments in parallel on 4 cores."""

    def __init__(self, logger: ThreadSafePipelineLogger, model_path_small: str, model_path_large: str):
        self.logger = logger
        self.model_path_small = model_path_small
        self.model_path_large = model_path_large
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

        self.logger.log_info(f"Found {len(datasets)} datasets: {datasets}")
        return sorted(datasets)

    def run_single_experiment(
        self,
        dataset_name: str,
        algorithm: str,
        iterations: int,
        run_id: int,
        output_dir: Path
    ) -> Dict:
        """
        Run a single LDA optimization experiment (one algorithm, one dataset, one seed).

        Args:
            dataset_name: Name of dataset
            algorithm: Algorithm to run (GA, ES, PABBO_Small, PABBO_Large)
            iterations: Number of optimization iterations
            run_id: Run identifier (0-9)
            output_dir: Where to save results

        Returns:
            Results dictionary
        """
        start_time = time.time()

        # Prepare data paths
        val_data = self.data_dir / f"X_{dataset_name}_val_bow.npz"

        if not val_data.exists():
            return {
                'status': 'FAILED',
                'dataset': dataset_name,
                'algorithm': algorithm,
                'run_id': run_id,
                'error': f"Validation data not found for {dataset_name}"
            }

        # Create output directory for this run
        run_dir = output_dir / dataset_name / algorithm / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Determine which model to use
        if algorithm == "PABBO_Small":
            model_path = self.model_path_small
            algo_name = "PABBO_Full"  # Use PABBO_Full optimizer with small model
        elif algorithm == "PABBO_Large":
            model_path = self.model_path_large
            algo_name = "PABBO_Full"  # Use PABBO_Full optimizer with large model
        else:
            model_path = None
            algo_name = algorithm

        # Build command (using run_no_early_stop.py for parallel execution)
        cmd = [
            sys.executable,
            str(self.lda_dir / "run_no_early_stop.py"),
            "--data", str(val_data),
            "--algorithms", algo_name,
            "--iterations", str(iterations),
            "--seed", str(42 + run_id),
            "--outdir", str(run_dir),
            "--init", str(self.lda_dir / "lda_init_population.json"),
        ]

        # Add PABBO model path if needed
        if model_path:
            cmd.extend(["--pabbo-model", model_path])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.lda_dir),
                capture_output=True,
                text=True,
                env=_subprocess_env(),
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                self.logger.log_error(f"Experiment failed: {dataset_name}/{algorithm}/run_{run_id}")
                return {
                    'status': 'FAILED',
                    'dataset': dataset_name,
                    'algorithm': algorithm,
                    'run_id': run_id,
                    'elapsed': elapsed,
                    'error': result.stderr
                }

            # Parse results
            results = self._parse_results(run_dir, algo_name)
            results.update({
                'status': 'SUCCESS',
                'dataset': dataset_name,
                'algorithm': algorithm,  # Keep original algorithm name (PABBO_Small/Large)
                'run_id': run_id,
                'elapsed': elapsed
            })

            return results

        except subprocess.TimeoutExpired:
            self.logger.log_error(f"Experiment timed out: {dataset_name}/{algorithm}/run_{run_id}")
            return {
                'status': 'TIMEOUT',
                'dataset': dataset_name,
                'algorithm': algorithm,
                'run_id': run_id,
                'error': 'Timeout'
            }
        except Exception as e:
            self.logger.log_error(f"Experiment error: {dataset_name}/{algorithm}/run_{run_id} - {e}")
            return {
                'status': 'ERROR',
                'dataset': dataset_name,
                'algorithm': algorithm,
                'run_id': run_id,
                'error': str(e)
            }

    def _parse_results(self, run_dir: Path, algorithm: str) -> Dict:
        """Parse results from a run."""
        summary_file = run_dir / algorithm / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                return {algorithm: json.load(f)}
        else:
            return {algorithm: {'status': 'NO_SUMMARY'}}

    def run_dataset_thread(
        self,
        dataset_name: str,
        algorithm: str,
        iterations: int,
        num_runs: int,
        output_dir: Path,
        results_queue: Queue,
        progress_queue: Queue
    ):
        """
        Thread worker: runs all experiments for one dataset with one algorithm.

        Args:
            dataset_name: Dataset name
            algorithm: Algorithm name (GA, ES, PABBO_Small, PABBO_Large)
            iterations: Optimization iterations
            num_runs: Number of runs (10)
            output_dir: Output directory
            results_queue: Queue to put results
            progress_queue: Queue for progress updates
        """
        self.logger.log_info(f"[{algorithm}] Thread started for dataset: {dataset_name}")

        for run_id in range(num_runs):
            result = self.run_single_experiment(
                dataset_name=dataset_name,
                algorithm=algorithm,
                iterations=iterations,
                run_id=run_id,
                output_dir=output_dir
            )
            results_queue.put(result)
            progress_queue.put(1)  # Signal progress

        self.logger.log_info(f"[{algorithm}] Thread finished for dataset: {dataset_name}")

    def run_algorithm_process(
        self,
        algorithm: str,
        iterations: int,
        num_runs: int,
        output_dir: Path,
        results_queue: Queue,
        progress_queue: Queue
    ):
        """
        Process worker: runs experiments for one algorithm across all datasets using threads.

        Args:
            algorithm: Algorithm name (GA, ES, PABBO_Small, PABBO_Large)
            iterations: Optimization iterations
            num_runs: Number of runs per dataset (10)
            output_dir: Output directory
            results_queue: Queue to collect results
            progress_queue: Queue for progress updates
        """
        self.logger.log_info(f"[{algorithm}] Process started")

        # Create threads for each dataset
        threads = []
        for dataset in self.datasets:
            thread = Thread(
                target=self.run_dataset_thread,
                args=(dataset, algorithm, iterations, num_runs, output_dir, results_queue, progress_queue),
                name=f"{algorithm}-{dataset}"
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        self.logger.log_info(f"[{algorithm}] Process finished")

    def run_all_experiments_parallel(
        self,
        algorithms: List[str],
        iterations: int,
        num_runs: int,
        output_dir: Path
    ) -> List[Dict]:
        """
        Run experiments in parallel:
        - 4 processes (GA, ES, PABBO_Small, PABBO_Large)
        - 4 threads per process (one per dataset)
        - 10 sequential runs per thread

        Args:
            algorithms: List of algorithms (GA, ES, PABBO_Small, PABBO_Large)
            iterations: Optimization iterations
            num_runs: Number of repetitions per dataset (10)
            output_dir: Where to save all results

        Returns:
            List of all results
        """
        self.logger.log_info("\n" + "="*80)
        self.logger.log_info("STAGE 5: Running LDA Experiments (4 CORES - PARALLEL)")
        self.logger.log_info(f"Processes: {len(algorithms)} (4 cores)")
        self.logger.log_info(f"Threads per process: {len(self.datasets)} (one per dataset)")
        self.logger.log_info(f"Runs per thread: {num_runs}")
        self.logger.log_info(f"Algorithms: {algorithms}")
        self.logger.log_info(f"Datasets: {self.datasets}")
        self.logger.log_info(f"Iterations per run: {iterations}")
        self.logger.log_info(f"Total experiments: {len(self.datasets) * num_runs * len(algorithms)}")
        self.logger.log_info("="*80)

        # Shared queues for results and progress
        manager = Manager()
        results_queue = manager.Queue()
        progress_queue = manager.Queue()

        start_time = time.time()

        # Create processes for each algorithm
        processes = []
        for algorithm in algorithms:
            process = Process(
                target=self.run_algorithm_process,
                args=(algorithm, iterations, num_runs, output_dir, results_queue, progress_queue),
                name=algorithm
            )
            processes.append(process)
            process.start()

        # Progress monitoring
        total_experiments = len(self.datasets) * num_runs * len(algorithms)
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            completed = 0
            while completed < total_experiments:
                try:
                    progress_queue.get(timeout=1)
                    completed += 1
                    pbar.update(1)
                except queue.Empty:
                    # Check if all processes are still alive
                    if not any(p.is_alive() for p in processes):
                        break

        # Wait for all processes to complete
        for process in processes:
            process.join()

        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.append(results_queue.get())

        total_time = time.time() - start_time

        # Count successes/failures
        successes = sum(1 for r in all_results if r['status'] == 'SUCCESS')
        failures = len(all_results) - successes

        self.logger.log_stage(
            "LDA_Experiments_Cluster_4Core",
            "COMPLETED",
            total_time,
            total_experiments=total_experiments,
            successes=successes,
            failures=failures,
            processes=len(algorithms),
            threads_per_process=len(self.datasets)
        )

        return all_results


class ResultsAggregator:
    """Aggregates and analyzes results from multiple runs."""

    def __init__(self, logger: ThreadSafePipelineLogger):
        self.logger = logger

    def _save_figure(self, output_dir: Path, filename: str):
        """
        Save figure in both PNG and SVG formats.

        Args:
            output_dir: Directory to save figures
            filename: Base filename without extension
        """
        # Save as PNG (high resolution for presentations)
        plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        # Save as SVG (vector format for publications)
        plt.savefig(output_dir / f"{filename}.svg", format='svg', bbox_inches='tight')
        plt.close()

    def aggregate_results(self, all_results: List[Dict], output_dir: Path):
        """
        Aggregate results from all runs and create comprehensive analysis.

        Args:
            all_results: List of all experiment results
            output_dir: Directory for aggregated results
        """
        self.logger.log_info("\n" + "="*80)
        self.logger.log_info("STAGE 6: Aggregating Results")
        self.logger.log_info("="*80)

        agg_dir = output_dir / "aggregated_results"
        agg_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = self._results_to_dataframe(all_results)
        if df.empty:
            self.logger.log_info("No successful experiment results to aggregate.")
            (agg_dir / "all_results.csv").write_text("status\nNO_SUCCESS\n")
            self.logger.log_stage("Results_Aggregation", "SKIPPED", 0, reason="no_successful_runs")
            return

        df.to_csv(agg_dir / "all_results.csv", index=False)

        # Generate statistics
        stats = self._compute_statistics(df)
        with open(agg_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

        # Create visualizations
        self._create_visualizations(df, agg_dir)

        self.logger.log_info(f"Aggregated results saved to {agg_dir}")
        self.logger.log_stage("Results_Aggregation", "SUCCESS", 0)

    def _results_to_dataframe(self, all_results: List[Dict]) -> pd.DataFrame:
        """Convert results list to pandas DataFrame."""
        rows = []

        for result in all_results:
            if result['status'] != 'SUCCESS':
                continue

            base = {
                'dataset': result['dataset'],
                'algorithm': result['algorithm'],
                'run_id': result['run_id'],
                'elapsed_time': result['elapsed']
            }

            # Extract algorithm results (stored under PABBO_Full key for both PABBO variants)
            algo_key = "PABBO_Full" if "PABBO" in result['algorithm'] else result['algorithm']

            if algo_key in result:
                row = base.copy()
                row['best_T'] = result[algo_key].get('best_T', np.nan)
                row['best_perplexity'] = result[algo_key].get('best_perplexity', np.nan)
                row['total_time'] = result[algo_key].get('total_time', np.nan)
                row['num_iterations'] = result[algo_key].get('num_iterations', np.nan)
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
        plt.rcParams['figure.figsize'] = (14, 8)

        # 1. Perplexity comparison across datasets
        self._plot_perplexity_comparison(df, output_dir)

        # 2. Time comparison
        self._plot_time_comparison(df, output_dir)

        # 3. Box plots for each dataset
        self._plot_boxplots(df, output_dir)

        self.logger.log_info(f"Visualizations saved to {output_dir}")

    def _plot_perplexity_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot perplexity comparison across algorithms and datasets."""
        datasets = df['dataset'].unique()
        n_datasets = len(datasets)
        n_cols = 2
        n_rows = (n_datasets + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_datasets == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, dataset in enumerate(datasets):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

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

        # Hide unused subplots
        for idx in range(n_datasets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        self._save_figure(output_dir, "perplexity_comparison")

    def _plot_time_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot execution time comparison."""
        fig, ax = plt.subplots(figsize=(14, 6))

        time_data = df.groupby(['dataset', 'algorithm'])['total_time'].mean().unstack()
        time_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title('Average Execution Time by Algorithm and Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.legend(title='Algorithm', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self._save_figure(output_dir, "time_comparison")

    def _plot_boxplots(self, df: pd.DataFrame, output_dir: Path):
        """Plot box plots for perplexity distribution."""
        datasets = df['dataset'].unique()
        n_datasets = len(datasets)
        n_cols = 2
        n_rows = (n_datasets + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_datasets == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, dataset in enumerate(datasets):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

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

        # Hide unused subplots
        for idx in range(n_datasets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        self._save_figure(output_dir, "perplexity_boxplots")


def main():
    """Main GPU-accelerated pipeline execution."""
    print("="*80)
    print("LDA HYPERPARAMETER OPTIMIZATION - GPU VERSION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"run_gpu_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    pipeline_logger = ThreadSafePipelineLogger(results_dir / "logs")
    pipeline_logger.log_info("GPU pipeline started")

    try:
        # =====================================================================
        # STAGE 0: Check GPU Availability
        # =====================================================================
        pipeline_logger.log_info("\n" + "="*80)
        pipeline_logger.log_info("STAGE 0: GPU Initialization")
        pipeline_logger.log_info("="*80)

        gpu_available, device_name, gpu_info = check_gpu_availability()

        if not gpu_available:
            pipeline_logger.log_error("GPU not available!")
            pipeline_logger.log_error(f"GPU Info: {gpu_info}")
            pipeline_logger.log_error("Falling back to CPU is not recommended for this script.")
            pipeline_logger.log_error("Please use for_klaster.py for CPU-only execution.")
            raise RuntimeError("GPU not available")

        pipeline_logger.log_info(f"✓ GPU Available: {device_name}")
        pipeline_logger.log_info(f"✓ GPU Count: {gpu_info['device_count']}")
        pipeline_logger.log_info(f"✓ CUDA Version: {gpu_info['cuda_version']}")
        pipeline_logger.log_info(f"✓ PyTorch Version: {gpu_info['pytorch_version']}")
        pipeline_logger.log_info(f"✓ Total GPU Memory: {gpu_info['total_memory_gb']:.2f} GB")

        device = "cuda"

        # =====================================================================
        # STAGE 1: Train PABBO Model (SMALL) on GPU
        # =====================================================================
        trainer = PABBOTrainerGPU(pipeline_logger, device=device)
        model_path_small, training_time_small = trainer.train_model("small")

        pipeline_logger.log_info(f"✓ PABBO SMALL model trained on GPU: {model_path_small}")
        pipeline_logger.log_info(f"✓ Training time: {training_time_small:.2f}s ({training_time_small/60:.2f}min)")

        # =====================================================================
        # STAGE 2: Evaluate PABBO Model (SMALL) on GPU
        # =====================================================================
        eval_metrics_small = trainer.evaluate_model(model_path_small, "small")
        pipeline_logger.log_info(f"✓ SMALL model evaluation completed: {eval_metrics_small['status']}")

        # =====================================================================
        # STAGE 3: Train PABBO Model (LARGE) on GPU
        # =====================================================================
        model_path_large, training_time_large = trainer.train_model("large")

        pipeline_logger.log_info(f"✓ PABBO LARGE model trained on GPU: {model_path_large}")
        pipeline_logger.log_info(f"✓ Training time: {training_time_large:.2f}s ({training_time_large/60:.2f}min)")

        # =====================================================================
        # STAGE 4: Evaluate PABBO Model (LARGE) on GPU
        # =====================================================================
        eval_metrics_large = trainer.evaluate_model(model_path_large, "large")
        pipeline_logger.log_info(f"✓ LARGE model evaluation completed: {eval_metrics_large['status']}")

        # Clear GPU cache before CPU experiments
        clear_gpu_cache()
        pipeline_logger.log_info("✓ GPU cache cleared before LDA experiments")

        # =====================================================================
        # STAGE 5: Run LDA Experiments in PARALLEL (4 processes, 20 iterations)
        # =====================================================================
        experiment_runner = ClusterLDAExperimentRunner(
            pipeline_logger,
            model_path_small,
            model_path_large
        )

        algorithms = ["GA", "ES", "PABBO_Small", "PABBO_Large"]
        iterations = 20  # Fixed 20 iterations, no early stopping
        num_runs = 10

        all_results = experiment_runner.run_all_experiments_parallel(
            algorithms=algorithms,
            iterations=iterations,
            num_runs=num_runs,
            output_dir=results_dir / "experiments"
        )

        # Save raw results
        with open(results_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

        # =====================================================================
        # STAGE 6: Aggregate Results and Create Visualizations
        # =====================================================================
        aggregator = ResultsAggregator(pipeline_logger)
        aggregator.aggregate_results(all_results, results_dir)

        # =====================================================================
        # Pipeline Complete
        # =====================================================================
        pipeline_logger.log_info("\n" + "="*80)
        pipeline_logger.log_info("GPU PIPELINE COMPLETED SUCCESSFULLY!")
        pipeline_logger.log_info("="*80)
        pipeline_logger.log_info(f"Results saved to: {results_dir}")
        pipeline_logger.log_info(f"GPU Used: {device_name}")
        pipeline_logger.log_info(f"Small model path: {model_path_small}")
        pipeline_logger.log_info(f"Large model path: {model_path_large}")
        pipeline_logger.log_info(f"Total experiments: {len(all_results)}")

        successes = sum(1 for r in all_results if r['status'] == 'SUCCESS')
        pipeline_logger.log_info(f"Successful runs: {successes}/{len(all_results)}")

        # Final GPU memory info
        pipeline_logger.log_gpu_memory("Final ")

        pipeline_logger.save_metrics()

        return 0

    except Exception as e:
        pipeline_logger.log_error(f"\n{'='*80}")
        pipeline_logger.log_error("GPU PIPELINE FAILED!")
        pipeline_logger.log_error(f"Error: {e}")
        pipeline_logger.log_error("="*80)

        import traceback
        pipeline_logger.log_error(traceback.format_exc())

        pipeline_logger.save_metrics()

        return 1


if __name__ == "__main__":
    sys.exit(main())