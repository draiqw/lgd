#!/bin/bash
#SBATCH --job-name=lda_pabbo_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# ============================================================================
# SLURM Script for GPU-Accelerated LDA Hyperparameter Optimization
# ============================================================================
#
# This script runs for_klaster_gpu.py on a GPU node
#
# Features:
# - Uses 1 GPU for transformer training
# - Uses 32 CPU cores for LDA experiments
# - Runs in ~8-12 hours (vs 20-24 hours on CPU)
#
# GPU Requirements:
# - 8GB+ GPU memory (minimum)
# - 16GB+ GPU memory (recommended)
#
# ============================================================================

# Configuration
CONTAINER_IMAGE="/scratch/$USER/llabs_lda_hyperopt_gpu.sqsh"
OUTPUT_BASE="/scratch/$USER/lda_results_gpu"
SCRIPT_NAME="for_klaster_gpu.py"

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "============================================================================"
echo "GPU-Accelerated LDA Hyperparameter Optimization Pipeline"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "============================================================================"

# Display GPU information
echo ""
echo "GPU Information:"
echo "----------------------------------------------------------------------------"
nvidia-smi
echo "============================================================================"

# Display CUDA environment
echo ""
echo "CUDA Environment:"
echo "----------------------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"
echo "============================================================================"

# Run the container
echo ""
echo "Starting Container..."
echo "============================================================================"

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$OUTPUT_BASE:/app/lda_pipeline_results" \
     --container-writable \
     python3 /app/"$SCRIPT_NAME"

EXIT_CODE=$?

echo "============================================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================================"

# Display final GPU state
echo ""
echo "Final GPU State:"
echo "----------------------------------------------------------------------------"
nvidia-smi
echo "============================================================================"

# Check if results were generated
if [ -d "$OUTPUT_BASE" ]; then
    echo ""
    echo "Results saved to: $OUTPUT_BASE"
    echo "Latest run:"
    ls -lht "$OUTPUT_BASE" | head -n 5
else
    echo ""
    echo "WARNING: Output directory not found!"
fi

exit $EXIT_CODE