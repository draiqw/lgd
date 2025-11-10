#!/bin/bash

#SBATCH --job-name=lda_hyperopt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

CONTAINER_IMAGE="./draiqws+llabs_lda_hyperopt+v1.sqsh"

RESULTS_DIR="/scratch/akramovrr/lda_results"

mkdir -p "$RESULTS_DIR"

echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID:              $SLURM_JOB_ID"
echo "Job Name:            $SLURM_JOB_NAME"
echo "Node:                $SLURM_NODELIST"
echo "CPUs per task:       $SLURM_CPUS_PER_TASK"
echo "Start time:          $(date)"
echo "Container image:     $CONTAINER_IMAGE"
echo "Results directory:   $RESULTS_DIR"
echo "============================================================================"
echo ""

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "$RESULTS_DIR:/app/lda_pipeline_results" \
     --container-workdir /app \
     bash -c "python3 for_klaster.py"

echo ""
echo "============================================================================"
echo "Job completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================================"