#!/bin/bash
# ============================================================================
# Cluster Configuration Template
# ============================================================================
# Скопируйте этот файл в cluster_config.sh и заполните своими данными:
#   cp cluster_config.example.sh cluster_config.sh
#
# Затем используйте в скриптах:
#   source cluster_config.sh
# ============================================================================

# ============================================================================
# Docker Hub Configuration
# ============================================================================

# Ваш username на Docker Hub
export DOCKER_USERNAME="YOUR_DOCKERHUB_USERNAME"

# Название образа
export IMAGE_NAME="llabs_lda_hyperopt"

# Версия образа (можно изменять для разных версий)
export IMAGE_VERSION="v1"

# ============================================================================
# Enroot VM Configuration
# ============================================================================

# SSH параметры для виртуалки с enroot
export ENROOT_HOST="188.44.41.125"
export ENROOT_PORT="2295"
export ENROOT_USER="mmp"

# ============================================================================
# SLURM Master Configuration
# ============================================================================

# SSH параметры для slurm-master
export SLURM_HOST="10.36.60.202"
export SLURM_USER="YOUR_SLURM_USERNAME"  # Например: g.skiba

# Рабочая директория на slurm-master
export SLURM_SCRATCH_DIR="/scratch/$SLURM_USER"

# ============================================================================
# Job Configuration
# ============================================================================

# Параметры задачи SLURM
export SLURM_JOB_NAME="lda_hyperopt"
export SLURM_CPUS="128"
export SLURM_GPUS="0"
export SLURM_TIME="24:00:00"  # Максимальное время выполнения (24 часа)

# ============================================================================
# Display current configuration
# ============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "============================================================================"
    echo "Cluster Configuration"
    echo "============================================================================"
    echo "Docker Hub:"
    echo "  Username:      $DOCKER_USERNAME"
    echo "  Image:         $IMAGE_NAME:$IMAGE_VERSION"
    echo ""
    echo "Enroot VM:"
    echo "  Host:          $ENROOT_USER@$ENROOT_HOST:$ENROOT_PORT"
    echo ""
    echo "SLURM Master:"
    echo "  Host:          $SLURM_USER@$SLURM_HOST"
    echo "  Scratch Dir:   $SLURM_SCRATCH_DIR"
    echo ""
    echo "Job Settings:"
    echo "  Name:          $SLURM_JOB_NAME"
    echo "  CPUs:          $SLURM_CPUS"
    echo "  GPUs:          $SLURM_GPUS"
    echo "  Max Time:      $SLURM_TIME"
    echo "============================================================================"
fi