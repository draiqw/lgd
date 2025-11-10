# ============================================================================
# Dockerfile for LDA Hyperparameter Optimization on Cluster
# ============================================================================
# This container includes all dependencies for running for_klaster.py
# on a SLURM cluster with CPU resources.
# ============================================================================

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Create directories for data and results
RUN mkdir -p /app/data && \
    mkdir -p /app/lda_pipeline_results && \
    mkdir -p /app/results

# Set Python path to include project directories
ENV PYTHONPATH="/app:/app/pabbo_method:/app/lda_hyperopt:${PYTHONPATH}"

# Make sure matplotlib uses non-interactive backend
ENV MPLBACKEND=Agg

# Set number of threads for numpy/scipy (prevent oversubscription)
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Default command (can be overridden in sbatch script)
CMD ["python3", "for_klaster.py"]
