FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p /app/data && \
    mkdir -p /app/lda_pipeline_results && \
    mkdir -p /app/results

ENV PYTHONPATH="/app:/app/pabbo_method:/app/lda_hyperopt"

ENV MPLBACKEND=Agg

ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

CMD ["python3", "for_klaster.py"]
