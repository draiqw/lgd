# 🚀 GPU Version Setup Guide

Руководство по запуску GPU-оптимизированной версии PABBO pipeline.

---

## Основные отличия от CPU версии

### `for_klaster_gpu.py` vs `for_klaster.py`

| Параметр | CPU (for_klaster.py) | GPU (for_klaster_gpu.py) |
|----------|---------------------|--------------------------|
| **Small Model Training** |
| n_steps | 2000 | 4000 |
| batch_size | 16 | 32 |
| device | cpu | cuda |
| **Large Model Training** |
| n_steps | 8000 | 12000 |
| batch_size | 32 | 64 |
| device | cpu | cuda |
| **Evaluation** |
| eval_num_query_points | 256 | 512 |
| max_num_ctx | 20 | 50 |

### Новые возможности

✅ Автоматическое определение GPU
✅ Мониторинг GPU памяти
✅ Автоматическая очистка GPU кэша между этапами
✅ Оптимизированные параметры для GPU
✅ Детальное логирование использования GPU

---

## Требования

### Минимальные требования

- **GPU**: NVIDIA GPU с 8+ GB памяти (минимум)
- **CUDA**: 11.0 или выше
- **PyTorch**: 2.0+ с CUDA support
- **Драйверы**: NVIDIA Driver 470+

### Рекомендуемые требования

- **GPU**: NVIDIA GPU с 16+ GB памяти (A100, V100, RTX 3090, RTX 4090)
- **CUDA**: 11.8 или 12.1
- **PyTorch**: 2.1+ с CUDA support

---

## Проверка GPU

### Быстрая проверка

```bash
# Проверка доступности CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Информация о GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Проверка версии CUDA
python -c "import torch; print('CUDA version:', torch.version.cuda)"
```

### Детальная информация

```python
import torch

if torch.cuda.is_available():
    print("✓ CUDA is available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("✗ CUDA is not available")
```

---

## Локальный запуск (для тестирования)

```bash
# Перейдите в директорию проекта
cd /path/to/Llabs

# Запустите GPU версию
python for_klaster_gpu.py
```

Скрипт автоматически:
1. Проверит доступность GPU
2. Выведет информацию о GPU
3. Обучит модели на GPU
4. Запустит LDA эксперименты
5. Создаст отчеты с GPU метриками

---

## Запуск на кластере

### 1. Создайте SLURM скрипт для GPU

Создайте файл `run_gpu.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=lda_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1                    # Запросить 1 GPU
#SBATCH --gres=gpu:1                # Альтернативный синтаксис
#SBATCH --mem=64G                   # Память
#SBATCH --time=12:00:00             # Меньше времени чем CPU версия

# Опционально: конкретный тип GPU
# #SBATCH --gres=gpu:a100:1
# #SBATCH --gres=gpu:v100:1
# #SBATCH --gres=gpu:rtx3090:1

# Load modules (если требуется на вашем кластере)
# module load cuda/11.8
# module load python/3.9

# Activate environment
source /path/to/venv/bin/activate

# Информация о системе
echo "==================================================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "==================================================================="

# Проверка GPU
nvidia-smi

# Запуск
python for_klaster_gpu.py

echo "==================================================================="
echo "Job finished at: $(date)"
echo "==================================================================="
```

### 2. Отправьте задачу

```bash
sbatch run_gpu.sh
```

### 3. Мониторинг

```bash
# Статус задачи
squeue -u $USER

# Лог в реальном времени
tail -f slurm-<jobid>.out

# GPU использование
ssh <compute-node> "nvidia-smi"
```

---

## Docker с GPU

### Dockerfile для GPU

Обновите существующий `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN pip3 install -r requirements.txt

# Copy project files
COPY . /app

# Set entrypoint
CMD ["python3", "for_klaster_gpu.py"]
```

### Запуск Docker контейнера с GPU

```bash
# Сборка
docker build -t llabs_gpu:v1 .

# Запуск локально
docker run --gpus all -v $(pwd)/results:/app/lda_pipeline_results llabs_gpu:v1

# Проверка GPU в контейнере
docker run --gpus all llabs_gpu:v1 nvidia-smi
```

### Конвертация в Enroot (для Slurm)

```bash
# На машине с enroot
sudo enroot import docker://yourusername/llabs_gpu:v1
sudo enroot create --name llabs_gpu llabs_gpu.sqsh

# Запуск через enroot
enroot start --rw llabs_gpu python3 /app/for_klaster_gpu.py
```

---

## Ожидаемое время выполнения

### На RTX 3090 (24GB)

- **Small model training**: ~5-8 минут (vs 30 минут на CPU)
- **Large model training**: ~15-20 минут (vs 60 минут на CPU)
- **Total pipeline**: ~8-10 часов (vs 20-24 часов на CPU)

### На A100 (40GB)

- **Small model training**: ~3-5 минут
- **Large model training**: ~8-12 минут
- **Total pipeline**: ~6-8 часов

### На V100 (32GB)

- **Small model training**: ~6-10 минут
- **Large model training**: ~18-25 минут
- **Total pipeline**: ~9-12 часов

---

## Мониторинг GPU

### Во время выполнения

Скрипт автоматически логирует:
- GPU device name
- CUDA version
- Total GPU memory
- Allocated memory после каждого этапа
- Reserved memory
- Max allocated memory

### Ручной мониторинг

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Детальная информация
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 1

# Логирование в файл
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 5 > gpu_log.csv
```

---

## Оптимизация GPU использования

### Если не хватает памяти

1. **Уменьшите batch_size**:
   ```python
   # В for_klaster_gpu.py, строки 288-289 и 295-296
   batch_size = 16  # вместо 32 для small
   batch_size = 32  # вместо 64 для large
   ```

2. **Уменьшите размер модели** (не рекомендуется):
   ```python
   # Уменьшите max_num_ctx
   data.max_num_ctx=30  # вместо 50
   ```

3. **Используйте gradient accumulation** (требует изменения train.py):
   ```python
   train.gradient_accumulation_steps=2
   ```

### Если нужно больше скорости

1. **Увеличьте batch_size** (если есть память):
   ```python
   batch_size = 64  # для small
   batch_size = 128  # для large
   ```

2. **Включите mixed precision** (требует изменения train.py):
   ```python
   train.use_amp=true  # Automatic Mixed Precision
   ```

3. **Используйте несколько GPU** (требует изменения скрипта):
   ```python
   # Добавьте в train command
   experiment.device="cuda:0,cuda:1"
   ```

---

## Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Решение**:
1. Уменьшите `batch_size` (см. выше)
2. Очистите GPU кэш: скрипт делает это автоматически, но можно добавить еще:
   ```python
   torch.cuda.empty_cache()
   ```
3. Используйте GPU с большей памятью

### GPU не обнаружен

```
GPU not available!
```

**Решение**:
1. Проверьте CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Переустановите PyTorch с CUDA:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. Проверьте CUDA_VISIBLE_DEVICES:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   ```

### Медленная работа на GPU

**Причины**:
1. Слишком маленький batch_size
2. CPU bottleneck (data loading)
3. Старая версия CUDA/PyTorch

**Решение**:
1. Увеличьте batch_size
2. Используйте более новую версию PyTorch
3. Проверьте что модель действительно на GPU:
   ```python
   print(next(model.parameters()).device)  # должно быть cuda:0
   ```

---

## Результаты

После выполнения в `lda_pipeline_results/run_gpu_YYYYMMDD_HHMMSS/` будет:

```
logs/
├── pipeline_main.log           # Логи с GPU метриками
└── pipeline_metrics.json       # Метрики (включая GPU memory)

experiments/                     # Результаты экспериментов
aggregated_results/              # Агрегированные результаты
all_results.json                 # Все результаты
```

В логах будет информация вида:
```
✓ GPU Available: NVIDIA A100-SXM4-40GB
✓ GPU Count: 1
✓ CUDA Version: 11.8
✓ PyTorch Version: 2.1.0
✓ Total GPU Memory: 40.00 GB
...
Before training: GPU Memory - Allocated: 0.00GB, Reserved: 0.00GB, Max: 0.00GB
After training: GPU Memory - Allocated: 2.34GB, Reserved: 3.50GB, Max: 3.45GB
```

---

## Сравнение CPU vs GPU

| Метрика | CPU (for_klaster.py) | GPU (for_klaster_gpu.py) |
|---------|---------------------|---------------------------|
| Small training | ~30 min | ~5-8 min |
| Large training | ~60 min | ~15-20 min |
| Total pipeline | ~20-24h | ~8-10h |
| Batch size (small) | 16 | 32 |
| Batch size (large) | 32 | 64 |
| Training steps (small) | 2000 | 4000 |
| Training steps (large) | 8000 | 12000 |

**Ускорение**: ~2-3x для полного пайплайна

---

## Лучшие практики

1. ✅ **Всегда проверяйте GPU** перед запуском длительных экспериментов
2. ✅ **Мониторьте GPU память** во время обучения
3. ✅ **Очищайте GPU кэш** между этапами (скрипт делает автоматически)
4. ✅ **Логируйте GPU метрики** для анализа производительности
5. ✅ **Тестируйте локально** перед запуском на кластере
6. ✅ **Используйте правильный batch_size** для вашей GPU
7. ✅ **Следите за температурой GPU** (nvidia-smi)

---

## FAQ

**Q: Можно ли использовать CPU версию, если есть GPU?**
A: Да, но это будет медленнее. GPU версия оптимизирована для GPU и использует больше параметров обучения.

**Q: Сколько GPU памяти нужно?**
A: Минимум 8GB, рекомендуется 16GB или больше.

**Q: Можно ли использовать несколько GPU?**
A: Текущая версия использует 1 GPU. Для multi-GPU нужны изменения в train.py.

**Q: Работает ли на AMD GPU?**
A: Нет, требуется NVIDIA GPU с CUDA.

**Q: Что если GPU недоступен?**
A: Скрипт выдаст ошибку и предложит использовать CPU версию (for_klaster.py).

---

**Готово! Используйте GPU версию для ускорения обучения! 🚀**