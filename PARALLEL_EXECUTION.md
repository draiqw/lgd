# LDA Parallel Execution

## Файлы

### `lda_parallel.py`
Параллельная версия `lda.py` с следующими особенностями:

**Архитектура параллелизации:**
- **3 процесса** (multiprocessing) - по одному на каждый алгоритм оптимизации:
  - Процесс 1: GA (Genetic Algorithm)
  - Процесс 2: ES (Evolution Strategy)
  - Процесс 3: PABBO_Full

- **4 потока** (threading) внутри каждого процесса - по одному на каждый датасет

- **10 последовательных запусков** в каждом потоке с разными сидами (42+0, 42+1, ..., 42+9)

**Ключевые отличия от lda.py:**
- ✅ **20 итераций** вместо 200
- ✅ **Отключен early stopping** - все алгоритмы выполняют ровно 20 итераций
- ✅ Использует `run_no_early_stop.py` вместо `run.py`
- ✅ Thread-safe логирование с указанием процесса/потока
- ✅ Синхронизация через multiprocessing.Queue
- ✅ Real-time прогресс через tqdm

### `lda_hyperopt/run_no_early_stop.py`
Специальная версия `run.py` для параллельного выполнения:

**Изменения:**
- `early_stop_eps_pct=0.0` - досрочная остановка отключена
- `max_no_improvement=999999` - практически бесконечное ожидание улучшений
- Все алгоритмы выполняют ровно заданное количество итераций

## Использование

### Запуск параллельной версии:
```bash
python lda_parallel.py
```

### Запуск обычной версии (для сравнения):
```bash
python lda.py
```

## Сравнение производительности

| Параметр | lda.py (последовательно) | lda_parallel.py (параллельно) |
|----------|-------------------------|------------------------------|
| Итераций на запуск | 20 | 20 |
| Early stopping | Нет | Нет |
| Процессы | 1 | 3 (GA, ES, PABBO_Full) |
| Потоки на процесс | 1 | 4 (по датасетам) |
| Общее ускорение | 1x | ~10-12x (зависит от CPU) |

**Обновление:** Оба файла (`lda.py` и `lda_parallel.py`) теперь используют **20 итераций без early stopping** для честного сравнения алгоритмов.

## Архитектура

```
lda_parallel.py
│
├── STAGE 1: Train PABBO Model
│   └── PABBOTrainer.train_light_model()
│
├── STAGE 2: Evaluate PABBO Model
│   └── PABBOTrainer.evaluate_model()
│
├── STAGE 3: Run Experiments (PARALLEL)
│   │
│   ├── Process 1 (GA)
│   │   ├── Thread 1: Dataset 1 → 10 runs (seed 42-51)
│   │   ├── Thread 2: Dataset 2 → 10 runs (seed 42-51)
│   │   ├── Thread 3: Dataset 3 → 10 runs (seed 42-51)
│   │   └── Thread 4: Dataset 4 → 10 runs (seed 42-51)
│   │
│   ├── Process 2 (ES)
│   │   ├── Thread 1: Dataset 1 → 10 runs (seed 42-51)
│   │   ├── Thread 2: Dataset 2 → 10 runs (seed 42-51)
│   │   ├── Thread 3: Dataset 3 → 10 runs (seed 42-51)
│   │   └── Thread 4: Dataset 4 → 10 runs (seed 42-51)
│   │
│   └── Process 3 (PABBO_Full)
│       ├── Thread 1: Dataset 1 → 10 runs (seed 42-51)
│       ├── Thread 2: Dataset 2 → 10 runs (seed 42-51)
│       ├── Thread 3: Dataset 3 → 10 runs (seed 42-51)
│       └── Thread 4: Dataset 4 → 10 runs (seed 42-51)
│
└── STAGE 4: Aggregate Results
    └── ResultsAggregator.aggregate_results()
```

## Технические детали

### Thread-safe операции
- `ThreadSafePipelineLogger` использует `threading.Lock()` для безопасного логирования
- `multiprocessing.Manager()` для создания shared queues между процессами
- Каждый поток/процесс имеет уникальное имя в логах

### Сбор результатов
- `results_queue` - для результатов всех экспериментов
- `progress_queue` - для отслеживания прогресса в реальном времени
- Результаты собираются после завершения всех процессов

### Выходные данные
Структура такая же как в `lda.py`:
```
lda_pipeline_results/
└── run_parallel_YYYYMMDD_HHMMSS/
    ├── logs/
    │   ├── pipeline_main.log
    │   └── pipeline_metrics.json
    ├── experiments/
    │   └── [dataset_name]/
    │       ├── GA/
    │       ├── ES/
    │       └── PABBO_Full/
    ├── aggregated_results/
    │   ├── all_results.csv
    │   ├── statistics.json
    │   └── visualizations/
    └── all_results.json
```

## Рекомендации

1. **CPU**: Рекомендуется минимум 8 ядер для эффективной работы (3 процесса × 4 потока = 12 параллельных задач)
2. **RAM**: Минимум 16GB, рекомендуется 32GB для больших датасетов
3. **Диск**: SSD для быстрой записи логов и результатов

## Отладка

Логи содержат информацию о процессе и потоке:
```
2024-11-04 10:15:30 [INFO] [GA-dataset1] Thread started for dataset: dataset1
```

Формат: `[ProcessName-ThreadName]`
