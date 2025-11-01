
# Инструкция по предобучению и использованию полного PABBO с Transformer

## Введение

Полный PABBO (Preference-Augmented Bayesian Black-Box Optimization) с Transformer - это сложная система, требующая предобучения нейронной сети. Этот гайд поможет вам настроить и запустить полный PABBO для тестов.

## Что уже сделано

✅ **1. Усложнена тестовая функция** в `test.py`:
   - Rastrigin-like функция с множеством локальных минимумов
   - Более долгий поиск оптимума
   - Диапазон: [-5.12, 5.12]

✅ **2. Одинаковая инициализация для всех алгоритмов**:
   - `shared_population` используется для GA, ES, и PABBO
   - Latin Hypercube Sampling для лучшего покрытия пространства

✅ **3. Создана конфигурация для быстрого обучения**:
   - `pabbo_method/configs/train_rastrigin1d_test.yaml`
   - Оптимизирована для быстрого обучения (~10-20 мин на CPU)
   - Уменьшено количество шагов: 2000 вместо 8000

✅ **4. Добавлена тестовая функция** в PABBO:
   - `rastrigin1D` в `pabbo_method/data/function.py`
   - Совместима с архитектурой PABBO

## Шаги для запуска полного PABBO

### Шаг 1: Установка зависимостей

```bash
# Убедитесь что установлены все зависимости для PABBO
cd /Users/draiqws/Llabs
pip install torch==2.0.1  # или более новая версия
pip install botorch gpytorch hydra-core wandb omegaconf
```

### Шаг 2: Предобучение модели PABBO (ВАЖНО!)

```bash
# Перейдите в директорию pabbo_method
cd /Users/draiqws/Llabs/pabbo_method

# Запустите обучение (займёт ~10-20 минут на CPU, ~5 минут на GPU)
python train.py --config-name train_rastrigin1d_test

# ПРИМЕЧАНИЕ: Обучение создаст модель в:
# results/PABBO/PABBO_rastrigin1d_test_quick_<timestamp>/ckpt.tar
```

**Прогресс обучения:**
- Шаги 1-800: burnin phase (только prediction task)
- Шаги 801-2000: full training (prediction + acquisition)
- Сохранение каждые 500 шагов

### Шаг 3: Создание адаптера для test.py

После обучения модели нужно создать адаптер, который будет использовать обученную модель в `test.py`. Это требует:

1. **Wrapper класс** для preference-based оптимизации
2. **Конвертацию** между integer T и continuous x
3. **Интеграцию** с preference-based подходом PABBO

Пример структуры адаптера (нужно реализовать):

```python
class PABBOTransformerOptimizer(BaseOptimizer):
    """
    PABBO optimizer using pre-trained Transformer model.
    """

    def __init__(self, obj, eval_func, T_bounds, seed, model_path, ...):
        # Load pre-trained model
        self.model = TransformerModel.load_from_checkpoint(model_path)
        # Setup preference-based evaluation
        ...

    def run(self, iterations, ...):
        # Convert to preference-based optimization
        # Use model for acquisition function
        # Return results in same format as other optimizers
        ...
```

### Шаг 4: Интеграция в test.py

После создания адаптера, обновите `test.py`:

```python
from exp_pabbo_transformer import PABBOTransformerOptimizer  # New!

# В main():
pabbo_transformer_result = run_algorithm_test(
    "PABBO-Transformer",
    PABBOTransformerOptimizer,
    x_range,
    T_range,
    iterations=iterations,
    pop_size=pop_size,
    seed=seed,
    initial_population=shared_population,
    model_path="pabbo_method/results/PABBO/.../ckpt.tar"
)
results_list.append(pabbo_transformer_result)
```

## Текущий статус

### ✅ Готово:
1. Тестовая функция усложнена
2. Одинаковая инициализация для всех алгоритмов
3. Конфигурация для обучения создана
4. Функция добавлена в PABBO

### ⏳ Требуется выполнить:
1. **Запустить предобучение** (~10-20 мин):
   ```bash
   cd pabbo_method
   python train.py --config-name train_rastrigin1d_test
   ```

2. **Создать адаптер** (`all_exps/exp_pabbo_transformer.py`):
   - Wrapper для обученной модели
   - Preference-based evaluation
   - Интеграция с test.py API

3. **Обновить test.py**:
   - Импортировать адаптер
   - Добавить PABBO-Transformer в список алгоритмов
   - Обновить графики

## Альтернатива: Улучшенный упрощённый PABBO

Если полное предобучение занимает слишком много времени, можно улучшить текущий упрощённый PABBO в `exp_pabbo.py`:

1. Добавить preference-based подход (без нейросети)
2. Улучшить acquisition function
3. Добавить более сложную стратегию exploration-exploitation

Это даст лучшие результаты чем текущий случайный поиск, но без сложности Transformer.

## Проверка корректности

После интеграции полного PABBO, проверьте:

1. **Одинаковая инициализация**: все алгоритмы стартуют с `shared_population`
2. **Convergence**: PABBO-Transformer должен показать лучшую сходимость
3. **Final results**: сравните best_y для всех алгоритмов
4. **True minimum**: -0.92 (approx) для Rastrigin-like на [-5.12, 5.12]

## Полезные ссылки

- PABBO paper: Preference-Augmented Bayesian Optimization
- Конфигурации: `pabbo_method/configs/`
- Evaluation script: `pabbo_method/evaluate_continuous.py`

## Контакты

Если возникнут вопросы или проблемы, создайте issue в репозитории или обратитесь к документации PABBO.
