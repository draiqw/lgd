# LDA Hyperparameter Optimization Experiments

Модульная реализация экспериментов по оптимизации гиперпараметров LDA с использованием трёх алгоритмов: Genetic Algorithm (GA), Evolution Strategy (ES) и PABBO-inspired Random Search.

## Структура проекта

```
all_exps/
├── README.md              # Этот файл
├── __init__.py            # Инициализация пакета
├── main.py                # Главный скрипт для запуска экспериментов
├── utils.py               # Вспомогательные функции и базовые классы
├── exp_ga.py              # Реализация Genetic Algorithm
├── exp_es.py              # Реализация Evolution Strategy
└── exp_pabbo.py           # Реализация PABBO-inspired алгоритма
```

## Описание алгоритмов

### 1. Genetic Algorithm (GA)
- **Файл**: `exp_ga.py`
- **Особенности**:
  - Бинарный кроссовер на уровне битов
  - Целочисленная мутация с ограниченным случайным блужданием
  - Турнирная селекция
  - Элитизм (лучшие особи всегда выживают)
  - Ранняя остановка на основе относительного улучшения

### 2. Evolution Strategy (ES)
- **Файл**: `exp_es.py`
- **Особенности**:
  - (μ + λ) стратегия селекции
  - Мутация без кроссовера (стандартный подход ES)
  - Целочисленная мутация с ограниченным случайным блужданием
  - Ранняя остановка

### 3. PABBO-inspired Random Search
- **Файл**: `exp_pabbo.py`
- **Особенности**:
  - Адаптивный случайный поиск
  - Баланс exploration-exploitation
  - Latin Hypercube Sampling для начальной выборки
  - Температурный контроль для постепенного сужения области поиска

## Использование

### Базовое использование

```bash
# Запуск всех алгоритмов на всех датасетах (последовательно)
python main.py --sequential

# Параллельный запуск
python main.py --parallel --max-workers 4

# Запуск только одного алгоритма
python main.py --algorithm ga --sequential

# Запуск на конкретных датасетах
python main.py --datasets 20news agnews --sequential

# Настройка параметров
python main.py --sequential --iterations 100 --pop-size 20 --seed 42
```

### Параметры командной строки

- `--parallel` - Параллельное выполнение экспериментов
- `--sequential` - Последовательное выполнение (по умолчанию)
- `--algorithm {ga,es,pabbo,all}` - Выбор алгоритма (по умолчанию: all)
- `--datasets DATASET [DATASET ...]` - Выбор датасетов
- `--iterations N` - Число итераций (по умолчанию: 200)
- `--pop-size N` - Размер популяции (по умолчанию: 10)
- `--seed N` - Random seed (по умолчанию: 42)
- `--output-dir DIR` - Директория для результатов (по умолчанию: results)
- `--max-workers N` - Число параллельных воркеров (по умолчанию: 4)

## Архитектура и ООП принципы

### Базовый класс `BaseOptimizer`

Все оптимизаторы наследуются от абстрактного класса `BaseOptimizer`, который определяет общий интерфейс:

```python
class BaseOptimizer(ABC):
    def __init__(self, obj, eval_func, T_bounds, seed, ...):
        # Инициализация

    @abstractmethod
    def run(self, iterations, writer, outdir, initial_population):
        # Основной метод оптимизации

    def decode(self, individual):
        # Декодирование особи в (T, alpha, eta)
```

### Справедливое сравнение

Для обеспечения справедливого сравнения алгоритмов:
1. **Общая начальная популяция**: все алгоритмы стартуют с одинаковых начальных точек
2. **Одинаковые параметры LDA**: max_iter, batch_size, learning_method
3. **Одинаковые критерии остановки**: early_stop_eps_pct, max_no_improvement
4. **Фиксированный seed**: для воспроизводимости результатов

## Логирование

### Уровни логирования

1. **Console logging**: основные этапы и прогресс
2. **File logging**: детальные логи каждого эксперимента (`experiment.log`)
3. **TensorBoard**: метрики в реальном времени
4. **CSV файлы**: история оптимизации, популяции
5. **JSON файлы**: итоговые результаты

### Просмотр логов TensorBoard

```bash
tensorboard --logdir results/
```

## Выходные файлы

Для каждого эксперимента создаётся следующая структура:

```
results/
├── {dataset}/
│   ├── ga/
│   │   ├── tensorboard/          # TensorBoard логи
│   │   ├── experiment.log         # Текстовый лог
│   │   ├── history.csv            # История оптимизации
│   │   ├── summary.json           # Итоговая сводка
│   │   ├── population_gen_*.csv   # Популяции по поколениям
│   │   ├── perplexity.png         # График perplexity
│   │   ├── T_over_time.png        # График T
│   │   ├── step_time.png          # График времени
│   │   └── population_stats.png   # Статистика популяции
│   ├── es/
│   │   └── ...                    # Аналогичная структура
│   └── pabbo/
│       └── ...                    # Аналогичная структура
└── all_results.json               # Сводные результаты всех экспериментов
```

## Датасеты

Используются 4 датасета в формате bag-of-words (scipy sparse матрицы):
- `20news` - 20 Newsgroups
- `agnews` - AG News
- `val_out` - Validation Out
- `yelp` - Yelp Reviews

Датасеты должны находиться в `../data/` относительно папки `all_exps/`.

## Методология оптимизации

**Правильный подход к подбору гиперпараметров LDA:**

1. ✅ Обучаем LDA на **validation** выборке
2. ✅ Оптимизируем для минимизации **perplexity** на той же validation выборке
3. ✅ Ищем оптимальное T (число топиков)
4. ✅ Alpha и eta устанавливаются как 1/T (стандартная эвристика)

## Производительность

### Время выполнения

- **Последовательно**: ~2-6 часов на все датасеты и алгоритмы (зависит от числа итераций)
- **Параллельно (4 воркера)**: ~0.5-2 часа

### Оптимизация скорости

1. Используйте кэширование (уже реализовано в `EVAL_CACHE`)
2. Уменьшите `max_iter_lda` для быстрых экспериментов
3. Уменьшите число итераций оптимизации
4. Используйте параллельный режим

## Примеры использования в коде

```python
from all_exps import GAOptimizer, load_bow_data, make_objective, make_eval_func

# Загрузка данных
Xval = load_bow_data("../data/X_20news_val_bow.npz")

# Создание функций
obj = make_objective(Xval, seed=42, max_iter=60)
eval_func = make_eval_func(Xval, seed=42, max_iter=60)

# Создание оптимизатора
ga = GAOptimizer(
    obj=obj,
    eval_func=eval_func,
    T_bounds=(2, 1000),
    seed=42
)

# Запуск оптимизации
results = ga.run(iterations=100, pop_size=10)

# Результаты
print(f"Best T: {results['best']['T']}")
print(f"Best perplexity: {results['best']['perplexity']:.4f}")
```

## Расширение функциональности

### Добавление нового алгоритма

1. Создайте новый файл `exp_new_algorithm.py`
2. Унаследуйтесь от `BaseOptimizer`
3. Реализуйте метод `run()`
4. Добавьте в `main.py`

```python
class NewAlgorithmOptimizer(BaseOptimizer):
    def __init__(self, obj, eval_func, **kwargs):
        super().__init__(obj, eval_func, **kwargs)
        # Специфичная инициализация

    def run(self, iterations, writer, outdir, initial_population):
        # Реализация алгоритма
        ...
        return results_dict
```

### Добавление новых метрик

Модифицируйте `_fit_eval_on_val` в `utils.py` для расчёта дополнительных метрик.

## Требования

- Python 3.7+
- numpy
- scipy
- scikit-learn
- matplotlib
- tensorboardX
- deap

## Часто задаваемые вопросы (FAQ)

**Q: Почему все алгоритмы используют одну начальную популяцию?**
A: Для справедливого сравнения. Это устраняет случайность начальной выборки как фактор, влияющий на результаты.

**Q: Можно ли оптимизировать alpha и eta отдельно?**
A: Да, но текущая реализация следует распространённой эвристике alpha=eta=1/T. Для независимой оптимизации нужно расширить пространство поиска.

**Q: Почему используется validation set для обучения и оценки?**
A: Это стандартный подход для подбора гиперпараметров. После нахождения оптимальных параметров, финальная модель обучается на train+val и оценивается на test.

**Q: Как ускорить эксперименты?**
A: Уменьшите `max_iter_lda` (например, до 30), число итераций оптимизации, или используйте параллельный режим.

## Лицензия

MIT License

## Контакты

Для вопросов и предложений создайте issue в репозитории.