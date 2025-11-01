# Optimization Algorithms Testing Framework

Тестирование трёх алгоритмов оптимизации (GA, ES, PABBO) на сложной недифференцируемой функции.

## 🎯 Характеристики системы

### Алгоритмы
- **GA (Genetic Algorithm)** - Генетический алгоритм
- **ES (Evolution Strategy)** - Эволюционная стратегия
- **PABBO (Preference-Augmented BBO)** - Упрощённая версия с адаптивным поиском

### Тестовая функция

**Сложная недифференцируемая функция:**

```
y(x) = |sin(2πx)| + 0.5*|cos(5πx)| + 0.3*x² + 2*|x| +
       |sin(10πx)|*0.2 + step_function(x)
```

**Характеристики:**
- ❌ Недифференцируемая (абсолютные значения)
- ⚡ Разрывная (step function)
- 🎢 Множество локальных минимумов (>5)
- 🎯 Глобальный минимум около x ≈ 0
- 📊 Диапазон: [-5, 5]

### Ключевые особенности

✅ **Одинаковая инициализация** - все алгоритмы стартуют с одного `shared_population`

✅ **Максимальное логирование:**
   - В терминал (консоль)
   - В файлы (`test_results/*.log`)
   - Детальная информация о каждой итерации

✅ **Графики от времени:**
   - Best value vs Iteration
   - Best value vs Time (wall-clock)
   - Best x vs Iteration
   - Best x vs Time

✅ **Latin Hypercube Sampling** - для равномерного покрытия пространства поиска

## 📋 Требования

```bash
# Python packages
numpy
matplotlib
scipy
deap
torch  # опционально для PABBO Transformer
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
cd /Users/draiqws/Llabs/all_exps
make install
```

### Запуск теста

```bash
# Простой запуск
make test

# Или напрямую
python test.py
```

### Просмотр результатов

```bash
# Показать сводку
make summary

# Показать логи
make logs

# Открыть графики
make visualize
```

## 📊 Результаты

После запуска `make test` создаются файлы:

```
test_results/
├── test_function.png      # Визуализация функции
├── convergence.png        # Графики сходимости (4 графика)
├── GA_test.log           # Логи GA
├── ES_test.log           # Логи ES
├── PABBO_test.log        # Логи PABBO
└── test_run.log          # Общий лог запуска
```

### Графики сходимости

1. **Best Value vs Iteration** - Лучшее значение от итерации
2. **Best Value vs Time** - Лучшее значение от времени (wall-clock)
3. **Best x vs Iteration** - Лучшая точка x от итерации
4. **Best x vs Time** - Лучшая точка x от времени

## 🔧 Конфигурация

Параметры в `test.py:335-340`:

```python
x_range = (-5.0, 5.0)     # Диапазон x
T_range = (1, 1000)       # Целочисленный диапазон T
iterations = 50           # Количество итераций
pop_size = 20             # Размер популяции
seed = 42                 # Случайное зерно
```

## 📖 Использование Makefile

```bash
# Показать справку
make help

# Установить зависимости
make install

# Запустить тест
make test

# Запустить на GPU (для будущего PABBO Transformer)
make test-gpu

# Обучить PABBO Transformer (~10-20 мин)
make train-pabbo

# Очистить результаты
make clean

# Проверить зависимости
make check-deps

# Показать сводку результатов
make summary

# Показать логи
make logs

# Открыть визуализацию
make visualize
```

## 🧪 Верификация

После запуска проверьте:

1. **Одинаковая инициализация:**
   - Откройте логи: `test_results/GA_test.log`, `ES_test.log`, `PABBO_test.log`
   - Проверьте "Initial population" - должно быть одинаково
   - Проверьте "Initial best: T=..." - должно быть одинаково

2. **Графики начинаются из одной точки:**
   - Откройте `test_results/convergence.png`
   - Все 3 линии должны начинаться с одного значения y

3. **Логирование:**
   - Терминал: детальный вывод прогресса
   - Файлы: полная история оптимизации

## 🎓 Структура кода

```
all_exps/
├── test.py                 # Главный тестовый скрипт
├── exp_ga.py              # Реализация GA
├── exp_es.py              # Реализация ES
├── exp_pabbo.py           # Реализация PABBO (упрощённая)
├── utils.py               # Утилиты (логирование, визуализация)
├── Makefile               # Автоматизация
├── README_TEST.md         # Эта документация
└── test_results/          # Результаты тестов
    ├── *.png             # Графики
    └── *.log             # Логи
```

## 📈 Пример вывода

```
================================================================================
OPTIMIZATION ALGORITHMS TEST
Testing on: Non-differentiable function with discontinuities and multiple local minima
================================================================================

Configuration:
  x range: (-5.0, 5.0)
  T range: (1, 1000)
  Iterations: 50
  Population size: 20
  Seed: 42

Visualizing test function...
Function plot saved to test_results/test_function.png
True minimum (approx): x=0.046126, y=0.419459

Creating shared initial population (seed=42)...
Shared population: [52, 880, 976, 681, ...]
  min=52, max=976, mean=491.6

================================================================================
Testing GA
================================================================================
GA Results:
  Best T: 512
  Best x: 0.123456
  Best y: 0.523451
  Total time: 0.45s
  Iterations: 23

...

================================================================================
SUMMARY
================================================================================
True minimum: x=0.046126, y=0.419459
--------------------------------------------------------------------------------
Algorithm    Best x       Best y       Error        Time (s)
--------------------------------------------------------------------------------
GA           0.123456     0.523451     0.103992     0.45
ES           0.098765     0.445123     0.025664     0.38
PABBO        0.052341     0.421234     0.001775     0.52
--------------------------------------------------------------------------------

🏆 Best algorithm: PABBO
   Found: x=0.052341, y=0.421234
   Error: 0.001775

✅ Test completed successfully!
   Results saved to: test_results/
```

## 🔬 Для исследователей

### Изменение функции

Отредактируйте `test.py:25-75` функцию `test_function(x)`.

### Изменение параметров алгоритмов

```python
# GA parameters (test.py:428-439)
elite=3,
cxpb=0.9,    # Crossover probability
mutpb=0.2    # Mutation probability

# ES parameters (test.py:444-456)
mu=5,        # Parents count
lmbda=10     # Offspring count

# PABBO parameters (test.py:459-470)
exploration_rate=0.3  # Exploration vs exploitation
```

## 📝 Логи

### Структура лог-файла

```
================================================================================
GA Optimization Test
================================================================================
Domain: x ∈ [-5.0, 5.0]
T range: [1, 1000]
Iterations: 50
Population size: 20
Seed: 42
Initial population: [52, 880, 976, ...]
================================================================================
GA initialized: cxpb=0.9, mutpb=0.2, elite=3, dT=5
...
Gen 1/50 | Best: T=512, perplexity=0.523 | Pop: mean=5.91, std=3.23 | Time: 0.01s
...
```

## 🚀 Запуск на GPU (для PABBO Transformer)

```bash
# 1. Обучить модель PABBO Transformer
make train-pabbo

# 2. Обновить test.py для использования модели
# См. PABBO_PRETRAIN_GUIDE.md

# 3. Запустить на GPU
make test-gpu
```

## ❓ FAQ

**Q: Как увеличить сложность теста?**
A: Увеличьте `iterations` и `pop_size` в `test.py:335-340`

**Q: Как добавить свой алгоритм?**
A: Создайте класс наследующий `BaseOptimizer`, добавьте в `test.py`

**Q: Где посмотреть детальную информацию?**
A: В логах `test_results/*.log`

**Q: Как сравнить алгоритмы по времени?**
A: График "Best Value vs Time" в `convergence.png`

## 📚 Дополнительные материалы

- `PABBO_PRETRAIN_GUIDE.md` - Инструкция по полному PABBO с Transformer
- Логи в `test_results/` - Детальная информация о работе алгоритмов

## 🤝 Контрибьюторы

Разработано для исследования алгоритмов оптимизации.

---

**Последнее обновление:** 2025-11-02
**Версия:** 1.0.0