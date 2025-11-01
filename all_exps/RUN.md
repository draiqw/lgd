# 🚀 ПРОСТАЯ ИНСТРУКЦИЯ ПО ЗАПУСКУ

## ⚡ Быстрый старт (3 шага)

### Шаг 1: Установка зависимостей

```bash
cd /Users/draiqws/Llabs/all_exps

# Установите базовые библиотеки
pip install numpy scipy matplotlib deap

# ИЛИ используйте requirements.txt
pip install -r requirements.txt
```

### Шаг 2: Запуск теста

```bash
# Просто запустите test.py
python test.py
```

### Шаг 3: Посмотреть результаты

```bash
# Откройте графики
open test_results/convergence.png
open test_results/test_function.png

# Или посмотрите логи
cat test_results/GA_test.log
cat test_results/ES_test.log
cat test_results/PABBO_test.log
```

**Вот и всё!** Тест запустится, выполнится и создаст результаты в `test_results/`.

---

## 📋 Полная информация

### Что запускается?

**3 алгоритма оптимизации:**
1. **GA** (Genetic Algorithm) - Генетический алгоритм
2. **ES** (Evolution Strategy) - Эволюционная стратегия
3. **PABBO** (Preference-Augmented BBO) - Упрощённая версия

**Тестовая функция:**
- Недифференцируемая (с разрывами)
- Множество локальных минимумов (>5)
- Один глобальный минимум около x≈0

### Необходимые библиотеки

**Обязательные:**
```bash
pip install numpy scipy matplotlib deap
```

**Опциональные (для полного PABBO с Transformer):**
```bash
pip install torch botorch gpytorch hydra-core omegaconf wandb
```

### Файлы в проекте

```
all_exps/
├── test.py              # Главный скрипт теста
├── exp_ga.py            # Реализация GA
├── exp_es.py            # Реализация ES
├── exp_pabbo.py         # Реализация PABBO (упрощённая)
├── utils.py             # Утилиты
├── requirements.txt     # Список зависимостей
├── RUN.md              # Эта инструкция
└── test_results/       # Результаты (создаётся автоматически)
    ├── convergence.png    # Графики сходимости
    ├── test_function.png  # График функции
    ├── GA_test.log        # Лог GA
    ├── ES_test.log        # Лог ES
    └── PABBO_test.log     # Лог PABBO
```

---

## 🔧 Настройка параметров

Откройте `test.py` и измените параметры (строки 381-386):

```python
x_range = (-5.0, 5.0)     # Диапазон поиска
T_range = (1, 1000)       # Целочисленный диапазон
iterations = 50           # Количество итераций
pop_size = 20             # Размер популяции
seed = 42                 # Случайное зерно
```

---

## 📊 Что создаётся после запуска?

### Графики (PNG)

1. **test_function.png** - Визуализация тестовой функции
   - Показывает все локальные минимумы
   - Отмечает глобальный минимум

2. **convergence.png** - 4 графика сходимости:
   - Best Value vs Iteration
   - **Best Value vs Time (wall-clock)**
   - Best x vs Iteration
   - **Best x vs Time (wall-clock)**

### Логи (TXT)

Каждый алгоритм создаёт свой лог-файл:

```
GA_test.log    - Детальный лог Genetic Algorithm
ES_test.log    - Детальный лог Evolution Strategy
PABBO_test.log - Детальный лог PABBO
```

**Содержимое лога:**
- Конфигурация эксперимента
- Initial population (одинаковая для всех!)
- Прогресс по итерациям
- Лучшие найденные значения
- Время выполнения

---

## ✅ Проверка правильности

### 1. Одинаковая инициализация

Проверьте что все 3 алгоритма начинают с одного:

```bash
grep "Initial population" test_results/*.log

# Должны увидеть ОДИНАКОВЫЙ список в 3 файлах:
# [732, 811, 32, 354, 777, ...]
```

### 2. Одинаковое начальное значение

```bash
grep "Initial best: T=" test_results/*.log

# Все 3 должны показать одинаковое значение:
# T=511, perplexity=0.9305
```

### 3. Графики начинаются из одной точки

Откройте `test_results/convergence.png` - все 3 линии должны начинаться с одной точки на графике.

---

## 🎯 Что делает каждый файл?

### test.py
Главный файл. Запускает все 3 алгоритма с одинаковой инициализацией.

**Что делает:**
1. Создаёт сложную недифференцируемую функцию
2. Создаёт одинаковую initial population
3. Запускает GA, ES, PABBO с этой population
4. Создаёт графики и логи
5. Выводит сводку результатов

### exp_ga.py
Реализация Genetic Algorithm (GA).

**Особенности:**
- Crossover probability: 0.9
- Mutation probability: 0.2
- Elite size: 3
- Early stopping при сходимости

### exp_es.py
Реализация Evolution Strategy (ES).

**Особенности:**
- mu (parents): 5
- lambda (offspring): 10
- Self-adaptive mutation
- Early stopping при сходимости

### exp_pabbo.py
Упрощённая реализация PABBO.

**Особенности:**
- Exploration rate: 0.3
- Temperature decay: 0.95
- Adaptive search
- Early stopping при сходимости

### utils.py
Вспомогательные функции:
- `setup_logger()` - Настройка логирования
- `ensure_dir()` - Создание директорий
- `plot_series()` - Построение графиков
- `BaseOptimizer` - Базовый класс оптимизатора

---

## 🐛 Решение проблем

### Ошибка: "ModuleNotFoundError: No module named 'deap'"

**Решение:**
```bash
pip install deap
```

### Ошибка: "ModuleNotFoundError: No module named 'matplotlib'"

**Решение:**
```bash
pip install matplotlib
```

### Тест не создаёт графики

**Проверьте:**
1. Установлен ли matplotlib: `pip install matplotlib`
2. Есть ли права на запись в `test_results/`

### Тест запускается но сразу завершается

Это нормально! Алгоритмы используют early stopping. Если они быстро находят решение, они останавливаются.

**Чтобы запустить больше итераций:**
Откройте `test.py` и уберите early stopping или увеличьте `iterations`.

---

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
True minimum (approx): x=-0.005005, y=0.602165

Creating shared initial population (seed=42)...
Shared population: [732, 811, 32, 354, ...]

================================================================================
Testing GA
================================================================================
GA Results:
  Best T: 491
  Best x: -0.095095
  Best y: 0.854995
  Total time: 0.00s

================================================================================
Testing ES
================================================================================
ES Results:
  Best T: 500
  Best x: -0.005005
  Best y: 0.602165
  Total time: 0.00s

================================================================================
Testing PABBO
================================================================================
PABBO Results:
  Best T: 511
  Best x: 0.041041
  Best y: 0.930495
  Total time: 0.00s

================================================================================
SUMMARY
================================================================================
True minimum: x=-0.005005, y=0.602165
--------------------------------------------------------------------------------
Algorithm    Best x       Best y       Error        Time (s)
--------------------------------------------------------------------------------
GA           -0.095095    0.854995     0.252830     0.00
ES           -0.005005    0.602165     0.000000     0.00
PABBO        0.041041     0.930495     0.328330     0.00
--------------------------------------------------------------------------------

🏆 Best algorithm: ES
   Found: x=-0.005005, y=0.602165
   Error: 0.000000

✅ Test completed successfully!
```

---

## 🚀 Запуск на компьютере с GPU

### Для текущей версии (упрощённый PABBO)

GPU **не требуется**. Всё работает на CPU.

```bash
python test.py
```

### Для полного PABBO с Transformer

Если хотите использовать полную версию PABBO с нейросетью:

**1. Установите PyTorch:**
```bash
# Для GPU (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Для CPU (если нет GPU)
pip install torch torchvision
```

**2. Установите остальные зависимости:**
```bash
pip install botorch gpytorch hydra-core omegaconf wandb
```

**3. Обучите модель:**
```bash
cd ../pabbo_method
python train.py --config-name train_rastrigin1d_test
```

**4. Интегрируйте в test.py**

См. `PABBO_PRETRAIN_GUIDE.md` для деталей.

---

## 📞 Контакты и поддержка

При проблемах проверьте:
1. ✅ Установлены ли зависимости: `pip install -r requirements.txt`
2. ✅ Запускается ли Python: `python --version`
3. ✅ Есть ли права на запись: `ls -la test_results/`

---

## 📚 Дополнительная документация

- `README_TEST.md` - Полная техническая документация
- `PABBO_PRETRAIN_GUIDE.md` - Инструкция по PABBO с Transformer
- Логи в `test_results/*.log` - Детальная информация

---

**Последнее обновление:** 2025-11-02

**Версия:** 1.0.0

**Лицензия:** MIT