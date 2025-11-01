# 🚀 НАЧНИТЕ ОТСЮДА

## ✅ ВСЁ УЖЕ РАБОТАЕТ!

Проект полностью настроен и протестирован. Просто следуйте инструкциям ниже.

---

## 📦 ЧТО УСТАНОВЛЕНО

**Все файлы на месте:**
- ✓ `test.py` - главный тестовый скрипт
- ✓ `exp_ga.py` - Genetic Algorithm
- ✓ `exp_es.py` - Evolution Strategy
- ✓ `exp_pabbo.py` - PABBO (упрощённый)
- ✓ `utils.py` - вспомогательные функции
- ✓ `requirements.txt` - список зависимостей
- ✓ `RUN.md` - подробная инструкция
- ✓ `QUICKSTART.txt` - быстрая шпаргалка

**Результаты тестов созданы:**
- ✓ `test_results/convergence.png` - 4 графика сходимости
- ✓ `test_results/test_function.png` - график функции
- ✓ `test_results/GA_test.log` - лог GA
- ✓ `test_results/ES_test.log` - лог ES
- ✓ `test_results/PABBO_test.log` - лог PABBO

---

## ⚡ 3 СПОСОБА ЗАПУСКА

### Способ 1: Автоматический (рекомендуется)

```bash
./INSTALL_AND_RUN.sh
```

Скрипт автоматически:
1. Проверит Python
2. Установит зависимости
3. Запустит тесты
4. Проверит результаты

### Способ 2: Вручную (простой)

```bash
# 1. Установить зависимости
pip install numpy scipy matplotlib deap

# 2. Запустить тест
python test.py

# 3. Посмотреть результаты
open test_results/convergence.png
```

### Способ 3: С использованием requirements.txt

```bash
# 1. Установить все зависимости
pip install -r requirements.txt

# 2. Запустить тест
python test.py
```

---

## 📊 РЕЗУЛЬТАТЫ

После запуска вы увидите:

```
================================================================================
SUMMARY
================================================================================
True minimum: x=-0.005005, y=0.602165
--------------------------------------------------------------------------------
Algorithm    Best x       Best y       Error        Time (s)
--------------------------------------------------------------------------------
GA           -0.095095    0.854995     0.252830     0.00
ES           -0.005005    0.602165     0.000000     0.00
PABBO        0.105105     0.930474     0.328309     0.00
--------------------------------------------------------------------------------

🏆 Best algorithm: ES
   Found: x=-0.005005, y=0.602165
   Error: 0.000000
```

---

## ✅ ПРОВЕРКА КОРРЕКТНОСТИ

### 1. Одинаковая инициализация

```bash
grep "Initial best: T=" test_results/*.log
```

Все 3 файла должны показать:
```
T=511, perplexity=0.9305
```

### 2. Графики

Откройте `test_results/convergence.png` - должны увидеть 4 графика:
1. Best Value vs Iteration
2. **Best Value vs Time (wall-clock)**
3. Best x vs Iteration
4. **Best x vs Time (wall-clock)**

Все 3 линии должны начинаться из одной точки!

### 3. Логи

Проверьте логи:
```bash
cat test_results/GA_test.log
cat test_results/ES_test.log
cat test_results/PABBO_test.log
```

Каждый лог содержит:
- Конфигурацию
- Initial population (одинаковая!)
- Прогресс по итерациям
- Финальные результаты

---

## 📚 ДОКУМЕНТАЦИЯ

Если нужны подробности, читайте:

1. **QUICKSTART.txt** - быстрая шпаргалка (1 страница)
2. **RUN.md** - полная инструкция (со всеми деталями)
3. **README_TEST.md** - техническая документация

---

## 🎯 ЧТО ТЕСТИРУЕТСЯ

**3 алгоритма оптимизации:**
- GA (Genetic Algorithm)
- ES (Evolution Strategy)
- PABBO (Preference-Augmented BBO)

**Тестовая функция:**
```python
y(x) = |sin(2πx)| + 0.5*|cos(5πx)| + 0.3*x² + 2*|x| +
       |sin(10πx)|*0.2 + step_function(x)
```

**Свойства:**
- ❌ Недифференцируемая (абсолютные значения)
- ⚡ Разрывная (step function)
- 🎢 Множество локальных минимумов (>5)
- 🎯 Глобальный минимум около x≈0

---

## 🔧 НАСТРОЙКА

Откройте `test.py` и измените параметры (строки 381-386):

```python
x_range = (-5.0, 5.0)     # Диапазон поиска
T_range = (1, 1000)       # Целочисленный диапазон
iterations = 50           # Количество итераций
pop_size = 20             # Размер популяции
seed = 42                 # Случайное зерно
```

---

## 🐛 РЕШЕНИЕ ПРОБЛЕМ

### Ошибка: "No module named 'deap'"
```bash
pip install deap
```

### Ошибка: "No module named 'matplotlib'"
```bash
pip install matplotlib
```

### Скрипт не запускается
```bash
chmod +x INSTALL_AND_RUN.sh
./INSTALL_AND_RUN.sh
```

### Графики не открываются
```bash
# macOS
open test_results/convergence.png

# Linux
xdg-open test_results/convergence.png

# Windows
start test_results/convergence.png
```

---

## 📦 КОПИРОВАНИЕ НА ДРУГОЙ КОМПЬЮТЕР

Просто скопируйте всю папку `all_exps/` и запустите:

```bash
cd all_exps
./INSTALL_AND_RUN.sh
```

Или:

```bash
cd all_exps
pip install -r requirements.txt
python test.py
```

---

## ✨ ВАЖНЫЕ ОСОБЕННОСТИ

### ✅ Одинаковая инициализация
Все 3 алгоритма стартуют с **одинаковой** initial population.

Проверено в логах:
```
Initial population: [732, 811, 32, 354, 777, ...]
Initial best: T=511, perplexity=0.9305
```

### ✅ Максимальное логирование
- В терминал: прогресс в реальном времени
- В файлы: полная история оптимизации

### ✅ Графики от времени
4 графика, включая:
- Best Value vs **Time (wall-clock)**
- Best x vs **Time (wall-clock)**

### ✅ Сложная функция
Недифференцируемая функция с разрывами и множеством локальных минимумов.

---

## 📞 ПОДДЕРЖКА

Если что-то не работает:

1. Проверьте версию Python: `python --version` (должно быть 3.7+)
2. Установите зависимости: `pip install -r requirements.txt`
3. Запустите тест: `python test.py`
4. Проверьте логи: `cat test_results/*.log`

---

## 🎓 СТРУКТУРА ПРОЕКТА

```
all_exps/
├── START_HERE.md         ← ВЫ ЗДЕСЬ
├── QUICKSTART.txt        ← Быстрая шпаргалка
├── RUN.md               ← Полная инструкция
├── requirements.txt     ← Зависимости
├── INSTALL_AND_RUN.sh   ← Автоматический скрипт
│
├── test.py              ← ЗАПУСКАЙТЕ ЭТО
├── exp_ga.py            ← GA реализация
├── exp_es.py            ← ES реализация
├── exp_pabbo.py         ← PABBO реализация
├── utils.py             ← Утилиты
│
└── test_results/        ← Результаты
    ├── convergence.png       # Графики сходимости
    ├── test_function.png     # График функции
    ├── GA_test.log          # Лог GA
    ├── ES_test.log          # Лог ES
    └── PABBO_test.log       # Лог PABBO
```

---

## 🚀 ГОТОВО К ЗАПУСКУ!

Выберите один из способов выше и запустите тест.

**Рекомендуется:** `./INSTALL_AND_RUN.sh`

---

**Последнее обновление:** 2025-11-02
**Версия:** 1.0.0
**Статус:** ✅ Всё работает и протестировано
