
# Quick Start Guide

## ✅ Проверка готовности

Все проверено и готово к работе:
- ✅ Все модули успешно импортируются
- ✅ Все 3 алгоритма (GA, ES, PABBO) работают корректно
- ✅ Тест на математической функции пройден (все алгоритмы нашли минимум)
- ✅ Все 4 датасета присутствуют
- ✅ ООП архитектура с BaseOptimizer работает
- ✅ Общая начальная популяция реализована
- ✅ Логирование на всех уровнях функционирует

## 🚀 Быстрый старт

### 1. Тест на простой функции (2 минуты)

```bash
cd all_exps
python test.py
```

**Что произойдет:**
- Все 3 алгоритма будут минимизировать функцию y(x) = sin(x) + e^cos(x) - 20 + 99*ln(x)
- Создастся папка `test_results/` с графиками
- Вы увидите сводную таблицу результатов
- Убедитесь, что всё работает корректно

### 2. Пример на одном датасете (5-10 минут)

```bash
cd all_exps
python example_usage.py
```

**Что произойдет:**
- Загрузится датасет 20news
- Запустятся все 3 алгоритма с небольшим числом итераций (10)
- Вы увидите процесс оптимизации в реальном времени
- Сравнение результатов в конце

### 3. Полные эксперименты

#### Последовательный запуск (безопасно, медленно)

```bash
cd all_exps
python main.py --sequential --iterations 200 --pop-size 10
```

**Время выполнения:** ~2-6 часов (зависит от hardware)

#### Параллельный запуск (быстро, рекомендуется)

```bash
cd all_exps
python main.py --parallel --max-workers 4 --iterations 200 --pop-size 10
```

**Время выполнения:** ~0.5-2 часа (с 4 воркерами)

#### Только один алгоритм

```bash
# Только GA
python main.py --algorithm ga --sequential

# Только ES
python main.py --algorithm es --sequential

# Только PABBO
python main.py --algorithm pabbo --sequential
```

#### Только определенные датасеты

```bash
python main.py --datasets 20news agnews --sequential
```

## 📊 Просмотр результатов

### TensorBoard (в реальном времени)

```bash
# Откройте в другом терминале во время или после экспериментов
tensorboard --logdir results/
```

Затем откройте http://localhost:6006 в браузере.

### Результаты в файлах

```
results/
├── 20news/
│   ├── ga/
│   │   ├── tensorboard/          # TensorBoard логи
│   │   ├── experiment.log         # Детальный текстовый лог
│   │   ├── history.csv            # История: iter, perplexity, T, time
│   │   ├── summary.json           # Итоговые результаты
│   │   ├── population_gen_*.csv   # Популяции по поколениям
│   │   └── *.png                  # Графики (perplexity, T, time)
│   ├── es/                        # Аналогично для ES
│   └── pabbo/                     # Аналогично для PABBO
├── agnews/                        # Аналогично для других датасетов
├── val_out/
├── yelp/
└── all_results.json               # СВОДКА ВСЕХ ЭКСПЕРИМЕНТОВ
```

### Анализ результатов

```python
import json

# Загрузить все результаты
with open('results/all_results.json', 'r') as f:
    results = json.load(f)

# Посмотреть лучший результат для датасета 20news
print(results['20news']['GA']['best'])
print(results['20news']['ES']['best'])
print(results['20news']['PABBO']['best'])
```

## 🎯 Параметры командной строки

```bash
python main.py --help
```

**Основные параметры:**
- `--parallel` - параллельное выполнение
- `--sequential` - последовательное (по умолчанию)
- `--algorithm {ga,es,pabbo,all}` - выбор алгоритма
- `--datasets DATASET [DATASET ...]` - выбор датасетов
- `--iterations N` - число итераций (по умолчанию: 200)
- `--pop-size N` - размер популяции (по умолчанию: 10)
- `--seed N` - random seed (по умолчанию: 42)
- `--output-dir DIR` - директория для результатов
- `--max-workers N` - число параллельных воркеров (по умолчанию: 4)

## 💡 Рекомендации

### Для быстрого тестирования

```bash
# Уменьшите число итераций и популяцию
python main.py --sequential --iterations 20 --pop-size 5 --datasets 20news
```

**Время:** ~5-10 минут

### Для серьезных экспериментов

```bash
# Увеличьте параметры
python main.py --parallel --max-workers 4 --iterations 200 --pop-size 20
```

**Время:** ~1-3 часа

### Для воспроизводимости

```bash
# Используйте фиксированный seed
python main.py --sequential --seed 42
```

## 🔧 Устранение проблем

### "ModuleNotFoundError"
```bash
# Убедитесь, что вы в правильной директории
cd all_exps
python test.py
```

### "Dataset not found"
```bash
# Проверьте наличие датасетов
ls -lh ../data/X_*_val_bow.npz
```

### Нехватка памяти
```bash
# Уменьшите batch_size или max_iter_lda
# Отредактируйте DEFAULT_CONFIG в main.py:
#   "max_iter_lda": 30,  # вместо 60
#   "batch_size": 1024,  # вместо 2048
```

### Слишком долго выполняется
```bash
# Уменьшите число итераций
python main.py --sequential --iterations 50 --pop-size 5
```

## 📈 Что оптимизируется

**Цель:** Найти оптимальное число топиков T для LDA модели

**Параметры:**
- T: [2, 1000] - число топиков (целое число)
- alpha = 1/T - doc-topic prior
- eta = 1/T - topic-word prior

**Метрика:** Perplexity (чем меньше, тем лучше)

**Методология:**
1. Обучаем LDA на validation выборке
2. Оцениваем perplexity на той же validation выборке
3. Минимизируем perplexity через оптимизацию T

## 📚 Дополнительная документация

- **README.md** - полная документация
- **IMPLEMENTATION_SUMMARY.md** - технические детали реализации
- **test.py** - исходный код теста на простой функции
- **example_usage.py** - пример использования API

## 🎉 Готово!

Ваша система готова к запуску. Начните с `python test.py` для проверки, затем переходите к полным экспериментам.

**Удачных экспериментов! 🚀**
