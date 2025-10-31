# LDA Hyperparameter Optimization

Оптимизация гиперпараметров LDA (Latent Dirichlet Allocation) с помощью генетического алгоритма.

## Подход

**Правильный способ подбора гиперпараметров:**
- Обучение LDA на validation выборке
- Оптимизация для минимизации средней перплексии на той же validation выборке
- Подбор оптимального числа топиков T (alpha и eta автоматически устанавливаются как 1/T)

## Структура проекта

```
.
├── lda_optimizer.py    # Основной скрипт оптимизации
├── data/               # Датасеты (.npz файлы)
├── requirements.txt    # Python зависимости
├── Dockerfile          # Docker образ
├── build.sh            # Скрипт сборки
├── run.sh              # Скрипт запуска
└── logs/               # Результаты (создается автоматически)
```

## Быстрый старт

### Локальный запуск

```bash
python lda_optimizer.py
```

### Docker запуск

```bash
# Сборка образа
./build.sh

# Запуск оптимизации
./run.sh
```

## Результаты

После выполнения в `logs/` будет создана структура:

```
logs/
├── 20news/
│   ├── tensorboard/
│   ├── history.csv
│   ├── summary.json
│   └── *.png
├── agnews/
├── val_out/
├── yelp/
└── all_results.json
```

## TensorBoard

Для просмотра метрик в реальном времени:

```bash
tensorboard --logdir=logs/
```

Откройте в браузере: http://localhost:6006

## Зависимости

- numpy
- scipy
- scikit-learn
- deap
- tensorboardX
- matplotlib
- pandas