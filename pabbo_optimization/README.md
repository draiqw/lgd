# PABBO Optimization Experiments

Эта директория содержит ноутбуки для запуска экспериментов с PABBO (Preferential Bayesian Optimization by Optimization).

## Структура

```
pabbo_optimization/
├── README.md                    # Этот файл
├── simple_experiment.ipynb      # Предобучение модели PABBO на GP1D
└── sinexp_experiment.ipynb      # Эксперимент с функцией sin(x) + e^x
```

## Быстрый старт

### 1. Предобучение модели

Сначала необходимо предобучить модель PABBO на синтетических одномерных функциях:

```bash
cd ../pabbo_method

python train.py --config-name=train \
    experiment.expid=PABBO_GP1D \
    data.name=GP1D \
    data.d_x=1 \
    data.x_range="[[-1,1]]" \
    data.min_num_ctx=1 \
    data.max_num_ctx=50 \
    train.num_query_points=100 \
    train.num_prediction_points=100 \
    train.n_random_pairs=100 \
    experiment.device=cpu \
    experiment.wandb=false \
    train.n_steps=1000 \
    train.n_burnin=300
```

**Примечание:** Для быстрого теста используются сокращенные параметры (`n_steps=1000` вместо `8000`). Для полноценного обучения увеличьте эти значения.

После завершения чекпоинт будет сохранен в:
```
../pabbo_method/results/evaluation/PABBO/PABBO_GP1D/ckpt.tar
```

### 2. Эксперимент с функцией sin(x) + e^x

После предобучения можно запустить оптимизацию кастомной функции:

```bash
cd ../pabbo_method

python evaluate_continuous.py --config-name=evaluate \
    experiment.model=PABBO \
    experiment.expid=PABBO_GP1D \
    experiment.device=cpu \
    eval.eval_max_T=30 \
    eval.eval_num_query_points=256 \
    eval.num_parallel=1 \
    data.name=sinexp1D \
    data.d_x=1 \
    data.x_range="[[-1,1]]" \
    data.Xopt="[[1.0]]" \
    data.yopt="[[2.718]]" \
    experiment.wandb=false \
    eval.num_seeds=5 \
    eval.num_datasets=1
```

Результаты будут сохранены в:
```
../pabbo_method/results/evaluation/sinexp1D/PABBO/PABBO_GP1D/
```

## Ноутбуки

### simple_experiment.ipynb

Этот ноутбук содержит:
- Настройку окружения
- Команды для предобучения модели
- Проверку результатов предобучения
- Базовую оценку модели на GP1D
- Функции для визуализации результатов

**Использование:**
1. Откройте ноутбук в Jupyter
2. Выполните все ячейки последовательно
3. Запустите предобучение (раскомментируйте соответствующую ячейку или запустите команду в терминале)
4. Дождитесь завершения (это может занять время)
5. Проверьте результаты

### sinexp_experiment.ipynb

Этот ноутбук демонстрирует:
- Визуализацию целевой функции y(x) = sin(x) + e^x
- Запуск оптимизации с предобученной моделью
- Загрузку и анализ результатов
- Визуализацию метрик оптимизации:
  - Simple Regret (основная метрика)
  - Immediate Regret
  - Kendall-Tau Correlation (качество ранжирования)
  - Policy Entropy (уверенность модели)

**Использование:**
1. Убедитесь, что предобучение завершено
2. Откройте ноутбук
3. Выполните ячейки для визуализации функции
4. Запустите оптимизацию
5. Проанализируйте результаты

## Добавление кастомных функций

Функция `sinexp1D` была добавлена в `/pabbo_method/data/function.py`:

```python
def sinexp1D(x: torch.Tensor, negate: bool = True, add_dim: bool = True):
    """Custom test function: y(x) = sin(x) + e^x.

    Args:
        x, (B, N, 1): input tensor
        negate, bool: whether to negate the function.
        add_dim, bool: whether to add dimension at the end.

    Returns:
        y, (B, N, 1) if add_dim else (B, N)
    """
    y = torch.sin(x) + torch.exp(x)
    if negate:
        y = -y
    return y if add_dim else y.squeeze(-1)
```

Для добавления своей функции:

1. Откройте `../pabbo_method/data/function.py`
2. Добавьте свою функцию по аналогии с `sinexp1D`
3. Добавьте имя функции в список `__all__`
4. Используйте `data.name=<имя_вашей_функции>` при запуске

## Параметры

### Основные параметры предобучения:

- `data.name` - тип данных (GP1D для синтетических функций)
- `data.d_x` - размерность пространства поиска
- `data.x_range` - диапазон значений для каждой размерности
- `train.n_steps` - количество шагов обучения (8000 для полного обучения)
- `train.n_burnin` - количество шагов до начала обучения acquisition (3000 для полного)
- `experiment.device` - устройство (cpu или cuda)
- `experiment.wandb` - использовать ли Weights & Biases для логирования

### Основные параметры оценки:

- `experiment.expid` - ID эксперимента с предобученной моделью
- `data.name` - имя целевой функции
- `eval.eval_max_T` - количество итераций оптимизации
- `eval.eval_num_query_points` - размер пространства поиска
- `eval.num_parallel` - количество параллельных запросов (1 для последовательного)
- `eval.num_seeds` - количество случайных запусков
- `data.Xopt` - известное положение оптимума (для вычисления regret)
- `data.yopt` - известное значение оптимума

## Метрики

- **Simple Regret**: `y* - max(y_observed)` - разница между оптимумом и лучшим найденным значением
- **Immediate Regret**: качество последнего запроса
- **Cumulative Regret**: сумма immediate regret за все итерации
- **Kendall-Tau Correlation**: корреляция между предсказанным и истинным ранжированием точек
- **Entropy**: энтропия политики выбора следующей точки
- **Inference Regret**: regret для точки с максимальным предсказанным значением acquisition

## Следующие шаги

1. **Эксперимент с LDA**: Создайте функцию для оптимизации количества тем в LDA модели
2. **Сравнение с baselines**: Добавьте другие методы оптимизации для сравнения
3. **Многомерные функции**: Попробуйте функции с `d_x > 1`
4. **Дискретная оптимизация**: Используйте `evaluate_discrete.py` для дискретных параметров

## Полезные ссылки

- [PABBO Paper](https://arxiv.org/abs/2010.04352) (если доступна)
- [Документация Hydra](https://hydra.cc/) - для работы с конфигурациями
- [PyTorch](https://pytorch.org/) - основной фреймворк

## Troubleshooting

### Ошибка: чекпоинт не найден
Убедитесь, что предобучение завершилось успешно и файл существует в `../pabbo_method/results/evaluation/PABBO/PABBO_GP1D/ckpt.tar`

### Ошибка: функция не найдена
Проверьте, что функция добавлена в `../pabbo_method/data/function.py` и импортируется в `__all__`

### Медленная работа
- Используйте `experiment.device=cuda` если доступен GPU
- Уменьшите `eval.eval_num_query_points` для более быстрой оценки
- Уменьшите `eval.num_seeds` и `eval.num_datasets`

### Out of Memory
- Уменьшите batch size
- Используйте меньше точек: `eval.eval_num_query_points`
- Используйте CPU вместо GPU

## Контакты и поддержка

Для вопросов и предложений создайте issue в репозитории проекта.