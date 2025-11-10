# SABO (Stochastic Adaptive Bayesian Optimization)

## Описание

SABO - метод оптимизации чёрного ящика, использующий:
- Гауссовское предложение N(μ, Σ) для генерации кандидатов
- Natural gradient для обновления параметров распределения
- Адаптивный learning rate
- Momentum для μ и Σ

## Реализация

Класс `SABOOptimizer` находится в:
- `lda_hyperopt/optimizers/sabo.py`
- `test_function_opt/optimizers/sabo.py`

Наследует `BaseOptimizer` и реализует метод `run()`.

## Ключевые параметры

- `rho`: контролирует размер окрестности (neighborhood size)
- `beta_t`: начальный learning rate
- `delta_Sigma`: momentum для обновления ковариации
- `delta_mu`: momentum для обновления среднего
- `N_batch`: размер батча (число образцов на итерацию)

## Алгоритм

1. **Инициализация**: μ₀ = 0.5, Σ₀ = 0.25 (в нормализованном пространстве [0,1])
2. **На каждой итерации**:
   - Сэмплируем N_batch точек из N(μₜ, Σₜ)
   - Оцениваем фитнесс для каждой точки
   - Вычисляем natural gradient: ∇μ, ∇Σ
   - Вычисляем adaptive step size: λₜ
   - Обновляем параметры с momentum:
     - μₜ₊₁ = μₜ - λₜ × δμ × ∇μ
     - Σₜ₊₁ = Σₜ - λₜ × δΣ × ∇Σ

## Математика

### Natural Gradient (упрощённый для 1D)

Стандартизованные остатки:
```
g_i = (z_i - μ) / Σ
```

Веса (инверсия для минимизации):
```
w_i = 1 / (f_norm_i + 0.1)
w_i = w_i / Σw_i  # нормализация
```

Градиенты:
```
∇μ = Σ(w_i × g_i)
∇Σ = 0.5 × (1/Σ - Σ(w_i × g_i²) / Σ)
```

### Adaptive Step Size

```
λₜ = βₜ / (1 + 0.1×t) × (1 + ||∇||)
```

где `||∇|| = sqrt(∇μ² + ∇Σ²)`

## Пример запуска

### Через lda_hyperopt

```bash
cd lda_hyperopt

# Запуск только SABO
python run_no_early_stop.py \
  --data ../data/X_20news_val_bow.npz \
  --init lda_init_population.json \
  --iterations 50 \
  --algorithms SABO \
  --outdir results/sabo_test

# Запуск всех алгоритмов (включая SABO)
python run_no_early_stop.py \
  --data ../data/X_20news_val_bow.npz \
  --init lda_init_population.json \
  --iterations 50 \
  --algorithms GA ES PABBO_Simple SABO \
  --outdir results/comparison
```

### Программный запуск

```python
from lda_hyperopt.optimizers import SABOOptimizer
from lda_hyperopt.utils import load_bow_data, make_objective, make_eval_func

# Загрузка данных
Xval = load_bow_data("data/X_20news_val_bow.npz")

# Создание objective функций
obj = make_objective(Xval, seed=42, max_iter=60)
eval_func = make_eval_func(Xval, seed=42, max_iter=60)

# Инициализация оптимизатора
optimizer = SABOOptimizer(
    obj=obj,
    eval_func=eval_func,
    T_bounds=(2, 1000),
    seed=42,
    rho=0.1,           # neighborhood size
    beta_t=0.01,       # learning rate
    delta_Sigma=0.5,   # covariance momentum
    delta_mu=0.5,      # mean momentum
    N_batch=10         # batch size
)

# Запуск оптимизации
result = optimizer.run(
    iterations=50,
    n_initial=10,
    outdir="results/sabo"
)

print(f"Best T: {result['best']['T']}")
print(f"Best perplexity: {result['best']['perplexity']:.4f}")
print(f"Final μ: {result['final_mu']:.4f}")
print(f"Final Σ: {result['final_Sigma']:.4f}")
```

## Выходные данные

Оптимизатор возвращает словарь с:
- `best`: лучшие параметры и метрики
- `history`: история по итерациям (включая μ, Σ, градиенты, λ)
- `total_time`: общее время оптимизации
- `total_evaluations`: число вызовов objective функции
- `final_mu`, `final_Sigma`: финальные параметры распределения

## Логирование

SABO логирует на каждой итерации:
- Best T и perplexity
- Текущие μ и Σ
- Градиенты ∇μ и ∇Σ
- Step size λ
- Счётчик no improvement

Также записывает в TensorBoard:
- `SABO/Parameters/mu`, `SABO/Parameters/Sigma`
- `SABO/Gradients/grad_mu`, `SABO/Gradients/grad_Sigma`
- `SABO/StepSize/lambda`
- Все стандартные метрики (perplexity, T, времена)

## Сравнение с другими методами

| Метод | Тип | Batch | Gradient | Адаптация |
|-------|-----|-------|----------|-----------|
| GA | Популяционный | Да (pop) | Нет | Через селекцию |
| ES | Популяционный | Да (λ) | Нет | Через мутацию |
| PABBO | Sequential | Нет | Нет | Через temperature |
| **SABO** | **Batch** | **Да (N_batch)** | **Да (natural)** | **μ, Σ адаптивны** |

## Преимущества

1. **Natural gradient** - более эффективное использование информации о геометрии
2. **Адаптивный step size** - автоматическая настройка скорости обучения
3. **Momentum** - стабилизация обновлений
4. **Батчевая оценка** - можно параллелизовать

## Настройка параметров

### rho (neighborhood size)
- Влияет на исследование/эксплуатацию
- Рекомендуемые значения: 0.05-0.2
- Меньше → больше локальный поиск
- Больше → больше исследование

### beta_t (learning rate)
- Начальный шаг обучения
- Рекомендуемые значения: 0.001-0.1
- Автоматически уменьшается со временем

### delta_mu, delta_Sigma (momentum)
- Контролируют инерцию обновлений
- Рекомендуемые значения: 0.3-0.7
- Больше → сильнее сглаживание

### N_batch
- Число образцов на итерацию
- Рекомендуемые значения: 5-20
- Больше → точнее градиент, но дороже

## Известные ограничения

1. **1D оптимизация**: текущая реализация для одного параметра T
2. **Дискретность**: T должен быть целым → используется округление
3. **Нормализация**: работает в [0,1], требует denormalize для T

## Планы развития

- [ ] Многомерная версия (оптимизация T, alpha, eta одновременно)
- [ ] Лучшая обработка дискретности
- [ ] Адаптивный N_batch
- [ ] Parallel batch evaluation
