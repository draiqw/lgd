# 🎓 Руководство по обучению PABBO Transformer

Полное руководство по предобучению модели PABBO для использования в оптимизации LDA.

---

## 📋 Оглавление

1. [Что такое обучение PABBO?](#что-такое-обучение-pabbo)
2. [Быстрый старт (10 минут)](#быстрый-старт)
3. [Полное обучение (30-60 минут)](#полное-обучение)
4. [Обучение для LDA](#обучение-для-lda)
5. [Как это работает внутри](#как-это-работает-внутри)
6. [Кастомизация](#кастомизация)
7. [Troubleshooting](#troubleshooting)

---

## 🤔 Что такое обучение PABBO?

**PABBO (Preference-Augmented Black-Box Optimization)** - это нейронная сеть (Transformer), которая учится **как искать минимум функции**.

### Простая аналогия

**Обычная оптимизация (GA, ES):**
- Каждый раз начинаем "с нуля"
- Не помним предыдущий опыт
- Как новичок решает каждую задачу

**PABBO после обучения:**
- Помнит паттерны из сотен задач
- "Видел такое раньше, знаю что делать"
- Как эксперт решает новую задачу

### Что происходит при обучении?

1. **Генерируются тысячи задач оптимизации** (разные функции)
2. **Модель пробует их решать**
3. **Учится на ошибках** (preference learning)
4. **Запоминает паттерны** что работает

После обучения модель может **быстрее находить минимум** новых функций.

---

## ⚡ Быстрый старт (10 минут)

### Шаг 1: Установка зависимостей

```bash
cd /Users/draiqws/Llabs/pabbo_method

# Если у вас нет PyTorch
pip install torch torchvision torchaudio

# Дополнительные зависимости
pip install botorch gpytorch hydra-core
```

**Проверка установки:**
```bash
python -c "import torch; import botorch; print('OK!')"
```

### Шаг 2: Быстрое тестовое обучение

```bash
# Обучение на 2000 шагов (~10 минут)
python train.py --config-name train_rastrigin1d_test

# Вы увидите:
# Step 0/2000 | Loss: 0.6931 | Best: 2.456
# Step 100/2000 | Loss: 0.4521 | Best: 1.234
# ...
# Training complete! Model saved to: policies/checkpoints/model_best.pt
```

### Шаг 3: Проверка модели

```bash
# Тестируем обученную модель
python evaluate_continuous.py \
  --model policies/checkpoints/model_best.pt \
  --function rastrigin \
  --n_trials 10 \
  --budget 50

# Результат:
# Mean best value: 0.45 ± 0.18
# Success rate: 90%
```

### Шаг 4: Использование в LDA

```bash
cd ../lda_hyperopt

# Запуск с обученной моделью
python run.py \
  --data data/val_bow.npz \
  --algorithms PABBO_Full \
  --pabbo-model ../pabbo_method/policies/checkpoints/model_best.pt \
  --iterations 50 \
  --outdir results_pabbo_full
```

✅ **Готово!** Теперь PABBO_Full использует обученную модель.

---

## 🎯 Полное обучение (30-60 минут)

Для **лучшего качества** используйте полное обучение:

```bash
cd pabbo_method

# Полное обучение (8000 шагов)
python train.py --config-name train_rastrigin1d
```

**Параметры полного обучения:**
- **Шагов**: 8000 (vs 2000 в быстром)
- **Модель**: d_model=64, 6 layers (vs 32, 3 layers)
- **Время**: ~30-60 минут (зависит от CPU/GPU)

**Результат:**
- Лучше обобщение на новые функции
- Более стабильная сходимость
- Выше success rate

---

## 🔬 Обучение для LDA (рекомендуется)

Для **оптимизации LDA** нужно обучить модель на **дискретных функциях** (T - целое число).

### Вариант 1: Обучение на похожих функциях

```bash
# Обучаем на дискретной версии Rastrigin
python train.py --config-name train_discrete

# В config нужно указать:
# - discrete: true
# - bounds: [2, 1000]  # как в LDA
```

### Вариант 2: Обучение на синтетических LDA задачах

Создайте `configs/train_lda_synthetic.yaml`:

```yaml
seed: 42
function: lda_synthetic  # Нужно добавить в data/function.py
n_steps: 8000
batch_size: 16

# Дискретное пространство как в LDA
discrete: true
bounds:
  T_min: 2
  T_max: 1000

model:
  d_model: 64
  n_heads: 8
  n_layers: 6
  dropout: 0.1

optimizer:
  type: adam
  lr: 1e-4
  weight_decay: 1e-5

training:
  n_episodes: 100
  budget: 20
  warmup_steps: 100
```

Запуск:
```bash
python train.py --config-name train_lda_synthetic
```

### Вариант 3: Multi-task обучение

Обучение на **нескольких функциях** для лучшей generalization:

```bash
# Обучаем на смеси функций
python train.py \
  --config-name train_multitask \
  functions=[rastrigin1D,forrester1D,sinexp1D] \
  n_steps=10000
```

---

## 🔍 Как это работает внутри

### Процесс обучения (пошагово)

#### Шаг 1: Генерация эпизода

```python
# 1. Создается случайная задача оптимизации
function = Rastrigin1D()

# 2. Начальная точка
x_init = random.uniform(-5, 5)
history = [(x_init, function(x_init))]

# 3. Цикл оптимизации (20 шагов)
for step in range(20):
    # Модель предсказывает следующую точку
    x_next = policy(history)  # Transformer!

    # Оценка
    y_next = function(x_next)

    # Добавление в историю
    history.append((x_next, y_next))
```

#### Шаг 2: Preference Learning

```python
# Из истории создаются пары для сравнения
pairs = []
for i in range(len(history)):
    for j in range(i+1, len(history)):
        x_i, y_i = history[i]
        x_j, y_j = history[j]

        # Если y_i лучше (меньше)
        if y_i < y_j:
            pairs.append((x_i, x_j, label=1))  # x_i > x_j
        else:
            pairs.append((x_j, x_i, label=1))  # x_j > x_i
```

#### Шаг 3: Обучение модели

```python
# Loss: научить модель правильно ранжировать
for (x_better, x_worse, _) in pairs:
    score_better = policy.score(x_better, history)
    score_worse = policy.score(x_worse, history)

    # Preference loss
    loss = -log(sigmoid(score_better - score_worse))

    # Backpropagation
    loss.backward()

optimizer.step()
```

### Архитектура Transformer

```
Input: [(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)]
   ↓
[Embedding Layer]
   x, y → d_model размерность
   ↓
[Positional Encoding]
   Добавляет информацию о порядке
   ↓
[Transformer Encoder]
   ┌─────────────────┐
   │ Self-Attention  │ ← Какие точки важны?
   │ Feed-Forward    │
   │ LayerNorm       │
   └─────────────────┘
   × 6 layers
   ↓
[Output Head]
   Linear → score для каждой точки
   ↓
Output: Scores или Next candidate xₙ₊₁
```

**Что учит модель:**
- Какие регионы пространства перспективны
- Как баланс exploration/exploitation
- Паттерны сходимости разных функций

---

## 🛠 Кастомизация

### 1. Изменить размер модели

**Маленькая модель (быстрее, но хуже):**
```bash
python train.py \
  --config-name train_rastrigin1d_test \
  model.d_model=32 \
  model.n_layers=3 \
  model.n_heads=4
```

**Большая модель (медленнее, но лучше):**
```bash
python train.py \
  --config-name train_rastrigin1d \
  model.d_model=128 \
  model.n_layers=8 \
  model.n_heads=8
```

### 2. Изменить learning rate

```bash
# Быстрее, но может diverge
python train.py --config-name train_rastrigin1d lr=1e-3

# Медленнее, но стабильнее
python train.py --config-name train_rastrigin1d lr=1e-5
```

### 3. Изменить количество шагов

```bash
# Короткое обучение (тест)
python train.py --config-name train_rastrigin1d n_steps=1000

# Длинное обучение (production)
python train.py --config-name train_rastrigin1d n_steps=20000
```

### 4. Обучение на своей функции

Создайте функцию в `data/function.py`:

```python
def my_lda_like_function(x: torch.Tensor, negate: bool = True, add_dim: bool = True):
    """
    Синтетическая функция похожая на LDA perplexity.

    Args:
        x: Input tensor (shape: [batch, dim])
        negate: If True, return -f(x)
        add_dim: If True, add output dimension

    Returns:
        Function values
    """
    # Ваша функция здесь
    # Например: симуляция LDA perplexity
    T = x.squeeze(-1)  # T из [2, 1000]

    # Симуляция: perplexity растет для очень маленьких/больших T
    y = 1000 / T + T / 10 + 100 * torch.sin(T / 50)

    if negate:
        y = -y

    if add_dim:
        y = y.unsqueeze(-1)

    return y
```

Затем в config:
```yaml
function: my_lda_like_function
discrete: true
bounds:
  T_min: 2
  T_max: 1000
```

---

## 📊 Мониторинг обучения

### Console output

```
[2024-11-03 10:00:00] Starting training...
[2024-11-03 10:00:00] Config: train_rastrigin1d_test
[2024-11-03 10:00:00] Model: d_model=32, n_layers=3, n_heads=4
[2024-11-03 10:00:00] Device: cpu

Step 0/2000 | Loss: 0.6931 | Best: 2.456 | Time: 0.12s
Step 100/2000 | Loss: 0.4521 | Best: 1.234 | Time: 12.5s
Step 200/2000 | Loss: 0.3215 | Best: 0.876 | Time: 24.8s
...
Step 2000/2000 | Loss: 0.1234 | Best: 0.245 | Time: 250s

Training complete!
Best model saved to: policies/checkpoints/model_best.pt
Final checkpoint: policies/checkpoints/model_step_2000.pt
```

### Что означают метрики?

- **Loss**: Preference loss (должна уменьшаться)
  - Начало: ~0.69 (случайная модель)
  - Конец: ~0.1-0.2 (хорошо обученная)

- **Best**: Лучшее найденное значение функции
  - Должно приближаться к глобальному минимуму
  - Для Rastrigin 1D: ~0 (идеально)

- **Time**: Время с начала обучения

### TensorBoard (опционально)

Если включен TensorBoard:
```bash
tensorboard --logdir policies/checkpoints/runs
```

Графики:
- Loss vs Steps
- Best Value vs Steps
- Learning Rate schedule

---

## 🔧 Troubleshooting

### Проблема 1: CUDA Out of Memory

**Симптомы:**
```
RuntimeError: CUDA out of memory
```

**Решение:**
```bash
# Уменьшите batch_size или размер модели
python train.py \
  --config-name train_rastrigin1d_test \
  batch_size=8 \
  model.d_model=32
```

### Проблема 2: Loss не уменьшается

**Симптомы:**
```
Step 1000 | Loss: 0.6931 (не меняется)
```

**Решения:**
1. Уменьшите learning rate:
   ```bash
   python train.py --config-name train_rastrigin1d lr=1e-5
   ```

2. Увеличьте warmup:
   ```bash
   python train.py --config-name train_rastrigin1d training.warmup_steps=500
   ```

3. Проверьте функцию (может быть слишком сложная)

### Проблема 3: Best value не улучшается

**Симптомы:**
```
Step 1000 | Best: 2.5 (застряло)
```

**Решения:**
1. Увеличьте exploration:
   - В коде модели: увеличьте temperature sampling

2. Дольше обучайте:
   ```bash
   python train.py --config-name train_rastrigin1d n_steps=16000
   ```

3. Увеличьте емкость модели:
   ```bash
   python train.py \
     --config-name train_rastrigin1d \
     model.d_model=128 \
     model.n_layers=8
   ```

### Проблема 4: Слишком медленное обучение

**Симптомы:**
- 1 step = 10+ секунд

**Решения:**
1. Используйте GPU (если доступна):
   ```bash
   python train.py --config-name train_rastrigin1d device=cuda
   ```

2. Уменьшите n_episodes:
   ```bash
   python train.py \
     --config-name train_rastrigin1d \
     training.n_episodes=50
   ```

3. Используйте маленькую модель для теста:
   ```bash
   python train.py --config-name train_rastrigin1d_test
   ```

---

## 📈 Оценка качества модели

### Тестирование на валидации

```bash
# Тест на той же функции
python evaluate_continuous.py \
  --model policies/checkpoints/model_best.pt \
  --function rastrigin \
  --n_trials 100 \
  --budget 50

# Результат:
# Mean: 0.45 ± 0.18
# Median: 0.38
# Success rate: 92% (< 0.5)
```

### Сравнение с baseline

```bash
# Сравнение: Random, BO, PABBO
python baseline.py \
  --function rastrigin \
  --methods random bo pabbo \
  --n_trials 100 \
  --budget 50 \
  --pabbo_model policies/checkpoints/model_best.pt

# Результат:
# Method          Best Value    Std Dev    Time (s)
# -----------------------------------------------------
# Random          2.34 ± 0.89              0.05
# BO (GP)         0.82 ± 0.34              1.23
# PABBO           0.45 ± 0.18              0.15
```

**Хорошие результаты:**
- PABBO лучше Random (значительно)
- PABBO comparable или лучше BO
- PABBO быстрее BO

---

## 🚀 Использование в production

### 1. Обучите финальную модель

```bash
# Длинное обучение для production
python train.py \
  --config-name train_rastrigin1d \
  n_steps=20000 \
  model.d_model=128 \
  model.n_layers=8

# Сохранится в: policies/checkpoints/model_best.pt
```

### 2. Протестируйте на валидации

```bash
python evaluate_continuous.py \
  --model policies/checkpoints/model_best.pt \
  --function rastrigin \
  --n_trials 500 \
  --budget 100
```

### 3. Используйте в LDA оптимизации

```bash
cd ../lda_hyperopt

python run.py \
  --data your_corpus.npz \
  --algorithms PABBO_Full \
  --pabbo-model ../pabbo_method/policies/checkpoints/model_best.pt \
  --iterations 100 \
  --outdir results
```

---

## 💡 Best Practices

1. **Начните с быстрого теста**
   ```bash
   python train.py --config-name train_rastrigin1d_test
   ```

2. **Проверьте что модель учится** (Loss уменьшается)

3. **Запустите полное обучение** если тест успешен

4. **Тестируйте на валидации** перед использованием

5. **Сохраняйте checkpoint'ы регулярно**

6. **Документируйте hyperparameters** которые работают

---

## 📚 Дополнительные ресурсы

### Доступные функции для обучения

Из `data/function.py`:
- `forrester1D` - 1D синтетическая
- `sinexp1D` - sin(x) + exp(x)
- `rastrigin1D` - много локальных минимумов
- `branin2D` - 2D классическая
- `beale2D` - 2D с крутым ландшафтом
- `hartmann6D` - 6D сложная
- `ackley6D` - 6D многомерная
- `rastrigin6D` - 6D Rastrigin
- `levy6D` - 6D Levy
- `rosenbrock6D` - 6D Rosenbrock

### Рекомендуемые конфигурации

**Для быстрого теста:**
```yaml
n_steps: 2000
model: {d_model: 32, n_layers: 3, n_heads: 4}
batch_size: 16
```

**Для production:**
```yaml
n_steps: 20000
model: {d_model: 128, n_layers: 8, n_heads: 8}
batch_size: 32
```

**Для LDA:**
```yaml
n_steps: 10000
discrete: true
bounds: {T_min: 2, T_max: 1000}
model: {d_model: 64, n_layers: 6, n_heads: 8}
```

---

## ✅ Checklist для успешного обучения

- [ ] Зависимости установлены (torch, botorch, etc.)
- [ ] Быстрый тест работает (`train_rastrigin1d_test`)
- [ ] Loss уменьшается во время обучения
- [ ] Best value улучшается
- [ ] Модель сохранена в `policies/checkpoints/model_best.pt`
- [ ] Протестировано на валидации
- [ ] Работает в `lda_hyperopt/run.py` с `--pabbo-model`

---

## 🎉 Готово!

Теперь вы знаете как:
1. ✅ Установить зависимости
2. ✅ Быстро протестировать обучение (10 мин)
3. ✅ Запустить полное обучение (30-60 мин)
4. ✅ Кастомизировать под LDA
5. ✅ Использовать модель в оптимизации

**Следующий шаг:**
```bash
cd pabbo_method
python train.py --config-name train_rastrigin1d_test
# Подождать 10 минут
cd ../lda_hyperopt
python run.py --algorithms PABBO_Full --pabbo-model ../pabbo_method/policies/checkpoints/model_best.pt
```

Удачи! 🚀
