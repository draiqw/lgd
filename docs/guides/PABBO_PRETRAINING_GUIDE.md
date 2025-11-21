# 🎓 Полное руководство по предобучению PABBO Full

Детальная инструкция по обучению и использованию Transformer-модели для PABBO Full.

---

## 📋 Оглавление

1. [Как работает предобучение](#как-работает-предобучение)
2. [Структура проекта](#структура-проекта)
3. [Процесс предобучения](#процесс-предобучения)
4. [Использование модели](#использование-модели)
5. [Продвинутые настройки](#продвинутые-настройки)

---

## 🧠 Как работает предобучение

### Концепция

PABBO использует **амортизированную оптимизацию** (amortized optimization):

```python
# Традиционный подход: обучаем модель для каждой задачи
for task in tasks:
    model = train_from_scratch(task)  # Медленно!

# PABBO: обучаем ОДНУ модель на ВСЕХ задачах
model = pretrain_on_many_tasks(1000s_of_tasks)  # Быстро потом!
for task in new_tasks:
    result = model.predict(task)  # Мгновенно!
```

### Что происходит при обучении

**train.py (line 23)** - основной пайплайн тренировки:

1. **Генерация задач** (train.py:144-160)
   - OptimizationSampler создает семейство функций из GP
   - Для GP1D: случайные 1D функции с гарантированным глобальным оптимумом
   - data/sampler.py (line 338) - класс OptimizationSampler

2. **Формирование эпизодов** (train.py:208-329)
   - Для каждой функции создаются пары точек для сравнения
   - Модель учится предпочтениям: "точка A лучше точки B"
   - Это preference learning - основа PABBO

3. **Обучение Transformer** (train.py:339-362)
   - policies/transformer.py (line 55) - архитектура модели
   - Кодирует контекстные дуэли и оценочные точки
   - Голова af_mlp выдает баллы для кандидатных пар (line 196)

4. **Сохранение чекпоинтов** (train.py:486-509)
   - **ВАЖНО**: Модель сохраняется в `results/PABBO/{expid}/ckpt.tar`
   - НЕ в `policies/checkpoints/`!

---

## 📂 Структура проекта

### Папки и файлы

```
pabbo_method/
├── train.py                    # Скрипт обучения
├── evaluate_continuous.py      # Оценка модели
├── configs/
│   ├── train.yaml             # Базовая конфигурация
│   ├── train_rastrigin1d_test.yaml  # Быстрый тест (10-20 мин)
│   └── evaluate.yaml          # Конфиг для оценки
├── data/
│   ├── sampler.py             # OptimizationSampler (line 338)
│   └── function.py            # Тестовые функции (line 27)
├── policies/
│   └── transformer.py         # TransformerModel (line 55)
├── results/                   # ← СОЗДАЕТСЯ АВТОМАТИЧЕСКИ
│   └── PABBO/
│       └── {expid}/
│           └── ckpt.tar       # ← МОДЕЛЬ ЗДЕСЬ!
└── datasets/                  # ← СОЗДАЕТСЯ АВТОМАТИЧЕСКИ
    └── evaluation/
```

### Где хранятся модели

**ВАЖНО**: Путь к модели формируется так:

```python
# train.py (line 26-28)
exp_path = results/PABBO/{expid}/
model_path = exp_path/ckpt.tar

# Например:
# results/PABBO/PABBO_rastrigin1d_test_quick_20241103_190000/ckpt.tar
```

**expid** формируется из конфига (configs/train.yaml:20):
```yaml
expid: ${experiment.model}_${data.name}_${now:%Y%m%d_%H%M%S}
```

---

## 🚀 Процесс предобучения

### Шаг 1: Быстрое тестирование (10-20 минут)

Для начала убедимся, что всё работает:

```bash
cd pabbo_method

# Запускаем быстрое обучение
python train.py --config-name=train_rastrigin1d_test

# Что вы увидите:
# - Создается папка results/
# - Каждые 100 шагов печатается прогресс
# - Каждые 500 шагов сохраняется чекпоинт
```

**Вывод будет выглядеть так:**

```
Experiment: PABBO_rastrigin1d_test_quick_20241103_190000
Total number of parameters: 125432

PABBO_rastrigin1d_test_quick step 100 lr 1.000e-03 [train_loss]
    loss 0.5234 cls_loss 0.5234 policy_loss 0.0000 acc 0.6543 kt_cor 0.4321

PABBO_rastrigin1d_test_quick step 500 lr 9.239e-04 [train_loss]
    loss 0.2145 cls_loss 0.2145 policy_loss 0.0000 acc 0.8234 kt_cor 0.7123

PABBO_rastrigin1d_test_quick step 1000 lr 7.071e-04 [train_loss]
    loss 0.1543 cls_loss 0.1543 policy_loss 0.0000 acc 0.8876 kt_cor 0.8234
```

**Параметры быстрого теста** (configs/train_rastrigin1d_test.yaml):
- `n_steps: 2000` - всего шагов обучения
- `n_burnin: 800` - первые 800 шагов только prediction task
- `train_batch_size: 64` - размер батча для prediction
- `ac_train_batch_size: 8` - размер батча для acquisition
- `model.d_model: 32` - размер модели (маленькая)
- `model.n_layers: 3` - слоёв Transformer

### Шаг 2: Найти обученную модель

```bash
# После завершения обучения
ls -la results/PABBO/

# Вы увидите папку с именем типа:
# PABBO_rastrigin1d_test_quick_20241103_190000/

# Модель находится здесь:
ls -la results/PABBO/PABBO_rastrigin1d_test_quick_20241103_190000/ckpt.tar

# Это и есть ваша обученная модель!
```

### Шаг 3: Полное обучение (если нужно лучшее качество)

Для production создайте свою конфигурацию:

```bash
# Создайте pabbo_method/configs/train_gp1d_full.yaml
cd pabbo_method

python train.py --config-name=train \
  experiment.expid=PABBO_GP1D_FULL \
  experiment.device=cpu \
  experiment.wandb=false \
  data.name=GP1D \
  data.d_x=1 \
  data.x_range="[[-1,1]]" \
  data.min_num_ctx=1 \
  data.max_num_ctx=50 \
  train.n_steps=8000 \
  train.n_burnin=3000 \
  train.train_batch_size=128 \
  train.ac_train_batch_size=16 \
  model.d_model=64 \
  model.n_layers=6 \
  model.nhead=4
```

**Время обучения:**
- CPU: ~1-2 часа
- GPU (cuda): ~20-30 минут

**Модель сохранится в:**
```
results/PABBO/PABBO_GP1D_FULL/ckpt.tar
```

### Шаг 4: Проверка модели

```bash
cd pabbo_method

# Укажите путь к вашей обученной модели
python evaluate_continuous.py --config-name=evaluate \
  experiment.model=PABBO \
  experiment.expid=PABBO_rastrigin1d_test_quick_20241103_190000 \
  experiment.device=cpu \
  experiment.wandb=false \
  data.name=rastrigin1D \
  data.d_x=1 \
  data.x_range="[[-5.12,5.12]]" \
  data.Xopt="[[0.0]]" \
  data.yopt="[[0.0]]" \
  eval.eval_max_T=60 \
  eval.eval_num_query_points=256
```

**Что проверяется:**

evaluate_continuous.py (line 28):
1. Загружает модель из `results/PABBO/{expid}/ckpt.tar` (line 44)
2. Тестирует на новых функциях
3. Считает метрики: regret, entropy, correlation
4. Сохраняет результаты в `results/evaluation/{data.name}/{model}/{expid}/`

---

## 🔧 Использование модели

### В lda_hyperopt (PABBO_FULL)

**ВАЖНО**: Укажите ПОЛНЫЙ путь к модели!

```bash
cd lda_hyperopt

# Замените на ваш expid из train.py
EXPID="PABBO_rastrigin1d_test_quick_20241103_190000"
MODEL_PATH="../pabbo_method/results/PABBO/${EXPID}/ckpt.tar"

python run.py \
  --data data.npz \
  --algorithms PABBO_Full \
  --pabbo-model "${MODEL_PATH}" \
  --iterations 50 \
  --seed 42
```

### Как PABBO_FULL загружает модель

lda_hyperopt/optimizers/pabbo_full.py (line 140-163):

```python
def _load_model(self, model_path: str):
    """Load trained Transformer model."""
    checkpoint = torch.load(model_path, map_location='cpu')

    # Пытаемся извлечь конфиг из чекпоинта
    if 'config' in checkpoint:
        config = checkpoint['config']
        self.model = TransformerModel(**config)
    else:
        # Используем дефолтный конфиг
        self.model = TransformerModel(
            d_model=64,
            n_heads=8,
            n_layers=6,
            dropout=0.1
        )

    # Загружаем веса
    if 'model' in checkpoint:
        self.model.load_state_dict(checkpoint['model'])
    else:
        self.model.load_state_dict(checkpoint)
```

**Структура чекпоинта** (train.py:491-497):

```python
ckpt = {
    "model": model_state_dict,      # Веса модели
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "expdir": expdir,
    "step": epoch + 1,
}
```

---

## ⚙️ Продвинутые настройки

### Настройка для конкретной задачи

Если вы хотите предобучить модель специально для LDA оптимизации:

#### 1. Определите диапазон T

```python
# В вашей задаче:
T_bounds = (2, 1000)  # Например

# Нормализуем в [-1, 1]:
x_range = [[-1, 1]]
```

#### 2. Создайте кастомную конфигурацию

```yaml
# pabbo_method/configs/train_lda_optimized.yaml

data:
  name: "GP1D_LDA"
  d_x: 1
  x_range: [[-1, 1]]  # Нормализованный диапазон
  min_num_ctx: 5
  max_num_ctx: 100   # Увеличено для LDA (больше истории)

experiment:
  expid: PABBO_LDA_OPTIMIZED
  device: cuda  # Используйте GPU если доступен

train:
  n_steps: 15000     # Больше шагов для лучшего качества
  n_burnin: 5000
  train_batch_size: 128
  ac_train_batch_size: 16
  max_T: 100         # Увеличено (больше итераций)

model:
  d_model: 128       # Большая модель
  n_layers: 8
  nhead: 8
  dim_feedforward: 256
```

#### 3. Обучите модель

```bash
cd pabbo_method
python train.py --config-name=train_lda_optimized
```

### Понимание двухфазного обучения

train.py использует две фазы (line 208-329):

#### Фаза 1: Burnin (шаги 1-n_burnin)

```python
# train.py (line 213-228)
if epoch <= n_burnin:
    # Только prediction task
    # Модель учится предсказывать функцию по парам
    X_pred, y_pred = sampler.sample(...)

    # Compute BCE loss на предсказание предпочтений
    cls_loss = preference_cls_loss(f=pred_tar_f, c=src_c)
```

**Цель**: Научить модель понимать функции через парные сравнения.

#### Фаза 2: Acquisition Learning (шаги n_burnin+1 до n_steps)

```python
# train.py (line 229-412)
else:
    # Prediction + Acquisition task
    X_pred, y_pred = sampler.sample(...)  # для prediction
    X_ac, y_ac = sampler.sample(...)      # для acquisition

    # Policy learning loop (оптимизация на max_T шагов)
    for t in range(1, max_T + 1):
        # Предлагаем следующую пару точек
        acq_values, next_pair = action(model, context_pairs, ...)

        # Получаем reward
        reward = get_reward(context_pairs_y, acq_values, ...)

    # Обновляем модель через policy gradient
    policy_loss = finish_episode(rewards, log_probs, ...)
    loss = policy_loss + loss_weight * cls_loss
```

**Цель**: Научить модель выбирать информативные пары для запроса.

### Важные гиперпараметры

| Параметр | Быстро | Баланс | Качество | Описание |
|----------|--------|--------|----------|----------|
| `n_steps` | 2000 | 8000 | 20000 | Шагов обучения |
| `n_burnin` | 800 | 3000 | 7000 | Шагов в фазе 1 |
| `d_model` | 32 | 64 | 128 | Размер модели |
| `n_layers` | 3 | 6 | 8-12 | Слоёв Transformer |
| `nhead` | 2 | 4 | 8 | Голов внимания |
| `max_T` | 30 | 64 | 100 | Шагов оптимизации |
| `train_batch_size` | 64 | 128 | 256 | Размер батча (фаза 1) |
| `ac_train_batch_size` | 8 | 16 | 32 | Размер батча (фаза 2) |

---

## 🎯 Куда класть модель

### Вариант 1: Использовать из results/ (рекомендуется)

```bash
# После обучения модель автоматически в:
pabbo_method/results/PABBO/{expid}/ckpt.tar

# Использовать напрямую:
cd lda_hyperopt
python run.py \
  --data data.npz \
  --algorithms PABBO_Full \
  --pabbo-model ../pabbo_method/results/PABBO/PABBO_GP1D_FULL/ckpt.tar
```

### Вариант 2: Скопировать в отдельную папку (опционально)

```bash
# Создайте папку для продакшн моделей
mkdir -p pabbo_method/trained_models

# Скопируйте лучшую модель
cp results/PABBO/PABBO_GP1D_FULL/ckpt.tar \
   trained_models/pabbo_gp1d_production.tar

# Используйте
cd lda_hyperopt
python run.py \
  --data data.npz \
  --algorithms PABBO_Full \
  --pabbo-model ../pabbo_method/trained_models/pabbo_gp1d_production.tar
```

---

## 📊 Мониторинг обучения

### Ключевые метрики

**1. Classification Loss (cls_loss)**
- Насколько хорошо модель предсказывает предпочтения
- Должна уменьшаться: 0.7 → 0.3 → 0.1
- train.py (line 349-352)

**2. Accuracy (acc)**
- Точность классификации предпочтений
- Должна расти: 0.5 → 0.7 → 0.85+
- train.py (line 355)

**3. Kendall Tau Correlation (kt_cor)**
- Корреляция предсказанных значений с истинными
- Должна расти: 0.3 → 0.6 → 0.8+
- train.py (line 356-360)

**4. Policy Loss (после burnin)**
- Потеря reinforcement learning
- Должна стабилизироваться
- train.py (line 431-437)

**5. Final Simple Regret**
- Разница между найденным и глобальным оптимумом
- Должна уменьшаться
- train.py (line 428)

### Признаки хорошего обучения

```
# В начале
step 100 | loss 0.6543 | cls_loss 0.6543 | acc 0.5234 | kt_cor 0.2156

# После burnin
step 800 | loss 0.2345 | cls_loss 0.2345 | acc 0.7823 | kt_cor 0.6543

# Начало acquisition learning
step 801 | loss 0.3456 | cls_loss 0.2123 | policy_loss 0.1333 | acc 0.8012

# В конце
step 8000 | loss 0.1234 | cls_loss 0.0876 | policy_loss 0.0358 | acc 0.9234 | kt_cor 0.8765
```

**Хорошие показатели:**
- ✅ cls_loss < 0.2
- ✅ acc > 0.85
- ✅ kt_cor > 0.75
- ✅ final_simple_regret < 0.1

---

## 🐛 Troubleshooting

### Loss не уменьшается

```bash
# Уменьшите learning rate
python train.py --config-name=train_rastrigin1d_test \
  train.lr=1e-4 \
  train.ac_lr=1e-5
```

### Out of Memory

```bash
# Уменьшите размеры
python train.py --config-name=train_rastrigin1d_test \
  train.train_batch_size=32 \
  train.ac_train_batch_size=4 \
  model.d_model=32 \
  model.n_layers=3
```

### Модель не загружается в PABBO_FULL

**Проверьте путь:**
```bash
# Найдите все модели
find pabbo_method/results -name "ckpt.tar"

# Проверьте, что файл читается
python -c "import torch; print(torch.load('path/to/ckpt.tar', map_location='cpu').keys())"
```

**Должны увидеть:**
```python
dict_keys(['model', 'optimizer', 'scheduler', 'expdir', 'step'])
```

---

## 📝 Пошаговый чеклист

### Предобучение с нуля

- [ ] 1. Установить зависимости: `pip install torch botorch gpytorch hydra-core`
- [ ] 2. Перейти в папку: `cd pabbo_method`
- [ ] 3. Запустить быстрый тест: `python train.py --config-name=train_rastrigin1d_test`
- [ ] 4. Дождаться завершения (~10-20 мин)
- [ ] 5. Найти модель: `ls -la results/PABBO/*/ckpt.tar`
- [ ] 6. Записать путь к модели
- [ ] 7. (Опционально) Запустить evaluate для проверки
- [ ] 8. Использовать в lda_hyperopt с флагом `--pabbo-model {путь}`

### Для production

- [ ] 1. Создать конфигурацию (скопировать train.yaml)
- [ ] 2. Настроить параметры под задачу
- [ ] 3. Запустить с GPU: `device=cuda`
- [ ] 4. Увеличить шаги: `n_steps=15000`
- [ ] 5. Увеличить модель: `d_model=128`, `n_layers=8`
- [ ] 6. Мониторить метрики
- [ ] 7. Сохранить лучшую модель в отдельную папку
- [ ] 8. Протестировать на валидации

---

## 🎓 Теория: почему это работает

### Preference-based Optimization

PABBO не оптимизирует функцию напрямую, а учится на **сравнениях**:

```python
# Вместо:
f(x1) = 2.5, f(x2) = 3.7  # Нужны точные значения

# PABBO использует:
f(x1) < f(x2)  # Только сравнение!
```

**Преимущества:**
1. Работает даже когда точные значения зашумлены
2. Robustness к масштабу функции
3. Естественно для человека-эксперта

### Amortized Optimization

```python
# Традиционный BO:
for task in tasks:
    gp = train_gp(task)       # 100 вычислений
    optimize(gp)              # 50 вычислений
    # Итого: 150 × N задач

# PABBO:
model = pretrain(1000_tasks)   # Делаем ОДИН раз
for task in tasks:
    optimize(model)            # 50 вычислений
    # Итого: претрейн + 50 × N задач
```

Если задач много (N > 20), PABBO **значительно эффективнее**.

### Transformer Architecture

policies/transformer.py (line 55-100):

```python
# Каждая пара точек становится токеном
token = embed([x1, x2, preference])

# Transformer обрабатывает последовательность токенов
context = [token_1, token_2, ..., token_t]
representation = Transformer(context)

# Голова предсказывает acquisition value для всех пар
scores = af_mlp(representation)  # (line 196)
```

**Почему Transformer?**
1. Обрабатывает переменное количество запросов
2. Attention механизм фокусируется на важных парах
3. Масштабируется на большие истории

---

## 🚀 Готовая команда для копипаста

```bash
#!/bin/bash
# Полный пайплайн предобучения PABBO

# 1. Быстрый тест
cd pabbo_method
python train.py --config-name=train_rastrigin1d_test

# 2. Найти модель
EXPID=$(ls -t results/PABBO/ | head -1)
echo "Модель обучена: results/PABBO/${EXPID}/ckpt.tar"

# 3. Протестировать
python evaluate_continuous.py --config-name=evaluate \
  experiment.model=PABBO \
  experiment.expid="${EXPID}" \
  experiment.device=cpu \
  experiment.wandb=false \
  data.name=rastrigin1D \
  data.d_x=1 \
  data.x_range="[[-5.12,5.12]]" \
  data.Xopt="[[0.0]]" \
  data.yopt="[[0.0]]"

# 4. Использовать в LDA
cd ../lda_hyperopt
python run.py \
  --data data.npz \
  --algorithms PABBO_Full \
  --pabbo-model "../pabbo_method/results/PABBO/${EXPID}/ckpt.tar" \
  --iterations 50
```

---

**Готово!** Теперь у вас есть полное понимание предобучения PABBO. 🎉

Начните с быстрого теста, чтобы убедиться что всё работает, затем переходите к полному обучению.