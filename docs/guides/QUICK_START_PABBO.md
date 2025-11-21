# ⚡ PABBO Quick Start - Шпаргалка

Быстрый старт для обучения и использования PABBO Full.

---

## 🚀 Самый быстрый способ (3 команды)

```bash
# 1. Установка (если нужно)
pip install torch botorch gpytorch hydra-core

# 2. Обучение (10 минут)
cd pabbo_method
python train.py --config-name train_rastrigin1d_test

# 3. Найти обученную модель (появится после шага 2)
EXPID=$(ls -t results/PABBO/ | head -1)
echo "Модель: results/PABBO/${EXPID}/ckpt.tar"

# 4. Использование в LDA
cd ../lda_hyperopt
python run.py \
  --data data.npz \
  --algorithms PABBO_Full \
  --pabbo-model "../pabbo_method/results/PABBO/${EXPID}/ckpt.tar"
```

**Готово!** 🎉

---

## 📖 Как это работает (простыми словами)

**Без обучения (PABBO Simple):**
- Случайный поиск + память о хороших точках
- Работает "методом проб и ошибок"

**После обучения (PABBO Full):**
- То же самое + **умная модель**
- Модель помнит паттерны из сотен задач
- Может быстрее найти минимум

**Аналогия:**
- PABBO Simple = новичок
- PABBO Full = эксперт с опытом

---

## 🔧 Основные команды

### Обучение

```bash
cd pabbo_method

# Быстрый тест (10 мин, CPU ok)
python train.py --config-name train_rastrigin1d_test

# Полное обучение (30-60 мин, лучше качество)
python train.py --config-name train_rastrigin1d

# С GPU (быстрее)
python train.py --config-name train_rastrigin1d device=cuda

# Большая модель (лучше, но медленнее)
python train.py \
  --config-name train_rastrigin1d \
  model.d_model=128 \
  model.n_layers=8 \
  n_steps=20000
```

### Проверка модели

```bash
# Найдите ваш expid (имя папки после обучения)
EXPID=$(ls -t results/PABBO/ | head -1)

# Тест на конкретной функции
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
```

### Использование в LDA

```bash
cd ../lda_hyperopt

# Найдите модель (или укажите конкретный путь)
EXPID=$(ls -t ../pabbo_method/results/PABBO/ | head -1)
MODEL_PATH="../pabbo_method/results/PABBO/${EXPID}/ckpt.tar"

# С PABBO Full
python run.py \
  --data data.npz \
  --algorithms PABBO_Full \
  --pabbo-model "${MODEL_PATH}" \
  --iterations 50

# Сравнить Simple и Full
python run.py \
  --data data.npz \
  --algorithms PABBO_Simple PABBO_Full \
  --pabbo-model "${MODEL_PATH}"
```

---

## 📊 Мониторинг обучения

### Что вы увидите

```
Step 0/2000 | Loss: 0.6931 | Best: 2.456
Step 500/2000 | Loss: 0.3215 | Best: 0.876
Step 1000/2000 | Loss: 0.2103 | Best: 0.534
Step 2000/2000 | Loss: 0.1234 | Best: 0.245

Training complete!
Model saved to: policies/checkpoints/model_best.pt
```

### Что означает

- **Loss** уменьшается: ✅ модель учится
- **Best** улучшается: ✅ находит минимум
- **Loss не меняется**: ❌ проблема (см. troubleshooting)

---

## 🔍 Что происходит внутри?

### 1. Генерация задач
```python
# Модель решает тысячи случайных задач
for episode in range(n_episodes):
    function = random_function()  # Rastrigin, Ackley, etc.
    optimize(function)  # Пробует найти минимум
```

### 2. Обучение на опыте
```python
# Модель учится на сравнениях
if point_A_better_than_point_B:
    train: score(A) > score(B)
```

### 3. Запоминание паттернов
```python
# Transformer запоминает:
# - Какие регионы обычно хороши
# - Как балансировать exploration/exploitation
# - Паттерны разных функций
```

### 4. Использование
```python
# При оптимизации LDA:
history = [(T=733, ppl=1250), (T=811, ppl=1180), ...]
T_next = transformer.predict(history)  # Умное предложение!
```

---

## ⚙️ Параметры (самые важные)

| Параметр | Быстро | Качественно | Описание |
|----------|--------|-------------|----------|
| `n_steps` | 2000 | 20000 | Шагов обучения |
| `d_model` | 32 | 128 | Размер модели |
| `n_layers` | 3 | 8 | Слоев Transformer |
| `lr` | 1e-4 | 1e-4 | Learning rate |
| `batch_size` | 16 | 32 | Размер батча |

**Для быстрого теста:**
```bash
python train.py --config-name train_rastrigin1d_test
# Использует: n_steps=2000, d_model=32, n_layers=3
```

**Для production:**
```bash
python train.py \
  --config-name train_rastrigin1d \
  n_steps=20000 \
  model.d_model=128 \
  model.n_layers=8
```

---

## 🐛 Troubleshooting

### Loss не уменьшается
```bash
# Уменьшите learning rate
python train.py --config-name train_rastrigin1d lr=1e-5
```

### Out of memory
```bash
# Уменьшите модель/batch
python train.py \
  --config-name train_rastrigin1d_test \
  batch_size=8 \
  model.d_model=32
```

### Слишком медленно
```bash
# 1. Используйте GPU
python train.py --config-name train_rastrigin1d device=cuda

# 2. Или маленькую модель
python train.py --config-name train_rastrigin1d_test
```

### Модель не загружается в LDA
```bash
# Найдите все модели
find pabbo_method/results -name "ckpt.tar"

# Проверьте путь к конкретной модели
ls -la pabbo_method/results/PABBO/*/ckpt.tar

# Проверьте что файл корректный
python -c "import torch; print(torch.load('path/to/ckpt.tar', map_location='cpu').keys())"
# Должны увидеть: dict_keys(['model', 'optimizer', 'scheduler', 'expdir', 'step'])
```

---

## 🎯 Когда использовать PABBO Full?

### ✅ Используйте если:
- Есть время обучить модель (10-60 мин)
- Хотите более быструю сходимость
- Решаете много похожих задач

### ❌ Используйте Simple если:
- Нет времени/ресурсов для обучения
- Одна уникальная задача
- Простота важнее скорости

### 🤔 В чем разница?

На практике **оба работают хорошо**. Full может быть на **10-30% быстрее** в сходимости, но требует предобучения.

---

## 📁 Где что лежит?

```
pabbo_method/
├── train.py                           # Обучение модели
├── evaluate_continuous.py             # Тестирование
├── baseline.py                        # Сравнение с BO
├── configs/
│   ├── train_rastrigin1d_test.yaml   # Быстрый тест
│   ├── train.yaml                     # Базовая конфигурация
│   └── evaluate.yaml                  # Конфиг для оценки
├── policies/
│   └── transformer.py                 # Архитектура модели
├── data/
│   ├── sampler.py                     # Генератор задач
│   └── function.py                    # Тестовые функции
└── results/                           # ← СОЗДАЕТСЯ ПРИ ОБУЧЕНИИ
    └── PABBO/
        └── {expid}/                   # Например: PABBO_rastrigin1d_test_quick_20241103_190000
            └── ckpt.tar              # ← МОДЕЛЬ ЗДЕСЬ!
```

**ВАЖНО**: Модель НЕ в `policies/checkpoints/`, а в `results/PABBO/{expid}/ckpt.tar`!

---

## 🎓 Полезные ссылки

- **Полное руководство**: `PABBO_TRAINING_GUIDE.md`
- **Параметры LDA**: `lda_hyperopt/HYPERPARAMETERS.md`
- **Отчет проверки**: `VERIFICATION_REPORT.md`

---

## ✨ One-liner для каждого случая

### Просто протестировать
```bash
cd pabbo_method && python train.py --config-name train_rastrigin1d_test
```

### Production модель
```bash
cd pabbo_method && python train.py --config-name train_rastrigin1d n_steps=20000 model.d_model=128
```

### Использовать в LDA
```bash
# Сначала найдите модель
EXPID=$(ls -t pabbo_method/results/PABBO/ | head -1)
# Затем используйте
cd lda_hyperopt && python run.py --data data.npz --algorithms PABBO_Full --pabbo-model "../pabbo_method/results/PABBO/${EXPID}/ckpt.tar"
```

### Проверить что работает
```bash
cd pabbo_method
EXPID=$(ls -t results/PABBO/ | head -1)
python evaluate_continuous.py --config-name=evaluate \
  experiment.model=PABBO experiment.expid="${EXPID}" \
  data.name=rastrigin1D data.d_x=1 data.x_range="[[-5.12,5.12]]" \
  data.Xopt="[[0.0]]" data.yopt="[[0.0]]"
```

---

## 🚦 Статус после обучения

**✅ Хорошо обученная модель:**
- Loss < 0.2
- Best value близко к глобальному минимуму
- На валидации: success rate > 80%

**❌ Плохо обученная:**
- Loss > 0.5 (застряло)
- Best value далеко от минимума
- На валидации: worse than random

**➡️ Если плохо:**
- Обучайте дольше (`n_steps=20000`)
- Увеличьте модель (`d_model=128`)
- Проверьте function (может быть слишком сложная)

---

## 💡 Pro Tips

1. **Начните с теста** - не тратьте час если что-то не работает
2. **GPU не обязателен** - CPU справляется за разумное время
3. **Checkpoint'ы** - сохраняются автоматически каждые 1000 шагов
4. **PABBO Simple уже хорош** - Full это апгрейд, но не обязательно

---

Готово! Теперь всё понятно? 🎉

**Начните здесь:**
```bash
cd pabbo_method
python train.py --config-name train_rastrigin1d_test
```