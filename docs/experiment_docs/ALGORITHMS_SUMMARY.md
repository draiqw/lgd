# Краткое описание алгоритмов для статьи

## Сравнение трех алгоритмов оптимизации

| Характеристика | GA | ES | PABBO Full |
|----------------|----|----|------------|
| **Тип** | Genetic Algorithm | Evolution Strategy | Bayesian Optimization + Transformer |
| **Популяция** | 20 кандидатов | μ=5 родителей, λ=10 потомков | История оценок (нет фиксированной популяции) |
| **Селекция** | Tournament selection | (μ,λ)-selection (comma) | Acquisition function |
| **Генерация кандидатов** | Crossover + Mutation | Mutation only | Transformer предсказания |
| **Кроссовер** | Да (prob=0.9, binary) | Нет | Нет |
| **Мутация** | Да (prob=0.2) | Да (Gaussian) | Implicit через exploration |
| **Elitism** | Да (топ-5) | Нет (comma-selection) | Неявно через историю |
| **Обратная связь** | Absolute fitness | Absolute fitness | Pairwise preferences |
| **Требует обучения** | Нет | Нет | Да (meta-learning) |
| **Adaptive** | Нет | Да (step size) | Да (через RL) |

---

## Genetic Algorithm (GA)

### Основная идея
Эволюция популяции решений через операторы селекции, кроссовера и мутации с сохранением лучших особей (elitism).

### Ключевые компоненты

**1. Инициализация:**
- Популяция: 20 значений T ∈ [2, 1000]
- Загружается из `lda_init_population.json`

**2. Selection (Tournament):**
- Выбирается k кандидатов случайно
- Лучший (lowest perplexity) проходит в следующее поколение

**3. Crossover (Binary, prob=0.9):**
```
T_parent1 = 733 → binary: 1011011101
T_parent2 = 355 → binary: 0101100011
                          ↓ crossover point
T_child1  = 1011011011    (берем первую часть от parent1, вторую от parent2)
T_child2  = 0101100101
```

**4. Mutation (prob=0.2):**
- Случайное изменение битов или добавление Gaussian noise
- Результат обрезается до [2, 1000]

**5. Elitism:**
- Топ-5 лучших решений автоматически переходят в следующее поколение

### Гиперпараметры
```python
pop_size = 20
cxpb = 0.9        # crossover probability
mutpb = 0.2       # mutation probability
elite = 5         # number of elite individuals
tournament_k = 3  # tournament size
```

### Псевдокод
```
Initialize population P_0 (20 random T values)
For t = 1 to 20:
    Evaluate fitness (perplexity) for all T in P_{t-1}

    # Select parents
    Parents = tournament_selection(P_{t-1})

    # Generate offspring
    Offspring = []
    for i in range(pop_size - elite):
        parent1, parent2 = random.choice(Parents, 2)
        if random() < cxpb:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        if random() < mutpb:
            child1 = mutate(child1)
        Offspring.append(child1)

    # Elitism: keep top-5
    Elite = top_k(P_{t-1}, k=5)

    # Form new population
    P_t = Elite + Offspring[:15]

Return best T* from P_20
```

---

## Evolution Strategy (ES)

### Основная идея
(μ, λ)-ES: μ родителей генерируют λ потомков через мутацию, и только лучшие μ потомков выживают (родители НЕ конкурируют с потомками).

### Ключевые компоненты

**1. Инициализация:**
- μ=5 родителей выбираются из initial population

**2. Offspring Generation:**
- Каждый родитель генерирует λ/μ = 2 потомка
- Мутация: T_offspring = T_parent + N(0, σ²)
- Результат clipped to [2, 1000] и округляется

**3. Selection (comma-selection):**
- Оцениваются все λ=10 потомков
- Выбираются μ=5 лучших
- Родители **НЕ** участвуют в отборе

**4. Adaptive Step Size (опционально):**
- σ может адаптироваться на основе успешности мутаций
- В текущей реализации: фиксированная или медленно убывающая σ

### Гиперпараметры
```python
mu = 5            # number of parents
lambda_ = 10      # number of offspring
sigma = 50.0      # initial mutation step size (adaptive)
```

### Псевдокод
```
Initialize parents P_0 (select 5 from initial population)
For t = 1 to 20:
    # Generate offspring
    Offspring = []
    for parent in P_{t-1}:
        for i in range(lambda / mu):
            child = parent + Gaussian(0, sigma^2)
            child = clip(round(child), 2, 1000)
            Offspring.append(child)

    # Evaluate all offspring
    fitness = [perplexity(T) for T in Offspring]

    # Select mu best offspring (comma-selection)
    P_t = top_k(Offspring, k=mu)

    # Optional: adapt sigma based on success rate

Return best T* from final history
```

### Отличие от (μ+λ)-ES
- **(μ,λ)**: только потомки конкурируют → избегает стагнации
- **(μ+λ)**: родители+потомки конкурируют → может застрять в локальном оптимуме

---

## PABBO Full (Preferential Amortized Black-Box Optimization)

### Основная идея
Bayesian Optimization с использованием **preference-based feedback** (pairwise comparisons) вместо абсолютных значений fitness. Transformer модель обучается на синтетических задачах и переносится на LDA оптимизацию.

### Ключевые компоненты

**1. Preference Learning:**
- Вместо `f(T) = perplexity` используется `T_i ≻ T_j` (T_i лучше T_j)
- Модель предсказывает вероятность: P(T_i ≻ T_j | history)

**2. Transformer Architecture:**
```
Input: History H_t = {(T_1, perplexity_1), ..., (T_t, perplexity_t)}

↓ Embedding (depth=2)
  Each (T, p) → vector in R^32

↓ Transformer Encoder (3 layers, 2 heads, d_model=32, d_ff=64)
  Self-attention over history + positional encoding

↓ Two heads:
  - Acquisition head: scores for candidate pairs
  - Prediction head: preference prediction (auxiliary loss)

Output: Next T to evaluate
```

**3. Acquisition Function:**
- На основе истории H_t предсказываются "обещающие" регионы
- Acquisition policy: π_θ(a_t | H_t) = softmax(scores)
- Балансировка exploration/exploitation через ε=0.3

**4. Meta-Training:**
- Модель обучена **offline** на:
  - 1D Rastrigin function
  - Gaussian Process samples (RBF kernel)
- Цель: научиться общему поведению оптимизации
- После обучения: zero-shot transfer на LDA

### Гиперпараметры
```python
# Model architecture
d_model = 32
n_layers = 3
nhead = 2
dim_feedforward = 64
emb_depth = 2

# Optimization
exploration_rate = 0.3
model_checkpoint = "ckpt.tar"
```

### Псевдокод
```
Load pre-trained Transformer model (ckpt.tar)
Initialize history H_0 with initial population evaluations

For t = 1 to 20:
    # Encode history
    context = Transformer.encode(H_{t-1})

    # Generate candidate pairs
    candidates = generate_candidates()  # e.g., grid or random sampling
    pairs = [(T_i, T_j) for i, j in combinations(candidates)]

    # Compute acquisition scores
    scores = AcquisitionHead(context, pairs)

    # Select next T (epsilon-greedy)
    if random() < epsilon:
        T_t = random_choice(candidates)  # exploration
    else:
        T_t = argmax(scores)              # exploitation

    # Evaluate
    perplexity_t = evaluate_LDA(T_t)

    # Update history
    H_t = H_{t-1} ∪ {(T_t, perplexity_t)}

Return T* = argmin_{T in H_20} perplexity(T)
```

### Преимущества PABBO
1. **Meta-learning**: обучается на множестве задач → быстрая адаптация
2. **Preference-based**: не требует точных численных оценок, достаточно сравнений
3. **Non-myopic**: Transformer учитывает всю историю, не только текущий шаг
4. **Transfer learning**: модель, обученная на синтетических функциях, работает на реальных задачах

### Недостатки
- Требует предварительного обучения модели (~30 минут)
- Более сложная реализация
- Зависит от качества мета-обучения

---

## Общие параметры всех алгоритмов

```python
# Search space
T_min = 2
T_max = 1000

# Optimization budget
iterations = 20
early_stopping = False  # DISABLED for fair comparison

# Initial population (seed=42)
initial_population = [
    733, 811, 133, 355, 777, 115, 452, 940, 879, 345,
    576, 153, 950, 602, 162, 238, 422, 511, 660, 285
]

# LDA hyperparameters
alpha = 1 / T
eta = 1 / T

# LDA training
lda_max_iter = 60
lda_batch_size = 2048
lda_learning_method = "online"

# Reproducibility
base_seed = 42
run_seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 runs
```

---

## Сравнение по критериям

### 1. Качество оптимизации (Perplexity)
- **GA**: хорошо благодаря crossover (рекомбинация признаков)
- **ES**: стабильно благодаря adaptive mutations
- **PABBO**: потенциально лучше благодаря meta-learning и non-myopic planning

### 2. Вычислительная эффективность
- **GA**: ~средняя (много evaluations из-за большой популяции)
- **ES**: ~быстрая (меньше evaluations: только 10 потомков)
- **PABBO**: +overhead на Transformer inference, но может быстрее сходиться

### 3. Стабильность (low variance)
- **GA**: может быть нестабильной из-за случайного crossover
- **ES**: стабильная благодаря comma-selection
- **PABBO**: зависит от качества meta-training

### 4. Interpretability
- **GA**: высокая (понятные операторы)
- **ES**: высокая (простая мутация)
- **PABBO**: низкая (black-box neural network)

### 5. Требования к ресурсам
- **GA**: низкие (только evaluations)
- **ES**: низкие (только evaluations)
- **PABBO**: средние (нужен Transformer, но CPU достаточно)

---

## Для статьи: ключевые моменты

### Algorithms section
1. **Мотивация**: LDA perplexity optimization — black-box задача без градиентов
2. **Три подхода**: GA (эволюция через crossover), ES (мутация+селекция), PABBO (Bayesian + meta-learning)
3. **Честное сравнение**: одинаковая initial population, фиксированный бюджет (20 iter)
4. **Детали**: описать ключевые операторы (crossover, mutation, acquisition function)

### Setup section
1. **Pipeline**: 4 этапа (train PABBO → evaluate → LDA optimize → aggregate)
2. **Датасеты**: реальные текстовые корпуса (20news, Reuters, ...)
3. **Протокол**: 10 повторений с разными seeds, no early stopping
4. **Метрики**: perplexity (основная), time (эффективность), convergence (траектории)
5. **Статистика**: Wilcoxon test, Friedman test, Cohen's d
6. **Воспроизводимость**: все seeds зафиксированы, код доступен

---

## LaTeX snippets для быстрой вставки

### Краткое описание GA
```latex
The Genetic Algorithm evolves a population of 20 candidates via tournament
selection, binary crossover (prob. 0.9), and mutation (prob. 0.2). Elitism
preserves the top 5 individuals across generations.
```

### Краткое описание ES
```latex
Evolution Strategy follows a $(\mu, \lambda)$-ES with $\mu=5$ parents
generating $\lambda=10$ offspring through Gaussian mutation. Comma-selection
ensures only offspring compete for survival, preventing stagnation.
```

### Краткое описание PABBO
```latex
PABBO leverages a pre-trained Transformer (3 layers, 32-dim) that learns
from preference-based comparisons. Meta-trained on synthetic functions,
it transfers zero-shot to LDA optimization via a non-myopic acquisition policy.
```

### Описание fair comparison
```latex
All algorithms share identical initial conditions: the same 20-point population
(seed 42), exact 20-iteration budget with no early stopping, and hyperparameters
$\alpha=\eta=1/T$. This ensures a fair comparison under a fixed computational budget.
```
