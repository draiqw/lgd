# Визуальная схема экспериментального pipeline

## Общая структура

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LDA HYPERPARAMETER OPTIMIZATION                   │
│                         EXPERIMENTAL PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ЭТАП 1: PABBO Model Training                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: config "train_rastrigin1d_test"                             │
│    │                                                                  │
│    ▼                                                                  │
│  ┌─────────────────────────────────┐                                │
│  │  Transformer Model Training     │                                │
│  │  • Architecture: 3-layer        │                                │
│  │  • d_model=32, nhead=2         │                                │
│  │  • Training on GP1D/Rastrigin  │                                │
│  └─────────────────────────────────┘                                │
│    │                                                                  │
│    ▼                                                                  │
│  Output: ckpt.tar (trained model weights)                           │
│                                                                       │
│  Duration: ~10-30 minutes                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ЭТАП 2: PABBO Model Evaluation                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: ckpt.tar                                                     │
│    │                                                                  │
│    ▼                                                                  │
│  ┌─────────────────────────────────┐                                │
│  │  Evaluation on GP1D             │                                │
│  │  • x_range: [-1, 1]            │                                │
│  │  • ctx points: 5-20            │                                │
│  │  • query points: 256           │                                │
│  └─────────────────────────────────┘                                │
│    │                                                                  │
│    ▼                                                                  │
│  Output: evaluation metrics (optional)                              │
│                                                                       │
│  Duration: ~5-10 minutes                                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ЭТАП 3: LDA Hyperparameter Optimization                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: ckpt.tar + validation datasets                              │
│    │                                                                  │
│    ├──► Load datasets: data/X_*_val_bow.npz                         │
│    │                                                                  │
│    └──► Load initial_population.json                                │
│            [733, 811, 133, 355, 777, ...]                           │
│            (20 values, seed=42)                                     │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FOR EACH DATASET:                                          │   │
│  │                                                              │   │
│  │    FOR run_id in [0, 1, 2, ..., 9]:  (10 repetitions)     │   │
│  │                                                              │   │
│  │      seed = 42 + run_id                                     │   │
│  │                                                              │   │
│  │      ┌────────────────────────────────────────────────┐    │   │
│  │      │  Parallel Execution of 3 Algorithms:          │    │   │
│  │      │                                                │    │   │
│  │      │  ┌──────────────────────────────────────┐     │    │   │
│  │      │  │ GA (Genetic Algorithm)              │     │    │   │
│  │      │  │ • pop_size: 20                      │     │    │   │
│  │      │  │ • cxpb: 0.9, mutpb: 0.2            │     │    │   │
│  │      │  │ • elite: 5                          │     │    │   │
│  │      │  │ • iterations: 20 (NO early stop)   │     │    │   │
│  │      │  │ • T_bounds: [2, 1000]              │     │    │   │
│  │      │  └──────────────────────────────────────┘     │    │   │
│  │      │                                                │    │   │
│  │      │  ┌──────────────────────────────────────┐     │    │   │
│  │      │  │ ES (Evolution Strategy)             │     │    │   │
│  │      │  │ • μ (parents): 5                    │     │    │   │
│  │      │  │ • λ (offspring): 10                 │     │    │   │
│  │      │  │ • iterations: 20 (NO early stop)   │     │    │   │
│  │      │  │ • T_bounds: [2, 1000]              │     │    │   │
│  │      │  └──────────────────────────────────────┘     │    │   │
│  │      │                                                │    │   │
│  │      │  ┌──────────────────────────────────────┐     │    │   │
│  │      │  │ PABBO_Full (with Transformer)       │     │    │   │
│  │      │  │ • model: ckpt.tar                   │     │    │   │
│  │      │  │ • exploration_rate: 0.3             │     │    │   │
│  │      │  │ • iterations: 20 (NO early stop)   │     │    │   │
│  │      │  │ • T_bounds: [2, 1000]              │     │    │   │
│  │      │  └──────────────────────────────────────┘     │    │   │
│  │      │                                                │    │   │
│  │      │  Each algorithm:                              │    │   │
│  │      │  1. Initialize from same population           │    │   │
│  │      │  2. Optimize T to minimize LDA perplexity    │    │   │
│  │      │  3. Run exactly 20 iterations                 │    │   │
│  │      │  4. Save: best_T, best_perplexity, history   │    │   │
│  │      └────────────────────────────────────────────────┘    │   │
│  │                                                              │   │
│  │      Output per run:                                        │   │
│  │        └─ {dataset}/run_{id}/{GA,ES,PABBO_Full}/          │   │
│  │              ├─ summary.json                                │   │
│  │              ├─ history.csv                                 │   │
│  │              └─ optimization_plots.png                      │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Total experiments: N_datasets × 10 runs × 3 algorithms             │
│  Duration: ~40-80 hours (sequential), ~4-8 hours (parallel)        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ЭТАП 4: Results Aggregation & Visualization                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: All summary.json + history.csv files                        │
│    │                                                                  │
│    ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ 4.1 Data Collection                                     │        │
│  │   • Collect all 10 runs × N datasets × 3 algorithms    │        │
│  │   • Filter successful experiments                       │        │
│  │   • Create DataFrame:                                   │        │
│  │     [dataset, run_id, algorithm, best_T,              │        │
│  │      best_perplexity, total_time, ...]                │        │
│  │   • Save: all_results.csv                              │        │
│  └─────────────────────────────────────────────────────────┘        │
│    │                                                                  │
│    ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ 4.2 Statistical Analysis                                │        │
│  │   FOR EACH (dataset, algorithm):                       │        │
│  │     • mean_perplexity ± std (over 10 runs)            │        │
│  │     • min_perplexity, max_perplexity                   │        │
│  │     • mean_time ± std                                  │        │
│  │   • Save: statistics.json                              │        │
│  └─────────────────────────────────────────────────────────┘        │
│    │                                                                  │
│    ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ 4.3 Visualization Generation                            │        │
│  │                                                          │        │
│  │   ┌─────────────────────────────────────────┐          │        │
│  │   │ perplexity_comparison.png/svg           │          │        │
│  │   │ • 2×2 grid: bar charts per dataset     │          │        │
│  │   │ • X: algorithms, Y: perplexity         │          │        │
│  │   │ • Error bars: std over 10 runs         │          │        │
│  │   └─────────────────────────────────────────┘          │        │
│  │                                                          │        │
│  │   ┌─────────────────────────────────────────┐          │        │
│  │   │ time_comparison.png/svg                 │          │        │
│  │   │ • Grouped bar chart                     │          │        │
│  │   │ • X: datasets, Y: time (s)             │          │        │
│  │   │ • Groups: by algorithm                  │          │        │
│  │   └─────────────────────────────────────────┘          │        │
│  │                                                          │        │
│  │   ┌─────────────────────────────────────────┐          │        │
│  │   │ perplexity_boxplots.png/svg             │          │        │
│  │   │ • 2×2 grid: box plots per dataset      │          │        │
│  │   │ • X: algorithms, Y: perplexity         │          │        │
│  │   │ • Shows distribution over 10 runs      │          │        │
│  │   └─────────────────────────────────────────┘          │        │
│  │                                                          │        │
│  │   All plots saved in PNG (high-res) + SVG (vector)    │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                       │
│  Output: aggregated_results/ directory with:                        │
│    • all_results.csv (raw data)                                    │
│    • statistics.json (summary stats)                               │
│    • *.png, *.svg (publication-ready figures)                      │
│                                                                       │
│  Duration: ~1-5 minutes                                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   FINAL OUTPUT    │
                    └───────────────────┘
```

---

## Детальная схема одного эксперимента

```
┌───────────────────────────────────────────────────────────────┐
│ Single Experiment: (Dataset, Run_ID)                          │
└───────────────────────────────────────────────────────────────┘

Initial Setup:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • seed = 42 + run_id
  • Load initial_population (20 values of T)
  • Load dataset: X_val (BoW matrix)
  • Create LDA objective function:
      obj(T) → perplexity
      where: α = η = 1/T

┌────────────────────────────────────────────────────────────────┐
│                    ALGORITHM EXECUTION                         │
└────────────────────────────────────────────────────────────────┘

FOR iteration in [0, 1, 2, ..., 19]:  (20 total)

  ┌──────────────────────────────────────┐
  │ 1. Generate candidate solutions      │
  │    • GA: crossover + mutation        │
  │    • ES: mutation from parents       │
  │    • PABBO: transformer prediction   │
  └──────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────┐
  │ 2. Evaluate each candidate           │
  │    FOR each T_candidate:             │
  │      • Set α = η = 1/T              │
  │      • Train LDA model               │
  │        (max_iter=60, batch=2048)    │
  │      • Compute perplexity on X_val  │
  └──────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────┐
  │ 3. Update population/model           │
  │    • GA: select best, elitism        │
  │    • ES: select μ best from λ       │
  │    • PABBO: update history          │
  └──────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────┐
  │ 4. Log iteration results             │
  │    • iter, T_best, best_perplexity  │
  │    • cumulative_time                 │
  │    • current_population              │
  └──────────────────────────────────────┘

END FOR

┌────────────────────────────────────────────────────────────────┐
│                       SAVE RESULTS                             │
└────────────────────────────────────────────────────────────────┘

  • summary.json:
      {
        "algorithm": "GA" | "ES" | "PABBO_Full",
        "best_T": 234,
        "best_alpha": 0.00427,
        "best_eta": 0.00427,
        "best_perplexity": 1234.56,
        "total_time": 567.89,
        "avg_step_time": 28.39,
        "num_iterations": 20,
        "stopped_early": false
      }

  • history.csv:
      iter, T_best, best_perplexity, cum_time
      0,    733,    1456.78,         12.34
      1,    811,    1423.45,         25.67
      ...
      19,   234,    1234.56,         567.89

  • optimization_plots.png/svg:
      [4 subplots showing convergence]
```

---

## Схема структуры данных

```
lda_pipeline_results/
└── run_20241113_150230/
    │
    ├── logs/
    │   ├── pipeline_main.log          # Главный лог
    │   └── pipeline_metrics.json      # Метрики всех этапов
    │
    ├── experiments/
    │   │
    │   ├── 20newsgroups/
    │   │   ├── run_0/
    │   │   │   ├── GA/
    │   │   │   │   ├── summary.json
    │   │   │   │   ├── history.csv
    │   │   │   │   ├── optimization_plots.png
    │   │   │   │   ├── optimization_plots.svg
    │   │   │   │   └── GA_optimization.log
    │   │   │   ├── ES/
    │   │   │   │   └── [same structure]
    │   │   │   ├── PABBO_Full/
    │   │   │   │   └── [same structure]
    │   │   │   ├── comparison.png
    │   │   │   ├── comparison.svg
    │   │   │   └── overall_summary.json
    │   │   ├── run_1/
    │   │   │   └── [same structure]
    │   │   ├── ...
    │   │   └── run_9/
    │   │       └── [same structure]
    │   │
    │   ├── reuters/
    │   │   └── [same structure as 20newsgroups]
    │   │
    │   └── [other datasets]/
    │       └── [same structure]
    │
    ├── all_results.json               # Все сырые результаты
    │
    └── aggregated_results/
        ├── all_results.csv            # Сводная таблица
        ├── statistics.json            # Статистика по датасетам
        ├── perplexity_comparison.png  # Основной график
        ├── perplexity_comparison.svg
        ├── time_comparison.png
        ├── time_comparison.svg
        ├── perplexity_boxplots.png
        └── perplexity_boxplots.svg
```

---

## Временная диаграмма выполнения

```
Time →
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STAGE 1: PABBO Training
├──────────────────────┤
0                    30 min

STAGE 2: Evaluation
                      ├───────┤
                     30      40 min

STAGE 3: LDA Experiments (Sequential)
                              ├────────────────────────────────────────────────────────────┤
                             40 min                                                    80 hours

OR with Parallelization (10 workers):
                              ├───────────┤
                             40 min     8 hours

STAGE 4: Aggregation
                                                                                            ├┤
                                                                                           ~5 min

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~80 hours (sequential) or ~8 hours (parallel)
```

---

## Схема алгоритмов оптимизации

### Genetic Algorithm (GA)
```
┌─────────────────────────────────────────┐
│ Population: [T₁, T₂, ..., T₂₀]         │
└─────────────────────────────────────────┘
         │
         ├─► Selection (tournament)
         │      ↓
         ├─► Crossover (prob=0.9)
         │      ↓
         ├─► Mutation (prob=0.2)
         │      ↓
         ├─► Evaluate fitness (perplexity)
         │      ↓
         └─► Elitism (keep 5 best)
                ↓
         New Population
```

### Evolution Strategy (ES)
```
┌─────────────────────────────────────────┐
│ Parents: [T₁, T₂, T₃, T₄, T₅]  (μ=5)  │
└─────────────────────────────────────────┘
         │
         ├─► Generate λ=10 offspring
         │      (mutation from parents)
         │      ↓
         ├─► Evaluate all offspring
         │      ↓
         └─► Select μ=5 best offspring
                ↓
         New Parents (NO parents survive)
```

### PABBO Full
```
┌─────────────────────────────────────────┐
│ History: [(T, perplexity), ...]        │
└─────────────────────────────────────────┘
         │
         ├─► Feed to Transformer
         │      ↓
         ├─► Predict promising region
         │      ↓
         ├─► Acquisition function
         │      (exploration vs exploitation)
         │      ↓
         ├─► Sample next T
         │      ↓
         └─► Evaluate & add to history
                ↓
         Updated History
```

---

## Формулы и метрики

### LDA Objective Function
```
obj(T) = perplexity(T)

где:
  perplexity(T) = exp(-log_likelihood / num_words)

  α = η = 1/T  (symmetric Dirichlet prior)

  LDA trained with:
    • max_iter = 60
    • batch_size = 2048
    • learning_method = "online"
    • seed = 42 + run_id
```

### Evaluation Metrics
```
For each (dataset, algorithm):

  mean_perplexity = (1/10) Σ perplexity_i
                           i=1..10

  std_perplexity = sqrt((1/9) Σ (perplexity_i - mean)²)
                              i=1..10

  min_perplexity = min(perplexity_i)
                   i=1..10

  max_perplexity = max(perplexity_i)
                   i=1..10

Similar for time metrics.
```

### Statistical Tests (Recommended)
```
1. Wilcoxon Signed-Rank Test (paired):
   H₀: median difference = 0
   For: GA vs ES, GA vs PABBO, ES vs PABBO

2. Friedman Test (multiple):
   H₀: all algorithms have same distribution
   For: GA, ES, PABBO_Full

3. Effect Size:
   Cohen's d = (mean₁ - mean₂) / pooled_std
```

---

## Контрольные точки для воспроизводимости

✅ **Seeds**:
   - Initial population: 42
   - Run 0: seed=42
   - Run 1: seed=43
   - ...
   - Run 9: seed=51

✅ **Initial Population** (20 values):
   [733, 811, 133, 355, 777, 115, 452, 940, 879, 345,
    576, 153, 950, 602, 162, 238, 422, 511, 660, 285]

✅ **LDA Parameters**:
   - max_iter: 60
   - batch_size: 2048
   - learning_method: "online"

✅ **Optimization Budget**:
   - iterations: 20 (exact, no early stopping)
   - T_bounds: [2, 1000]

✅ **Algorithm Hyperparameters**:
   - GA: cxpb=0.9, mutpb=0.2, elite=5
   - ES: μ=5, λ=10
   - PABBO: exploration_rate=0.3

---

## Команды для запуска

```bash
# Полный pipeline (все этапы автоматически)
python lda.py

# Только обучение PABBO (этап 1)
cd pabbo_method
python train.py --config-name=train_rastrigin1d_test experiment.wandb=false

# Только evaluation PABBO (этап 2)
cd pabbo_method
python evaluate_continuous.py --config-name=evaluate experiment.expid={EXPID}

# Только LDA оптимизация (этап 3)
cd lda_hyperopt
python run_no_early_stop.py \
  --data ../data/X_20news_val_bow.npz \
  --algorithms GA ES PABBO_Full \
  --iterations 20 \
  --seed 42 \
  --pabbo-model {MODEL_PATH} \
  --init lda_init_population.json \
  --outdir results/

# Проверка результатов
ls -R lda_pipeline_results/run_*/
```
