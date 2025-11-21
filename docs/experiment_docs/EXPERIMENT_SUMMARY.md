# Краткая сводка экспериментов LDA Hyperparameter Optimization

## Цель исследования
Сравнить эффективность трех методов оптимизации для подбора числа топиков (T) в Latent Dirichlet Allocation (LDA).

---

## Методология

### Сравниваемые алгоритмы
1. **GA** (Genetic Algorithm) - классический генетический алгоритм
2. **ES** (Evolution Strategy) - (μ,λ)-эволюционная стратегия
3. **PABBO_Full** - Bayesian Optimization с обученной Transformer моделью

### Задача оптимизации
- **Цель**: минимизация perplexity на валидационном корпусе
- **Пространство поиска**: T ∈ [2, 1000] (целые числа)
- **Связанные параметры**: α = η = 1/T (symmetric Dirichlet prior)

### Экспериментальный протокол
- **Инициализация**: все алгоритмы стартуют с одинаковой популяции (20 точек)
- **Бюджет оптимизации**: ровно 20 итераций (без early stopping)
- **Повторения**: 10 независимых запусков с разными seeds
- **Датасеты**: множественные текстовые корпуса в BoW формате

---

## Pipeline из 4 этапов

```
ЭТАП 1: Обучение PABBO модели
    └─> Transformer (d_model=32, 3 layers) на синтетических функциях
    └─> Выход: ckpt.tar

ЭТАП 2: Валидация модели
    └─> Evaluation на GP1D
    └─> Выход: метрики качества

ЭТАП 3: LDA Optimization
    └─> Запуск GA, ES, PABBO_Full на каждом датасете
    └─> 10 повторений × N датасетов × 3 алгоритма
    └─> Выход: best_T, best_perplexity, траектории

ЭТАП 4: Агрегация и визуализация
    └─> Статистический анализ (mean, std, min, max)
    └─> Графики: bar charts, box plots, convergence curves
    └─> Выход: tables + figures для статьи
```

---

## Ключевые метрики

### Качество оптимизации
- **Mean perplexity** - основная метрика (ниже = лучше)
- **Std perplexity** - стабильность алгоритма
- **Min perplexity** - лучший результат из 10 runs

### Вычислительная эффективность
- **Total time** - время до завершения 20 итераций
- **Avg step time** - среднее время одной итерации

### Сходимость
- **Convergence speed** - скорость достижения оптимума
- **History trajectory** - поведение по итерациям

---

## Обеспечение честности сравнения

✅ **Одинаковая начальная популяция** для всех алгоритмов
✅ **Фиксированный бюджет** (20 итераций без early stopping)
✅ **Одинаковые LDA параметры** (max_iter=60, batch_size=2048)
✅ **Фиксированные seeds** (42 для population, 42+run_id для runs)
✅ **Одинаковые датасеты** для всех методов

---

## Воспроизводимость

Все параметры зафиксированы и сохранены:
- Initial population в `lda_init_population.json`
- Seeds: 42 (base), 42-51 (runs)
- Конфигурации алгоритмов в коде
- Полное логирование в `pipeline_main.log`
- История оптимизации в `history.csv`

---

## Статистический анализ

### Для каждой пары (датасет, алгоритм):
- Mean ± Std perplexity (по 10 runs)
- Min/Max perplexity
- Mean ± Std time

### Статистические тесты (рекомендуется):
- **Wilcoxon signed-rank test** - парное сравнение алгоритмов
- **Friedman test** - сравнение всех трех методов
- **Effect size** - величина различий (Cohen's d)

---

## Визуализация для статьи

### Основные графики (в PNG + SVG):
1. **Perplexity comparison** - bar charts с error bars
2. **Time comparison** - grouped bar chart по датасетам
3. **Box plots** - распределение perplexity по runs
4. **Convergence curves** - траектории оптимизации

### Рекомендуемая структура Results секции:

```
Section 4: Experimental Results

4.1 Setup and Methodology
    - Datasets description
    - Algorithms configuration
    - Evaluation protocol

4.2 Optimization Quality
    - Table 1: Mean perplexity comparison
    - Figure 1: Perplexity bar charts
    - Figure 2: Box plots (stability)

4.3 Computational Efficiency
    - Table 2: Time comparison
    - Figure 3: Time vs Algorithms

4.4 Convergence Analysis
    - Figure 4: Convergence curves
    - Discussion of convergence speed

4.5 Statistical Significance
    - Table 3: p-values (Wilcoxon tests)
    - Effect sizes

4.6 Discussion
    - Best performing algorithm
    - Trade-offs (quality vs speed)
    - Recommendations
```

---

## Краткие результаты (template)

**Research Questions:**
1. **RQ1**: Какой алгоритм находит лучшие гиперпараметры (lowest perplexity)?
2. **RQ2**: Какой алгоритм наиболее стабилен (lowest std)?
3. **RQ3**: Какой алгоритм наиболее эффективен по времени?
4. **RQ4**: Есть ли значимые различия между алгоритмами?

**Expected format:**
```
Algorithm | Mean Perplexity | Std  | Mean Time | Rank
----------|-----------------|------|-----------|------
GA        | XXXX ± YY      | ZZ   | AAA s     | ?
ES        | XXXX ± YY      | ZZ   | BBB s     | ?
PABBO     | XXXX ± YY      | ZZ   | CCC s     | ?
```

---

## Вычислительные требования

- **Общее время**: ~40-80 часов (sequential), ~4-8 часов (parallel по 10 runs)
- **CPU**: все вычисления на CPU
- **Memory**: зависит от размера датасетов (~4-8GB recommended)
- **Disk space**: ~1-5 GB на полный run (logs + plots + models)

---

## Команда для запуска

```bash
python lda.py
```

Автоматически выполнит все 4 этапа:
1. Train PABBO model
2. Evaluate model
3. Run LDA optimization (all datasets × 10 runs × 3 algorithms)
4. Aggregate results and generate plots

Результаты в: `lda_pipeline_results/run_{timestamp}/`

---

## Checklist для статьи

Experimental Section:
- [ ] Описать задачу оптимизации LDA
- [ ] Обосновать выбор метрик (perplexity)
- [ ] Описать датасеты (размер, vocabulary, preprocessing)
- [ ] Детализировать параметры алгоритмов
- [ ] Объяснить initial population
- [ ] Указать seeds и воспроизводимость
- [ ] Обосновать выбор бюджета (20 итераций)
- [ ] Обосновать число повторений (10 runs)

Results Section:
- [ ] Таблица с mean ± std для всех датасетов
- [ ] Графики perplexity comparison
- [ ] Графики convergence curves
- [ ] Статистические тесты (p-values)
- [ ] Анализ стабильности (box plots)
- [ ] Анализ времени выполнения

Discussion:
- [ ] Интерпретация результатов
- [ ] Объяснение превосходства одного метода
- [ ] Trade-offs между качеством и скоростью
- [ ] Ограничения исследования
- [ ] Практические рекомендации
- [ ] Future work

---

## Ссылки на детали

- Полная спецификация: `EXPERIMENT_SPECIFICATION.md`
- Код pipeline: `lda.py`
- Код оптимизации: `lda_hyperopt/run_no_early_stop.py`
- Initial population: `lda_hyperopt/lda_init_population.json`
