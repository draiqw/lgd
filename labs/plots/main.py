#!/usr/bin/env python3
"""
Полный pipeline агрегации и визуализации экспериментальных данных.

Структура:
1. Агрегация данных по runs
2. Вычисление mean/std по iterations
3. Агрегация данных по времени с интерполяцией
4. Создание 12 графиков (4 датасета × 3 типа)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import shutil
import json

print("Загрузка matplotlib...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("✓ Matplotlib загружен", flush=True)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
algorithms = ["ES", "GA", "PABBO", "SABBO"]
datasets = ["20news", "agnews", "val_out", "yelp"]
num_runs = 10
time_step = 10  # секунд для временной сетки

# Пути
script_dir = Path(__file__).parent
base_path = script_dir.parent / "run_20251104_135335" / "experiments"
summary_tables_dir = script_dir / "summary_tables"

# Папки для результатов
first_step_dir = script_dir / "first_step"
second_step_dir = script_dir / "second_step"
time_agg_dir = script_dir / "time_aggregated_data"
plots_dir = script_dir / "plots"
png_dir = plots_dir / "png"
svg_dir = plots_dir / "svg"
total_steps = 5

# Цвета для алгоритмов
colors = {
    "ES": "#1f77b4",
    "GA": "#ff7f0e",
    "PABBO": "#2ca02c",
    "SABBO": "#d62728"
}

print("="*80)
print("ЗАПУСК ПОЛНОГО PIPELINE АГРЕГАЦИИ И ВИЗУАЛИЗАЦИИ")
print("="*80)
print(f"Алгоритмы: {', '.join(algorithms)}")
print(f"Датасеты: {', '.join(datasets)}")
print("="*80)

# ============================================================================
# ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ
# ============================================================================
print(f"\n[1/{total_steps}] Очистка старых результатов...")
for folder in [first_step_dir, second_step_dir, time_agg_dir, plots_dir, summary_tables_dir]:
    if folder.exists():
        shutil.rmtree(folder)
        print(f"  ✓ Удалена папка: {folder.name}")

# Создаем папки
first_step_dir.mkdir(exist_ok=True)
second_step_dir.mkdir(exist_ok=True)
time_agg_dir.mkdir(exist_ok=True)
png_dir.mkdir(parents=True, exist_ok=True)
svg_dir.mkdir(parents=True, exist_ok=True)
summary_tables_dir.mkdir(exist_ok=True)

# ============================================================================
# ШАГ 2: ТАБЛИЦА ПО ЛУЧШЕЙ ПЕРПЛЕКСИИ
# ============================================================================
print(f"\n[2/{total_steps}] Формирование таблицы с лучшей перплексией (mean +/- std)...")

perplexity_rows = []
perplexity_stats = {}

for dataset in datasets:
    row = {"dataset": dataset}
    perplexity_stats[dataset] = {}

    for algorithm in algorithms:
        values = []

        for run_idx in range(num_runs):
            summary_file = base_path / dataset / f"run_{run_idx}" / algorithm / "summary.json"

            if not summary_file.exists():
                continue

            try:
                with open(summary_file) as f:
                    summary_data = json.load(f)
                if "best_perplexity" in summary_data:
                    values.append(float(summary_data["best_perplexity"]))
            except Exception as e:
                print(f"  Ошибка при чтении {summary_file}: {e}")
                continue

        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            perplexity_stats[dataset][algorithm] = {
                "mean": mean_val,
                "std": std_val,
                "num_runs": len(values)
            }
            row[algorithm] = f"{mean_val:.2f} +/- {std_val:.2f}"
        else:
            perplexity_stats[dataset][algorithm] = None
            row[algorithm] = "N/A"

    perplexity_rows.append(row)

perplexity_df = pd.DataFrame(perplexity_rows)
perplexity_csv = summary_tables_dir / "best_perplexity_mean_std.csv"
perplexity_json = summary_tables_dir / "best_perplexity_mean_std.json"

perplexity_df.to_csv(perplexity_csv, index=False)
with open(perplexity_json, "w") as f:
    json.dump(perplexity_stats, f, indent=2)

print(f"  ✓ Таблица сохранена: {perplexity_csv.name}")
print(f"  ✓ Подробная статистика: {perplexity_json.name}")

# ============================================================================
# ШАГ 3: АГРЕГАЦИЯ ДАННЫХ ПО RUNS
# ============================================================================
print(f"\n[3/{total_steps}] Агрегация данных по runs...")
print("-"*80)

all_tables = {}

for dataset in datasets:
    for algorithm in algorithms:
        aggregated_data = None

        for run_idx in range(num_runs):
            history_file = base_path / dataset / f"run_{run_idx}" / algorithm / "history.csv"

            if not history_file.exists():
                continue

            try:
                df = pd.read_csv(history_file)
                df_selected = df[['iter', 'best_perplexity', 'T_best', 'step_time']].copy()

                # Нормализуем iteration: сдвигаем так, чтобы минимум был 0
                min_iter = df_selected['iter'].min()
                df_selected['iter'] = df_selected['iter'] - min_iter

                # Переименовываем столбцы
                df_selected.columns = [
                    'iteration',
                    f'run_{run_idx}_best_perplexity',
                    f'run_{run_idx}_T_best',
                    f'run_{run_idx}_step_time'
                ]

                if aggregated_data is None:
                    aggregated_data = df_selected
                else:
                    aggregated_data = pd.merge(
                        aggregated_data,
                        df_selected,
                        on='iteration',
                        how='outer'
                    )
            except Exception as e:
                print(f"  Ошибка при обработке {history_file}: {e}")
                continue

        if aggregated_data is not None:
            aggregated_data = aggregated_data.sort_values('iteration').reset_index(drop=True)
            table_name = f"{dataset}_{algorithm}"
            all_tables[table_name] = aggregated_data

            # Сохраняем
            output_file = first_step_dir / f"{table_name}.csv"
            aggregated_data.to_csv(output_file, index=False)

print(f"✓ Создано таблиц: {len(all_tables)}")

# ============================================================================
# ШАГ 4: ВЫЧИСЛЕНИЕ MEAN/STD ПО ITERATIONS
# ============================================================================
print(f"\n[4/{total_steps}] Вычисление mean/std по iterations...")
print("-"*80)

for csv_file in first_step_dir.glob("*.csv"):
    df = pd.read_csv(csv_file)
    result_df = pd.DataFrame()
    result_df['iteration'] = df['iteration']

    # Находим все run'ы
    run_columns = [col for col in df.columns if col.startswith('run_')]
    run_indices = set()
    for col in run_columns:
        parts = col.split('_')
        if len(parts) >= 2 and parts[0] == 'run' and parts[1].isdigit():
            run_indices.add(int(parts[1]))

    num_runs_found = len(run_indices)

    # Для каждой метрики вычисляем mean и std
    for metric in ['best_perplexity', 'T_best', 'step_time']:
        metric_columns = [f'run_{i}_{metric}' for i in sorted(run_indices)]
        metric_data = df[metric_columns]

        result_df[f'{metric}_mean'] = metric_data.mean(axis=1)
        result_df[f'{metric}_std'] = metric_data.std(axis=1, ddof=1)

    # Вычисляем cumulative time
    result_df['cum_time_mean'] = result_df['step_time_mean'].cumsum()
    result_df['cum_time_std'] = result_df['step_time_std']

    # Сохраняем
    output_file = second_step_dir / csv_file.name
    result_df.to_csv(output_file, index=False)

print(f"✓ Обработано файлов: {len(list(second_step_dir.glob('*.csv')))}")

# ============================================================================
# ШАГ 5: АГРЕГАЦИЯ ДАННЫХ ПО ВРЕМЕНИ С ИНТЕРПОЛЯЦИЕЙ
# ============================================================================
print(f"\n[5/{total_steps}] Агрегация данных по времени...")
print("-"*80)

for dataset in datasets:
    for algorithm in algorithms:
        # Собираем данные для всех run'ов
        run_functions = []
        all_times = []

        for run_idx in range(num_runs):
            history_file = base_path / dataset / f"run_{run_idx}" / algorithm / "history.csv"

            if not history_file.exists():
                continue

            df = pd.read_csv(history_file)
            cum_time = df['step_time'].cumsum().values
            perplexity = df['best_perplexity'].values

            all_times.extend(cum_time)

            # Создаем кусочно-линейную функцию
            interp_func = interp1d(
                cum_time,
                perplexity,
                kind='linear',
                bounds_error=False,
                fill_value=(perplexity[0], perplexity[-1])
            )

            run_functions.append({
                'run_idx': run_idx,
                'function': interp_func,
                'min_time': cum_time[0],
                'max_time': cum_time[-1]
            })

        if not run_functions:
            continue

        # Находим общий диапазон времени
        min_time_global = min(rf['min_time'] for rf in run_functions)
        max_time_global = max(rf['max_time'] for rf in run_functions)

        # Создаем временную сетку
        time_grid = np.arange(min_time_global, max_time_global + time_step, time_step)

        # Для каждого момента времени вычисляем значения
        results = []
        for t in time_grid:
            values = [rf['function'](t) for rf in run_functions]
            mean_value = np.mean(values)
            std_value = np.std(values, ddof=1) if len(values) > 1 else 0.0

            results.append({
                'time': t,
                'best_perplexity_mean': mean_value,
                'best_perplexity_std': std_value,
                'num_runs': len(values)
            })

        # Сохраняем
        result_df = pd.DataFrame(results)
        output_file = time_agg_dir / f"{dataset}_{algorithm}_time_aggregated.csv"
        result_df.to_csv(output_file, index=False)

print(f"✓ Создано файлов: {len(list(time_agg_dir.glob('*.csv')))}")

# ============================================================================
# СОЗДАНИЕ ГРАФИКОВ
# ============================================================================
print("\n" + "="*80)
print("СОЗДАНИЕ ГРАФИКОВ (12 штук)")
print("="*80)

total_graphs = 0

for dataset in datasets:
    print(f"\nДатасет: {dataset}")

    # Читаем данные по iterations
    data_iter = {}
    for algorithm in algorithms:
        csv_file = second_step_dir / f"{dataset}_{algorithm}.csv"
        if csv_file.exists():
            data_iter[algorithm] = pd.read_csv(csv_file)

    # Читаем данные по времени
    data_time = {}
    for algorithm in algorithms:
        csv_file = time_agg_dir / f"{dataset}_{algorithm}_time_aggregated.csv"
        if csv_file.exists():
            data_time[algorithm] = pd.read_csv(csv_file)

    # ========================================================================
    # 1. BEST PERPLEXITY VS ITERATION
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for algorithm in algorithms:
        if algorithm not in data_iter:
            continue

        df = data_iter[algorithm]
        mean = df['best_perplexity_mean']
        std = df['best_perplexity_std']
        x_data = df['iteration']
        color = colors[algorithm]

        ax.plot(x_data, mean, color=color, linewidth=2, label=algorithm)
        ax.fill_between(x_data, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Perplexity', fontsize=12)
    ax.set_title(f'{dataset} - Best Perplexity vs Iteration', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    if data_iter:
        max_iter = max([data_iter[alg]['iteration'].max() for alg in data_iter.keys()])
        ax.set_xticks(range(0, int(max_iter) + 1, 1))
        ax.set_xticks(range(0, int(max_iter) + 1, 2), minor=True)
        ax.grid(True, which='minor', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.grid(True, which='major', alpha=0.15, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    filename = f"{dataset}_best_perplexity_vs_iteration"
    plt.savefig(png_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(svg_dir / f"{filename}.svg", format='svg', bbox_inches='tight')
    plt.close()
    total_graphs += 1
    print(f"  ✓ {filename}")

    # ========================================================================
    # 2. T_BEST VS ITERATION
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for algorithm in algorithms:
        if algorithm not in data_iter:
            continue

        df = data_iter[algorithm]
        mean = df['T_best_mean']
        std = df['T_best_std']
        x_data = df['iteration']
        color = colors[algorithm]

        ax.plot(x_data, mean, color=color, linewidth=2, label=algorithm)
        ax.fill_between(x_data, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('T Best', fontsize=12)
    ax.set_title(f'{dataset} - T Best vs Iteration', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    if data_iter:
        max_iter = max([data_iter[alg]['iteration'].max() for alg in data_iter.keys()])
        ax.set_xticks(range(0, int(max_iter) + 1, 1))
        ax.set_xticks(range(0, int(max_iter) + 1, 2), minor=True)
        ax.grid(True, which='minor', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.grid(True, which='major', alpha=0.15, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    filename = f"{dataset}_T_best_vs_iteration"
    plt.savefig(png_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(svg_dir / f"{filename}.svg", format='svg', bbox_inches='tight')
    plt.close()
    total_graphs += 1
    print(f"  ✓ {filename}")

    # ========================================================================
    # 3. BEST PERPLEXITY VS TIME
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    for algorithm in algorithms:
        if algorithm not in data_time:
            continue

        df = data_time[algorithm]
        time = df['time']
        mean = df['best_perplexity_mean']
        std = df['best_perplexity_std']
        color = colors[algorithm]

        ax.plot(time, mean, color=color, linewidth=2.5, label=algorithm)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Time (seconds)', fontsize=13)
    ax.set_ylabel('Best Perplexity', fontsize=13)
    ax.set_title(f'{dataset} - Best Perplexity vs Time',
               fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    plt.tight_layout()
    filename = f"{dataset}_best_perplexity_vs_time"
    plt.savefig(png_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(svg_dir / f"{filename}.svg", format='svg', bbox_inches='tight')
    plt.close()
    total_graphs += 1
    print(f"  ✓ {filename}")

# ============================================================================
# ФИНАЛ
# ============================================================================
print("\n" + "="*80)
print(f"✓ ВСЕГО СОЗДАНО ГРАФИКОВ: {total_graphs}")
print(f"  - Best Perplexity vs Iteration: 4")
print(f"  - T Best vs Iteration: 4")
print(f"  - Best Perplexity vs Time: 4")
print(f"\nГрафики сохранены в:")
print(f"  - PNG: {png_dir} ({total_graphs} файлов)")
print(f"  - SVG: {svg_dir} ({total_graphs} файлов)")
print("="*80)
