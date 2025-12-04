#!/usr/bin/env python3
"""
Generate a table of best perplexity (mean +/- std) for each dataset/algorithm.

Data source: summary.json files inside
labs/run_20251104_135335/experiments/{dataset}/run_{i}/{algorithm}/summary.json
for i in [0, num_runs).

Outputs (in the same directory as this script by default):
  - best_perplexity_mean_std.csv
  - best_perplexity_mean_std.json
  - best_perplexity_mean_std.md (readable table)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_ALGORITHMS = ["GA", "ES", "PABBO", "SABBO"]


def read_best_perplexities(
    root: Path, dataset: str, algorithm: str, num_runs: int
) -> List[float]:
    """Collect best_perplexity values for a dataset/algorithm across runs."""
    values: List[float] = []
    for run_idx in range(num_runs):
        summary_file = root / dataset / f"run_{run_idx}" / algorithm / "summary.json"
        if not summary_file.exists():
            continue
        try:
            with summary_file.open() as f:
                data = json.load(f)
            if "best_perplexity" in data:
                values.append(float(data["best_perplexity"]))
        except Exception as exc:  # noqa: BLE001 - diagnostic output
            print(f"[warn] failed to read {summary_file}: {exc}")
    return values


def compute_stats(values: List[float]) -> Optional[Dict[str, float]]:
    """Return mean/std/num_runs or None when no values."""
    if not values:
        return None
    mean_val = statistics.fmean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean_val, "std": std_val, "num_runs": len(values)}


def format_cell(stats: Optional[Dict[str, float]], precision: int = 2) -> str:
    """Format stats for CSV/MD table."""
    if stats is None:
        return "N/A"
    mean_val = stats["mean"]
    std_val = stats["std"]
    return f"{mean_val:.{precision}f} +/- {std_val:.{precision}f}"


def write_csv(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in header) + "\n")


def write_md(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join("---" for _ in header) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(col, "")) for col in header) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build best perplexity tables.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "run_20251104_135335" / "experiments",
        help="Path to experiments directory (default: labs/run_20251104_135335/experiments)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to store generated tables (default: tables folder)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        help="Algorithms to include (default: GA ES PABBO SABBO)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs to scan (default: 10 -> run_0..run_9)",
    )
    args = parser.parse_args()

    experiments_root: Path = args.experiments_root
    out_dir: Path = args.out_dir
    algorithms: List[str] = args.algorithms
    num_runs: int = args.runs

    if not experiments_root.exists():
        raise SystemExit(f"Experiments root not found: {experiments_root}")

    datasets = sorted([p.name for p in experiments_root.iterdir() if p.is_dir()])
    if not datasets:
        raise SystemExit(f"No datasets found in {experiments_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    rows_for_table: List[Dict[str, str]] = []
    stats_full: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}

    for dataset in datasets:
        row: Dict[str, str] = {"dataset": dataset}
        stats_full[dataset] = {}

        for algorithm in algorithms:
            vals = read_best_perplexities(experiments_root, dataset, algorithm, num_runs)
            stats = compute_stats(vals)
            stats_full[dataset][algorithm] = stats
            row[algorithm] = format_cell(stats)

        rows_for_table.append(row)

    header = ["dataset"] + algorithms

    csv_path = out_dir / "best_perplexity_mean_std.csv"
    json_path = out_dir / "best_perplexity_mean_std.json"
    md_path = out_dir / "best_perplexity_mean_std.md"

    write_csv(csv_path, header, rows_for_table)
    write_md(md_path, header, rows_for_table)
    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(stats_full, f_json, indent=2)

    print(f"[ok] CSV written:   {csv_path}")
    print(f"[ok] JSON written:  {json_path}")
    print(f"[ok] Markdown written: {md_path}")


if __name__ == "__main__":
    main()
