from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rag.config import load_benchmark_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate results table and plots")
    parser.add_argument("--config", default=None, help="Path to benchmark config YAML")
    parser.add_argument("--results-dir", default=None, help="Results directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    return parser.parse_args()


def load_results(results_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.json"
    preds_path = results_dir / "predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found in {results_dir}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    predictions = None
    if preds_path.exists():
        predictions = pd.read_csv(preds_path)

    return metrics, predictions


def print_results_table(metrics: dict) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    rows = []
    for method, m in sorted(metrics.items()):
        row = {"Method": method}
        for metric_name in ("mae", "rmse", "qwk", "within_0.5", "within_1.0"):
            if metric_name in m:
                row[metric_name.upper()] = f"{m[metric_name]['value']:.4f}"
        if "elapsed_seconds" in m:
            row["Time (s)"] = f"{m['elapsed_seconds']:.1f}"
        rows.append(row)

    df_table = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df_table.to_string(index=False))
    return df_table


def plot_metrics_comparison(
    metrics: dict,
    output_dir: str | Path,
    filename: str = "metrics_comparison.png",
) -> None:
    methods = sorted(metrics.keys())
    metric_names = ["mae", "rmse", "qwk", "within_0.5", "within_1.0"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for ax, metric_name in zip(axes, metric_names):
        values = []
        for method in methods:
            m = metrics[method]
            if metric_name in m:
                values.append(m[metric_name]["value"])
            else:
                values.append(0)

        bars = ax.bar(methods, values, color=colors)
        ax.set_title(metric_name.upper())
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    path = Path(output_dir) / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_predicted_vs_true(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "pred_vs_true.png",
) -> None:
    if "true" not in predictions.columns:
        return

    true_vals = predictions["true"].values
    method_cols = [c for c in predictions.columns if c.startswith("pred_")]

    n_methods = len(method_cols)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, col in zip(axes, method_cols):
        pred_vals = predictions[col].values
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=10)
        lims = [min(np.min(true_vals), np.min(pred_vals)) - 0.5,
                max(np.max(true_vals), np.max(pred_vals)) + 0.5]
        ax.plot(lims, lims, "r--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True Score")
        ax.set_ylabel("Predicted Score")
        method_name = col.replace("pred_", "")
        ax.set_title(method_name)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_error_histogram(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "error_histogram.png",
) -> None:
    if "true" not in predictions.columns:
        return

    true_vals = predictions["true"].values
    method_cols = [c for c in predictions.columns if c.startswith("pred_")]

    fig, axes = plt.subplots(1, len(method_cols), figsize=(5 * len(method_cols), 4))
    if len(method_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, method_cols):
        errors = np.abs(true_vals - predictions[col].values)
        ax.hist(errors, bins=15, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Absolute Error")
        ax.set_ylabel("Frequency")
        method_name = col.replace("pred_", "")
        ax.set_title(f"{method_name}\nMean Error: {np.mean(errors):.3f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)

    results_dir = Path(args.results_dir or resolve_path(config["evaluation"]["output_dir"]))
    output_dir = Path(args.output_dir or results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, predictions = load_results(results_dir)
    print_results_table(metrics)

    plot_metrics_comparison(metrics, output_dir)

    if predictions is not None:
        plot_predicted_vs_true(predictions, output_dir)
        plot_error_histogram(predictions, output_dir)

    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
