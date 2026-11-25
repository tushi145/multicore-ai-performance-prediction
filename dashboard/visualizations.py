"""Matplotlib figures built from real benchmark / ML data. Every function returns a Figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config

# Validated categorical palette (fixed order, colour follows the entity, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential: input size
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

ALGO_COLORS = {"merge_sort": SERIES[0], "quick_sort": SERIES[1], "timsort": SERIES[2],
               "parallel_merge_sort": SERIES[3]}
WORKER_COLORS = {1: SERIES[3], 2: SERIES[4], 4: SERIES[5], 8: SERIES[6], 16: SERIES[7]}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 6, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "axes.axisbelow": True,
})


def _series_key(algo: str, workers: int) -> str:
    return f"Parallel merge sort ({int(workers)}w)" if algo == "parallel_merge_sort" else config.ALGORITHM_LABELS[algo]


def _series_color(algo: str, workers: int) -> str:
    if algo == "parallel_merge_sort":
        return WORKER_COLORS.get(int(workers), SERIES[7])
    return ALGO_COLORS[algo]


def _size_colors(sizes: list[int]) -> dict[int, str]:
    idx = np.linspace(0, len(BLUE_RAMP) - 1, num=max(len(sizes), 1)).round().astype(int)
    return {s: BLUE_RAMP[i] for s, i in zip(sorted(sizes), idx)}


def _empty(msg: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2)
    return fig


def plot_time_vs_size(summary: pd.DataFrame) -> Figure:
    if summary.empty:
        return _empty("No benchmark data")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ((algo, w), g) in enumerate(summary.groupby(["algorithm", "workers"], sort=False)):
        g = g.sort_values("input_size")
        ax.plot(g["input_size"], g["median_time_s"], marker=MARKERS[i % len(MARKERS)],
                ls="--" if algo == config.OPTIMIZED_BASELINE else "-",  # reference baseline, not a contender
                color=_series_color(algo, w), label=_series_key(algo, w))
    ax.set(xscale="log", yscale="log", xlabel="Input size (elements, log scale)",
           ylabel="Median execution time (s, log scale)", title="Execution time vs input size")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def _par_lines(summary: pd.DataFrame, col: str, title: str, ylabel: str, ideal: str | None) -> Figure:
    par = summary[summary["algorithm"] == "parallel_merge_sort"]
    if par.empty:
        return _empty("No parallel benchmark data")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = _size_colors(par["input_size"].unique().tolist())
    workers = sorted(par["workers"].unique())
    if ideal == "linear":
        ax.plot(workers, workers, ls="--", lw=1.2, color=INK2, label="Ideal (linear)")
    elif ideal == "one":
        ax.axhline(1.0, ls="--", lw=1.2, color=INK2, label="Ideal (100%)")
    for i, (n, g) in enumerate(par.groupby("input_size")):
        g = g.sort_values("workers")
        ax.plot(g["workers"], g[col], marker=MARKERS[i % len(MARKERS)], color=colors[n], label=f"n = {int(n):,}")
    ax.set(xlabel="Worker processes", ylabel=ylabel, title=title)
    ax.set_xticks(workers)
    ax.legend(fontsize=8)
    return fig


def plot_speedup_vs_workers(summary: pd.DataFrame) -> Figure:
    return _par_lines(summary, "speedup", "Speedup vs worker count (baseline: serial merge sort)",
                      "Speedup (x)", "linear")


def plot_efficiency_vs_workers(summary: pd.DataFrame) -> Figure:
    fig = _par_lines(summary, "efficiency", "Parallel efficiency vs worker count", "Efficiency (speedup / workers)", "one")
    if fig.axes and fig.axes[0].lines:
        fig.axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    return fig


def plot_serial_vs_parallel(summary: pd.DataFrame, input_size: int | None = None) -> Figure:
    if summary.empty:
        return _empty("No benchmark data")
    n = input_size or int(summary["input_size"].max())
    sub = summary[summary["input_size"] == n].copy()
    if sub.empty:
        return _empty(f"No data for n = {n:,}")
    sub["label"] = [_series_key(a, w) for a, w in zip(sub["algorithm"], sub["workers"])]
    sub = sub.sort_values("median_time_s")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(sub) + 1.5))
    y = np.arange(len(sub))
    ax.barh(y, sub["median_time_s"], xerr=sub["std_time_s"].fillna(0), height=0.6,
            color=[_series_color(a, w) for a, w in zip(sub["algorithm"], sub["workers"])],
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for yi, t in zip(y, sub["median_time_s"]):
        ax.text(t, yi, f"  {t * 1000:,.1f} ms", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, sub["label"])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Median execution time (s), error bar = std", title=f"Serial vs parallel at n = {n:,}")
    ax.margins(x=0.18)
    return fig


def plot_cpu_utilization(summary: pd.DataFrame, input_size: int | None = None) -> Figure:
    """Effective logical CPUs used (process-tree CPU time / wall time) and aggregate system CPU %."""
    if summary.empty:
        return _empty("No benchmark data")
    n = input_size or int(summary["input_size"].max())
    sub = summary[summary["input_size"] == n].copy()
    if sub.empty:
        return _empty(f"No data for n = {n:,}")
    sub["label"] = [_series_key(a, w) for a, w in zip(sub["algorithm"], sub["workers"])]
    colors = [_series_color(a, w) for a, w in zip(sub["algorithm"], sub["workers"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(sub) + 1.8), sharey=True)
    y = np.arange(len(sub))
    for ax, col, title in ((axes[0], "mean_effective_cores_used", "Effective logical CPUs used\n(benchmark process tree)"),
                           (axes[1], "mean_system_cpu_percent", "Aggregate system CPU utilisation (%)\n(all logical CPUs, incl. background)")):
        ax.barh(y, sub[col], height=0.6, color=colors)
        for yi, v in zip(y, sub[col]):
            ax.text(v, yi, f" {v:.2f}" if "cores" in col else f" {v:.1f}%", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.2)
    axes[0].set_yticks(y, sub["label"])
    fig.suptitle(f"CPU utilisation at n = {n:,}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_overhead_breakdown(summary: pd.DataFrame, input_size: int | None = None) -> Figure:
    par = summary[summary["algorithm"] == "parallel_merge_sort"]
    if par.empty:
        return _empty("No parallel benchmark data")
    n = input_size or int(par["input_size"].max())
    sub = par[par["input_size"] == n].sort_values("workers")
    if sub.empty:
        return _empty(f"No parallel data for n = {n:,}")
    compute = (sub["mean_parallel_phase_time_s"] - sub["mean_communication_overhead_s"]).clip(lower=0)
    parts = [("Split", sub["mean_split_time_s"], SERIES[0]), ("Worker sorting (slowest)", compute, SERIES[1]),
             ("IPC / pickling / scheduling", sub["mean_communication_overhead_s"], SERIES[2]),
             ("k-way merge (parent)", sub["mean_merge_time_s"], SERIES[3])]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    for label, vals, color in parts:
        v = vals.to_numpy()
        ax.bar(x, v, bottom=bottom, width=0.6, color=color, label=label, edgecolor=SURFACE, linewidth=1.5)
        bottom += v
    for xi, top, st in zip(x, bottom, sub["mean_startup_time_s"]):
        ax.text(xi, top, f"pool startup\n{st * 1000:.0f} ms (excluded)", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"{int(w)} worker{'s' if w > 1 else ''}" for w in sub["workers"]])
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Mean time (s)", title=f"Parallel merge sort phase breakdown at n = {n:,}")
    ax.margins(y=0.2)
    ax.legend(fontsize=8)
    return fig


def plot_actual_vs_predicted(pred: pd.DataFrame, model: str, label: str, r2: float | None = None) -> Figure:
    col = f"pred_{model}"
    if pred.empty or col not in pred.columns:
        return _empty("No predictions")
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (algo, g) in enumerate(pred.groupby("algorithm")):
        ax.scatter(g["actual_s"], g[col], s=36, marker=MARKERS[i], color=ALGO_COLORS.get(algo, SERIES[7]),
                   edgecolor=SURFACE, linewidth=1, label=config.ALGORITHM_LABELS.get(algo, algo))
    lo = min(pred["actual_s"].min(), pred[col].min()) * 0.8
    hi = max(pred["actual_s"].max(), pred[col].max()) * 1.25
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=INK2, label="Perfect prediction")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual time (s)",
           ylabel="Predicted time (s)",
           title=f"{label}: actual vs predicted (test set)" + (f"\nR² = {r2:.4f}" if r2 is not None else ""))
    ax.legend(fontsize=8)
    return fig


def plot_feature_importance(imp: pd.DataFrame, label: str) -> Figure:
    if imp.empty:
        return _empty("No feature importance available")
    imp = imp.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.5))
    ax.barh(imp["feature"], imp["importance"], height=0.6, color=SERIES[0])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=imp["kind"].iloc[0], title=f"{label}: feature importance")
    return fig


def plot_model_comparison(metrics: pd.DataFrame, metric: str = "RMSE") -> Figure:
    sub = metrics[metrics["split"].isin(["cv", "test"])]
    if sub.empty:
        return _empty("No model metrics")
    models = list(dict.fromkeys(sub["model"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    for j, (split, color) in enumerate((("cv", SERIES[0]), ("test", SERIES[1]))):
        vals = [sub[(sub["model"] == m) & (sub["split"] == split)][metric].mean() for m in models]
        ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=color,
               label="Cross-validation (mean)" if split == "cv" else "Held-out test")
    ax.set_xticks(x, [m.replace("_", " ").title() for m in models])
    ax.grid(axis="x", visible=False)
    ax.set(ylabel=metric + (" (s)" if metric in ("MAE", "RMSE") else ""), title=f"Model comparison: {metric}")
    ax.legend(fontsize=8)
    return fig


def save_analysis_plots(summary: pd.DataFrame, out_dir: Path = config.PLOTS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "time_vs_input_size": plot_time_vs_size(summary),
        "speedup_vs_workers": plot_speedup_vs_workers(summary),
        "efficiency_vs_workers": plot_efficiency_vs_workers(summary),
        "serial_vs_parallel_largest": plot_serial_vs_parallel(summary),
        "cpu_utilization_largest": plot_cpu_utilization(summary),
        "parallel_overhead_breakdown": plot_overhead_breakdown(summary),
    }
    paths = []
    for name, fig in figs.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
