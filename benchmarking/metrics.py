"""Performance metrics (speedup, efficiency, statistics) and data-driven analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config

GROUP_KEYS = ["algorithm", "execution_mode", "input_size", "workers"]


def _parallel_efficiency(df: pd.DataFrame) -> pd.Series:
    """speedup / workers; defined only for parallel executions (NaN for serial algorithms)."""
    return (df["speedup"] / df["workers"]).where(df["execution_mode"] == config.PARALLEL)


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row ``speedup`` and ``efficiency``.

    speedup    = median T(serial merge sort) for the same run & input size / T(row)
                 (merge-sort family only; NaN for quick sort and the Timsort baseline)
    efficiency = speedup / workers   (parallel rows only; NaN for serial algorithms)
    If the run has no serial merge-sort baseline for that size, the dataset-wide median for the
    size is used; if none exists at all the values are NaN.
    """
    if df.empty:
        return df.assign(speedup=pd.Series(dtype=float), efficiency=pd.Series(dtype=float))
    out = df.copy()
    base = out[out["algorithm"] == config.BASELINE_ALGORITHM]
    run_base = base.groupby(["run_id", "input_size"])["exec_time_s"].median().rename("_rb")
    size_base = base.groupby("input_size")["exec_time_s"].median().rename("_sb")
    out = out.join(run_base, on=["run_id", "input_size"]).join(size_base, on="input_size")
    baseline = out["_rb"].fillna(out["_sb"])
    out["speedup"] = (baseline / out["exec_time_s"]).where(out["algorithm"].isin(config.SPEEDUP_FAMILY))
    out["efficiency"] = _parallel_efficiency(out)
    return out.drop(columns=["_rb", "_sb"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repetitions per configuration: mean/median/min/max/std of execution time and
    median-based speedup and efficiency."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(GROUP_KEYS)
    s = g["exec_time_s"].agg(["count", "mean", "median", "min", "max", "std"])
    s.columns = ["n_runs", "mean_time_s", "median_time_s", "min_time_s", "max_time_s", "std_time_s"]
    means = g[["startup_time_s", "split_time_s", "parallel_phase_time_s", "merge_time_s",
               "communication_overhead_s", "load_imbalance", "system_cpu_percent",
               "process_cpu_time_s", "effective_cores_used"]].mean()
    s = s.join(means.add_prefix("mean_")).reset_index()
    s["cv_percent"] = 100 * s["std_time_s"] / s["mean_time_s"]

    def median_of(algo: str) -> pd.Series:
        m = s[s["algorithm"] == algo].set_index("input_size")["median_time_s"]
        return s["input_size"].map(m)

    family = s["algorithm"].isin(config.SPEEDUP_FAMILY)
    s["speedup"] = (median_of(config.BASELINE_ALGORITHM) / s["median_time_s"]).where(family)
    s["efficiency"] = _parallel_efficiency(s)
    s["speedup_vs_timsort"] = median_of(config.OPTIMIZED_BASELINE) / s["median_time_s"]
    # end-to-end view: a one-off parallel sort must also pay the pool startup cost
    s["end_to_end_time_s"] = s["median_time_s"] + s["mean_startup_time_s"].fillna(0)
    s["end_to_end_speedup"] = (median_of(config.BASELINE_ALGORITHM) / s["end_to_end_time_s"]).where(family)
    return s.sort_values(["input_size", "execution_mode", "algorithm", "workers"]).reset_index(drop=True)


def _fmt_s(x: float) -> str:
    return f"{x * 1000:.1f} ms" if x < 1 else f"{x:.3f} s"


def compute_insights(summary: pd.DataFrame) -> dict[str, Any]:
    """Derive analysis findings strictly from measured summary data."""
    if summary.empty:
        return {"findings": ["No benchmark data available."], "details": {}}
    findings: list[str] = []
    details: dict[str, Any] = {}
    par = summary[summary["algorithm"] == "parallel_merge_sort"]
    sizes = sorted(summary["input_size"].unique())

    # 1. input-size scaling: empirical growth exponent from log-log fit
    exps = {}
    for algo, grp in summary.groupby(["algorithm", "workers"]):
        if grp["input_size"].nunique() >= 2:
            slope = np.polyfit(np.log(grp["input_size"]), np.log(grp["median_time_s"]), 1)[0]
            exps[f"{algo[0]} (w={algo[1]})"] = round(float(slope), 3)
    details["growth_exponents"] = exps
    if exps:
        findings.append("Input size: empirical time growth exponent (log-log slope; 1.0=linear, "
                        "n log n appears as ~1.05-1.15): "
                        + ", ".join(f"{k}: {v}" for k, v in exps.items()) + ".")

    if not par.empty:
        largest = par["input_size"].max()
        pl = par[par["input_size"] == largest].sort_values("workers")
        details["speedup_at_largest"] = dict(zip(pl["workers"].astype(int), pl["speedup"].round(3)))
        findings.append(f"Worker count at n={largest:,}: speedup vs serial merge sort "
                        + ", ".join(f"{int(w)}w={sp:.2f}x (eff {e:.0%})"
                                    for w, sp, e in zip(pl["workers"], pl["speedup"], pl["efficiency"])) + ".")

        best = par.loc[par.groupby("input_size")["median_time_s"].idxmin()]
        details["best_workers_per_size"] = dict(zip(best["input_size"].astype(int), best["workers"].astype(int)))
        findings.append("Best worker count per input size: "
                        + ", ".join(f"n={int(n):,}: {int(w)}" for n, w in zip(best["input_size"], best["workers"])) + ".")

        # saturation: doubling workers yields < 1.25x additional speedup
        sat = []
        for n, grp in par.groupby("input_size"):
            grp = grp.sort_values("workers")
            w, t = grp["workers"].to_numpy(), grp["median_time_s"].to_numpy()
            for i in range(1, len(w)):
                gain = t[i - 1] / t[i]
                if gain < 1.25:
                    sat.append((int(n), int(w[i - 1]), int(w[i]), round(float(gain), 3)))
                    break
        details["saturation_points"] = sat
        if sat:
            findings.append("Saturation (going to more workers gave <1.25x extra gain): "
                            + "; ".join(f"n={n:,}: {a}->{b} workers gain {g}x" for n, a, b, g in sat) + ".")

        startup = par.groupby("workers")["mean_startup_time_s"].mean()
        details["startup_by_workers"] = startup.round(4).to_dict()
        findings.append("Process-pool startup overhead (measured separately, excluded from exec time): "
                        + ", ".join(f"{int(w)}w={_fmt_s(v)}" for w, v in startup.items()) + ".")

        pl_total = pl["median_time_s"]
        comm = (pl["mean_communication_overhead_s"] / pl_total).mean()
        merge = (pl["mean_merge_time_s"] / pl_total).mean()
        details["comm_fraction_largest"] = round(float(comm), 4)
        details["merge_fraction_largest"] = round(float(merge), 4)
        findings.append(f"Overheads at n={largest:,}: IPC/pickling ~{comm:.1%} and final k-way merge in the "
                        f"parent ~{merge:.1%} of parallel execution time on average across worker counts; the "
                        "merge is serial and grows with worker count, limiting scalability (Amdahl's law).")

        imb = par.groupby("workers")["mean_load_imbalance"].mean()
        details["load_imbalance_by_workers"] = imb.round(4).to_dict()
        findings.append("Load balance (max/mean worker sort time, 1.00=perfect): "
                        + ", ".join(f"{int(w)}w={v:.3f}" for w, v in imb.items())
                        + ". Chunks differ by at most one element, so imbalance comes from OS scheduling/"
                          "SMT sharing rather than uneven work.")

        # when serial wins
        serial_wins = []
        for n in sizes:
            sub = summary[summary["input_size"] == n]
            ms = sub[sub["algorithm"] == "merge_sort"]["median_time_s"]
            ts = sub[sub["algorithm"] == "timsort"]["median_time_s"]
            pp = sub[sub["algorithm"] == "parallel_merge_sort"]
            if pp.empty:
                continue
            best_par = pp["median_time_s"].min()
            best_e2e = pp["end_to_end_time_s"].min()
            entry = {"input_size": int(n), "best_parallel_s": float(best_par)}
            if not ms.empty:
                entry["merge_sort_beats_parallel_exec"] = bool(ms.iloc[0] < best_par)
                entry["merge_sort_beats_parallel_end_to_end"] = bool(ms.iloc[0] < best_e2e)
            if not ts.empty:
                entry["timsort_beats_parallel"] = bool(ts.iloc[0] < best_par)
            serial_wins.append(entry)
        details["serial_vs_parallel"] = serial_wins
        e2e_losses = [e["input_size"] for e in serial_wins if e.get("merge_sort_beats_parallel_end_to_end")]
        if e2e_losses:
            findings.append("Serial merge sort beats a one-off parallel sort (including pool startup) at n = "
                            + ", ".join(f"{n:,}" for n in e2e_losses) + ".")
        w1 = par[par["workers"] == 1]
        if not w1.empty:
            findings.append("Parallel merge sort with 1 worker is slower than serial merge sort by "
                            f"{(1 / w1['speedup']).mean() - 1:.1%} on average: pure overhead of "
                            "pickling data to/from a worker process.")
        tw = [e["input_size"] for e in serial_wins if e.get("timsort_beats_parallel")]
        if tw:
            findings.append("Built-in Timsort (C implementation, serial) beats the best pure-Python parallel "
                            f"merge sort at {len(tw)}/{len(serial_wins)} input sizes: implementation efficiency "
                            "outweighs multicore parallelism here.")
    return {"findings": findings, "details": details}
