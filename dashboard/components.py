"""Reusable dashboard building blocks: data loading, subprocess experiment runners, downloads."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import config
from benchmarking.benchmark import load_results
from benchmarking.metrics import compute_insights, summarize

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
    border-radius: 10px; padding: 0.7rem 0.9rem;}
div[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
</style>
"""


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_raw_cached(path: str, mtime: float) -> pd.DataFrame:
    return load_results(Path(path))


def load_raw() -> pd.DataFrame:
    return _load_raw_cached(str(config.RAW_BENCHMARK_CSV), _mtime(config.RAW_BENCHMARK_CSV))


@st.cache_data(show_spinner=False)
def _summary_cached(mtime: float) -> pd.DataFrame:
    return summarize(load_raw())


def load_summary() -> pd.DataFrame:
    return _summary_cached(_mtime(config.RAW_BENCHMARK_CSV))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def load_model_metadata() -> dict[str, Any] | None:
    if not config.MODEL_METADATA_JSON.exists():
        return None
    return json.loads(config.MODEL_METADATA_JSON.read_text())


def show_fig(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def write_analysis_outputs() -> list[Path]:
    """Persist summary CSV, insights JSON and analysis plots from the current raw data."""
    from dashboard.visualizations import save_analysis_plots
    summary = summarize(load_results())
    if summary.empty:
        return []
    summary.to_csv(config.SUMMARY_CSV, index=False)
    config.INSIGHTS_JSON.write_text(json.dumps(compute_insights(summary), indent=2, default=str))
    return save_analysis_plots(summary)


def run_single_subprocess(algorithm: str, size: int, workers: int, repetition: int, seed: int,
                          save: bool) -> dict[str, Any]:
    """Run one benchmark in a fresh interpreter so the dashboard's own threads and memory do not
    contaminate the measurement and Windows ``spawn`` multiprocessing works reliably."""
    cmd = [sys.executable, str(config.PROJECT_ROOT / "main.py"), "single", "--algorithm", algorithm,
           "--size", str(size), "--workers", str(workers), "--repetition", str(repetition),
           "--seed", str(seed), "--json"] + (["--save"] if save else [])
    proc = subprocess.run(cmd, cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON="):])
    raise RuntimeError(f"No result returned by benchmark process:\n{proc.stdout}\n{proc.stderr}")


def run_suite_subprocess(sizes: list[int], workers: list[int], reps: int, seed: int, algorithms: list[str],
                         on_line: Callable[[str], None]) -> int:
    cmd = [sys.executable, "-u", str(config.PROJECT_ROOT / "main.py"), "benchmark",
           "--sizes", *map(str, sizes), "--workers", *map(str, workers), "--reps", str(reps),
           "--seed", str(seed), "--algorithms", *algorithms]
    with subprocess.Popen(cmd, cwd=config.PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            on_line(line.rstrip())
        return proc.wait()


def plots_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(config.PLOTS_DIR.glob("*.png")):
            zf.write(p, arcname=p.name)
    return buf.getvalue()


def download_file_button(label: str, path: Path, mime: str = "text/csv", key: str | None = None) -> None:
    if path.exists():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=key or str(path))
    else:
        st.caption(f"{label}: not available yet ({path.name} missing)")


def parse_int_list(text: str) -> list[int]:
    values = [int(v.strip().replace("_", "")) for v in text.replace(";", ",").split(",") if v.strip()]
    if not values:
        raise ValueError("Enter at least one value")
    return values
