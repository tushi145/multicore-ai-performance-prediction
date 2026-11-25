"""Central configuration: paths, benchmark defaults and algorithm registry metadata."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RAW_BENCHMARK_CSV = RAW_DIR / "benchmark_results.csv"
SUMMARY_CSV = PROCESSED_DIR / "benchmark_summary.csv"
INSIGHTS_JSON = RESULTS_DIR / "performance_insights.json"
MODEL_METRICS_CSV = RESULTS_DIR / "model_metrics.csv"
MODEL_PREDICTIONS_CSV = RESULTS_DIR / "test_predictions.csv"
FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"
MODEL_METADATA_JSON = MODELS_DIR / "model_metadata.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardware limits
# ---------------------------------------------------------------------------
LOGICAL_CORES: int = os.cpu_count() or 1

# ---------------------------------------------------------------------------
# Benchmark defaults (all overridable from the CLI / dashboard)
# ---------------------------------------------------------------------------
DEFAULT_SIZES: list[int] = [10_000, 50_000, 100_000, 250_000, 500_000]
DEFAULT_WORKER_COUNTS: list[int] = [w for w in (1, 2, 4, 8) if w <= LOGICAL_CORES]
DEFAULT_REPETITIONS: int = 5
DEFAULT_SEED: int = 42
VALUE_RANGE: tuple[int, int] = (-1_000_000, 1_000_000)  # includes negatives and duplicates
MAX_INPUT_SIZE: int = 5_000_000  # hard safety cap for pure-Python sorts

# ---------------------------------------------------------------------------
# Algorithm registry metadata
# ---------------------------------------------------------------------------
SERIAL = "serial"
PARALLEL = "parallel"

ALGORITHM_MODES: dict[str, str] = {
    "merge_sort": SERIAL,
    "quick_sort": SERIAL,
    "timsort": SERIAL,
    "parallel_merge_sort": PARALLEL,
}
ALGORITHM_LABELS: dict[str, str] = {
    "merge_sort": "Merge Sort (serial)",
    "quick_sort": "Quick Sort (serial)",
    "timsort": "Timsort / sorted() (optimized C baseline)",
    "parallel_merge_sort": "Parallel Merge Sort",
}
BASELINE_ALGORITHM = "merge_sort"  # speedup = T(serial merge sort) / T(parallel merge sort)
# Speedup/efficiency are only reported within the equivalent merge-sort family (same algorithm,
# serial vs parallel). Quick sort is a serial comparison point; Timsort (C implementation) is an
# optimized reference baseline compared separately via ``speedup_vs_timsort``.
SPEEDUP_FAMILY = ("merge_sort", "parallel_merge_sort")
OPTIMIZED_BASELINE = "timsort"

RANDOM_STATE = 42
