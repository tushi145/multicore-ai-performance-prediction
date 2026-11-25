"""Execution-time prediction with input validation, range warnings and historical comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd

import config
from benchmarking.benchmark import load_results
from benchmarking.metrics import summarize
from benchmarking.system_monitor import get_system_info
from ml.preprocessing import build_features

MODEL_NAMES = ["linear_regression", "decision_tree", "random_forest"]


class ModelNotTrainedError(FileNotFoundError):
    pass


@dataclass
class Prediction:
    model_name: str
    predicted_s: float
    warnings: list[str] = field(default_factory=list)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    exact_match: dict[str, Any] | None = None

    @property
    def predicted_ms(self) -> float:
        return self.predicted_s * 1000


def load_model(name: str = "best") -> dict[str, Any]:
    path = config.BEST_MODEL_PATH if name == "best" else config.MODELS_DIR / f"{name}.joblib"
    if name != "best" and name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'. Choose from 'best', {MODEL_NAMES}")
    if not path.exists():
        raise ModelNotTrainedError(f"Model file not found: {path}. Train the models first.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"{path} is not a valid model bundle")
    return bundle


def validate_request(input_size: Any, algorithm: str, mode: str, workers: Any) -> tuple[int, int]:
    if algorithm not in config.ALGORITHM_MODES:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Valid: {list(config.ALGORITHM_MODES)}")
    if mode not in (config.SERIAL, config.PARALLEL):
        raise ValueError("mode must be 'serial' or 'parallel'")
    if config.ALGORITHM_MODES[algorithm] != mode:
        raise ValueError(f"'{algorithm}' is a {config.ALGORITHM_MODES[algorithm]} algorithm, not {mode}")
    for name, value in (("input_size", input_size), ("workers", workers)):
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(f"{name} must be a whole number")
    try:
        n, w = int(input_size), int(workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_size and workers must be integers") from exc
    if not 1 <= n <= config.MAX_INPUT_SIZE:
        raise ValueError(f"input_size must be between 1 and {config.MAX_INPUT_SIZE:,}")
    if mode == config.SERIAL and w != 1:
        raise ValueError("serial algorithms use exactly 1 worker")
    if not 1 <= w <= config.LOGICAL_CORES:
        raise ValueError(f"workers must be between 1 and {config.LOGICAL_CORES} (logical CPUs)")
    return n, w


def _range_warnings(meta: dict[str, Any], algorithm: str, n: int, w: int, hw: dict[str, Any]) -> list[str]:
    warns: list[str] = []
    ranges = meta.get("training_ranges", {}).get(algorithm)
    if ranges is None:
        return [f"'{algorithm}' was not in the training data; prediction is unreliable."]
    lo, hi = ranges["input_size_min"], ranges["input_size_max"]
    if n < lo or n > hi:
        warns.append(f"Input size {n:,} is outside the trained range [{lo:,}, {hi:,}] for {algorithm}: "
                     "this is extrapolation. Tree-based models cannot predict beyond the times seen in "
                     "training (they plateau), and all models become less reliable.")
    elif n not in ranges["input_sizes"]:
        warns.append(f"Input size {n:,} was not benchmarked; prediction interpolates between "
                     f"trained sizes {ranges['input_sizes']}.")
    if w not in ranges["workers"]:
        warns.append(f"Worker count {w} was never benchmarked for {algorithm} (trained: {ranges['workers']}).")
    thw = meta.get("training_hardware", {})
    if thw and (thw.get("physical_cores") != hw["physical_cores"] or thw.get("logical_cores") != hw["logical_cores"]):
        warns.append(f"Current hardware ({hw['physical_cores']}C/{hw['logical_cores']}T) differs from training "
                     f"hardware ({thw.get('physical_cores')}C/{thw.get('logical_cores')}T); predictions do not "
                     "transfer across machines (single-machine training data has constant hardware "
                     "features, so the model has not learned hardware effects).")
    return warns


def predict_execution_time(input_size: Any, algorithm: str, mode: str, workers: Any = 1,
                           model_name: str = "best", history: pd.DataFrame | None = None) -> Prediction:
    n, w = validate_request(input_size, algorithm, mode, workers)
    bundle = load_model(model_name)
    meta = bundle.get("metadata", {})
    hw = get_system_info()
    row = pd.DataFrame([{"algorithm": algorithm, "execution_mode": mode, "input_size": n, "workers": w,
                         "physical_cores": hw["physical_cores"], "logical_cores": hw["logical_cores"],
                         "ram_total_gb": hw["ram_total_gb"]}])
    pred = float(bundle["pipeline"].predict(build_features(row))[0])
    if pred <= 0:
        raise RuntimeError(f"Model produced a non-positive prediction ({pred})")

    raw = load_results() if history is None else history
    hist = pd.DataFrame()
    exact = None
    if not raw.empty:
        summ = summarize(raw)
        hist = summ[(summ["algorithm"] == algorithm) & (summ["workers"] == w)][
            ["input_size", "workers", "n_runs", "median_time_s", "mean_time_s", "min_time_s", "std_time_s"]]
        hit = hist[hist["input_size"] == n]
        if not hit.empty:
            exact = hit.iloc[0].to_dict()
            exact["prediction_error_pct"] = 100 * (pred - exact["median_time_s"]) / exact["median_time_s"]
    return Prediction(model_name=bundle["name"], predicted_s=pred,
                      warnings=_range_warnings(meta, algorithm, n, w, hw),
                      history=hist.reset_index(drop=True), exact_match=exact)
