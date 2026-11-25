"""Train, validate, compare and persist the execution-time regression models.

Strategy
--------
* Rows = real benchmark repetitions. Group = (algorithm, workers, input_size) configuration.
* Hold-out test set: ``GroupShuffleSplit`` on configurations (no repetition of a test
  configuration is ever seen during training -> no leakage).
* Model selection: ``GroupKFold`` cross-validation on the training portion, ranked by RMSE.
* The best model (by CV RMSE) is then evaluated once on the held-out test configurations.
* All models predict log(time) internally (``TransformedTargetRegressor``) because times
  span several orders of magnitude; every reported metric is in seconds.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import config
from benchmarking.benchmark import load_results
from dashboard.visualizations import plot_actual_vs_predicted, plot_feature_importance
from ml.evaluate import extrapolation_check, feature_importance, regression_metrics
from ml.preprocessing import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, InsufficientDataError,
                              make_preprocessor, prepare_dataset)

MIN_GROUPS = 8          # minimum distinct configurations to train + test at all
MIN_TRAIN_GROUPS_CV = 3

MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "linear_regression": lambda: LinearRegression(),
    "decision_tree": lambda: DecisionTreeRegressor(min_samples_leaf=2, random_state=config.RANDOM_STATE),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=1,
                                                   random_state=config.RANDOM_STATE, n_jobs=1),
}
MODEL_LABELS = {"linear_regression": "Linear Regression", "decision_tree": "Decision Tree",
                "random_forest": "Random Forest"}


def build_pipeline(name: str) -> Pipeline:
    reg = TransformedTargetRegressor(regressor=MODEL_FACTORIES[name](), func=np.log, inverse_func=np.exp)
    return Pipeline([("pre", make_preprocessor()), ("model", reg)])


def model_path(name: str):
    return config.MODELS_DIR / f"{name}.joblib"


@dataclass
class TrainingResult:
    metrics: pd.DataFrame
    best_model: str
    predictions: pd.DataFrame
    importances: pd.DataFrame
    metadata: dict[str, Any]
    plots: list[str] = field(default_factory=list)


def _training_metadata(raw: pd.DataFrame, X: pd.DataFrame, groups: pd.Series, train_idx, test_idx) -> dict[str, Any]:
    tr = X.iloc[train_idx]
    algos = {}
    for algo, g in tr.groupby("algorithm"):
        algos[algo] = {"input_size_min": int(g["input_size"].min()), "input_size_max": int(g["input_size"].max()),
                       "input_sizes": sorted(int(v) for v in g["input_size"].unique()),
                       "workers": sorted(int(v) for v in g["workers"].unique())}
    hw = raw.iloc[0]  # prepare_dataset guarantees at least one valid row
    return {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(X)), "n_configurations": int(groups.nunique()),
        "n_train_samples": int(len(train_idx)), "n_test_samples": int(len(test_idx)),
        "test_configurations": sorted(groups.iloc[test_idx].unique().tolist()),
        "feature_columns": FEATURE_COLUMNS, "categorical_features": CATEGORICAL_FEATURES,
        "training_ranges": algos,
        "training_hardware": {"cpu_model": str(hw.get("cpu_model", "unknown")),
                              "physical_cores": int(hw["physical_cores"]), "logical_cores": int(hw["logical_cores"]),
                              "ram_total_gb": float(hw["ram_total_gb"])},
        "n_distinct_hardware_configs": int(raw[["physical_cores", "logical_cores", "ram_total_gb"]]
                                           .drop_duplicates().shape[0]),
    }


def train_models(raw: pd.DataFrame | None = None, test_size: float = 0.25, save: bool = True,
                 progress: Callable[[str], None] | None = None) -> TrainingResult:
    log = progress or (lambda msg: None)
    raw = load_results() if raw is None else raw
    X, y, groups = prepare_dataset(raw)
    n_groups = groups.nunique()
    if n_groups < MIN_GROUPS:
        raise InsufficientDataError(
            f"Only {n_groups} distinct configurations available; at least {MIN_GROUPS} are needed "
            "(run benchmarks with more input sizes / worker counts).")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, y_tr, g_tr = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    n_train_groups = g_tr.nunique()
    if n_train_groups < MIN_TRAIN_GROUPS_CV:
        raise InsufficientDataError("Too few training configurations for cross-validation.")
    cv = GroupKFold(n_splits=min(5, n_train_groups))
    log(f"Dataset: {len(X)} measurements, {n_groups} configurations; "
        f"train {len(X_tr)} rows / test {len(X_te)} rows ({groups.iloc[test_idx].nunique()} held-out configs); "
        f"{cv.get_n_splits()}-fold GroupKFold CV")

    scoring = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error",
               "RMSE": "neg_root_mean_squared_error", "R2": "r2",
               "MAPE": "neg_mean_absolute_percentage_error"}
    metric_rows, pred_frame, fitted = [], pd.DataFrame({"actual_s": y_te.to_numpy()}), {}
    pred_frame = pd.concat([X_te[["algorithm", "execution_mode", "input_size", "workers"]].reset_index(drop=True),
                            pred_frame], axis=1)
    for name in MODEL_FACTORIES:
        log(f"Cross-validating {MODEL_LABELS[name]} ...")
        cvres = cross_validate(build_pipeline(name), X_tr, y_tr, groups=g_tr, cv=cv, scoring=scoring)
        cv_metrics = {k: float(-cvres[f"test_{k}"].mean()) if k != "R2" else float(cvres["test_R2"].mean())
                      for k in scoring}
        metric_rows.append({"model": name, "split": "cv", **cv_metrics,
                            "RMSE_std": float(cvres["test_RMSE"].std())})
        log(f"Fitting {MODEL_LABELS[name]} on full training set ...")
        pipe = build_pipeline(name).fit(X_tr, y_tr)
        fitted[name] = pipe
        pred = pipe.predict(X_te)
        pred_frame[f"pred_{name}"] = pred
        metric_rows.append({"model": name, "split": "test", **regression_metrics(y_te, pred)})

    metrics = pd.DataFrame(metric_rows)
    cv_rank = metrics[metrics["split"] == "cv"].sort_values("RMSE")
    best = str(cv_rank.iloc[0]["model"])
    metrics["selected_by_cv"] = metrics["model"] == best
    log(f"Best model by CV RMSE: {MODEL_LABELS[best]}")

    extra = extrapolation_check(build_pipeline(best), X, y)
    if extra:
        metrics = pd.concat([metrics, pd.DataFrame([{"model": best, "split": "extrapolation_largest_size",
                                                      **{k: extra[k] for k in ("MAE", "MSE", "RMSE", "R2", "MAPE")},
                                                      "selected_by_cv": True}])], ignore_index=True)

    importances = pd.concat([feature_importance(p).assign(model=n) for n, p in fitted.items()], ignore_index=True)
    meta = _training_metadata(raw, X, groups, train_idx, test_idx)
    meta.update(best_model=best, extrapolation_check=extra,
                metrics=metrics.replace({np.nan: None}).to_dict("records"))

    plots: list[str] = []
    if save:
        log("Saving models, metrics and plots ...")
        for name, pipe in fitted.items():
            joblib.dump({"name": name, "pipeline": pipe, "metadata": meta}, model_path(name))
        joblib.dump({"name": best, "pipeline": fitted[best], "metadata": meta}, config.BEST_MODEL_PATH)
        config.MODEL_METADATA_JSON.write_text(json.dumps(meta, indent=2, default=str))
        metrics.to_csv(config.MODEL_METRICS_CSV, index=False)
        pred_frame.to_csv(config.MODEL_PREDICTIONS_CSV, index=False)
        importances.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)
        test_m = metrics[metrics["split"] == "test"].set_index("model")
        for name in fitted:
            fig = plot_actual_vs_predicted(pred_frame, name, MODEL_LABELS[name], test_m.loc[name, "R2"])
            path = config.PLOTS_DIR / f"ml_actual_vs_predicted_{name}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig); plots.append(str(path))
            fig = plot_feature_importance(importances[importances["model"] == name], MODEL_LABELS[name])
            path = config.PLOTS_DIR / f"ml_feature_importance_{name}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig); plots.append(str(path))
    log("Training complete")
    return TrainingResult(metrics, best, pred_frame, importances, meta, plots)


def models_trained() -> bool:
    return config.BEST_MODEL_PATH.exists() and config.MODEL_METADATA_JSON.exists()
