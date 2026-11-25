"""Regression metrics, feature importance extraction and extrapolation diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Importances for tree models; absolute standardized coefficients for linear regression
    (on the log-time scale). Returns columns: feature, importance, kind."""
    pre = pipeline.named_steps["pre"]
    reg = pipeline.named_steps["model"].regressor_
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    if hasattr(reg, "feature_importances_"):
        values, kind = reg.feature_importances_, "impurity importance"
    elif hasattr(reg, "coef_"):
        values, kind = np.abs(np.ravel(reg.coef_)), "|standardized coefficient| (log-time)"
    else:
        return pd.DataFrame(columns=["feature", "importance", "kind"])
    return (pd.DataFrame({"feature": names, "importance": values, "kind": kind})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def extrapolation_check(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:
    """Refit a clone on all input sizes except the largest and test on the largest one.

    Quantifies how well the model extrapolates to unseen, larger inputs. Returns None when
    fewer than 3 distinct input sizes exist.
    """
    sizes = np.sort(X["input_size"].unique())
    if len(sizes) < 3:
        return None
    largest = sizes[-1]
    train_mask = X["input_size"] < largest
    model = clone(pipeline).fit(X[train_mask], y[train_mask])
    pred = model.predict(X[~train_mask])
    return {"held_out_input_size": int(largest), "trained_up_to": int(sizes[-2]),
            "n_test": int((~train_mask).sum()), **regression_metrics(y[~train_mask], pred)}
