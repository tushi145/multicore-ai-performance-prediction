"""Feature engineering and preprocessing shared by training and prediction.

The exact same ``build_features`` + fitted ``ColumnTransformer`` (stored inside the saved
pipeline) are used at prediction time, so train/predict preprocessing cannot drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

TARGET = "exec_time_s"
CATEGORICAL_FEATURES = ["algorithm", "execution_mode"]
BASE_NUMERIC_FEATURES = ["input_size", "workers", "physical_cores", "logical_cores", "ram_total_gb"]
# Derived from input_size/workers only (no target information): complexity-aware features
DERIVED_FEATURES = ["log2_input_size", "log2_workers", "log2_work_per_worker"]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_RAW_COLUMNS = ["algorithm", "execution_mode", "input_size", "workers",
                        "physical_cores", "logical_cores", "ram_total_gb"]


class InsufficientDataError(ValueError):
    """Raised when there is not enough benchmark data to train/evaluate models reliably."""


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[REQUIRED_RAW_COLUMNS].copy()
    X["input_size"] = X["input_size"].astype(float)
    X["workers"] = X["workers"].astype(float)
    n = X["input_size"].clip(lower=2)
    X["log2_input_size"] = np.log2(n)
    X["log2_workers"] = np.log2(X["workers"].clip(lower=1))
    # log2(n * log2(n) / workers): the theoretical per-worker comparison work of merge sort
    X["log2_work_per_worker"] = np.log2(n * np.log2(n) / X["workers"].clip(lower=1))
    return X[FEATURE_COLUMNS]


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def config_group_ids(df: pd.DataFrame) -> pd.Series:
    """One group per experimental configuration; repeated measurements share a group so they
    never straddle the train/test boundary (prevents leakage)."""
    return (df["algorithm"].astype(str) + "|w" + df["workers"].astype(int).astype(str)
            + "|n" + df["input_size"].astype(int).astype(str))


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate raw benchmark rows and return (X, y, groups). Only real, correct measurements are used."""
    if df is None or df.empty:
        raise InsufficientDataError("No benchmark data found. Run benchmarks first.")
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column '{TARGET}'")
    data = df.copy()
    if "correct" in data.columns:
        data = data[data["correct"].astype(bool)]
    data = data.dropna(subset=REQUIRED_RAW_COLUMNS + [TARGET])
    data = data[(data[TARGET] > 0) & data["algorithm"].isin(config.ALGORITHM_MODES)]
    data = data.reset_index(drop=True)
    if data.empty:
        raise InsufficientDataError("No valid benchmark rows after cleaning.")
    return build_features(data), data[TARGET].astype(float), config_group_ids(data)
