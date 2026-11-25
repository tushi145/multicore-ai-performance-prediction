import numpy as np
import pandas as pd
import pytest

import config
from ml.preprocessing import FEATURE_COLUMNS, InsufficientDataError, build_features, config_group_ids, prepare_dataset


def test_build_features(small_benchmark_df):
    X = build_features(small_benchmark_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert np.allclose(X["log2_input_size"], np.log2(small_benchmark_df["input_size"]))


def test_groups_keep_repetitions_together(small_benchmark_df):
    g = config_group_ids(small_benchmark_df)
    assert g.nunique() == 20  # 4 sizes x 5 configurations
    assert (small_benchmark_df.groupby(g).size() == 2).all()


def test_prepare_rejects_empty_and_bad_rows(small_benchmark_df):
    with pytest.raises(InsufficientDataError):
        prepare_dataset(pd.DataFrame())
    bad = small_benchmark_df.copy()
    bad.loc[0, "correct"] = False
    X, y, _ = prepare_dataset(bad)
    assert len(X) == len(bad) - 1 and (y > 0).all()


def test_insufficient_configurations_raise(small_benchmark_df):
    from ml.train import train_models
    tiny = small_benchmark_df[small_benchmark_df["input_size"] == 1000].head(6)
    with pytest.raises(InsufficientDataError):
        train_models(tiny, save=False)


def test_train_no_leakage_and_metrics(small_benchmark_df):
    from ml.train import train_models
    res = train_models(small_benchmark_df, save=False)
    assert set(res.metrics["model"]) == {"linear_regression", "decision_tree", "random_forest"}
    for col in ("MAE", "MSE", "RMSE", "R2"):
        assert res.metrics[col].notna().all()
    test_cfgs = set(res.metadata["test_configurations"])
    train_cfgs = set(config_group_ids(small_benchmark_df)) - test_cfgs
    assert test_cfgs and not (test_cfgs & train_cfgs)
    assert res.best_model == res.metrics[res.metrics["split"] == "cv"].sort_values("RMSE").iloc[0]["model"]


def test_save_load_and_predict(small_benchmark_df, isolated_paths):
    from ml.predict import load_model, predict_execution_time
    from ml.train import train_models
    res = train_models(small_benchmark_df, save=True)
    assert config.BEST_MODEL_PATH.exists() and config.MODEL_METRICS_CSV.exists()
    assert len(list(config.PLOTS_DIR.glob("ml_*.png"))) == 6
    for name in ("best", "linear_regression", "decision_tree", "random_forest"):
        bundle = load_model(name)
        assert "pipeline" in bundle
    p = predict_execution_time(4000, "merge_sort", "serial", 1, history=small_benchmark_df)
    assert p.model_name == res.best_model and p.predicted_s > 0
    assert p.predicted_ms == pytest.approx(p.predicted_s * 1000)
    assert p.exact_match is not None and not p.history.empty
    out_of_range = predict_execution_time(500_000, "parallel_merge_sort", "parallel", 2, history=small_benchmark_df)
    assert any("outside the trained range" in w for w in out_of_range.warnings)
    unseen_workers = predict_execution_time(4000, "parallel_merge_sort", "parallel", 3, "random_forest",
                                            history=small_benchmark_df)
    assert any("never benchmarked" in w for w in unseen_workers.warnings)


@pytest.mark.parametrize("args", [
    (1000, "bogo", "serial", 1),
    (1000, "merge_sort", "parallel", 1),
    (1000, "parallel_merge_sort", "serial", 1),
    (1000, "timsort", "serial", 2),
    (0, "timsort", "serial", 1),
    (10.5, "timsort", "serial", 1),
    (1000, "parallel_merge_sort", "parallel", 0),
])
def test_invalid_prediction_inputs(args):
    from ml.predict import validate_request
    with pytest.raises(ValueError):
        validate_request(*args)


def test_missing_model_is_reported(isolated_paths):
    from ml.predict import ModelNotTrainedError, load_model
    with pytest.raises(ModelNotTrainedError):
        load_model("best")
