import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def small_benchmark_df():
    """Real (small) benchmark measurements, not written to the project dataset."""
    from benchmarking.benchmark import BenchmarkConfig, run_benchmark_suite
    cfg = BenchmarkConfig(sizes=[1000, 2000, 4000, 8000], worker_counts=[1, 2], repetitions=2, warmup=False)
    return run_benchmark_suite(cfg, output_csv=None)


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every output path in config to a temp directory."""
    import config
    models, results, plots = tmp_path / "models", tmp_path / "results", tmp_path / "results" / "plots"
    for d in (models, results, plots):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "PLOTS_DIR", plots)
    monkeypatch.setattr(config, "BEST_MODEL_PATH", models / "best_model.joblib")
    monkeypatch.setattr(config, "MODEL_METADATA_JSON", models / "model_metadata.json")
    monkeypatch.setattr(config, "MODEL_METRICS_CSV", results / "model_metrics.csv")
    monkeypatch.setattr(config, "MODEL_PREDICTIONS_CSV", results / "test_predictions.csv")
    monkeypatch.setattr(config, "FEATURE_IMPORTANCE_CSV", results / "feature_importance.csv")
    monkeypatch.setattr(config, "RAW_BENCHMARK_CSV", tmp_path / "raw.csv")
    return tmp_path
