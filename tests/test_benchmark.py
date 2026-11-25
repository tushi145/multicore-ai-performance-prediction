import math

import pandas as pd
import pytest

import config
from benchmarking.benchmark import (BenchmarkConfig, append_results, generate_dataset, load_results,
                                    run_single_benchmark)
from benchmarking.metrics import add_speedup, compute_insights, summarize
from benchmarking.system_monitor import CpuSampler, get_system_info


def test_dataset_reproducible_and_has_negatives_and_duplicates():
    a, b = generate_dataset(50_000, 42, 0), generate_dataset(50_000, 42, 0)
    assert a == b
    assert generate_dataset(50_000, 42, 1) != a
    assert min(a) < 0 and len(set(a)) < len(a)


def test_system_info():
    info = get_system_info()
    assert info["logical_cores"] >= info["physical_cores"] >= 1
    assert info["ram_total_gb"] > 0


def test_cpu_sampler_short_interval_is_nan_not_zero():
    s = CpuSampler()
    s.start()
    usage = s.stop()
    assert math.isnan(usage.system_cpu_percent) and math.isnan(usage.effective_cores_used)


@pytest.mark.parametrize("algo,workers", [("merge_sort", 1), ("quick_sort", 1), ("timsort", 1),
                                          ("parallel_merge_sort", 2)])
def test_single_benchmark_row(algo, workers):
    row = run_single_benchmark(algo, 5_000, workers)
    assert row["correct"] and row["exec_time_s"] > 0
    assert row["algorithm"] == algo and row["workers"] == workers
    assert row["execution_mode"] == config.ALGORITHM_MODES[algo]
    if algo == "parallel_merge_sort":
        assert row["startup_time_s"] > 0 and row["merge_time_s"] >= 0


def test_serial_with_multiple_workers_rejected():
    with pytest.raises(ValueError):
        run_single_benchmark("timsort", 100, workers=2)


def test_config_validation():
    with pytest.raises(ValueError):
        BenchmarkConfig(sizes=[0]).validate()
    with pytest.raises(ValueError):
        BenchmarkConfig(algorithms=["bogo_sort"]).validate()
    with pytest.raises(ValueError):
        BenchmarkConfig(worker_counts=[10_000]).validate()


def test_suite_collects_all_configs(small_benchmark_df):
    df = small_benchmark_df
    # 4 sizes x (3 serial + 2 worker counts) x 2 reps
    assert len(df) == 4 * 5 * 2
    assert df["correct"].all()
    required = {"algorithm", "execution_mode", "input_size", "workers", "exec_time_s", "system_cpu_percent",
                "physical_cores", "logical_cores", "speedup", "efficiency", "startup_time_s"}
    assert required <= set(df.columns)
    base = df[df["algorithm"] == "merge_sort"]
    assert base["speedup"].notna().all()


def test_speedup_and_efficiency_formula():
    df = pd.DataFrame({"run_id": "r", "algorithm": ["merge_sort", "merge_sort", "parallel_merge_sort"],
                       "execution_mode": ["serial", "serial", "parallel"],
                       "input_size": 100, "workers": [1, 1, 4], "exec_time_s": [2.0, 2.0, 0.5]})
    out = add_speedup(df)
    par = out.iloc[2]
    assert par["speedup"] == pytest.approx(4.0)
    assert par["efficiency"] == pytest.approx(1.0)
    assert math.isnan(out.iloc[0]["efficiency"])


def test_summary_statistics(small_benchmark_df):
    s = summarize(small_benchmark_df)
    assert len(s) == 4 * 5
    row = s.iloc[0]
    assert row["min_time_s"] <= row["median_time_s"] <= row["max_time_s"]
    assert (s["n_runs"] == 2).all()
    ms = s[s["algorithm"] == "merge_sort"]
    assert ms["speedup"].round(9).eq(1.0).all()
    insights = compute_insights(s)
    assert insights["findings"]


def test_csv_roundtrip(small_benchmark_df, tmp_path):
    path = tmp_path / "bench.csv"
    append_results(small_benchmark_df, path)
    append_results(small_benchmark_df, path)
    loaded = load_results(path)
    assert len(loaded) == 2 * len(small_benchmark_df)


def test_load_missing_file_returns_empty(tmp_path):
    assert load_results(tmp_path / "missing.csv").empty


def test_execution_order_is_randomized_and_reproducible():
    """Configurations are interleaved in a seeded random order (not grouped by algorithm)."""
    from benchmarking.benchmark import run_benchmark_suite

    def order(seed):
        msgs = []
        run_benchmark_suite(BenchmarkConfig(sizes=[500, 700], worker_counts=[1, 2], repetitions=3, seed=seed,
                                            warmup=False), output_csv=None,
                            progress=lambda m, _: msgs.append(m.split(":")[0]))
        return [m for m in msgs if "rep" in m]

    a, b = order(7), order(7)
    assert a == b, "same seed must reproduce the same order"
    labels = [m.split(" n=")[0] for m in a[:5]]
    assert labels != sorted(labels) or len(set(m.split(" rep")[0] for m in a[:5])) == 5
    per_rep = [m.split(" n=")[0] for m in a if "n=500 " in m]
    reps = [per_rep[i:i + 5] for i in range(0, 15, 5)]
    assert len({tuple(r) for r in reps}) > 1, "order must be reshuffled per repetition"


def test_speedup_only_for_merge_sort_family(small_benchmark_df):
    s = summarize(small_benchmark_df)
    other = s[~s["algorithm"].isin(config.SPEEDUP_FAMILY)]
    assert other["speedup"].isna().all() and other["efficiency"].isna().all()
    assert s[s["algorithm"].isin(config.SPEEDUP_FAMILY)]["speedup"].notna().all()
    assert s["speedup_vs_timsort"].notna().all()
    assert small_benchmark_df[small_benchmark_df["algorithm"] == "timsort"]["speedup"].isna().all()
