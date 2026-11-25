"""Command-line entry point.

    python main.py benchmark [--sizes ...] [--workers ...] [--reps N] [--seed S] [--no-warmup]
    python main.py analyze
    python main.py train
    python main.py predict --size N --algorithm A --mode serial|parallel [--workers W] [--model best]
    python main.py single --algorithm A --size N [--workers W] [--save] [--json]
    python main.py all
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import config


def cmd_benchmark(args: argparse.Namespace) -> None:
    from benchmarking.benchmark import BenchmarkConfig, run_benchmark_suite
    cfg = BenchmarkConfig(sizes=args.sizes, worker_counts=args.workers, repetitions=args.reps,
                          seed=args.seed, algorithms=args.algorithms, warmup=not args.no_warmup)
    print(f"Benchmark: sizes={cfg.sizes} workers={cfg.worker_counts} reps={cfg.repetitions} "
          f"algorithms={cfg.algorithms}", flush=True)
    df = run_benchmark_suite(cfg, progress=lambda msg, frac: print(f"[{frac:6.1%}] {msg}", flush=True))
    print(f"Recorded {len(df)} measurements -> {config.RAW_BENCHMARK_CSV}")


def cmd_analyze(_: argparse.Namespace) -> None:
    from benchmarking.benchmark import load_results
    from benchmarking.metrics import compute_insights, summarize
    from dashboard.visualizations import save_analysis_plots
    raw = load_results()
    if raw.empty:
        sys.exit("No benchmark data. Run: python main.py benchmark")
    summary = summarize(raw)
    summary.to_csv(config.SUMMARY_CSV, index=False)
    insights = compute_insights(summary)
    config.INSIGHTS_JSON.write_text(json.dumps(insights, indent=2, default=str))
    paths = save_analysis_plots(summary)
    cols = ["algorithm", "input_size", "workers", "n_runs", "median_time_s", "mean_time_s", "min_time_s",
            "std_time_s", "speedup", "efficiency", "mean_startup_time_s"]
    with __import__("pandas").option_context("display.width", 200, "display.max_rows", 200):
        print(summary[cols].round(4).to_string(index=False))
    print("\nFindings:")
    for f in insights["findings"]:
        print(" -", f)
    print(f"\nSummary -> {config.SUMMARY_CSV}\nPlots -> {[p.name for p in paths]}")


def cmd_train(_: argparse.Namespace) -> None:
    from ml.train import train_models
    res = train_models(progress=lambda m: print(m, flush=True))
    with __import__("pandas").option_context("display.width", 200):
        print(res.metrics.round(6).to_string(index=False))
    print(f"\nBest model (CV RMSE): {res.best_model}")
    print(f"Models -> {config.MODELS_DIR}\nMetrics -> {config.MODEL_METRICS_CSV}")


def cmd_predict(args: argparse.Namespace) -> None:
    from ml.predict import predict_execution_time
    p = predict_execution_time(args.size, args.algorithm, args.mode, args.workers, args.model)
    print(f"Model: {p.model_name}\nPredicted execution time: {p.predicted_s:.6f} s ({p.predicted_ms:.3f} ms)")
    if p.exact_match:
        print(f"Measured median for this configuration: {p.exact_match['median_time_s']:.6f} s "
              f"(prediction error {p.exact_match['prediction_error_pct']:+.2f}%)")
    for w in p.warnings:
        print("WARNING:", w)
    if not p.history.empty:
        print("\nHistorical measurements (same algorithm & workers):")
        print(p.history.round(6).to_string(index=False))


def cmd_single(args: argparse.Namespace) -> None:
    from benchmarking.benchmark import append_results, run_single_benchmark
    from benchmarking.metrics import add_speedup
    import pandas as pd
    workers = args.workers if config.ALGORITHM_MODES[args.algorithm] == config.PARALLEL else 1
    row = run_single_benchmark(args.algorithm, args.size, workers, args.repetition, args.seed)
    if args.save:
        from benchmarking.benchmark import load_results
        hist = load_results()
        both = add_speedup(pd.concat([hist, pd.DataFrame([row])], ignore_index=True))
        row = both.iloc[-1].to_dict()
        append_results(pd.DataFrame([row])[both.columns])
    if args.json:
        clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        print("RESULT_JSON=" + json.dumps(clean, default=str))
    else:
        print(json.dumps(row, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="run the benchmark grid and append to the raw CSV")
    b.add_argument("--sizes", type=int, nargs="+", default=config.DEFAULT_SIZES)
    b.add_argument("--workers", type=int, nargs="+", default=config.DEFAULT_WORKER_COUNTS)
    b.add_argument("--reps", type=int, default=config.DEFAULT_REPETITIONS)
    b.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    b.add_argument("--algorithms", nargs="+", default=list(config.ALGORITHM_MODES), choices=list(config.ALGORITHM_MODES))
    b.add_argument("--no-warmup", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    sub.add_parser("analyze", help="summary statistics, insights and plots").set_defaults(func=cmd_analyze)
    sub.add_parser("train", help="train, validate and save the ML models").set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="predict execution time")
    pr.add_argument("--size", type=int, required=True)
    pr.add_argument("--algorithm", required=True, choices=list(config.ALGORITHM_MODES))
    pr.add_argument("--mode", required=True, choices=[config.SERIAL, config.PARALLEL])
    pr.add_argument("--workers", type=int, default=1)
    pr.add_argument("--model", default="best")
    pr.set_defaults(func=cmd_predict)

    s = sub.add_parser("single", help="run one benchmark (used by the dashboard)")
    s.add_argument("--algorithm", required=True, choices=list(config.ALGORITHM_MODES))
    s.add_argument("--size", type=int, required=True)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--repetition", type=int, default=0)
    s.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    s.add_argument("--save", action="store_true", help="append the result to the benchmark dataset")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_single)

    a = sub.add_parser("all", help="benchmark (defaults) -> analyze -> train")
    a.set_defaults(func=lambda ns: [f(ns) for f in (cmd_benchmark, cmd_analyze, cmd_train)],
                   sizes=config.DEFAULT_SIZES, workers=config.DEFAULT_WORKER_COUNTS,
                   reps=config.DEFAULT_REPETITIONS, seed=config.DEFAULT_SEED,
                   algorithms=list(config.ALGORITHM_MODES), no_warmup=False)
    return p


if __name__ == "__main__":  # required for multiprocessing 'spawn' on Windows
    ns = build_parser().parse_args()
    ns.func(ns)
