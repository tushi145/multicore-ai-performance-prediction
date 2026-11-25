"""Benchmarking engine: reproducible datasets, isolated timed runs, correctness checks, CSV output."""
from __future__ import annotations

import gc
import random
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from algorithms.parallel_sort import ParallelMergeSorter, PhaseTimings, validate_worker_count
from algorithms.serial_sort import SERIAL_ALGORITHMS
from benchmarking.metrics import add_speedup
from benchmarking.system_monitor import CpuSampler, CpuUsage, get_system_info

WARMUP_SIZE = 20_000  # warm-up runs use a small slice: they only prime imports/caches/pool
# Before EVERY timed run, the same configuration is warmed up for at least this long (untimed).
# Measured on this laptop: without it, an 8-worker sort following single-core work or idle time ran
# ~45% slower (cores parked / clocks ramping), which biased the randomized order against parallel runs.
WARMUP_SECONDS = 0.1


class BenchmarkCorrectnessError(RuntimeError):
    """Raised when an algorithm returns an output different from ``sorted(data)``."""


@dataclass
class BenchmarkConfig:
    sizes: list[int] = field(default_factory=lambda: list(config.DEFAULT_SIZES))
    worker_counts: list[int] = field(default_factory=lambda: list(config.DEFAULT_WORKER_COUNTS))
    repetitions: int = config.DEFAULT_REPETITIONS
    seed: int = config.DEFAULT_SEED
    algorithms: list[str] = field(default_factory=lambda: list(config.ALGORITHM_MODES))
    warmup: bool = True

    def validate(self) -> None:
        unknown = set(self.algorithms) - set(config.ALGORITHM_MODES)
        if unknown:
            raise ValueError(f"Unknown algorithms: {sorted(unknown)}")
        if not self.sizes:
            raise ValueError("At least one input size is required")
        for n in self.sizes:
            if not 1 <= n <= config.MAX_INPUT_SIZE:
                raise ValueError(f"Input size {n} outside [1, {config.MAX_INPUT_SIZE}]")
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if "parallel_merge_sort" in self.algorithms:
            if not self.worker_counts:
                raise ValueError("worker_counts required for parallel_merge_sort")
            for w in self.worker_counts:
                validate_worker_count(w)
        self.sizes = sorted(set(self.sizes))
        self.worker_counts = sorted(set(self.worker_counts))


def generate_dataset(size: int, seed: int = config.DEFAULT_SEED, repetition: int = 0) -> list[int]:
    """Deterministic random integers (with negatives and duplicates) for a (seed, size, repetition)."""
    rng = np.random.default_rng([seed, size, repetition])
    lo, hi = config.VALUE_RANGE
    return rng.integers(lo, hi, size=size, endpoint=True).tolist()


def _warm_up(func: Callable[[list[int]], Any], data: list[int]) -> None:
    """Untimed: repeatedly sort a small slice for >= WARMUP_SECONDS so every configuration starts its
    timed run from a comparable machine state (cores awake, clocks ramped, caches primed)."""
    sample = data[:WARMUP_SIZE]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < WARMUP_SECONDS:
        func(sample)


def _timed(func: Callable[[list[int]], Any], data: list[int], extra_pids: list[int]) -> tuple[Any, float, CpuUsage]:
    """Run ``func(data)`` in isolation: full GC beforehand, GC disabled during timing."""
    gc.collect()
    sampler = CpuSampler(extra_pids)
    gc.disable()
    try:
        sampler.start()
        t0 = time.perf_counter()
        result = func(data)
        elapsed = time.perf_counter() - t0
        usage = sampler.stop()
    finally:
        gc.enable()
    return result, elapsed, usage


def _make_row(*, algorithm: str, size: int, workers: int, repetition: int, seed: int, elapsed: float,
              usage: CpuUsage, sysinfo: dict[str, Any], run_id: str, startup: float | None = None,
              timings: PhaseTimings | None = None) -> dict[str, Any]:
    nan = float("nan")
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "algorithm": algorithm,
        "execution_mode": config.ALGORITHM_MODES[algorithm],
        "input_size": size,
        "workers": workers,
        "repetition": repetition,
        "seed": seed,
        "exec_time_s": elapsed,
        "startup_time_s": startup if startup is not None else 0.0,
        "split_time_s": timings.split if timings else nan,
        "parallel_phase_time_s": timings.parallel_phase if timings else nan,
        "merge_time_s": timings.merge if timings else nan,
        "worker_compute_max_s": timings.worker_compute_max if timings else nan,
        "worker_compute_mean_s": timings.worker_compute_mean if timings else nan,
        "communication_overhead_s": timings.communication_overhead if timings else nan,
        "load_imbalance": timings.load_imbalance if timings else nan,
        "system_cpu_percent": usage.system_cpu_percent,
        "process_cpu_time_s": usage.process_cpu_time_s,
        "effective_cores_used": usage.effective_cores_used,
        "physical_cores": sysinfo["physical_cores"],
        "logical_cores": sysinfo["logical_cores"],
        "cpu_model": sysinfo["cpu_model"],
        "ram_total_gb": sysinfo["ram_total_gb"],
        "os": sysinfo["os"],
        "python_version": sysinfo["python_version"],
        "correct": True,
    }


def _check(result: Sequence[int], reference: list[int], label: str) -> None:
    if result != reference:
        raise BenchmarkCorrectnessError(f"{label} produced incorrect output; result not recorded")


def run_single_benchmark(algorithm: str, size: int, workers: int = 1, repetition: int = 0,
                         seed: int = config.DEFAULT_SEED, run_id: str | None = None,
                         sorter: ParallelMergeSorter | None = None) -> dict[str, Any]:
    """Run, validate and measure one benchmark. Raises BenchmarkCorrectnessError on wrong output."""
    if algorithm not in config.ALGORITHM_MODES:
        raise ValueError(f"Unknown algorithm '{algorithm}'")
    if not 1 <= size <= config.MAX_INPUT_SIZE:
        raise ValueError(f"size must be in [1, {config.MAX_INPUT_SIZE}]")
    sysinfo = get_system_info()
    run_id = run_id or f"single-{uuid.uuid4().hex[:8]}"
    data = generate_dataset(size, seed, repetition)
    reference = sorted(data)
    label = f"{algorithm} n={size} w={workers}"

    if config.ALGORITHM_MODES[algorithm] == config.SERIAL:
        if workers != 1:
            raise ValueError("Serial algorithms run with exactly 1 worker")
        _warm_up(SERIAL_ALGORITHMS[algorithm], data)
        result, elapsed, usage = _timed(SERIAL_ALGORITHMS[algorithm], data, [])
        _check(result, reference, label)
        return _make_row(algorithm=algorithm, size=size, workers=1, repetition=repetition, seed=seed,
                         elapsed=elapsed, usage=usage, sysinfo=sysinfo, run_id=run_id)

    owns_sorter = sorter is None
    sorter = sorter or ParallelMergeSorter(workers)
    try:
        startup = sorter.start()
        _warm_up(sorter.sort, data)
        (result, timings), elapsed, usage = _timed(sorter.sort_with_timings, data, sorter.worker_pids)
        _check(result, reference, label)
        return _make_row(algorithm=algorithm, size=size, workers=sorter.workers, repetition=repetition,
                         seed=seed, elapsed=elapsed, usage=usage, sysinfo=sysinfo, run_id=run_id,
                         startup=startup, timings=timings)
    finally:
        if owns_sorter:
            sorter.close()


def run_benchmark_suite(cfg: BenchmarkConfig, output_csv: Path | None = config.RAW_BENCHMARK_CSV,
                        progress: Callable[[str, float], None] | None = None) -> pd.DataFrame:
    """Run the full experiment grid. Rows are appended to ``output_csv`` after each input size.

    Every (size, repetition) pair uses one fixed dataset shared by all algorithm configurations.
    To reduce order/drift bias (CPU boost, thermal state, background load), the order of input
    sizes and, within every repetition, the order of all algorithm configurations are shuffled
    with an RNG seeded by ``cfg.seed`` (reproducible); each timed run is preceded by an untimed
    warm-up of the same configuration (``cfg.warmup``). Each parallel configuration has its own
    pool, started before any timed run; its startup time is recorded separately.
    """
    cfg.validate()
    sysinfo = get_system_info()
    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    serial_algos = [a for a in cfg.algorithms if config.ALGORITHM_MODES[a] == config.SERIAL]
    parallel_workers = cfg.worker_counts if "parallel_merge_sort" in cfg.algorithms else []
    configs = [(a, 1) for a in serial_algos] + [("parallel_merge_sort", w) for w in parallel_workers]
    total_runs = len(cfg.sizes) * len(configs) * cfg.repetitions
    order_rng = random.Random(cfg.seed)
    sizes = list(cfg.sizes)
    order_rng.shuffle(sizes)
    done = 0
    all_rows: list[dict[str, Any]] = []

    def report(msg: str) -> None:
        if progress:
            progress(msg, done / total_runs if total_runs else 1.0)

    for size in sizes:
        datasets = [generate_dataset(size, cfg.seed, r) for r in range(cfg.repetitions)]
        references = [sorted(d) for d in datasets]
        size_rows: list[dict[str, Any]] = []
        sorters: dict[int, ParallelMergeSorter] = {}
        try:
            for w in parallel_workers:  # idle pools consume no CPU while other configurations run
                sorters[w] = ParallelMergeSorter(w)
                sorters[w].start()
            for r, (data, ref) in enumerate(zip(datasets, references)):
                order = list(configs)
                order_rng.shuffle(order)
                for algo, w in order:
                    if cfg.warmup:
                        _warm_up(sorters[w].sort if algo == "parallel_merge_sort" else SERIAL_ALGORITHMS[algo], data)
                    if algo == "parallel_merge_sort":
                        s = sorters[w]
                        (result, timings), elapsed, usage = _timed(s.sort_with_timings, data, s.worker_pids)
                        startup = s.startup_time or 0.0
                    else:
                        result, elapsed, usage = _timed(SERIAL_ALGORITHMS[algo], data, [])
                        timings, startup = None, None
                    _check(result, ref, f"{algo} n={size} w={w} rep={r}")
                    size_rows.append(_make_row(algorithm=algo, size=size, workers=w, repetition=r, seed=cfg.seed,
                                               elapsed=elapsed, usage=usage, sysinfo=sysinfo, run_id=run_id,
                                               startup=startup, timings=timings))
                    done += 1
                    report(f"{algo} w={w} n={size:,} rep {r + 1}/{cfg.repetitions}: {elapsed:.4f}s")
        finally:
            for s in sorters.values():
                s.close()

        del datasets, references
        size_df = add_speedup(pd.DataFrame(size_rows))
        if output_csv is not None:
            append_results(size_df, output_csv)
        all_rows.extend(size_df.to_dict("records"))

    report("Benchmark suite complete")
    return pd.DataFrame(all_rows)


def append_results(df: pd.DataFrame, path: Path = config.RAW_BENCHMARK_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
        if existing_cols != df.columns.tolist():
            df = df.reindex(columns=existing_cols)
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def load_results(path: Path = config.RAW_BENCHMARK_CSV) -> pd.DataFrame:
    """Load raw benchmark measurements; returns an empty DataFrame if none exist yet."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "correct" in df.columns:
        df = df[df["correct"].astype(bool)]
    return df.reset_index(drop=True)
