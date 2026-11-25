"""Parallel merge sort using a persistent ``multiprocessing`` worker pool.

Pipeline: split input into ``workers`` balanced contiguous chunks -> each worker process
sorts one chunk with the serial merge sort -> the parent performs an O(n log k) k-way
merge of the sorted chunks (``heapq.merge``).

Windows uses the ``spawn`` start method, so worker functions live at module level and any
script that creates a pool must be protected with ``if __name__ == "__main__":``.
The pool is created once and reused across many sorts to avoid unnecessary process
creation; its startup cost is measured separately from the sorting time.
"""
from __future__ import annotations

import gc
import heapq
import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, asdict
from typing import Any

from algorithms.serial_sort import merge_sort

_START_BARRIER: Any = None
_BARRIER_TIMEOUT_S = 120.0


@dataclass
class PhaseTimings:
    """Wall-clock breakdown of one parallel sort (seconds)."""
    split: float
    parallel_phase: float      # scatter + worker sorting + gather (pickling/IPC included)
    merge: float
    worker_compute_max: float  # slowest worker's pure sorting time
    worker_compute_mean: float
    chunk_sizes: list[int]

    @property
    def total(self) -> float:
        return self.split + self.parallel_phase + self.merge

    @property
    def communication_overhead(self) -> float:
        """Time in the parallel phase not spent sorting by the slowest worker (IPC, pickling, scheduling)."""
        return max(0.0, self.parallel_phase - self.worker_compute_max)

    @property
    def load_imbalance(self) -> float:
        """max/mean worker compute time; 1.0 means perfectly balanced."""
        return self.worker_compute_max / self.worker_compute_mean if self.worker_compute_mean > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(total=self.total, communication_overhead=self.communication_overhead,
                 load_imbalance=self.load_imbalance)
        return d


# --------------------------------------------------------------------------- worker side
def _init_worker(barrier: Any) -> None:
    global _START_BARRIER
    _START_BARRIER = barrier


def _wait_ready(_: int) -> int:
    """Blocks until every worker is alive, guaranteeing each process has started and imported."""
    _START_BARRIER.wait(timeout=_BARRIER_TIMEOUT_S)
    return os.getpid()


def _sort_chunk(chunk: list[Any]) -> tuple[list[Any], float]:
    # Same conditions as the timed serial baseline (GC paused during sorting) for a fair comparison.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        result = merge_sort(chunk)
        return result, time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()


# --------------------------------------------------------------------------- helpers
def split_into_chunks(data: Sequence[Any], parts: int) -> list[list[Any]]:
    """Split into at most ``parts`` contiguous, non-empty chunks whose sizes differ by <= 1."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    n = len(data)
    parts = min(parts, n) or 1
    base, extra = divmod(n, parts)
    chunks, start = [], 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        chunks.append(list(data[start:end]))
        start = end
    return chunks


def merge_sorted_chunks(chunks: list[list[Any]]) -> list[Any]:
    """k-way merge of sorted lists in O(n log k)."""
    if not chunks:
        return []
    if len(chunks) == 1:
        return list(chunks[0])
    return list(heapq.merge(*chunks))


def validate_worker_count(workers: int) -> int:
    max_workers = os.cpu_count() or 1
    if not isinstance(workers, int) or isinstance(workers, bool):
        raise TypeError(f"workers must be an int, got {type(workers).__name__}")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers > max_workers:
        raise ValueError(f"workers={workers} exceeds available logical CPUs ({max_workers})")
    return workers


# --------------------------------------------------------------------------- public API
class ParallelMergeSorter:
    """Reusable parallel merge sorter bound to a persistent process pool.

    Usage::

        with ParallelMergeSorter(workers=4) as sorter:
            result = sorter.sort(data)
    """

    def __init__(self, workers: int):
        self.workers = validate_worker_count(workers)
        self._pool: Any = None
        self.worker_pids: list[int] = []
        self.startup_time: float | None = None

    def start(self) -> float:
        """Create the pool and wait until all workers are ready. Returns startup time (s)."""
        if self._pool is not None:
            return self.startup_time or 0.0
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()
        barrier = ctx.Barrier(self.workers)
        self._pool = ctx.Pool(processes=self.workers, initializer=_init_worker, initargs=(barrier,))
        try:
            self.worker_pids = sorted(set(self._pool.map(_wait_ready, range(self.workers), chunksize=1)))
        except Exception:
            self.close()
            raise
        self.startup_time = time.perf_counter() - t0
        return self.startup_time

    def sort(self, data: Sequence[Any]) -> list[Any]:
        return self.sort_with_timings(data)[0]

    def sort_with_timings(self, data: Sequence[Any]) -> tuple[list[Any], PhaseTimings]:
        if self._pool is None:
            self.start()
        t0 = time.perf_counter()
        chunks = split_into_chunks(data, self.workers)
        t1 = time.perf_counter()
        results = self._pool.map(_sort_chunk, chunks, chunksize=1)
        t2 = time.perf_counter()
        merged = merge_sorted_chunks([r[0] for r in results])
        t3 = time.perf_counter()
        compute = [r[1] for r in results] or [0.0]
        timings = PhaseTimings(
            split=t1 - t0, parallel_phase=t2 - t1, merge=t3 - t2,
            worker_compute_max=max(compute), worker_compute_mean=sum(compute) / len(compute),
            chunk_sizes=[len(c) for c in chunks],
        )
        return merged, timings

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self.worker_pids = []

    def __enter__(self) -> "ParallelMergeSorter":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def parallel_merge_sort(data: Sequence[Any], workers: int | None = None) -> list[Any]:
    """One-shot convenience wrapper (creates and tears down a pool).

    For repeated sorts reuse a :class:`ParallelMergeSorter` instead.
    """
    workers = validate_worker_count(workers or min(4, os.cpu_count() or 1))
    with ParallelMergeSorter(workers) as sorter:
        return sorter.sort(data)
