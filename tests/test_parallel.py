import os
import random

import psutil
import pytest

from algorithms.parallel_sort import (ParallelMergeSorter, merge_sorted_chunks, parallel_merge_sort,
                                      split_into_chunks, validate_worker_count)

WORKER_COUNTS = [w for w in (1, 2, 4) if w <= (os.cpu_count() or 1)]
rng = random.Random(7)
CASES = {
    "empty": [],
    "single": [42],
    "fewer_than_workers": [3, 1],
    "duplicates": [rng.choice([-1, 0, 1]) for _ in range(3_001)],
    "negatives": [rng.randint(-10**6, -1) for _ in range(2_000)],
    "random": [rng.randint(-10**6, 10**6) for _ in range(20_001)],
    "sorted": list(range(5_000)),
    "reverse": list(range(5_000, 0, -1)),
}


@pytest.fixture(scope="module", params=WORKER_COUNTS)
def sorter(request):
    with ParallelMergeSorter(request.param) as s:
        yield s


@pytest.mark.parametrize("case", list(CASES))
def test_parallel_matches_sorted(sorter, case):
    data = CASES[case]
    original = list(data)
    assert sorter.sort(data) == sorted(data)
    assert data == original


def test_large_array(sorter):
    data = [random.Random(1).randint(-10**6, 10**6) for _ in range(300_000)]
    result, timings = sorter.sort_with_timings(data)
    assert result == sorted(data)
    assert sum(timings.chunk_sizes) == len(data)
    assert len(timings.chunk_sizes) == sorter.workers
    assert timings.total > 0 and timings.load_imbalance >= 1.0


def test_pool_uses_real_separate_processes(sorter):
    """Genuine multiprocessing: workers are distinct OS processes, not threads."""
    assert len(sorter.worker_pids) == sorter.workers
    assert os.getpid() not in sorter.worker_pids
    for pid in sorter.worker_pids:
        assert psutil.pid_exists(pid)


def test_pool_is_reused_across_sorts(sorter):
    pids_before = list(sorter.worker_pids)
    sorter.sort([5, 4, 3])
    sorter.sort([9, 8])
    assert sorter.worker_pids == pids_before


def test_one_shot_wrapper():
    data = [rng.randint(-100, 100) for _ in range(1_000)]
    assert parallel_merge_sort(data, workers=2) == sorted(data)


def test_close_terminates_workers():
    s = ParallelMergeSorter(2)
    s.start()
    pids = list(s.worker_pids)
    s.close()
    for pid in pids:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("n,parts", [(10, 3), (10, 4), (3, 8), (0, 4), (100, 1), (7, 7)])
def test_split_balanced(n, parts):
    data = list(range(n))
    chunks = split_into_chunks(data, parts)
    assert [x for c in chunks for x in c] == data
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1
    assert len(chunks) == max(1, min(parts, n))


def test_merge_sorted_chunks():
    assert merge_sorted_chunks([]) == []
    assert merge_sorted_chunks([[1, 4], [2, 3], [], [0, 5]]) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("bad", [0, -1, (os.cpu_count() or 1) + 1])
def test_invalid_worker_counts(bad):
    with pytest.raises(ValueError):
        validate_worker_count(bad)


def test_worker_count_type():
    with pytest.raises(TypeError):
        validate_worker_count(2.0)
