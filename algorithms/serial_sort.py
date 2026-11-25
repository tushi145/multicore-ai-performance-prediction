"""Serial sorting algorithms. All functions return a new sorted list and never mutate the input."""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any


def merge(left: Sequence[Any], right: Sequence[Any]) -> list[Any]:
    """Stable two-way merge of two already-sorted sequences."""
    result: list[Any] = []
    append = result.append
    i = j = 0
    n_left, n_right = len(left), len(right)
    while i < n_left and j < n_right:
        if right[j] < left[i]:
            append(right[j])
            j += 1
        else:
            append(left[i])
            i += 1
    if i < n_left:
        result.extend(left[i:])
    if j < n_right:
        result.extend(right[j:])
    return result


def _merge_sort(items: list[Any]) -> list[Any]:
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    return merge(_merge_sort(items[:mid]), _merge_sort(items[mid:]))


def merge_sort(data: Sequence[Any]) -> list[Any]:
    """Classic top-down recursive merge sort, O(n log n), stable. Recursion depth is log2(n)."""
    return _merge_sort(list(data))


def quick_sort(data: Sequence[Any], seed: int = 0) -> list[Any]:
    """In-place iterative quick sort on a copy of ``data``.

    Uses a seeded random pivot (deterministic, avoids worst case on sorted input) and
    Dijkstra 3-way partitioning (efficient with many duplicates). An explicit stack,
    always processing the smaller partition first, bounds extra memory to O(log n).
    """
    a = list(data)
    rng = random.Random(seed)
    stack: list[tuple[int, int]] = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        while lo < hi:
            pivot = a[rng.randint(lo, hi)]
            lt, i, gt = lo, lo, hi
            while i <= gt:
                x = a[i]
                if x < pivot:
                    a[lt], a[i] = x, a[lt]
                    lt += 1
                    i += 1
                elif x > pivot:
                    a[i], a[gt] = a[gt], x
                    gt -= 1
                else:
                    i += 1
            # a[lo:lt] < pivot, a[lt:gt+1] == pivot, a[gt+1:hi+1] > pivot
            if lt - lo < hi - gt:
                stack.append((gt + 1, hi))
                hi = lt - 1
            else:
                stack.append((lo, lt - 1))
                lo = gt + 1
    return a


def timsort(data: Sequence[Any]) -> list[Any]:
    """Python's built-in Timsort (C implementation) via ``sorted()``."""
    return sorted(data)


SERIAL_ALGORITHMS: dict[str, Callable[[Sequence[Any]], list[Any]]] = {
    "merge_sort": merge_sort,
    "quick_sort": quick_sort,
    "timsort": timsort,
}
