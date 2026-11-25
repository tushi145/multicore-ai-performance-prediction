import random

import pytest

from algorithms.serial_sort import SERIAL_ALGORITHMS, merge, merge_sort, quick_sort

ALGOS = list(SERIAL_ALGORITHMS.items())
rng = random.Random(123)

CASES = {
    "empty": [],
    "single": [7],
    "two": [2, 1],
    "duplicates": [5, 3, 5, 1, 3, 3, 5, 0, 0],
    "all_equal": [4] * 500,
    "negatives": [-3, 10, -50, 0, 7, -1, -1, 99],
    "random": [rng.randint(-10_000, 10_000) for _ in range(5_000)],
    "sorted": list(range(3_000)),
    "reverse": list(range(3_000, 0, -1)),
    "few_unique": [rng.choice([1, 2, 3]) for _ in range(4_000)],
    "floats": [rng.uniform(-1, 1) for _ in range(1_000)],
}


@pytest.mark.parametrize("name,func", ALGOS)
@pytest.mark.parametrize("case", list(CASES))
def test_matches_builtin_sorted(name, func, case):
    data = CASES[case]
    original = list(data)
    assert func(data) == sorted(data)
    assert data == original, "input must not be mutated"


@pytest.mark.parametrize("name,func", ALGOS)
def test_large_array(name, func):
    data = [random.Random(9).randint(-10**6, 10**6) for _ in range(200_000)]
    assert func(data) == sorted(data)


@pytest.mark.parametrize("name,func", ALGOS)
def test_returns_new_list(name, func):
    data = [3, 1, 2]
    out = func(data)
    assert out is not data and isinstance(out, list)


def test_merge_is_stable_and_correct():
    left = [(1, "a"), (3, "a")]
    right = [(1, "b"), (2, "b")]
    keyed = merge([x[0] for x in left], [x[0] for x in right])
    assert keyed == [1, 1, 2, 3]
    assert merge([], [1, 2]) == [1, 2] and merge([1], []) == [1]


def test_quick_sort_deep_input_no_recursion_error():
    data = list(range(100_000))  # adversarial for naive pivots; iterative implementation must cope
    assert quick_sort(data) == data
    assert merge_sort(data[::-1]) == data
