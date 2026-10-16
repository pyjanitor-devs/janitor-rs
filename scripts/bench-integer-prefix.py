"""Compare starts/ends integer sums and counts through PyO3 and join_agg.

Use explicit release libraries; --compare-library alternates both in one process
outside the timed intervals to reduce drift between separate runs. Reports
JSON lines with independent direct NumPy oracle checks, median time, Python/NumPy
traced peak, and analytical Rust table bytes (not measured RSS). Fixtures,
imports, oracles, and one warmup are outside timing. The same script works on the
base commit: no new Python endpoint or benchmark-only export is required.
"""

import argparse
import gc
import importlib.machinery
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
import warnings
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--pyjanitor", type=Path, required=True)
    parser.add_argument(
        "--compare-library",
        type=Path,
        help="Other strategy's library for alternating timings",
    )
    parser.add_argument("--strategy", choices=["base", "prefix"], required=True)
    parser.add_argument(
        "--aggregation", choices=["sum", "count", "sum-count"], default="sum"
    )
    parser.add_argument(
        "--baseline-stage",
        choices=["pre-sum", "pre-count"],
        default="pre-sum",
        help="Whether the base library already contains integer prefix sums",
    )
    parser.add_argument(
        "--count-candidate-cutoff",
        choices=["tree", "prefix"],
        default="tree",
        help="Candidate count guard; prefix reproduces the rejected count experiment",
    )
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--case", action="append", help="Restrict to named fixtures")
    parser.add_argument(
        "--endpoint", action="append", choices=["pyo3", "join_agg", "range_join_agg"]
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    def load_library(path):
        loader = importlib.machinery.ExtensionFileLoader(
            "janitor_rs", str(path.resolve())
        )
        spec = importlib.util.spec_from_loader("janitor_rs", loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules["janitor_rs"] = module
        loader.exec_module(module)
        return module

    rust = load_library(args.library)
    backends = [(args.strategy, rust)]
    if args.compare_library:
        other = "prefix" if args.strategy == "base" else "base"
        backends.append((other, load_library(args.compare_library)))
        assert backends[0][1] is not backends[1][1]
    sys.modules["janitor_rs"] = rust
    sys.path.insert(0, str(args.pyjanitor.resolve()))

    import janitor  # noqa: F401 - registers DataFrame methods
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def select_backend(module):
        nonlocal rust
        rust = module
        sys.modules["janitor_rs"] = module
        # Downstream modules import the extension namespace. Switch those
        # bindings outside timing, retaining identical pandas inputs.
        for name, loaded in list(sys.modules.items()):
            if name.startswith("janitor.") and any(
                getattr(loaded, "janitor_rs", None) is m for _, m in backends
            ):
                loaded.janitor_rs = module

    def check(result):
        if endpoint == "pyo3":
            np.testing.assert_array_equal(result[0], left_index)
            np.testing.assert_array_equal(result[1], np.ones(q, dtype=bool))
            actual_outputs = result[2]
        else:
            actual_outputs = [result.iloc[:, i] for i in range(result.shape[1])]
        assert len(actual_outputs) == len(expected_outputs)
        for actual, (result_dtype, expected) in zip(actual_outputs, expected_outputs):
            assert str(actual.dtype).lower() == result_dtype
            if endpoint != "pyo3":
                actual = actual.to_numpy(dtype=result_dtype)
            np.testing.assert_array_equal(actual, expected)

    def measure(run):
        times = {strategy: [] for strategy, _ in backends}
        for strategy, module in backends:
            select_backend(module)
            check(run())
        batch = 100 if endpoint == "pyo3" and q <= 4 else 10 if n <= 4096 else 1
        for trial in range(args.repeats):
            # Reverse the order on alternate trials, so neither library
            # consistently receives a warmer CPU or cache.
            for strategy, module in backends[:: 1 if trial % 2 == 0 else -1]:
                select_backend(module)
                # Like timeit, exclude unrelated cyclic-GC pauses. Collect
                # between samples so large pandas temporaries do not build up.
                gc.collect()
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    begin = time.perf_counter_ns()
                    for _ in range(batch):
                        result = run()
                        del result
                    elapsed = time.perf_counter_ns() - begin
                finally:
                    if gc_was_enabled:
                        gc.enable()
                times[strategy].append(elapsed / batch / 1e6)
        measurements = []
        for strategy, module in backends:
            select_backend(module)
            tracemalloc.start()
            result = run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del result
            measurements.append((strategy, statistics.median(times[strategy]), peak))
        return measurements

    fixtures = []
    for name, n, q, width in [
        ("tiny", 32, 2, 8),
        ("large", 4096, 128, 512),
        ("very_large", 65536, 512, 2048),
        ("super_large", 1048576, 4096, 2048),
    ]:
        for repeated in [False, True]:
            repeat = 8 if repeated else 1
            starts = (
                (np.arange(q, dtype=np.int64) * 7919) % ((n - width) // repeat + 1)
            ) * repeat
            ends = starts + width
            fixtures.append((name, n, starts, ends, repeat, True))
    fixtures.extend(
        [
            (
                "compact",
                1000000,
                496000 + np.arange(2000) % 4 * 2048,
                498048 + np.arange(2000) % 4 * 2048,
                1,
                False,
            ),
            (
                "many_groups",
                524288,
                np.arange(16384) // 8 * 256 + np.arange(16384) % 8 * 4,
                np.arange(16384) // 8 * 256 + np.arange(16384) % 8 * 4 + 224,
                1,
                False,
            ),
            (
                "sparse",
                1048576,
                np.array([1, 10001, 500001, 900001]),
                np.array([9, 10009, 500009, 900009]),
                1,
                False,
            ),
        ]
    )
    million_starts = (np.arange(1048576, dtype=np.int64) * 7919) % (1048576 - 2000 + 1)
    fixtures.append(
        ("million_queries", 1048576, million_starts, million_starts + 2000, 1, False)
    )
    for width in [3, 4, 5]:
        fixtures.append(
            (
                f"cutoff_{width}",
                16,
                np.array([0, 1, 2, 3]),
                np.array([16, 3, 4, 3 + width]),
                1,
                False,
            )
        )

    if args.aggregation != "sum":
        # These widths fall between the prefix and tree cutoffs: the count
        # change should switch direct scans to a packed prefix table.
        for name, n, q, width in [
            ("crossover_small", 4096, 128, 64),
            ("crossover_large", 65536, 512, 512),
            ("crossover_super", 1048576, 4096, 512),
        ]:
            for repeat in [1, 8]:
                starts = (
                    (np.arange(q, dtype=np.int64) * 7919) % ((n - width) // repeat + 1)
                ) * repeat
                fixtures.append((name, n, starts, starts + width, repeat, True))
        starts = np.arange(8192) // 4 * 256 + np.arange(8192) % 4 * 4
        fixtures.append(("crossover_groups", 524288, starts, starts + 32, 1, False))

    for name, n, starts, ends, repeat, downstream in fixtures:
        if args.case and name not in args.case:
            continue
        q = len(starts)
        spans = []
        for start, end in sorted(zip(starts.tolist(), ends.tolist())):
            if spans and start <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
        packed = sum(end - start for start, end in spans)
        total_width = int(np.sum(ends - starts))
        tree = q > 3 and total_width > 5 * packed + 2 * q * (packed - 1).bit_length()
        prefix = q > 3 and total_width > packed + 2 * q
        nulls = np.arange(n) % 10 == 0
        keys = np.arange(n, dtype=np.int64) // repeat
        dtypes = ["int64"] if args.aggregation == "count" else ["int64", "uint64"]
        for dtype in dtypes:
            # Large positive/negative payloads force wraparound. Every tenth
            # value is null; the direct NumPy oracle skips masks explicitly.
            bits = np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
            values = bits.view(np.int64) if dtype == "int64" else bits
            if name == "million_queries":
                # Constant MAX payloads give an independent closed-form
                # oracle for every row without a billion-element rescan.
                values.fill(np.iinfo(dtype).max)
                valid = ends - starts - ((ends + 9) // 10 - (starts + 9) // 10)
                expected = (
                    valid.astype(np.uint64) * np.uint64(np.iinfo(dtype).max)
                ).view(dtype)
                for row in range(0, q, 1024):
                    s, e = starts[row], ends[row]
                    assert expected[row] == np.sum(
                        values[s:e][~nulls[s:e]], dtype=dtype
                    )
            else:
                expected = np.array(
                    [
                        np.sum(values[s:e][~nulls[s:e]], dtype=dtype)
                        for s, e in zip(starts, ends)
                    ],
                    dtype=dtype,
                )
            left_index = np.arange(q, dtype=np.int64)
            right_index = np.arange(n, dtype=np.int64)
            expected_counts = (
                np.array(
                    [np.count_nonzero(~nulls[s:e]) for s, e in zip(starts, ends)],
                    dtype=np.int64,
                )
                if name != "million_queries"
                else valid.astype(np.int64)
            )
            aggregations = []
            aggfunc = []
            expected_outputs = []
            if args.aggregation != "count":
                aggregations.append((values, nulls, "sum"))
                aggfunc.append(("v", "sum"))
                expected_outputs.append((dtype, expected))
            if args.aggregation != "sum":
                aggregations.append(("*", nulls, "count"))
                aggfunc.append(("v", "count"))
                expected_outputs.append(("int64", expected_counts))

            def raw(
                left_index=left_index,
                right_index=right_index,
                starts=starts,
                ends=ends,
                aggregations=aggregations,
            ):
                return rust.equi_aggregate(
                    left_index,
                    right_index,
                    starts.copy(),
                    ends.copy(),
                    [],
                    [],
                    aggregations,
                    True,
                    False,
                )

            runs = [("pyo3", raw)]
            if downstream:
                # Shuffle physical rows so downstream timing includes sorting
                # and real preparation. Duplicate cases repeat predicate keys
                # and dataframe index labels; Rust positions remain unique.
                permutation = np.random.default_rng(246).permutation(n)
                nullable = pd.array(
                    values, dtype="Int64" if dtype == "int64" else "UInt64"
                )
                nullable[nulls] = pd.NA
                right = pd.DataFrame(
                    {
                        "g": np.zeros(n, dtype=np.int64),
                        "x": keys[permutation],
                        "y": keys[permutation],
                        "v": nullable.take(permutation),
                    }
                )
                left = pd.DataFrame(
                    {
                        "g": np.zeros(q, dtype=np.int64),
                        "lo": starts // repeat,
                        "hi": ends // repeat,
                    }
                )
                if repeat > 1:
                    left.index = np.arange(q) // repeat
                    right.index = np.arange(n) // repeat

                def join(left=left, right=right, aggfunc=aggfunc):
                    return left.join_agg(
                        right,
                        ("g", "g", "=="),
                        ("lo", "x", "<="),
                        ("hi", "y", ">"),
                        aggfunc=aggfunc,
                    )

                runs.append(("join_agg", join))

                def range_join(left=left, right=right, aggfunc=aggfunc):
                    return left.join_agg(
                        right,
                        ("lo", "x", "<="),
                        ("hi", "y", ">"),
                        aggfunc=aggfunc,
                    )

                runs.append(("range_join_agg", range_join))
            for endpoint, run in runs:
                if args.endpoint and endpoint not in args.endpoint:
                    continue
                for strategy, median, peak in measure(run):
                    sum_prefix = (
                        prefix
                        if strategy == "prefix" or args.baseline_stage == "pre-count"
                        else False
                    )
                    sum_tree = (
                        tree
                        if strategy == "base" and args.baseline_stage == "pre-sum"
                        else False
                    )
                    sum_bytes = (
                        0
                        if args.aggregation == "count"
                        else (
                            8 * (packed + 1)
                            if sum_prefix
                            else 16 * packed
                            if sum_tree
                            else 0
                        )
                    )
                    count_prefix = (
                        prefix
                        if strategy == "prefix"
                        and args.count_candidate_cutoff == "prefix"
                        else tree
                    )
                    count_bytes = (
                        8 * (packed + 1)
                        if args.aggregation != "sum" and count_prefix
                        else 0
                    )
                    sum_route = (
                        "absent"
                        if args.aggregation == "count"
                        else "prefix"
                        if sum_prefix
                        else "tree"
                        if sum_tree
                        else "direct"
                    )
                    count_route = (
                        "absent"
                        if args.aggregation == "sum"
                        else "prefix"
                        if count_prefix
                        else "direct"
                    )
                    route = count_route if args.aggregation == "count" else sum_route
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "distribution": "duplicate" if repeat > 1 else "unique",
                                "dtype": dtype,
                                "endpoint": endpoint,
                                "rows": n,
                                "queries": q,
                                "strategy": strategy,
                                "aggregation": args.aggregation,
                                "baseline_stage": args.baseline_stage,
                                "count_candidate_cutoff": args.count_candidate_cutoff,
                                "sum_route_model": sum_route,
                                "count_route_model": count_route,
                                "packed_len": packed,
                                "route_model": route,
                                "median_ms": median,
                                "python_numpy_peak_bytes": peak,
                                "rust_sum_table_bytes_model": sum_bytes,
                                "rust_count_table_bytes_model": count_bytes,
                                "rust_table_peak_bytes_model": max(
                                    sum_bytes, count_bytes
                                ),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
