"""Measure pyjanitor range/equi joins against an explicitly selected Rust library.

Run once for a baseline build and once for the proposed build in fresh processes.
Pass the matching pyjanitor checkout and a release ``libjanitor_rs.dylib`` (or
extension ``.so``) explicitly so an unrelated installed wheel cannot be timed.
Fixtures and the NumPy oracle are built outside timing. Each JSON line reports
median runtime and peak traced allocation for one complete ``get_join_indices``
call, including pandas preparation, sorting, Rust selection and result creation.
Tracemalloc does not see Rust-owned buffers; those must be reported separately.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
import warnings
from functools import partial
from pathlib import Path


def main() -> None:
    """Load the selected library and compare each join with a NumPy oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--pyjanitor", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    loader = importlib.machinery.ExtensionFileLoader(
        "janitor_rs", str(args.library.resolve())
    )
    spec = importlib.util.spec_from_loader("janitor_rs", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["janitor_rs"] = module
    loader.exec_module(module)
    sys.path.insert(0, str(args.pyjanitor.resolve()))

    import numpy as np
    import pandas as pd
    from janitor import get_join_indices

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    for name, length, queries, width in [
        ("tiny", 32, 2, 8),
        ("large", 4096, 128, 2048),
        ("very_large", 65536, 512, 8192),
        ("super_large", 1048576, 1024, 32768),
    ]:
        for repeated in [False, True]:
            # Repeated predicate keys are valid, but physical row positions
            # stay unique. Each key is repeated eight times in this fixture.
            values = (np.arange(length, dtype=np.int64) * 7919) % length
            if repeated:
                values //= 8
            domain = length // 8 if repeated else length
            key_width = max(width // 8, 1) if repeated else width
            starts = (np.arange(queries, dtype=np.int64) * 7919) % (
                domain - key_width + 1
            )
            left = pd.DataFrame(
                {
                    "lo": starts - 1,
                    "hi": starts + key_width,
                    "group": np.zeros(queries, dtype=np.int64),
                }
            )
            right = pd.DataFrame(
                {"x": values, "y": values, "group": np.zeros(length, dtype=np.int64)}
            )
            for kind in ["range", "equi"]:
                conditions = [("lo", "x", "<"), ("hi", "y", ">")]
                if kind == "equi":
                    conditions.insert(0, ("group", "group", "=="))
                for keep in ["first", "last"]:
                    run = partial(get_join_indices, left, right, *conditions, keep=keep)

                    # The oracle is deliberately independent of range windows,
                    # binary searches and the adaptive extrema implementation.
                    expected = []
                    for lo, hi in zip(left.lo, left.hi):
                        matches = np.flatnonzero((values > lo) & (values < hi))
                        expected.append(matches[0] if keep == "first" else matches[-1])
                    result = run()
                    np.testing.assert_array_equal(
                        result["left_index"], np.arange(queries)
                    )
                    np.testing.assert_array_equal(result["right_index"], expected)
                    times = []
                    for _ in range(args.repeats):
                        before = time.perf_counter()
                        result = run()
                        times.append(time.perf_counter() - before)
                        del result
                    tracemalloc.start()
                    result = run()
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    del result
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "distribution": "repeated" if repeated else "unique",
                                "kind": kind,
                                "keep": keep,
                                "median_ms": statistics.median(times) * 1000,
                                "python_numpy_peak_bytes": peak,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
