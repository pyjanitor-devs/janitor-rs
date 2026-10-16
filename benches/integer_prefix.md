# Integer starts/ends prefix sums (#246)

Base: `origin/main` at `827ed2041eb03e100e4e7d4668e9249fce2469c8`.
Proposed source: the implementation in this change. No Python API changes.
Pyjanitor: default `dev` at `5b4e755e53368ea7e031055c871ef627f5466b7f`.
Environment: macOS 27.0.1 arm64, Rust 1.98.0, CPython 3.12.13,
NumPy 2.3.5, pandas 3.0.3. Both libraries were built with `maturin build --release`.

## Method

[`bench-integer-prefix.py`](../scripts/bench-integer-prefix.py) loads the two
explicit release libraries and alternates their timing order on successive
samples. Downstream modules' extension bindings are switched outside timing;
the same dataframe objects are reused. It reports medians of 21 samples, with
100 calls/sample for tiny PyO3 requests, 10 for other small cases, and one for
large cases. Cyclic garbage collection runs between samples and is disabled
during timing, as in `timeit`. Imports, fixtures, oracles, and warmup are excluded.
Result construction and destruction are included. No other tests or builds ran
concurrently with the final measurements.

The `pyo3` endpoint calls the production `equi_aggregate` with prepared boundaries,
including copies of its mutable starts/ends arrays. `join_agg` includes pandas
preparation, sorting, and result creation for equality plus two range conditions.
`range_join_agg` measures two ranges without the equality condition. Physical right
rows are shuffled before both downstream endpoints. Duplicate fixtures repeat both
predicate keys and dataframe index labels eight times; internal positions stay
unique. Both nullable `Int64` and `UInt64` payloads have 10% nulls and large values
that force wrapping.

Every fixture's outputs agree with direct masked NumPy sums. For the million-query
case, constant MAX payloads permit an independent closed-form answer for every row;
1,024 sampled rows additionally use direct NumPy sums. Correctness comparisons are
outside timing. Rust tests supply actual branch-execution evidence; benchmark
`route_model` fields predict dispatch and are not instrumentation.

## Results

All 62 final paired comparisons passed their independent result checks. The
following ratios are baseline time / proposed time; values above 1 mean faster.
Ranges span both dtypes and both label distributions. Full timings and memory
columns are in [`integer_prefix.csv`](integer_prefix.csv), `run=paired`.

| Source / workload | Queries | PyO3 speed ratio | Equi join_agg | Two-range join_agg |
| --- | ---: | ---: | ---: | ---: |
| Tiny: 32 rows, width 8 | 2 | 0.99–1.01x | 0.99–1.01x | 1.00–1.00x |
| Large: 4,096 rows, width 512 | 128 | 1.22–1.40x | 0.99–1.01x | 0.99–1.01x |
| Very large: 65,536 rows, width 2,048 | 512 | 1.19–1.44x | 1.00–1.01x | 0.99–1.01x |
| Super large: 1,048,576 rows, width 2,048 | 4,096 | 1.01–1.21x | 1.00–1.01x | 0.99–1.00x |
| Million-query case: 1,048,576 rows, width 2,000 | 1,048,576 | 1.47–1.52x | — | — |
| 2,048 separated groups, width 224 | 16,384 | 1.19–1.20x | — | — |
| Sparse million-row source, width 8 | 4 | 1.00–1.01x | — | — |
| Three cutoff fixtures, source length 16 | 4 | 0.99–1.02x | — | — |

The million-query signed call fell from **50.84 ms to 33.46 ms**; unsigned fell
from **49.25 ms to 33.57 ms**. The benefit is smaller than the issue's standalone
microbenchmark because these calls also validate, pack/sort windows, cross PyO3,
and materialize outputs. Shuffled pandas preparation dominates complete calls:
these fixtures show effectively neutral end-to-end performance, including small
measured slowdowns (largest in the main paired matrix is about 1.5%).

The compact-group signed measurement was unstable in the full run: **0.101 ms
base / 0.682 ms proposed**. An isolated 101-sample paired recheck measured
**0.0685 / 0.0586 ms** (1.17x); unsigned measured **0.0708 / 0.0617 ms** (1.15x).
Both the anomalous original and the recheck are retained in the CSV
(`run=paired` and `run=compact_recheck`). The large slowdown did not reproduce;
its cause is unverified, so no stable compact-group speedup is claimed.

Exploratory separate-process results (`run=separate_exploratory`) are also
retained. They predate the GC control and showed apparent end-to-end slowdowns
even for unchanged tiny direct paths. Alternating libraries and excluding cyclic
GC pauses reduced that variation. They are not the basis for the claims above.

## Memory

The CSV reports measured peak Python/NumPy traced allocations for each endpoint.
`tracemalloc` does not observe Rust-owned buffers. The separate sum-table columns
are exact allocation-size calculations, not measured RSS: the old sum tree uses
`16 * p` bytes and the new prefix table uses `8 * (p + 1)`, where `p` is the packed
queried length. Outputs, ranges, clusters, and span construction remain shared
costs. Trees for other requested operations are unaffected.

For a full 1,048,576-element packed source, sum scratch falls from 16 MiB to
8 MiB + 8 bytes. The compact-group fixture needs only 8,192 packed values:
128 KiB becomes 64 KiB + 8 bytes, regardless of its million-row source. The new
lower threshold can allocate a prefix where the old implementation scanned:
`cutoff_5` adds 136 bytes. Sparse/tiny direct paths allocate no sum table.

## Reproduce

Create a detached worktree at the base commit and build both revisions with the
same Python and Rust toolchains. For each checkout:

```sh
uvx maturin build --release --interpreter /path/to/python3.12 --out /tmp/wheels
```

Extract each wheel into its own directory, then run from a pyjanitor Pixi
workspace with a CPython 3.12 environment containing NumPy and pandas:

```sh
pixi run -e py312 python /path/to/proposed/scripts/bench-integer-prefix.py \
  --library /path/to/base/janitor_rs/janitor_rs.cpython-312-darwin.so \
  --compare-library /path/to/proposed/janitor_rs/janitor_rs.cpython-312-darwin.so \
  --pyjanitor /path/to/pyjanitor-dev --strategy base --repeats 21
```

The loader accepts the matching `.so` or `.dylib` on the current platform; use the
extension filename produced by that build. Omit `--compare-library` to run one
library in a separate process. `--case` and `--endpoint` restrict fixtures.

Measured wheel SHA-256 values:

- Base: `ee12a222540137322462846f78b60cb7ecfa79f74c9443e98269badecb953f00`
- Proposed: `7a598234483e3d786d6a07926f468740448018e3127b11218815252a8dd4bfbb`

The measured sum algorithms and dispatch match the current proposal. A later
review aligned non-null mask counts with the same prefix cutoff; these benchmarks
request sums only and do not measure that count optimization. The current wheel
was separately checked against the downstream compatibility suite.

## Verification and limits

- Rust test wrapper (`cargo test --no-default-features`): **106 passed**.
- Proposed wheel against pyjanitor's entire aggregation-kernel test module:
  **138 passed**.
- `cargo fmt --check`, Clippy across all targets/features with `-D warnings`, and
  `cargo bench --no-default-features --no-run`: passed.
- Benchmark script Ruff checks/format and new-document Markdown lint: passed.

These are single-machine measurements, not statistical guarantees or a promise
of the standalone reproduction's 14x speedup. Span construction still sorts
queries, and pandas preparation can dominate complete calls. RSS and native
allocator peak were not measured; Rust scratch figures above describe only the
sum table. Reverse paths, residual-filtered candidates, and float accumulation
are covered by correctness suites rather than performance claims here.
