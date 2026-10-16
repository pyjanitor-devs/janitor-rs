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

All 62 main paired comparisons passed their independent result checks. The
following ratios are baseline time / proposed time; values above 1 mean faster.
Ranges span both dtypes and both label distributions. Full timings and memory
columns are in [`integer_prefix.csv`](integer_prefix.csv), `run=paired`.

| Source / workload | Queries | PyO3 speed ratio | Equi join_agg | Two-range join_agg |
| --- | ---: | ---: | ---: | ---: |
| Tiny: 32 rows, width 8 | 2 | 0.99–1.01x | 0.99–1.01x | 1.00–1.00x |
| Large: 4,096 rows, width 512 | 128 | 1.22–1.40x | 0.99–1.01x | 0.99–1.01x |
| Very large: 65,536 rows, width 2,048 | 512 | 1.19–1.44x | 1.00–1.01x | 0.99–1.01x |
| Super large: 1,048,576 rows, width 2,048 | 4,096 | superseded; see focused rechecks below | 1.00–1.01x | 0.99–1.00x |
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

A focused recheck was added after review identified a possible regression in
the super-large aggregation call. Both commits were rebuilt with the same Rust
1.98 toolchain, then compared in three independent invocations with 101
alternating samples per library; one invocation reversed library load order.
All 12 result comparisons passed their independent oracles.

| Distribution | Signed baseline/proposed | Unsigned baseline/proposed |
| --- | ---: | ---: |
| Unique keys | 0.98–1.03x | **0.91–0.96x** |
| Duplicate keys | 0.98x | **0.84–0.88x** |

The initial focused measurements identified a repeatable pre-fix u64 regression
for this shape: about 4–10% longer runtime with unique keys and 14–19% with
duplicate keys. Signed sums were near parity. A reviewer's Rust-level harness
measured 1.26x for signed sums across two runs; the pre-fix Python-to-Rust
recheck measured parity, reflecting a different measurement method. Those
results are recorded as `run=recheck_*` in the CSV.

The register-total prefix-loop patch was then tested at the Python boundary in
two independent invocations with 101 alternating samples each. All eight
comparisons passed their independent oracles:

| Distribution | Signed baseline/proposed | Unsigned baseline/proposed |
| --- | ---: | ---: |
| Unique keys | **1.68–1.73x** | **1.24–1.29x** |
| Duplicate keys | **1.55–1.57x** | **1.19x** |

The patch keeps the dispatch rules and memory model unchanged. It keeps the
running total in a local, selects each signed source dtype once per call, and
writes each prefix entry once. The new results are recorded as
`run=register_recheck_*` in the CSV; the pre-fix regression is resolved for
these Python-level workloads.

The committed harness also supports `--random-nulls`. Each fixture gets its
own reproducible mask from `np.random.default_rng([246, n, repeat])`, so the
mask does not depend on which other fixtures are selected; the million-query
oracle derives valid counts from that mask. Across 101 alternating samples, the
baseline/proposed ratios were **1.43x signed** and **1.21x u64** for unique
keys, and **1.37x signed** and **1.08x u64** for duplicate keys. All four
comparisons matched direct masked NumPy oracles. These results are recorded as
`run=random_nulls_recheck` in the CSV.

To reproduce the recorded focused run (after building the two wheels described
below), run:

```sh
pixi run -e py312 python scripts/bench-integer-prefix.py \
  --library /path/to/base/janitor_rs/janitor_rs.cpython-312-darwin.so \
  --compare-library /path/to/patched/janitor_rs/janitor_rs.cpython-312-darwin.so \
  --pyjanitor /path/to/pyjanitor --strategy base --aggregation sum \
  --repeats 101 --case super_large --endpoint pyo3 --random-nulls
```

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

## Count cutoff experiment: retain the conservative guard

Measured the review suggestion independently: baseline `c554504` already contains
integer prefix sums but keeps the existing mask-count cutoff; experimental
`0d2d5bd` changes counts to `use_prefix`. Only that count policy differs in
production behavior. The baseline has the same production count code as
`3afe8fe`, immediately before the change. Both are release wheels and use the
same alternating-library, 21-sample protocol above. These are measurements of
an experiment, **not an optimization retained in the final proposal**.

All **105 paired comparisons** passed independent result checks: 35 count-only
and 70 mixed sum/count comparisons. Count-only uses the dtype-independent mask
request; mixed requests cover signed and unsigned sums. Both distributions repeat
the earlier unique/duplicate predicate-key and index-label cases. Additional
fixtures deliberately fall between the sum-prefix and tree cutoffs.

| Source rows / query width | Queries | Count-only PyO3 baseline / experiment | Mixed sum/count PyO3 baseline / experiment |
| --- | ---: | ---: | ---: |
| 4,096 / 64 | 128 | 0.68–0.72x | 0.82–0.86x |
| 65,536 / 512 | 512 | 0.61–0.62x | 0.71–0.76x |
| 1,048,576 / 512 | 4,096 | 0.16–0.20x | 0.72–0.76x |
| 2,048 separated groups / 32 | 8,192 | 0.97x | 1.11–1.57x |

Ratios below 1 indicate a regression. Million-row count-only calls changed from
**0.231 ms to 1.166–1.491 ms**, roughly **5–6.4x slower**. Mixed calls also
regressed in the three contiguous-source shapes, although the separated-group
mixed fixture improved. End-to-end calls showed no consistent benefit; the full
CSV retains their timings, including small regressions. Tiny/sparse controls keep
direct scanning, and the broad `large` control already uses count prefixes in
both builds. The three small cutoff fixtures test both sides of the new guard.

A sum cutoff cannot assume the same per-element costs for mask counting. Direct
counting avoids storing a running total for each packed value; the measured
tradeoff differs from summing numeric payloads. This experiment does not establish
the compiler-level cause of the difference. **The count change was reverted:**
mask counts retain their previous conservative cutoff and have no new branch
instrumentation. Sum branch counters and the original 1×2 direct overflow test
alongside the 4×5 prefix test remain covered.

Unlike replacing a sum tree, this count change adds scratch allocation where the
old path scans: approximately **8 MiB** for the million-row count-only fixtures.
In mixed requests the sum already allocates a table of that size, so the model's
peak table budget is unchanged, but another count table is allocated sequentially.
These are table-size calculations, not measured RSS. Per-endpoint traced
Python/NumPy peaks and both table models are in
[`count_prefix.csv`](count_prefix.csv).

To reproduce, build the two commits above and invoke the current harness with
`--baseline-stage pre-count --count-candidate-cutoff prefix --aggregation count`
or `--aggregation sum-count` with the same stage/cutoff options. The explicit
candidate cutoff reproduces the rejected build; the default `tree` setting models
the final proposal's conservative count policy.
For the table's targeted cases, add:

```sh
--case crossover_small --case crossover_large --case crossover_super \
  --case crossover_groups
```

Also measured with `--case tiny --case large --case sparse --case cutoff_3
--case cutoff_4 --case cutoff_5` as controls. The command's `--library` is the
baseline wheel's extracted extension; `--compare-library` is the experimental
extension; use `--strategy base --repeats 21` with both. Experimental wheel
SHA-256: `b890230166840b78304bdd0c8eb9ea3ef38d3df196451e3d912e119a5925e907`.
The baseline is the earlier proposed wheel (`7a598234…`) listed below.

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

The measured sum algorithms and dispatch match the current proposal. Count
measurements above isolate the subsequently rejected lower count cutoff. The
final proposal retains the earlier mask-count policy and was separately checked
against the downstream compatibility suite.

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
