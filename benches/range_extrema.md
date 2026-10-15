# Adaptive range extrema measurements

Measured on macOS 27.0.1 arm64, Rust 1.98.0, release profile, 2026-10-15.
Rust baseline: `5ed8b104b047f0251303024f361e6ae012aeba60`.
Proposed source: `2c7b88b` (this PR's implementation and benchmarks).
Pyjanitor: `99d9d49fcbfdd2c741604e036936da820ce4c156`,
pandas 3.0.5, NumPy 2.5.2, Python 3.12.

## Fixtures and scope

| Case | Right rows | Queries | Width |
| --- | ---: | ---: | ---: |
| tiny | 32 | 2 | 8 |
| large | 4,096 | 128 | 2,048 |
| very_large | 65,536 | 512 | 8,192 |
| super_large | 1,048,576 | 1,024 | 32,768 |
| sparse_large_column | 1,048,576 | 64 | 8 |

Windows start at `row * 7919 % (length - width + 1)`. Unique labels
use `position * 7919 % length`. Repeated helper labels use that value
modulo `min(97, length / 4)`, exercising ties. These repeated physical
labels are a helper stress case; real Python position maps remain unique.

The end-to-end script instead repeats each predicate key eight times.
Its equi fixture has one shared equality group plus two opposing range
conditions. The range fixture uses the same two range conditions.
Every result is checked against an independent NumPy pair-selection oracle.
Both first and last modes passed for all 32 end-to-end fixture/mode pairs
against each library. No pyjanitor source or installed wheel was modified.

## Rust helper timing

Criterion compares the adaptive implementation with a simple left-to-right
scan of the same labels and ranges. Input creation and result checks are
outside timing; answer allocation and destruction are included.
The reference uses slices and omits range validation, so this measures the
algorithm and guard overhead rather than a complete old wrapper call.

Values below are Criterion mean estimates for first/minimum, in microseconds.
Last/maximum is also benchmarked and checked in every fixture.

| Case | Labels | Direct µs | Adaptive µs | Direct/adaptive |
| --- | --- | ---: | ---: | ---: |
| tiny | unique | 0.0151 | 0.0158 | 0.96x |
| tiny | repeated | 0.0145 | 0.0157 | 0.92x |
| large | unique | 312.8244 | 7.8001 | 40.11x |
| large | repeated | 348.4564 | 7.8502 | 44.39x |
| very_large | unique | 2215.7618 | 99.9614 | 22.17x |
| very_large | repeated | 2824.4697 | 99.9516 | 28.26x |
| super_large | unique | 13559.3681 | 1387.3715 | 9.77x |
| super_large | repeated | 16726.2104 | 1418.7043 | 11.79x |
| sparse_large_column | unique | 0.2300 | 0.3117 | 0.74x |
| sparse_large_column | repeated | 0.2275 | 0.3025 | 0.75x |

The tiny path adds roughly one nanosecond. The sparse-column case adds
roughly 80 nanoseconds per batch (about 35% here), retaining a scan without
a whole-column tree allocation. These are real guard/view overheads; this
change does not improve every query shape. Broad first/minimum cases show
roughly 10–44x helper speedups in this run. End-to-end gains are smaller
because Python preparation and sorting remain part of the call.

The guard uses `queries > 3` and a strict workload comparison:
`total_width > 5 * span_len + 2 * queries * ceil(log2(span_len))`.
For eight queries over span 64, the cutoff is 416. The dedicated fixtures
total 415, 416 and 417 visits; only the last takes the tree branch.
Mean times below are nanoseconds, with D = direct and A = adaptive.

| Cutoff case | Labels | First D | First A | Last D | Last A |
| --- | --- | ---: | ---: | ---: | ---: |
| cutoff_below | unique | 171.8 | 165.0 | 410.6 | 560.0 |
| cutoff_below | repeated | 173.1 | 164.3 | 416.1 | 559.3 |
| cutoff_at | unique | 172.7 | 167.2 | 408.7 | 558.8 |
| cutoff_at | repeated | 178.4 | 167.1 | 408.5 | 558.7 |
| cutoff_above | unique | 171.7 | 168.2 | 408.2 | 175.6 |
| cutoff_above | repeated | 172.0 | 166.6 | 404.5 | 169.1 |

Thread-local test counters independently prove first/last tree execution
through the helper, range materializer and equi materializer. Tests also
cover thread isolation, both sides and equality at the cutoff, empty and
invalid windows, inactive signed windows, strides, ties, offset spans and
non-power-of-two trees.

## Pyjanitor end-to-end timing

Median milliseconds over seven calls after an oracle/warm-up call.
Each measurement includes pandas preparation, sorting, Rust execution
and result creation in `get_join_indices`. Fixture construction is excluded.
The tables summarize first mode; the script also measures and checks last.

| Case | Keys | Range baseline → proposed ms | Equi baseline → proposed ms |
| --- | --- | ---: | ---: |
| tiny | unique | 0.337 → 0.308 | 0.550 → 0.539 |
| tiny | repeated | 0.322 → 0.302 | 0.564 → 0.549 |
| large | unique | 0.898 → 0.502 | 1.417 → 1.026 |
| large | repeated | 0.880 → 0.498 | 1.245 → 0.761 |
| very_large | unique | 10.038 → 4.949 | 24.635 → 18.440 |
| very_large | repeated | 9.537 → 4.377 | 12.136 → 4.819 |
| super_large | unique | 102.399 → 77.792 | 386.191 → 325.491 |
| super_large | repeated | 104.709 → 79.977 | 201.082 → 144.594 |

## Memory

Peak Python/NumPy allocation traced during one separate end-to-end call,
in MiB, excluding fixture creation. This is neither RSS nor total native
memory: tracemalloc cannot observe Rust vectors. Small equi differences
reflect Python allocation noise.

| Case | Keys | Range baseline → proposed MiB | Equi baseline → proposed MiB |
| --- | --- | ---: | ---: |
| tiny | unique | 0.015 → 0.015 | 0.031 → 0.030 |
| tiny | repeated | 0.015 → 0.015 | 0.030 → 0.030 |
| large | unique | 0.265 → 0.265 | 0.391 → 0.391 |
| large | repeated | 0.265 → 0.265 | 0.337 → 0.337 |
| very_large | unique | 3.637 → 3.637 | 6.347 → 6.348 |
| very_large | repeated | 3.637 → 3.637 | 5.222 → 5.222 |
| super_large | unique | 57.075 → 57.075 | 101.269 → 101.269 |
| super_large | repeated | 57.075 → 57.075 | 87.269 → 87.269 |

For the core helper, requested Rust vector payload is exactly
`8 * queries` bytes for a direct answer on this 64-bit host.
The adaptive tree adds `16 * span_len` bytes to that answer. This is
calculated from actual fixture spans, not an RSS or allocator measurement:

| Case | Direct bytes | Adaptive bytes |
| --- | ---: | ---: |
| tiny | 16 | 16 |
| large | 1,024 | 66,512 |
| very_large | 4,096 | 1,052,448 |
| super_large | 8,192 | 16,772,800 |
| sparse_large_column | 512 | 512 |

Input range buffers, materialized outputs, allocator bookkeeping and Python
preparation are excluded from this core calculation. The production range
path additionally collects one native range pair per query; equi retains
row/range buffers reserved from the left-row upper bound. These costs
buy reuse across broad overlapping windows; they do not imply reduced
process memory.

## Reproduction

Run Criterion with the same Python/NumPy and dynamic-library environment
used by `scripts/test-rust.sh`:

```sh
cargo bench --no-default-features --bench kernels -- \
  range_extrema --warm-up-time 0.1 --measurement-time 0.3 --sample-size 10
```

The short Criterion sampling window is intended for review evidence, not a
stable cross-machine performance guarantee. Re-run with longer defaults
for capacity planning.

Build both Rust revisions as Python extensions with release optimization.
On macOS the local equivalent used here was:

```sh
cargo rustc --release --lib -- \
  -C link-arg=-undefined -C link-arg=dynamic_lookup
```

Keep a separate copy of each built library. In a matching pyjanitor Pixi
environment, run the following once per library in fresh processes:

```sh
pixi run python /path/to/janitor-rs/scripts/bench-range-extrema.py \
  --library /path/to/libjanitor_rs.dylib \
  --pyjanitor /path/to/pyjanitor --repeats 7
```

The script loads that exact library before importing pyjanitor. A full
published-wheel release validation is outside this internal optimization.
