# janitor-rs

Rust kernels behind performance-critical `pyjanitor` operations, compiled
into a Python extension module (`janitor_rs`) via [PyO3](https://pyo3.rs)
and [maturin](https://www.maturin.rs/). It's a dependency of
[pyjanitor](https://github.com/pyjanitor-devs/pyjanitor), not a standalone
tool -- everything here is called from
`janitor/functions/_conditional_join/` on the Python side.

## Building the wheel

Wheels are built and published by `.github/workflows/release.yml` via
`maturin`. To build one locally: `maturin build --release`.

## Testing

```sh
cargo test --no-default-features
```

`--no-default-features` disables the `extension-module` pyo3 feature. That
feature tells pyo3 not to link against libpython, because the real wheel
is `dlopen()`'d *by* a Python interpreter that already provides those
symbols -- a standalone `cargo test` binary has no such interpreter to
borrow symbols from, so it needs the feature off to link at all (a
[known pyo3 pitfall](https://pyo3.rs/latest/faq.html#i-cant-run-cargo-test)).
The real wheel build (via `maturin`) is unaffected -- `extension-module`
is still the *default* feature, `maturin` doesn't pass `--no-default-features`.

The tests do not start a Python interpreter, but PyO3 still needs a linkable
Python library. On macOS, Xcode's framework Python can compile successfully
and then fail at runtime with `Library not loaded: @rpath/Python3.framework`.
Point the loader at Xcode's framework directory in that environment:

```sh
DYLD_FRAMEWORK_PATH="/Applications/Xcode.app/Contents/Developer/"\
"Library/Frameworks" \
  cargo test --no-default-features
```

Tests live as `#[cfg(test)] mod tests` at the bottom of the files that
define the kernel they're testing (e.g. `src/bin_search/bin_search_lt.rs`),
next to a `pub fn <name>_core(...)` -- a plain-Rust extraction of that
kernel's algorithm, taking `ndarray::ArrayView1` instead of PyO3's
`PyReadonlyArray1`, so it needs no Python interpreter to call. The
`#[pyfunction]`-wrapped, per-dtype entry points that Python actually calls
are then thin wrappers around that one core function. Not every kernel has
been extracted this way yet -- see "What's covered" below.

### What's covered

As a foundation (issue [#21](https://github.com/pyjanitor-devs/janitor-rs/issues/21)),
one or two representative kernels are covered per family, not the whole
crate:

| Family | Kernel | Where |
| --- | --- | --- |
| Binary search | `binary_search_lt_core` | `src/bin_search/bin_search_lt.rs` |
| Comparison | `compare_start_end_core` | `src/compare/comp.rs` |
| Index building | `repeat_index_core`, `trim_index_core` | `src/index_builder.rs` |
| Aggregation | `sum_start_core`, `sum_end_core`, `sum_start_end_core` | `src/aggs/sum/sum_starts.rs`, `sum_ends.rs`, `sum_starts_ends.rs` |

Each covers: empty arrays, zero matches, duplicate values, boundary
positions, integer overflow/wraparound, and (for the aggregation kernels)
null masks. Integer and float range-sum paths also lock in the same `-1`
"no match" sentinel behavior: the range contributes zero before the signed
bound can be cast into an invalid array position. This is meant to be
extended kernel-by-kernel as other issues touch them (see "Relationship to
other issues" below) -- it is not a one-time exhaustive pass.

### How this relates to pyjanitor's own tests

`pyjanitor`'s test suite (`tests/functions/test_conditional_join.py`) already
exercises these kernels indirectly, through hypothesis-based property tests
that compare `join_agg`/`conditional_join` output against a
`pandas.merge().groupby().agg()` ground truth. That's real, valuable
coverage, but it means every kernel bug has to be diagnosed by first
reproducing it through the full Python join pipeline -- pandas, dtype
reconstruction, index building, and the Rust kernel all at once.

The tests in this repo complement that: they isolate one kernel's
algorithm and its edge cases directly, with no pandas/pyjanitor/join
machinery involved. Neither replaces the other -- pyjanitor's tests catch
integration-level regressions (wrong dtype reconstruction, wrong
aggregation dispatch, wrong null handling in the full pipeline); this
repo's tests catch kernel-level regressions (an off-by-one at a range
boundary, an overflow behavior change, a duplicate-value edge case) in
isolation, and much faster.

## Benchmarking

```sh
cargo bench --no-default-features
```

Runs `benches/kernels.rs` (a [`criterion`](https://bheisler.github.io/criterion.rs/book/)
harness) against the same `*_core` functions the unit tests cover, at a
small (100-row) and large (100,000-row) size, with no Python interpreter
or pyjanitor checkout required. The sum group also includes one tiny `u32`
suffix query over each column size. That sparse case protects cast-on-access:
an accidental whole-column widening is visible there instead of being hidden
inside an `n`-query throughput workload.

### Benchmarking a change that moves the Python/Rust boundary

Several issues here (e.g. [#26](https://github.com/pyjanitor-devs/janitor-rs/issues/26))
are about moving logic across the Python/Rust boundary -- replacing a Rust
kernel with NumPy, or vice versa. For that kind of change, a Rust-only
`cargo bench` number in isolation isn't the whole story: what matters is
the end-to-end call from Python. The process used for
pyjanitor-devs/pyjanitor#1673 (moving three integer sum kernels from Rust
to NumPy) is the worked example to follow:

1. Benchmark the kernel(s) in isolation first -- here, via `cargo bench`
   (Rust) or the equivalent in pyjanitor (NumPy/Python), at both a small
   and a large size.
2. Benchmark the real end-to-end call downstream in pyjanitor (e.g.
   `join_agg(..., aggfunc=[...])`), not just the kernel -- boundary
   crossings and index-building overhead can dominate at small sizes even
   when the kernel itself is faster in isolation.
3. Record both sets of numbers in the PR description (see pyjanitor PR
   #1673 for the format), so a reviewer can see the kernel-level and
   end-to-end pictures without having to reproduce either locally.

### Conditional-join survivor masks

Comparison kernels that receive an existing `matches` tape update it in
place. The first predicate still creates the tape; each later predicate
clears entries that fail and returns updated counts. This preserves the
flat `int8` representation and removes one full-width result allocation per
additional predicate.

The paired `compare_start_end_allocating_vs_in_place` benchmark in
`benches/kernels.rs` measures the two cores on identical inputs. In one local
run, in-place filtering was approximately 17% faster for dense masks at both
2.5M candidates (1.88 ms versus 2.27 ms) and 10M candidates (7.53 ms versus
9.16 ms). With 25% of mask entries already dead, it was approximately 21%
faster at 2.5M candidates (1.60 ms versus 2.02 ms) and 20% faster at 10M
candidates (6.38 ms versus 7.97 ms). It also reduced the per-call allocation
from 2.5 MB/10 MB to 4 KB/8 KB, with one allocation instead of two.

The same paired run for starts-only and ends-only was consistent: dense
2.5M-candidate runs improved from 2.27 ms to 1.88 ms and from 2.47 ms to
1.95 ms respectively; 25%-dead runs improved from 2.02 ms to 1.60 ms and
from 2.04 ms to 1.62 ms. These side-only shapes had the same allocation
reduction.

The nullable `!=` cores showed the same direction at 2.5M candidates. Dense
starts-only improved from 4.18 ms to 3.08 ms, ends-only from 3.67 ms to 3.08
ms, and starts+ends from 2.83 ms to 2.51 ms. With 25% dead entries, the
corresponding improvements were 3.48 ms to 2.51 ms, 3.02 ms to 2.51 ms, and
2.31 ms to 2.05 ms. Each nullable shape reduced the mask allocation from
2.5 MB/2 allocations to 4 KB/1 allocation.

Mutation is limited to masks owned by pyjanitor's internal join pipeline.
The Python caller must provide a writable, one-dimensional `int8` NumPy array
with the expected flat-tape width. Read-only arrays are
rejected by these mutable entry points; caller-owned buffers should not be
passed to them. The logical tape position still advances over dead entries,
so row ranges and their cumulative widths remain aligned.

## Linting

```sh
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

The default `too_many_arguments` threshold remains active for handwritten
functions. Macro-generated `#[pyfunction]` wrappers have a local
`#[allow(clippy::too_many_arguments)]` because their separate arrays, masks,
and flags are the Python-facing API and should not be bundled into an
internal struct solely to satisfy the linter. A small number of existing
low-level comparison and aggregation cores have the same targeted allow
because their public Rust signatures mirror those kernel inputs. New
handwritten functions remain subject to Clippy's default threshold. Every
other lint, and `-D warnings` itself, still applies in full.

## Relationship to other issues

This is foundational, low-level test/bench scaffolding -- it's meant to
land *before* the broader kernel changes already planned:
[#23](https://github.com/pyjanitor-devs/janitor-rs/issues/23) (deterministic
reverse aggregations), [#24](https://github.com/pyjanitor-devs/janitor-rs/issues/24)
(binary-search/comparison kernel improvements), [#25](https://github.com/pyjanitor-devs/janitor-rs/issues/25)
(index-builder hardening), and [#26](https://github.com/pyjanitor-devs/janitor-rs/issues/26)
(adaptive range-sum kernels) all touch kernels covered here. Expect those
PRs to update or extend these tests/benchmarks as the kernels themselves
change -- they are not meant to freeze the current implementation in
place.
