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

## Linting

```sh
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

`clippy.toml` raises the `too_many_arguments` threshold to 10 for the
whole crate: every `#[pyfunction]` kernel entry point intentionally
exposes each input array/mask/length as a separate named Python-facing
argument, which is a Python API-design choice, not something to fix by
bundling arguments into an internal struct. Every other lint, and `-D
warnings` itself, still applies in full.

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
