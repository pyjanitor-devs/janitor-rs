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

For a reproducible local environment, use the project wrapper instead. It
creates a `uv`-managed Python 3.12 environment and points PyO3 at that
interpreter so macOS does not accidentally select the Xcode
Command Line Tools Python framework:

```sh
./scripts/test-rust.sh
```

Additional Cargo test arguments are forwarded by the wrapper, for example:

```sh
./scripts/test-rust.sh single_range_predicate
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

Tests live as `#[cfg(test)] mod tests` at the bottom of the module that owns
the endpoint. The fused join tests use `Python::attach` because they exercise
the actual PyO3 boundary, typed dispatch, result tuples, and Python-facing
validation contracts. Plain Rust helpers are tested directly where a helper
has been extracted from that boundary.

### What's covered

The current fused join families are tested in the modules that define their
PyO3 endpoints. The suite covers range windows, equality groups, null-aware
`!=` traversal, region sweeps, residual filtering, physical position maps,
matched masks, no-match results, aggregation identities, null masks, integer
wrapping, and representative unsigned and floating-point inputs. The
operation and endpoint matrix is maintained in
[`AGGREGATION_MIGRATION_MATRIX.md`](AGGREGATION_MIGRATION_MATRIX.md).

The matrix is deliberately explicit about where coverage is direct and where
macro-generated dtype variants share the same typed implementation. It is an
audit record, not a promise that deleted optimization modules remain part of
the crate.

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

## Range-first Python/Rust contract

The range-first conditional-join family is split deliberately across the two
repositories. `pyjanitor` owns pandas preparation and `janitor-rs` owns typed
candidate traversal and aggregation:

1. Python removes null anchor values and sorts the right range values.
2. Python keeps the original physical row positions beside every prepared
   value. Sorting changes the search order, not the position identity.
3. The Rust index kernel uses the sorted values to produce half-open windows
   and evaluates residual predicates against compact offsets.
4. Rust returns physical left/right positions, so Python can recover the
   original dataframe rows without reconstructing the sorted layout.

For range-first aggregation, Python passes the compact anchor arrays plus the
full left/right source lengths. Aggregation columns remain in their original
full layouts. Rust translates each surviving compact candidate through the
anchor position arrays before updating the accumulator. Forward aggregation
uses physical right positions as source slots and physical left positions as
output slots; reverse aggregation swaps those roles.

This distinction is important because a sorted right value offset is not a
valid index into an original right aggregation column. The position arrays are
therefore part of the ABI, not an optimization hint. They must remain unique
physical positions and must never be replaced with sorted offsets or compact
array positions.

The range-first extended endpoints use separate physical position maps. The
first predicate is a three-field anchor, followed by any residual predicate
tuples:

```text
(left_values, right_values, operator)
(residual_left_values, residual_right_values, residual_operator)
```

`left_index` and `right_index` are separate endpoint arguments. They contain
physical positions in the original Python layouts, while the value arrays
may be compact or sorted. Aggregation source arrays use the aligned compact
layout expected by the endpoint. Residual predicate tuples are evaluated
before aggregation and before `keep` selection, so `first` and `last` always
refer to surviving candidates rather than the unfiltered range window.

## Benchmarking

Benchmark rebaselining is intentionally outside issue #206. This audit records
observable correctness and endpoint coverage; it makes no release-profile or
performance claim.

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
