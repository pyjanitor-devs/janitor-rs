# AGENTS.md

This file provides guidance to LLM agents working with code in this repository.
It serves as the agent's "constitution" for `janitor-rs` development.

---

## Agent Constitution

**Before reviewing any PR in this repo** (whether asked directly, via a
`/code-review`-style tool, or by forking a review subagent), read this
file in full first, and make sure the review itself -- including any
subagent or forked reviewer -- has it in context, not just the diff. The
first review of PR #28 missed a real bug (see "Learned Patterns" and the
adversarial-review Core Principle below) partly because this file didn't
yet contain the crate's sentinel-value convention or the adversarial-input
expectation; a reviewer with no access to this file is reviewing blind to
exactly the kind of contract this crate depends on.

### Self-Improvement Protocol

**CRITICAL RULE**: This file is a living document. Agents MUST update it when:

1. **User Corrections**: If the user corrects you on anything, immediately record
   the correction in this file (AGENTS.md) in an appropriate section, then
   continue with what you were doing, applying the correction.
2. **Discovered Patterns**: If you discover a pattern, convention, or gotcha not
   documented here while working on the codebase, add it to the appropriate
   section (the "Non-Obvious Gotchas" section exists specifically for this).
3. **Command Updates**: If a command changes, is deprecated, or a better
   alternative exists, update the Commands section.

**How to Update**: Add new learnings to the `## Learned Patterns` section at
the bottom of this file.

### Core Principles

- **Read before edit**: understand the existing macro pattern in a file before
  changing it -- nearly every kernel family here is one `macro_rules!` block
  instantiated per dtype, not hand-written per dtype.
- **Minimal changes**: this crate has ~100 near-identical macro-generated
  functions; don't "fix" all of them when the task only needs one.
- **Test after every change**: `cargo test --no-default-features` (see below
  for why the flag is required).
- **Preserve numerical contracts**: overflow/wraparound behavior, null-skip
  semantics, and dtype-widening rules here are load-bearing for pyjanitor's
  correctness, not incidental implementation details. See "Non-Obvious
  Gotchas".
- **ELI5 code comments, generously**: any non-obvious logic gets a `///
  ELI5: <plain-language analogy>` line (see existing examples throughout
  `src/`) - not just on the one line of code that's tricky, but on
  invariants, sentinel values, and *why a guard is safe* wherever a future
  reader would otherwise have to re-derive it. Err on the side of adding
  one rather than assuming the surrounding code is self-explanatory: this
  crate's failure modes tend to be exactly the kind of off-by-one/cast/
  sentinel subtlety that's obvious for five minutes after you fix it and
  opaque forever after. This applies to doc comments on public functions
  and to inline comments on guards/branches alike.
- **Review every PR adversarially, not just for plausibility**: a review
  that asks "does this look right?" is not the same review as one that
  asks "what input breaks this?" The first code review of PR #28 (8
  independent agent angles plus a manual pass) reported "no correctness
  bugs found" and missed a real one: `sum_end_core`, `sum_start_end_core`,
  and `compare_start_end_core` all cast `start`/`end` to `usize`
  unconditionally, and this crate's own established `-1` "no match"
  sentinel (already guarded in `binary_search_lt_core`) casts to
  `usize::MAX` and walks the loop off the end of the array -- see the
  `-1` sentinel entry in "Learned Patterns" for the full story. It was
  only found once someone deliberately tried the sentinel value against
  each new `_core` function, rather than reading the code and judging it
  self-consistent. Concretely, for every `_core` function touched or
  added: explicitly try `-1`/sentinel values, `0`, the exact boundary
  (`start == len`, `start == end`), and values one past whatever the
  "obviously safe" case is -- for each *input*, not just the ones the
  existing tests already cover. Trust a "no bugs found" review only as
  far as the adversarial inputs it actually tried.

---

## Project Overview

`janitor-rs` is a compiled PyO3 extension module (`janitor_rs`) that
[pyjanitor](https://github.com/pyjanitor-devs/pyjanitor) depends on for
performance-critical kernels behind `conditional_join`/`join_agg` -- binary
search, ragged comparisons, index building, and aggregation (sum/min/max/
prod/size, each in a "forward" and "reverse" flavor). It has **no standalone
purpose**; every exported function exists to be called from
`janitor/functions/_conditional_join/` in pyjanitor.

**Key structural fact**: almost every file under `src/` follows the same
shape -- a `macro_rules!` block defines one `#[pyfunction]` body once, then
that macro is instantiated once per supported dtype (`int64`, `int32`,
`int16`, `int8`, `uint64`, `uint32`, `uint16`, `uint8`, `f32`, `f64`).
Integer and float variants are usually separate macros (`generic_compute` vs
`generic_compute_floats`) because floats need Neumaier/Kahan compensated
summation, ints don't.

---

## Development Environment

- **Toolchain**: stable Rust (`rustup`), no nightly features used.
- **Build tool for the actual wheel**: [`maturin`](https://www.maturin.rs/)
  (`.github/workflows/release.yml`), not plain `cargo build`.
- **No Python process or pyjanitor checkout is needed for direct Rust tests** --
  see "Testing" below. PyO3 still needs a linkable Python library; macOS may
  require the framework path documented under "Non-Obvious Gotchas". You
  only need to run Python/pyjanitor to validate an end-to-end boundary change
  (dtype dispatch changes, new exported functions, signature changes).

---

## Commands Reference

| Task | Command |
| --- | --- |
| Run kernel unit tests | `cargo test --no-default-features` |
| Run benchmarks | `cargo bench --no-default-features` |
| Compile-check benches (fast) | `cargo bench --no-default-features --no-run` |
| Lint | `cargo clippy --all-targets --all-features -- -D warnings` |
| Format check | `cargo fmt --check` |
| Format | `cargo fmt` |
| Build the real wheel locally | `maturin build --release` |

See `README.md` for the full explanation of each command and why the
`--no-default-features`/`--all-features` split exists (short version: it's
the `extension-module` PyO3 linker gotcha below).

---

## Project Structure

```text
janitor-rs/
├── src/
│   ├── lib.rs              # #[pymodule] registration -- one add_function
│   │                        # call per exported dtype variant (huge, ~2000
│   │                        # lines; issue #22 tracks modularizing this)
│   ├── bin_search/          # binary search kernels (lt, gt, ge, le, ...
│   │                        # x first/regions variants)
│   ├── compare/              # ragged comparison kernels (op-coded: 0=`>`
│   │                         # 1=`>=` 2=`<` 3=`<=` 4=`==` 5=`!=`)
│   ├── index_builder.rs      # index-building helpers (repeat_index,
│   │                         # trim_index, build_positional_index, ...)
│   ├── left_le_right.rs      # positions where left region <= right region
│   └── aggs/
│       ├── sum/ sum_rev/     # forward and reverse sum kernels
│       ├── min/ min_rev/     # ditto for min
│       ├── max/ max_rev/     # ditto for max
│       ├── prod/ prod_rev/   # ditto for prod
│       └── size_rev/         # size (count) kernels, reverse-only
├── benches/kernels.rs        # criterion benchmarks for the extracted
│                             # `*_core` functions (see below)
├── clippy.toml                # too-many-arguments-threshold override
└── .github/workflows/
    ├── release.yml            # builds/publishes the wheel (maturin)
    └── ci.yml                 # cargo test/clippy/fmt/bench --no-run
```

---

## Development Patterns

### The `*_core` extraction pattern (issue #21)

Most kernel logic lives *inline* inside a `macro_rules!` body, operating on
PyO3 types (`PyReadonlyArray1`) that need a live Python interpreter to
construct -- that's why, historically, none of it had direct Rust tests.

Where a kernel has been given test/benchmark coverage, the pattern is:

1. Extract the algorithm into a `pub fn <name>_core(...)` that takes
   `numpy::ndarray::ArrayView1`/`Array1` (no PyO3 types at all).
2. The `#[pyfunction]` macro body becomes a thin wrapper: `.as_array()` then
   a call to the core function. For integer aggregation kernels, preserve
   cast-on-access semantics inside the queried ranges; do not use `.mapv()`
   to widen the whole input column before a potentially tiny range query.
3. Add `#[cfg(test)] mod tests` at the bottom of the same file, testing the
   core function directly.
4. If it's one of the four representative kernels in `benches/kernels.rs`,
   the core function needs `pub` (not `pub(crate)`) visibility, and its
   module needs `pub mod` in `lib.rs` -- see "Non-Obvious Gotchas".

**Not every kernel has been extracted this way** -- as of issue #21, only
one or two representative kernels per family have (`binary_search_lt_core`,
`compare_start_end_core`, `repeat_index_core`/`trim_index_core`,
`sum_start_core`/`sum_end_core`/`sum_start_end_core`). Extending this to
other kernels is expected to happen naturally as other issues touch them
(see `README.md`'s "Relationship to other issues"), not as one large sweep.

### Adding a new dtype-generic kernel

1. Write the `_core` function first, generic or `i64`-only as appropriate
   (see existing examples for the pattern). Comment its non-obvious logic
   per the "ELI5 code comments" Core Principle above (null-skip semantics,
   why an inverted range is free, why a flat `matches` tape still advances
   on a skip, etc.).
2. Write `#[cfg(test)]` tests covering: empty input, zero matches,
   duplicate values, boundary positions (start=0, start=len, start>end),
   and -- for aggregation kernels specifically -- null masks and integer
   overflow/wraparound.
3. Wrap it in the `macro_rules!` per-dtype boilerplate, same as existing
   siblings in the file.
4. Register each dtype instantiation in `src/lib.rs`'s `#[pymodule]`
   function (follow the existing `wrap_pyfunction!` pattern nearby).

---

## Non-Obvious Gotchas

### 1. `extension-module` breaks `cargo test`/`cargo bench` linking

PyO3's `extension-module` feature deliberately avoids linking against
libpython, because the shipped `.so`/`.pyd` is `dlopen()`'d *by* a Python
interpreter that already provides those symbols. A standalone `cargo test`
or `cargo bench` binary has no such interpreter, so with the feature on,
linking fails with "symbol(s) not found" for the Python C API (`__Py_Dealloc`,
`_PyUnicode_...`, etc.) -- a
[documented pyo3 pitfall](https://pyo3.rs/latest/faq.html#i-cant-run-cargo-test).

Fix in this repo: `extension-module` is a `default` Cargo feature (see
`[features]` in `Cargo.toml`), not an unconditional dependency feature.
Always run:

```sh
cargo test --no-default-features
cargo bench --no-default-features
```

`cargo clippy` is unaffected (it type-checks, it doesn't link), so
`cargo clippy --all-targets --all-features` is correct and exercises the
exact feature set the real wheel builds with.

### 2. `clippy` doesn't fully lint inside `#[pyfunction]`-macro bodies

Discovered while extracting `sum_start_end_core`: the exact same
`starts.into_iter().zip(ends.into_iter())` `useless_conversion` pattern was
**not** flagged by `cargo clippy --all-targets --all-features -- -D
warnings` while it lived inline inside the `#[pyfunction]`-decorated macro
body, but **was** flagged the moment it was moved into a plain `pub fn`.
Don't assume a clean `cargo clippy` run means macro-body code is actually
lint-clean -- extracting logic into a plain function (per the `*_core`
pattern above) is the only way to get full clippy coverage on it. This is
also a real, incidental benefit of doing the extraction, not just a
testing nicety.

### 3. Make intentional integer overflow explicit

The published wheel uses two's-complement wraparound for integer aggregation,
matching NumPy `i64` arithmetic. Encode that contract locally with
`wrapping_add`/`wrapping_sub`; do not disable overflow checks for the whole
test profile. Keeping Rust's default checked test arithmetic lets unrelated
index, counter, and allocation-length overflow fail loudly while the intended
aggregation behavior stays identical in debug, test, and release builds.

### 4. `benches/` needs `rlib` crate-type and `pub mod`, not just `mod`

`crate-type = ["cdylib"]` alone (the historical setting) can't be linked
against by an external binary target like `benches/kernels.rs` -- Cargo
needs an `rlib` artifact for that. Also, `pub(crate) fn` isn't visible
across the crate boundary even with `rlib` added; a function a bench needs
must be `pub`, and every module in its path (down from `lib.rs`) must be
declared `pub mod`, not `mod`. Neither of these expose anything new to
*Python* -- the Python-facing surface is only ever what `#[pymodule] fn
janitor_rs(...)` registers in `lib.rs`, unchanged by this.

### 5. macOS-only: Xcode's framework Python needs an explicit rpath locally

Purely a local macOS dev-machine issue, not a CI concern (CI runs on
`ubuntu-latest`, no framework Python): if `python3` resolves to Xcode's
bundled framework Python (`which python3` → `/usr/bin/python3`, version
tied to Xcode), `cargo test --no-default-features` can compile and link
but then fail at **runtime** with `Library not loaded: @rpath/Python3
.framework/...`. Work around it locally with:

```sh
DYLD_FRAMEWORK_PATH="/Applications/Xcode.app/Contents/Developer/Library/Frameworks" \
  cargo test --no-default-features
```

### 6. Aggregation benchmarks: bound the interval width, don't scale it with `n`

The forward `sum_*` kernels here are literally the O(sum of interval
widths) algorithm that pyjanitor-devs/pyjanitor#1673 replaced (on the
Python side) with an O(n + m) NumPy prefix sum, and janitor-rs#26 proposes
doing the same on the Rust side for the dense case. A "large" benchmark
fixture whose interval width scales with `n` (e.g. `starts[i] = i % n`,
averaging width n/2) makes total work O(n²) -- at n=100,000 that's
billions of element-touches and the benchmark effectively hangs. Keep
per-row width bounded by a small constant regardless of `n` (see
`SUM_BENCH_WIDTH` in `benches/kernels.rs`) unless you're deliberately
benchmarking the dense/adaptive-crossover behavior itself.

---

## Relationship to Other Repos and Issues

- **pyjanitor** (`pyjanitor-devs/pyjanitor`) is the only consumer of this
  crate. A signature change, dtype-dispatch change, or export removal here
  requires a coordinated pyjanitor-side change and release -- see
  pyjanitor-devs/pyjanitor#1648's acceptance criteria for the pattern
  ("remove the superseded ... exports only after the pyjanitor change is
  released or the dependency transition is otherwise coordinated").
- Several open issues here touch the same kernels this file's patterns
  apply to: **#23** (deterministic reverse aggregations -- note `*_rev`
  kernels currently emit via `HashMap` iteration, which Rust intentionally
  randomizes; don't write a test asserting a specific iteration order for
  those until #23 lands), **#24** (binary-search/comparison kernel
  improvements), **#25** (index-builder hardening, unsafe-cast/length
  contract), **#26** (adaptive dense/sparse range-sum kernels). Expect
  those PRs to extend or modify the `*_core` functions and tests this file
  describes -- they are foundation, not a frozen contract.
- **#22** (modularizing `#[pymodule]` registration in `lib.rs`) is
  unrelated to the `*_core` extraction pattern above; don't conflate the
  two when touching `lib.rs`.

---

## Learned Patterns

<!--
This section is for agents to record new learnings.
Add entries in the format:

### [Date] Learning Title

**Context**: What you were doing
**Learning**: What you discovered
**Recommendation**: How to apply this learning
-->

### [2026-08-22] Foundation PR for issue #21

**Context**: Adding the first Rust unit tests, benchmarks, and CI to this
crate, which previously had none.
**Learning**: All six gotchas in "Non-Obvious Gotchas" above were discovered
in the course of this one PR -- they're exactly the kind of thing that's
expensive to rediscover, so they're recorded here rather than only in the
PR description.
**Recommendation**: Read "Non-Obvious Gotchas" before touching build
config, adding a benchmark, or writing a test that checks overflow
behavior in this crate.

### [2026-08-22] Preserve sparse range costs when extracting integer sums

**Context**: Reviewing the integer `sum_*` core extraction in PR #28.
**Learning**: Widening an entire input with `.mapv(|v| v as i64)` before
calling a range core changes a sparse query from O(queried width) work to
O(array length) work and allocates a full-size temporary array.
**Recommendation**: Pass the original typed view into a generic internal
loop and cast only values visited by the requested ranges. Keep regression
tests that count conversions for tiny prefix, suffix, and interval queries.

### [2026-08-22] Scope intentional overflow semantics locally

**Context**: Re-reviewing the test-profile configuration in PR #28.
**Learning**: Disabling overflow checks for the entire test profile makes
aggregation wraparound tests pass but also hides accidental overflow in every
other kernel and future test.
**Recommendation**: Use explicit wrapping operations only where modular
arithmetic is part of the contract, and retain Rust's checked test arithmetic
everywhere else.

### [2026-08-22] `-1` sentinel must be checked before, not after, the `usize` cast

**Context**: Adversarial review of PR #28 found `sum_end_core`,
`sum_start_end_core`, and `compare_start_end_core` cast `start`/`end` to
`usize` unconditionally, then relied on a post-cast `start_ >= end_` check
to catch invalid ranges "for free". A lone `-1` (this crate's established
sentinel, already guarded in `binary_search_lt_core`) casts to
`usize::MAX`, which is *larger* than any real `start`, so the post-cast
check never fires and the loop walks off the end of the array.
**Learning**: A doc comment claiming a range check is "free" because Rust
ranges handle `start >= end` is only true if both bounds were compared
*before* any lossy/reinterpreting cast. Checking the same-looking
condition after casting to `usize` can silently invert it for sentinel
values.
**Recommendation**: Guard `start == -1 || end == -1 || start >= end` in
`i64` space, before either bound is cast to `usize` - the same pattern
`binary_search_lt_core` already uses. When extracting or reviewing a new
`_core` function that takes `start`/`end` as `i64`, check for this pattern
explicitly rather than trusting that an existing doc comment's safety
claim was verified against the sentinel case.

### [2026-08-22] Generalized the ELI5-comment convention to a Core Principle

**Context**: The "ELI5 code comments" guidance previously lived only inside
"Adding a new dtype-generic kernel," framed as something to do when writing
a brand-new kernel.
**Learning**: The same guidance applies just as much to fixes, guards, and
refactors on existing code - e.g. the sentinel-cast fix above needed exactly
this kind of comment, and had none before this review. Scoping the
convention to "new kernels only" left every other kind of change without
the same expectation.
**Recommendation**: Keep ELI5-comment guidance as one canonical Core
Principle, referenced (not restated) from narrower sections like "Adding a
new dtype-generic kernel." If you find another place in this file
restating it, consolidate rather than adding a third copy.
