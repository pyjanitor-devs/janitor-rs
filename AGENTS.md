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
│   ├── lib.rs              # top-level composition point only -- calls each
│   │                        # family's own `register(m)` (issue #22); no
│   │                        # per-function add_function calls live here
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
4. If it's one of the representative cores in `benches/kernels.rs`,
   the core function needs `pub` (not `pub(crate)`) visibility and an explicit
   re-export from `lib.rs`'s narrow `bench_support` facade -- see
   "Non-Obvious Gotchas".

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
4. Register each dtype instantiation in that same file's own `register(m)`
   function (see "Module registration" below) -- not in `src/lib.rs`, which
   only composes each family's top-level `register` call.

### Module registration (issue #22)

Python export registration is owned by the file/module that defines the
functions, not by `src/lib.rs`. Three layers, each a `pub(crate) fn
register(m: &Bound<'_, PyModule>) -> PyResult<()>`:

1. **Leaf file** (e.g. `aggs/sum/sum_starts.rs`, `bin_search/bin_search_lt.rs`,
   `index_builder.rs`): one `m.add_function(wrap_pyfunction!(..., m)?)?;`
   per dtype variant it defines, same names/order as before -- just moved
   out of `lib.rs` into the file that owns the functions.
2. **Family `mod.rs`** (e.g. `aggs/sum/mod.rs`, `bin_search/mod.rs`,
   `compare/mod.rs`): calls `child::register(m)?;` once per `pub mod`
   child declared in that file.
3. **`aggs/mod.rs`**: an extra layer above (2) for the aggregation
   family specifically, since it has sum/sum_rev/min/min_rev/max/max_rev/
   prod/prod_rev/size_rev as its own `pub mod` children, each of which is
   itself a directory following (2).

`src/lib.rs`'s `#[pymodule] fn janitor_rs` only calls the five top-level
family registers (`bin_search`, `compare`, `index_builder`, `left_le_right`,
`aggs`) -- it never names an individual dtype-specialized function.
`index_builder.rs` and `left_le_right.rs` are single files (not
directories), so their `register` lives directly in that file with no
extra layer.

When adding a brand-new leaf file (a new kernel shape, not just a new
dtype instantiation of an existing one), remember to also add its
`pub mod <name>;` declaration to the parent `mod.rs` and a
`<name>::register(m)?;` call to that parent's own `register` function --
a new leaf file with no caller anywhere in this chain compiles fine but
silently never reaches Python.

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

### 4. `benches/` needs `rlib` and a narrow public facade

`crate-type = ["cdylib"]` alone (the historical setting) can't be linked
against by an external binary target like `benches/kernels.rs` -- Cargo
needs an `rlib` artifact for that. Also, `pub(crate) fn` isn't visible
across the crate boundary even with `rlib` added; a function a bench needs
must be `pub`. Do **not** make every implementation module in its path public:
those trees already contain hundreds of public PyO3 wrappers, so doing that
turns all of them into an accidental Rust API. Keep the trees private and
re-export only benchmark targets through `lib.rs`'s `bench_support` module.
None of this changes the *Python* surface, which remains only what
`#[pymodule] fn janitor_rs(...)` registers.

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
benchmarking the dense/adaptive-crossover behavior itself. Also keep a
large-column, single-query case for a non-`i64` dtype: an `n`-query throughput
fixture can hide an accidental O(array length) cast behind O(n * width) useful
work, while one width-eight query makes that regression obvious.

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
- **#22** (modularizing `#[pymodule]` registration) landed: `lib.rs` is now
  a ~40-line composition point, and each family owns its own `register(m)`
  (see "Module registration" above). It was unrelated to the `*_core`
  extraction pattern above; don't conflate the two when touching `lib.rs`
  or a family's `register` function.

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

### [2026-08-22] Validate sentinel bounds before dtype-specific loops

**Context**: A follow-up adversarial review found that integer range-sum
wrappers handled the `-1` sentinel while their float siblings still cast it
to `usize::MAX` and panicked.
**Learning**: Fixing a shared index contract in only one dtype macro creates
input-dependent behavior even when the public functions otherwise represent
the same operation.
**Recommendation**: Put sentinel/range validation in a helper shared by the
integer and float paths, and add a direct regression test for each distinct
loop implementation.

### [2026-08-23] Position aggregations use `-1` for zero matches

**Context**: Hardening the forward min/max `*_matches` kernels in PR #29.
**Learning**: Their result arrays were zero-initialized, so a `count == 0`
branch returned position `0`, even though pyjanitor treats `-1` as the
no-result sentinel and masks that position to a missing aggregate value.
Mixed rows with zero matches could therefore receive the first right-side
value instead of a missing value.
**Recommendation**: Initialize position-valued aggregation results to `-1`.
When skipping a zero-count row on a flat match tape, preserve `-1` while still
advancing the tape by the row's candidate-range width, and test a following
non-empty row to catch alignment regressions.

### [2026-08-23] Empty product ranges preserve the identity value `1`

**Context**: Adversarially reviewing sentinel guards for forward
`prod_positions` kernels in PR #33.
**Learning**: Zero-initializing the result and continuing on a rejected range
changed valid empty ranges and the already-safe `start == -1` case from the
established multiplicative identity `1` to `0`. A bounds guard must not silently
change the aggregation's empty-range identity while preventing a panic.
**Recommendation**: Initialize product results to `1` (for both integer and
float paths), use explicit `wrapping_mul` for integer accumulation, and test
empty, sentinel, and overflow cases whenever extracting a product core.

### [2026-08-23] ELI5/doc comments must live on the function that's actually compiled

**Context**: A follow-up adversarial review of PR #33 found the guard
rationale (why `checked_range`/`checked_index` reject a row before it's
used to index) attached to `sum_positions_core`/`prod_positions_core`,
which are `#[cfg(test)]`-only convenience wrappers around the real
`..._core_with_cast` functions the production `#[pyfunction]` entry points
actually call.
**Learning**: `cargo doc` on a normal (non-test) build never sees
`#[cfg(test)]` items, so the documentation was invisible to anyone reading
generated docs or following the production code path -- exactly the
audience ELI5 comments are for.
**Recommendation**: When a kernel is split into a `#[cfg(test)]` typed
wrapper plus an unconditionally-compiled `_with_cast`/generic
implementation (see "Adding a new dtype-generic kernel"), put the doc/ELI5
comment on the unconditionally-compiled function. The test-only wrapper
should at most doc-link to it, not duplicate or hold the only copy of the
rationale.

### [2026-08-23] `booleans` null-mask length is an unvalidated whole-call assumption

**Context**: Same PR #33 adversarial review pass: every aggregation kernel
indexes `booleans[nn]` (or `booleans[indexer_]` in the `_positions`
family) with a bound checked only against `arr.len()`, never against
`booleans.len()` itself -- 49 call sites across every kernel family
(min/max/sum/prod, forward and `_rev`), not just the files PR #33 touched.
**Learning**: This is a different bug shape than the `-1`-sentinel
guards in #27/#32: those are per-row bounds fixable with
`checked_range`/`checked_index` inside the loop. `booleans.len() ==
arr.len()` is a whole-call invariant, so validating it belongs once per
call, not sprinkled into every row's loop body -- and retrofitting it
kernel-by-kernel the way #27/#32 were fixed would be the wrong shape of
fix here.
**Recommendation**: Tracked as issue #34 rather than folded into #33 or
fixed ad hoc. Decide deliberately (validate once at the `#[pyfunction]`/
`_core` boundary, or explicitly document the caller contract if
pyjanitor's Python call sites already guarantee it) instead of leaving 49
call sites relying on an implicit, unstated assumption.

### [2026-08-23] Validate parallel-array shapes once before aggregation loops

**Context**: Issue #36 hardens aggregation entry points that accept parallel
`starts` and `ends` arrays. Their `zip` loops silently truncated to the shorter
input and could return plausible partial results.
**Learning**: Cross-array shape is a whole-call contract, not a per-row bounds
condition. Checking two already-created ndarray views with `len()` is O(1) and
keeps the validation cost outside the hot loop.
**Recommendation**: Use the shared `ensure_equal_lengths` helper once at the
PyO3 boundary and return `PyResult` so mismatches become a normal Python
`ValueError`. Do not add the comparison inside a row or candidate loop.

### [2026-08-25] Deterministic reverse output need not be key-sorted

**Context**: Reviewing a compact ordinal-vector alternative for issue #23's
reverse aggregations.
**Learning**: Issue #23 requires deterministic output with every aggregate
correctly paired to its returned index; it does not require indices to be in
ascending key order. Emitting in a stable structural order, such as the order
of a deterministic candidate-index suffix, satisfies the contract without a
sorting pass.
**Recommendation**: Do not add `O(k log k)` key sorting merely to make reverse
aggregation indices ascending. Prefer the cheapest stable emission order that
preserves each `(index, aggregate)` association unless a separate public
ordering contract explicitly requires sorted keys.

### [2026-08-25] Benchmark validation passes before fusing them

**Context**: Comparing iterator-based reverse-sum boundary validation with a
manual pass that validated inputs while computing `min_start`/`max_end`.
**Learning**: The fused manual pass was 1-7% slower across narrow and dense
`sum_rev_starts`/`sum_rev_ends` workloads, despite doing fewer conceptual
scans. Iterator-based validation and bound discovery were better optimized.
**Recommendation**: Keep validation outside the aggregation loop, but retain
the iterator form unless a representative benchmark demonstrates a benefit
from fusing validation with `min`/`max` discovery.

### [2026-08-23] Issue #34 resolved by reusing #36's `ensure_equal_lengths`, not a bespoke guard

**Context**: Issue #34's initial fix (this branch, before rebasing onto #36)
added a separate `booleans_len_ok` predicate that returned each kernel's own
"nothing computed" value (`-1` for position kernels, `0` for `sum`, and a
special-cased `1` fill for `prod`'s non-`_matches` shapes, since their
empty-range identity isn't the zero-initialized default). That special-casing
was itself a symptom: silently synthesizing a "nothing happened" result for a
whole-call contract violation forces every family to separately re-derive what
"nothing" means for its own accumulator.
**Learning**: `booleans.len() == arr.len()` is the exact same shape of
whole-call invariant as `starts.len() == ends.len()` (#36) -- `ensure_equal_lengths`
doesn't care what the two lengths represent, only that a name/length pair
matches another. Raising `PyValueError` via `?` sidesteps the identity-value
problem entirely: there is no result to synthesize because the function never
starts computing one.
**Recommendation**: Rebase/stack a whole-call-invariant fix on top of any
sibling fix already adding this kind of shared helper rather than developing
a parallel one -- check `ensure_equal_lengths` (or its successor) for a name/
length pair before inventing a new predicate. Call it once in the
`#[pyfunction]` macro wrapper, before any `_core` function is invoked, not
inside the `_core` function itself: the `_core` functions stay plain,
PyO3-free, and directly Rust-testable (`ensure_equal_lengths` is covered by
its own tests in `aggs::adversarial_bounds_tests`, generically -- there is no
need to duplicate that coverage per call site or per kernel family).

### [2026-08-23] The `_rev`/`size_rev` families never got #27/#32's per-row range guard

**Context**: Adversarial review of #37 (touching `_rev`/`size_rev` `starts_ends`/
`positions`/`starts_ends_matches` shapes to add the `starts`/`ends` length
check) found that every one of those functions casts `start`/`end` straight to
`usize` and indexes `index[item]` (or, for the `positions` shape, `positions[nn]`
then `index[indexer as usize]`) with **no** bound against `index.len()` /
`positions.len()` at all -- not even the `-1`-sentinel handling #27/#32 already
established for the forward family's identical shapes. Reproduced directly:
`compute_max_rev_start_end_int64(arr=[5], starts=[0], ends=[5], index=[0],
booleans=[False], length=1)` panics (`ndarray: index out of bounds`) because
`ends` is never checked against `index.len()`. Confirmed via `git show
b6aa541:...` that this predates #33/#36/#37 entirely; #37 just happens to edit
these exact functions without addressing it, the same way #33's review spun
out #34 rather than folding it in.
**Learning**: A negative/sentinel `start` self-protects here (casts to a huge
`usize`, so `start_..end_` becomes empty -- no panic), but a negative/sentinel
**`end`** does not: `start_=0, end_=(-1 as usize)` produces `0..usize::MAX`,
which panics on `index[0]` almost immediately. This is the crate's own
documented `-1`-sentinel-before-cast gotcha (see the entry above about
`sum_end_core`/`sum_start_end_core`/`compare_start_end_core`), just
rediscovered in a family that never got the fix. A whole-call min/max scan
over `ends` can't safely replace `checked_range` here for the same reason --
it would have to reimplement the same signed-before-cast check to be sound,
while losing the ability to skip one bad row and still compute the valid
ones.
**Recommendation**: Fixed by reusing `checked_range(*start, *end, index.len())`
(or `positions.len()` for the `positions` shape, plus `checked_index` for the
`positions[nn]` -> `index[..]` indirection) exactly as the forward family
already does, skipping the row (no `n` advancement, matching "a row rejected
here never had any of its own tape entries") rather than failing the whole
call -- this is a per-row bound, not a whole-call one, so `ensure_equal_lengths`
cannot substitute for it. Also extended the equal-length checks in these same
functions to cover `arr`/`starts`/`ends`/`booleans` (and `counts`, where
present) as one mutually-consistent set, not just the `starts`/`ends` pair
#37 added -- `izip!` over all of them would otherwise silently truncate on a
mismatch, the exact #36 failure shape. While in the file, also found and fixed
`compute_size_rev_end`/`compute_size_rev_start` sizing their output arrays
from the `length` parameter (a capacity hint) instead of `dictionary.len()`
like every sibling function -- an independent bug (an out-of-bounds *write*,
not a read) that a naive fix to only the read side would have left live.
`_rev`/`size_rev` functions have no extracted `_core` functions and no
existing `Python::attach`-based test scaffolding; validated this fix
end-to-end via `maturin develop` plus a Python script instead, per the
"Kernels not yet extracted..." guidance above -- adding that scaffolding from
scratch is a larger, separate undertaking, not part of this fix.
**Follow-up**: The `_rev`/*_no_range.rs` shape (`arr[*index_left as usize]`,
`booleans[*index_left as usize]`, indexed directly by caller-supplied
`left_index`/`right_index` values with no bound check at all) has the same
underlying problem but a different fix shape -- filed separately rather than
folded in here.

### [2026-08-23] Issues #40/#41: a flat `matches` tape needs a whole-call width check, not a per-row one

**Context**: Every kernel that consumes a flat "match tape" (`matches:
ArrayView1<i8>`, one entry per candidate position across *every* row
combined) indexed it as `matches[n]` with no check that `matches.len()`
was actually large enough for the total tape width every row's
`(start, end)`/`(0, end)`/`(start, len)` range implies -- 43 call sites
across 27 files (24 aggregation `_matches` files, `comp.rs`,
`index_builder.rs`'s 9 `matches`-consuming functions, and 3
`size_rev/computes.rs` functions). `ensure_equal_lengths` (#34/#36) can't
substitute: `matches.len()` isn't comparable to any *single* other array's
length, only to the **sum of every row's own interval width**, which isn't
known until every row has been looked at.
**Learning**: The fix shape is neither `ensure_equal_lengths` nor
`checked_range`/`checked_index` alone -- it's a new helper,
`ensure_tape_width(expected_width, matches_len)`, fed a *pre-computed sum*.
Each call site sums its own rows' widths with a small pre-pass that
mirrors whatever per-row rejection that call site's main loop already
applies (`checked_range`'s `Some`/`None`, `checked_index`, or no rejection
at all where the loop has none) -- a row that contributes zero tape
entries in the main loop must also contribute zero to the pre-pass sum, or
the check would reject calls the main loop actually handles fine.
`index_builder.rs`'s 9 functions had no `PyResult` return type or
`ensure_equal_lengths` at all before this fix (unlike every aggregation
family), so adding the check there also meant converting
`Bound<'py, PyArray1<i64>>` returns to `PyResult<Bound<'py, PyArray1<i64>>>`
-- a pure win for callers (panic becomes catchable `ValueError`) since
PyO3's calling convention is unchanged on the success path.
**Learning (perf)**: measured the pre-pass in isolation (`Instant`-timed,
not committed as a permanent bench) at 1K/200K/2M rows: ~0.5-0.75 ns/row,
constant per-row cost across three orders of magnitude -- i.e. genuinely
`O(rows)`, not accidentally `O(rows^2)`. At 2M rows/10M tape entries it
cost under 1ms against an 8.6ms full `index_starts_and_ends` call (the
fastest affected function, no `HashMap`) and 663ms for a `HashMap`-based
`_rev` aggregation -- negligible next to pre-existing work in every case.
**Learning (docs)**: `#40` found the ELI5 comment borrowed onto the
single-bound `_ends_matches` `_rev` guard ("a row rejected here never had
any of its own tape entries... See issue #34") was wrong for that shape --
it's accurate for the dual-bound `_starts_ends` shape (whose own producer,
`compare_start_end_core`, really does have an invalid-row concept), but
the single-bound producers (`comp_ends.rs`/`comp_starts.rs`) have none;
what actually keeps `n` aligned there is `bin_search_gt_first`/
`bin_search_lt_first` dropping zero-match rows before `ends`/`starts` is
ever built, a cross-module invariant the local `checked_range` call is
defense-in-depth for, not a condition the real call path can trigger.
Fixed the comment in the 4 confirmed `_rev`/`*_ends_matches.rs` files
(`sum_rev`, `max_rev`, `min_rev`, `prod_rev`) to describe that mechanism
instead. Left the equivalent forward (`max`/`min`) comments alone -- they
already attribute correctness to "the caller only emits entries for rows
it already knows are valid" rather than claiming a local invalid-row
concept, so they weren't actually wrong.
**Recommendation**: When a new kernel consumes a flat multi-row tape
indexed by a running cursor, validate the *total* expected cursor
advancement against the tape's length once, before the loop -- not by
bounds-checking the cursor inline on every iteration (which either panics
or needs its own per-element error path) and not by comparing the tape's
length to any single row-count-shaped array (wrong shape of check, per
`ensure_tape_width`'s own doc comment). Reuse `ensure_tape_width`.

**Follow-up (review of PR #43)**: 19 of the pre-pass sums above computed a
row's width as plain `end - start` (or `end_ - start_`) for the *unguarded*
shapes -- the ones whose main loop has no `checked_range`/`checked_index`
call at all, just a raw `for x in start_..end_`. That's unsafe: the `-1`
"no match" sentinel (or any `start` past `end`) casts to a huge `usize`,
and unlike a real `Range<usize>` -- whose element count is `end.saturating_
sub(start)`, i.e. simply `0` when `start >= end`, not a panic or a wrapped
value -- plain `usize` subtraction either panics (debug) or silently wraps
to a huge, wrong width (release). Confirmed via
`janitor_rs.index_starts_only(index=[10,20], starts=[-1,1],
matches=[1], length=1)`: the valid second row needs 1 tape entry, but the
wrapped width from row 0 made the precheck demand 4, rejecting an
otherwise-correct call. **Fixed** by replacing every such formula with
`.saturating_sub(...)`, which is provably identical to the main loop's own
`Range<usize>` element count for *any* input, sentinel or not -- not with
`checked_range`/`checked_index` (the reviewer's suggested fix), since the
unguarded main loops don't validate `start`/`end` against anything either;
reusing those helpers would silently add validation the existing loop
never had, a bigger behavior change than the bug needs. The 3 *guarded*
shapes that already `filter_map`/`filter` through `checked_range` before
subtracting (`comp.rs`, and any `_starts_ends_matches`/`_ends_matches`
file using it) were never affected -- the filter already excludes exactly
the rows that would underflow. **Recommendation**: when a pre-pass formula
mirrors a `for x in a..b` loop that has no explicit bounds guard, use
`b.saturating_sub(a)`, never plain `b - a` -- it is the actual definition
of a `Range<usize>`'s length and the only way to match unguarded-loop
semantics for every possible input, including cast-from-negative
sentinels.

### [2026-08-24] Modularized `#[pymodule]` registration (issue #22)

**Context**: `src/lib.rs` had grown to ~3,570 lines, almost entirely
`m.add_function(wrap_pyfunction!(<fully-qualified-path>, m)?)?;` calls, one
per dtype-specialized export (884 total across 89 leaf modules).
**Learning**: The registrations were purely mechanical -- grouped by
leaf-module path and always in the same per-dtype order as the file's
`macro_rules!` instantiations -- which made this a safe fit for a scripted
transform rather than hand-editing 90 files: extract every
`(module_path, fn_name)` pair from `lib.rs` in order, group by module
path, and generate a `pub(crate) fn register(m: &Bound<'_, PyModule>) ->
PyResult<()>` in the file that already owns those functions. Two gotchas
the first pass missed: (1) directory `mod.rs` files that previously
contained only `pub mod x;` declarations had no `use pyo3::prelude::*;`,
so the generated `register` signature didn't compile until that import was
added; (2) `left_le_right.rs`'s single export uses
`#[pyfunction(name = "get_positions_where_left_le_right")]`, so its
Python-visible name differs from the Rust item name -- a register-writing
script (or reviewer) that assumes item name == exported name will get this
one wrong.
**Recommendation**: `lib.rs` is now a ~40-line composition point calling
five family-level `register` functions (see "Module registration" above).
When adding a new leaf module, register it in its own file, wire it into
the parent `mod.rs`'s `register`, and never add a `wrap_pyfunction!` call
to `lib.rs` directly. Before assuming a Python export name matches a Rust
function name, grep the file for `#[pyfunction(name = ...)]`.

### [2026-08-24] Issue #24 (part 1): one-pass `*_first` output, no marker needed

**Context**: `binary_search_{lt,gt,ge,le}_first.rs` each searched every row
into a `left.len()`-sized `Array1` using an internal "no match" marker (`0`
or `right.len()`, whichever value that specific operator's search can
never produce as a genuine result), then made a second pass over that
array to copy only the surviving rows into exactly-sized output arrays.
**Learning**: The marker step is unnecessary. Since the loop already knows,
at the exact moment it decides a row didn't match, whether to keep it, it
can just not add it: push `(search index, left_index[i])` straight into
two grow-on-demand `Vec`s instead of writing into a full-length array and
filtering it later. Do not eagerly reserve `left.len()` for both outputs:
sparse/no-match workloads would otherwise allocate twice as much space as
the old marker array before discovering that most rows are dropped. This
also removes the marker-value bookkeeping (whether
`0` or `right.len()` is safe to reuse as "no match" isn't obvious on its
own -- it depends on that specific operator's algorithm always keeping
genuine results away from that boundary).
**Learning (perf)**: measured with a custom `#[global_allocator]` wrapper
in `benches/kernels.rs` (`count_allocations`, before/after byte delta
around a single un-criterion'd call -- criterion itself only reports
timing, not allocation): the one-pass version grows only as rows survive,
so sparse/no-match inputs do not pay for two full output buffers. The
all-match case may perform several geometric `Vec` growth allocations,
but still avoids the old full-length marker plus compacted arrays. The
benchmark should report both sparse and dense cases; a dense-only result
does not characterize memory use for the common sparse case.
**Recommendation**: When a kernel's row-processing loop already knows at
decision time whether a row survives, prefer pushing into grow-on-demand
output `Vec`s over writing a full-length array with an internal sentinel
and filtering it in a second pass -- the sentinel approach costs an extra
full scan and can retain a large marker allocation for rows that are all
dropped. This is scoped to
one of #24's three opportunities (one-pass output); the shared/validated
comparison-operator enum and the contiguous-array fast path are tracked
separately and not addressed by this entry.

### [2026-08-24] Issue #38: a raw caller-supplied index (not a range) has no natural "empty" fallback

**Context**: `max_rev/max_no_range.rs`, `min_rev/min_no_range.rs`,
`sum_rev/sum_no_range.rs`, and `prod_rev/prod_no_range.rs` each read
`index_left` straight from a caller-supplied `left_index` array and used
it to index `arr`/`booleans` (`arr[*index_left as usize]`) with no bound
check at all -- `left_index` had already gained `arr`/`booleans`
equal-length validation (`ensure_equal_lengths`, `PyResult`) from an
earlier sweep, but nothing validated `index_left` itself before using it.
While fixing this, a broader sweep for the same *shape* of gap (found
folded into this PR rather than filed as a separate issue, per explicit
direction) turned up four more:
`compare/comp_no_range.rs` and `comp_no_range_ne.rs` only guarded the `-1`
sentinel before indexing `right`/`right_booleans` by `right_pos`, never
the upper bound; `index_builder::build_positional_index` only guarded
`position < 0`, same gap; `index_builder::reorder_index` had no guard at
all (not even `-1`), on *two* chained reads (`starts`/`counts` by `val`,
then `result` by the `pos` derived from those reads).
**Learning**: This is a different shape from both #27/#32's
`start..end` range guard and #40/#41's tape-width pre-pass. A single index
read from a caller-supplied array, used directly to index another array,
has no natural "empty" fallback the way an inverted `Range<usize>` does
(see the `saturating_sub` entry above) -- there's no arithmetic trick that
makes an invalid single index safe, it must be rejected outright before
use, via `checked_index`. Where the same `right_pos`-shaped value gates
*two* separate arrays (`right`/`right_booleans` in `comp_no_range_ne.rs`),
their lengths need a one-time `ensure_equal_lengths` check so a single
`checked_index` call safely covers both, matching how `arr`/`booleans` are
validated together elsewhere in `aggs/`.
**Learning (perf)**: measured `checked_index`'s added cost directly (built
wheel, timed from Python) across 100/1M/10M rows: `compare_no_range`,
`build_positional_index`, and `reorder_index` held flat ~0.5-1.4 ns/row
across all three sizes -- the guard is genuinely O(1) per element, not a
hidden O(n) or O(n^2) cost. `max_rev_no_range`'s per-row cost grows with
`n` (15 ns/row at 1M, 50 ns/row at 10M), but that's pre-existing
`HashMap`-rehashing cost from its dictionary-based grouping as the number
of distinct keys grows, not something this fix introduced -- confirmed by
the other three (no `HashMap` involved) staying flat.
**Recommendation**: When auditing for this class of bug, grep for
`\[\*[a-zA-Z_]* as usize\]` across the whole tree, not just the specific
files a filed issue names -- greenfield sentinel-only guards (`if *x ==
-1`) are exactly as unsafe as no guard at all against a positive
out-of-range value, and are easy to mistake for "already handled" on a
skim. Prefer `checked_index`/`checked_range`/`checked_end` over a hand-
rolled comparison so the missing-upper-bound mistake can't recur.

### [2026-08-24] The `-1` sentinel isn't safe for every consumer -- check what the caller does with it

**Context**: A first pass at guarding `index_builder::reorder_index`
(issue #38 follow-up) rejected out-of-range mappings by leaving the
crate's usual `-1` "no match" sentinel in the output and returning `Ok`,
matching how most other kernels here skip a bad row. An adversarial
review of that PR caught that this specific function's sole caller
(pyjanitor) does an unfiltered `right.iloc[reordered_positions]` on the
result -- and pandas treats `-1` as the *last* row, not "no match". A
malformed mapping therefore produced a wrong-but-plausible reordered
`DataFrame` (a duplicated row) instead of an error or an obviously broken
one.
**Learning**: The `-1` sentinel convention (see the `-1` sentinel entry
earlier in this file) is safe when callers treat it as an opaque "no
value" marker -- a boolean mask, an equality check, a Python-side `!= -1`
filter. It is *not* safe wherever the crate's own output feeds directly
into positional indexing (`.iloc`, `.take`, raw pointer/offset
arithmetic) without the caller filtering `-1` out first, because those
consumers reinterpret a negative index as "count from the end" rather
than "absent". Before choosing "skip and sentinel" vs. "reject the whole
call" for a new guard, check how the Python side actually consumes the
function's output, not just what every sibling function in this crate
happens to do.
**Recommendation**: `reorder_index` now returns `PyResult` and raises
`ValueError` on any unresolvable mapping (out-of-range bucket, or an
overflowing `starts[bucket] + counts[bucket]` via `checked_add` instead
of plain `+=`, which previously could panic in debug builds and
silently wrap in release) rather than ever emitting a `-1` into a result
consumed by positional indexing. When adding a new guard, ask "does this
function's output get positionally indexed downstream without a filter
step first?" -- if yes, prefer erroring over sentinel-and-skip.

### [2026-08-24] #45's own per-row bounds fix missed the whole-call `starts`/`ends`/`counts` shape check

**Context**: Adversarial review of #45 (which added `checked_end`/`checked_range`-style
per-row guards to `index_builder.rs`) found that `index_starts_and_ends`,
`index_starts_and_ends_keep_first`, `index_starts_and_ends_keep_last`,
`build_positional_index_first`, and `build_positional_index_last` all `zip`
`starts` against `ends` (and, for four of the five, `counts` too) with no
`ensure_equal_lengths` check -- the exact whole-call shape gap #36 closed for
the `sum`/`min`/`max`/`prod` families and #34 reused rather than
special-casing. A mismatched pair silently truncates to the shorter array
instead of raising, unlike the `ensure_equal_lengths("right", ...,
"right_booleans", ...)` guard this same PR added for the analogous
`comp_no_range_ne.rs` case.
**Learning**: Landing a per-row bounds fix (#38's `checked_end`/`checked_range`
family) in a file does not automatically cover that file's whole-call
`ensure_equal_lengths` shape contract (#36) -- they are separate invariants
guarding separate failure modes, and a PR scoped to one can leave the other
unaudited in the very same functions it touches.
**Recommendation**: When a PR adds per-row (`checked_*`) guards to a function,
also check every parallel array it zips for a whole-call `ensure_equal_lengths`
guard, not just the arrays the per-row fix happens to bound. `index_builder.rs`
now calls `ensure_equal_lengths("starts", ..., "ends", ...)` (and, where
present, `"starts"`/`"counts"`) at the top of all five functions, matching the
`prod_starts_ends.rs`-style call site pattern; regression coverage lives in
`aggs::adversarial_bounds_tests::index_builder_starts_ends_functions_reject_mismatched_lengths`.

### [2026-08-25] Use formal GitHub sub-issues when requested

**Context**: A cross-repository pyjanitor coordination task for issue #23 was
initially created only as a linked standalone issue when the requested object
was a formal GitHub sub-issue.
**Learning**: GitHub supports attaching an existing issue as a formal
cross-repository sub-issue through the GraphQL `addSubIssue` mutation.
**Recommendation**: When a task is requested as a sub-issue, create or reuse
the appropriately scoped issue and attach it to the parent with `addSubIssue`;
do not rely only on textual cross-links or checklist comments.

### [2026-08-25] Treat no-range right labels as arbitrary unless guaranteed otherwise

**Context**: Planning a dense-slot replacement for the reverse `no_range` aggregation kernels.
**Learning**: `right_index` in a `no_range` kernel may contain arbitrary labels such as `[20, 40, 20]`; it cannot be used directly as a vector index merely because an output length of `3` is known.
**Recommendation**: Use direct slots only under an explicit positional-index contract. Otherwise retain a label-to-ordinal mapping or another label-compression step, and keep the original labels for output.

### [2026-08-25] Treat no-range left indices as a strict contract

**Context**: Reviewing the new `sum_rev_no_range` hybrid implementation.
**Learning**: `left_index` is required to contain valid nonnegative positions into `arr`; `-1` sentinels and out-of-range values are not valid inputs for this shape. Per-row skipping can hide an upstream contract violation.
**Recommendation**: Reject any invalid `left_index` entry and never silently skip it. For hot no-range kernels, perform that validation immediately before the corresponding `arr`/`booleans` access so the check remains one-pass; zero remains a valid index.

### [2026-08-25] Prefer one-pass no-range index validation for hot kernels

**Context**: Benchmarking the `sum_rev_no_range` hybrid implementation showed that a separate full validation pass added approximately 1–14% runtime on valid inputs.
**Learning**: A per-row explicit negative/bounds check followed immediately by indexing preserves safety and early errors without requiring a second full scan.
**Recommendation**: For hot no-range kernels, use one-pass validation with `index < 0 || index as usize >= arr.len()` before each access, unless the API specifically requires rejecting all invalid inputs before doing any aggregation work.
