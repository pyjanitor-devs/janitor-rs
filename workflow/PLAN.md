# Issue #246: wrapping prefix sums for starts/ends

Base: freshly fetched `origin/main`, `827ed20` (includes #243 / PR #250).

## Scope and contract

Use packed wrapping prefix differences for `State::Sum` and `State::SumU64`
when cheaper than direct traversal. Preserve null masks, integer result widths,
invalid/empty range identities, matched metadata, and output ordering. Products,
extrema, floating reductions, reverse aggregation, and Python APIs retain their
existing behavior. Reuse the packed union of queried spans; never scan gaps.

## Execution

1. Add the independent prefix cost guard and signed/unsigned packed helpers.
   Update comments and simplify product trees once sums no longer use them.
2. Test direct/prefix dispatch at threshold boundaries, signed/unsigned wrapping,
   masks, strides, disjoint spans, invalid boundaries, and shared callers.
   Prove route execution with isolated thread-local counters.
3. Benchmark the same fixtures against base and proposal: small through million-row
   sources, unique/repeated keys, compact groups, separated groups, signed/unsigned
   sums. Include low-level Python calls and downstream `join_agg`, correctness
   comparisons, runtime, and explicit memory measurements/estimates.
4. Run the Rust test wrapper, fmt, Clippy, and benchmark compilation. Apply the
   adversarial and code-quality review skills; inspect final diff and record
   benchmark reproduction instructions and compatibility evidence.

## Completion

All required checks pass, the prefix route has empirical execution evidence,
results agree with independent wrapping reference sums and the base, and timings
are recorded with regressions and limitations. This is a Rust-internal optimization
under #225; no Python signature or result-contract change is intended.

Status: complete locally. See [validation](VALIDATION.md) and the
[benchmark report](../benches/integer_prefix.md) for results and limitations.
