# Issue #246 validation and review

Comparison base: `827ed2041eb03e100e4e7d4668e9249fce2469c8` (`origin/main`).

## Contract and branch evidence

Let `p` be packed queried-source length, `q` all output queries, and `w` the
saturating sum of valid widths. The new sum route is selected exactly when
`q > 3 && p > 0 && w > p + 2q`, with saturating cost arithmetic. The existing
tree predicate still governs products, extrema, and non-null counts.

| Matrix cell | Evidence |
| --- | --- |
| Width below / at / above prefix cutoff | `prefix_dispatch_thresholds_and_saturating_costs`: p=16, q=4, w=23/24/25; exact counter deltas 0/0/2 |
| Query count 3 / 4 | Same test: repeated full-width requests, exact deltas 0/2 |
| Empty source, empty batch, zero packed length, saturated costs | Dispatch test and exhaustive-window test |
| Signed and unsigned overflow and subtraction wraparound | Direct scalar-update oracle versus prefix results for all intervals in a 17-element source |
| Partial/all-null masks, positive/negative strides | Exhaustive-window test, both sum states; exact two-helper delta |
| -1 on either bound, 0, len, len+1, empty/inverted windows | Exhaustive-window test; matched, count, success, and output identities checked |
| Disjoint spans with nonzero offsets | `prefix_packed_disjoint_spans_cancel_earlier_overflow`; packed length explicitly 6; exact two-helper delta |
| i64/i32/i16/i8/u32/u16/u8 source representations | `prefix_signed_state_widens_every_integer_source_representation`; reversed sources/masks; exact one-helper delta per dtype |
| Equality-led caller | Existing duplicate-group/null/mixed-reduction test now asserts exact one-helper delta |
| Two-range caller | Overflow regression enlarged to four queries/five values; exact one-helper delta; sum and product wraparound checked |
| Product trees | Direct wrapping product reference, including zero, even factors, nulls, all-null and absent ranges |
| Float sums and other aggregation shapes | Existing Rust suite and downstream aggregation suite; float dispatch unchanged |

Counters are test-only thread-local cells, measured by before/after deltas. A
different parallel test cannot supply this test's branch evidence. Release builds
contain no counters. The actual prefix helpers, rather than a dispatch prediction,
increment them. Benchmark `route_model` fields are explicitly analytical models.

## Invariant and code-quality review

Applied `adversarial-pr-review`, then `code-review-and-quality` as a self-review.
No outstanding correctness findings. Each valid request belongs wholly to one
packed span; subtracting its two prefix entries cancels all preceding packed
values modulo 2^64. Offsets use source coordinates only for reads, and packed
coordinates only for prefix lookups. Invalid ranges leave initialized outputs
untouched. Both direct and prefix helpers overwrite valid output slots, preserving
the existing one-batch contract. Masks remain authoritative. Unsigned values never
pass through a signed accumulator.

The only production policy change is integer sum selection. Tree sum modes became
unused and were removed from the two product helpers; their multiplication and
query traversal remain equivalent to the old `product=true` specialization. New
tests live in a separate file to avoid growing the already large state module.
No dependencies, Python exports, tuple layouts, or validation order changed.
Index and allocation arithmetic retain checked Rust semantics. Shared span
construction still costs O(q log q).

## Downstream compatibility (#225)

Classification: Rust-internal performance optimization, with identical public
signatures, wrapping results, masks, result dtypes, and output metadata. No
pyjanitor implementation change or paired PR is needed. Both signed and unsigned
sums were compared with independent NumPy oracles through the Python endpoint
and downstream equality-led and two-range `join_agg`.

The proposed CPython 3.12 wheel was extracted and explicitly loaded into a fresh
Python process before importing pyjanitor at
`5b4e755e53368ea7e031055c871ef627f5466b7f` (current `origin/dev`). Its complete
`tests/functions/test_conditional_join_aggregation_kernels.py` suite passed:
**138 passed**, including forward/reverse, dtype, mask, residual, and result-shape
compatibility. No installed Rust module was substituted for the proposed wheel.

## Required checks

The final check results and measured performance are recorded in
[`benches/integer_prefix.md`](../benches/integer_prefix.md).
