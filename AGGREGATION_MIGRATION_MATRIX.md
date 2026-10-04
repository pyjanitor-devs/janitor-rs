# Fused aggregation migration matrix

This matrix records the Rust-side aggregation surface audited for issue
[#206](https://github.com/pyjanitor-devs/janitor-rs/issues/206). It describes
the current fused architecture at commit `088db8c`; it is not a compatibility
map for deleted implementation modules.

## Supported aggregation contract

All fused aggregation endpoints use the same aggregation request parser:

| Request | Meaning | Null-mask behavior |
| --- | --- | --- |
| `(values, mask, "sum")` | Sum valid source values | `true` mask entries are skipped; integer arithmetic preserves the source-width wrapping contract |
| `(values, mask, "prod")` or `"product"` | Product valid source values | Identity is `1`; masked values are skipped |
| `(values, mask, "min")` | Position of the minimum valid value | Empty/unmatched output uses the family sentinel |
| `(values, mask, "max")` | Position of the maximum valid value | Empty/unmatched output uses the family sentinel |
| `("*", mask, "count")` | Count valid candidates | Uses only the authoritative mask |
| `("*", "size")` | Count all candidates | Ignores value nullness |

The value and mask arrays are aligned to the source layout supplied by
Pyjanitor. Rust does not infer nulls from numeric values. `return_matched`
controls whether the result includes one boolean per output slot; it does not
change which candidates contribute to an aggregation.

Supported typed inputs are `int64`, `int32`, `int16`, `int8`, `uint64`,
`uint32`, `uint16`, `uint8`, `float64`, and `float32`. Typed anchor endpoints
are instantiated for all ten dtypes. The generic multi-anchor endpoints parse
each anchor independently and therefore support mixed anchor dtypes.

## Pyjanitor path to Rust endpoint

| Pyjanitor dispatch path | Rust endpoint(s) | Direction | Window/candidate shape | Direct Rust coverage |
| --- | --- | --- | --- | --- |
| `_single_range_predicate.py` single range | `single_range_aggregate_*`, `single_range_aggregate_reverse_*` | Forward / reverse | One binary-search window per left query; reverse writes one slot per right row | `single_range_predicate::tests::single_aggregation_covers_all_operations_and_null_masks`, plus sorted-layout and reverse-alignment tests |
| `_single_range_predicate.py` range-first residual | `range_anchor_extended_aggregate`, `range_anchor_extended_aggregate_reverse` | Forward / reverse | One anchor window, then residual filtering before state updates | `single_range_predicate::tests::extended_aggregation_uses_aligned_anchor_layouts`; extended index keep tests |
| `_maybe_range_join.py` dual range | `range_join_aggregate`, `range_join_aggregate_reverse` | Forward / reverse | Intersection of two range windows | `range_join::aggregation_tests::range_aggregation_covers_all_operations_maps_nulls_and_return_modes`, `forward_two_range_aggregation_uses_intersected_windows`, `reverse_two_range_aggregation_uses_intersected_windows` |
| `_maybe_range_join.py` dual range with residuals | `range_join_extended_aggregate`, `range_join_extended_aggregate_reverse` | Forward / reverse | Intersected windows, then residual filtering | `range_join::aggregation_tests::extended_range_aggregation_filters_before_forward_and_reverse_updates` |
| `_equi_join.py` equality-led | `equi_join_aggregate` | Forward / reverse flag | Unique or duplicate equi candidates, optional range/residual filters | `equi_join::aggregation_tests` (unique, duplicate, reverse, residual, malformed and no-match cases) |
| `_not_equals_only.py` all `!=` | `not_equals_aggregate_*`, `not_equals_aggregate_reverse_*` | Forward / reverse | Null-aware non-equality traversal without candidate materialization | `not_equals_only::tests::forward_and_reverse_aggregation_use_physical_output_slots`; matched and no-matched-output tests, null and exhaustive oracle tests |
| `_not_equals_only.py` all `!=` with residuals | `not_equals_extended_aggregate_*`, `not_equals_extended_aggregate_reverse_*` | Forward / reverse | Null-aware anchor traversal, residual filtering, then aggregation | `not_equals_only::tests::extended_residuals_are_applied_before_keep` plus all-`!=` traversal tests |
| `_regions.py` dual region | `region_aggregate`, `region_aggregate_reverse` | Forward / reverse | Region sweep over two inequality anchors | `regions::aggregation_tests` (boundaries, maps, reverse alignment, operations, masks) |
| `_regions.py` dual region with residuals | `region_extended_aggregate`, `region_extended_aggregate_reverse` | Forward / reverse | Region candidates, then residual filtering | `regions::aggregation_tests::residuals_use_source_positions_after_reversal`, malformed-residual coverage |

The `*_reverse_*` endpoints aggregate left-side source values into right-side
output slots. Forward endpoints aggregate right-side source values into
left-side output slots. Every endpoint preserves physical position maps when
the right search layout is sorted or compacted.

## Behavior and boundary coverage

| Contract | Evidence |
| --- | --- |
| Integer wrapping and aggregation identities | Range aggregation overflow test; operation matrix tests in range, regions, and equi families |
| Floating-point and unsigned dispatch | Range and regions unsigned/float tests; ten-dtype macro registration |
| Authoritative null masks | Range operation matrix, regions operation matrix, and `AggregationInput` parser contract |
| `count` versus `size` | Range and regions operation matrices; both wildcard forms are parsed directly |
| `min`/`max` sentinels | Range operation matrix and empty-window boundary tests |
| Matched output and unmatched slots | Forward/reverse range, equi, region, and not-equal tests with `return_matched` both enabled and disabled |
| No-match behavior | Empty range intersections, equi no-match, not-equal exhaustive layouts, and single-range empty/no-match tests |
| Physical output alignment | Single-range sorted-layout tests, range reverse-layout tests, region map/reversal tests, and not-equal physical-slot tests |
| Residual ordering before `keep` | Range extended, equi filtered, regions extended, and not-equal residual tests |
| Duplicate labels/positions and adversarial boundaries | Equi duplicate-code tests, region duplicate-label tests, range boundary tests, and not-equal exhaustive null-layout oracle |

## Export and legacy audit

The only registered join families are the five modules called by
`src/lib.rs`: `equi_join`, `not_equals_only`, `range_join`,
`single_range_predicate`, and `regions`. Their registrations are owned by the
defining module. The old `anchor_non_equi_join`, `multi_join_indices`, and
separate legacy aggregation modules are not present in this worktree and are
not referenced by the current Rust source.

The public-wheel and cross-repository installation check remains outside this
matrix and belongs to issue
[#76](https://github.com/pyjanitor-devs/janitor-rs/issues/76). Benchmark
rebaselining is also intentionally excluded from this audit.

## Audit conclusion

No missing fused Rust endpoint was identified for the aggregation paths
currently dispatched by PyJanitor. The direct tests exercise every endpoint
family, every aggregation operation, both directions where supported, and
the important layout/null/residual boundaries. The ten typed wrappers are
macro-generated from the same implementation pattern; signed, unsigned,
`f32`, and `f64` dispatch is exercised at the family level rather than by
duplicating the full operation matrix ten times.

An exhaustive cross-product of every operation, dtype, join shape, and
residual arrangement would be redundant with the shared parser/state tests
and is not a release gate for this issue. If maintainers require that matrix,
it should be added as a separate generated-test or property-test issue rather
than presented as performance or wheel validation here.

## Verification command

On macOS installations using Xcode's framework Python:

```sh
DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/Developer/Library/Frameworks \
  cargo test --no-default-features
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```
