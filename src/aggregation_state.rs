//! Shared aggregation state for fused conditional-join comparisons.
//!
//! The comparison kernels own candidate traversal. They report successful
//! `(source_position, output_position)` pairs to [`AggregationSet`], which
//! keeps one accumulator per requested operation and emits one result per
//! output position.
//!
//! The state is deliberately unaware of whether a comparison is forward or
//! reverse. In a forward join, the source is a right-side candidate and the
//! output is a left-side row. In a reverse join, the source is a left-side row
//! and the output is a right-side position. Keeping that distinction in the
//! comparison-path files avoids duplicating dtype dispatch and operation
//! semantics here.
//!
//! The three reverse range-only methods near the middle of this file are a
//! slightly different entry point. Their comparison work has already been
//! completed by the caller, so they receive one boundary (or one pair of
//! boundaries) per source row and write directly into dense right-side slots.
//! Keeping those methods here is intentional: the dtype dispatch, null-mask
//! contract, identities, and Kahan compensation are the same state concerns
//! used by the comparison-driven methods, while the traversal strategy is
//! specific to starts-only, ends-only, or starts/ends ranges.
//!
//! The Python boundary normalizes numeric inputs before they reach this state:
//! signed integer `sum` and `prod` inputs are `int64`, unsigned inputs are
//! `uint64`, and floating inputs arrive as `f32` or `f64`. Integer accumulators
//! therefore use the pandas result widths, and floating states preserve the
//! input `f32` or `f64` width through materialization. Position, length, and
//! allocation arithmetic remains checked and must not wrap.

use super::aggregation_input::{AggregationInput, AggregationOp};
use numpy::ndarray::{Array1, ArrayView1};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aggregation_common::adaptive::{
    should_use_prefix_difference, should_use_running_aggregation, should_use_segment_tree,
    MAX_DIRECT_QUERY_COUNT,
};
use crate::aggregation_common::ensure_equal_lengths;

/// A borrowed view of one supported NumPy dtype.
///
/// The enum is the Rust equivalent of a tagged union: the tag records which
/// concrete numeric type is present, and the matching branch lets the hot
/// comparison loop read values without allocating or converting through
/// Python. The source array remains borrowed for the lifetime of the view.
enum Values<'a> {
    I64(ArrayView1<'a, i64>),
    I32(ArrayView1<'a, i32>),
    I16(ArrayView1<'a, i16>),
    I8(ArrayView1<'a, i8>),
    U64(ArrayView1<'a, u64>),
    U32(ArrayView1<'a, u32>),
    U16(ArrayView1<'a, u16>),
    U8(ArrayView1<'a, u8>),
    F64(ArrayView1<'a, f64>),
    F32(ArrayView1<'a, f32>),
}

/// The source values and their null metadata for one requested aggregation.
///
/// The value array is expected to be null-free; null tracking is supplied
/// separately by `nulls`. A `true` entry means the corresponding value is
/// null, while `false` means it is valid. The mask is authoritative: this code
/// does not inspect a value or infer nullness from sentinels or special values.
/// Null metadata is used by value-based operations (`sum`, `product`, `min`,
/// `max`, and column-based `count`). Count-all/`size` intentionally counts
/// the comparison event regardless of this mask.
struct View<'a> {
    values: Values<'a>,
    nulls: ArrayView1<'a, bool>,
}

/// Runtime state for one operation, including its output buffer.
///
/// Each variant has a fixed output type dictated by the aggregation contract:
/// signed values use `i64`, unsigned values use `u64`, floating-point values
/// preserve `f32`/`f64`, and positions/counts use `i64`.
enum State<'a> {
    // A state variant stores both the typed source view and the output
    // buffer. Keeping them together prevents an update from accidentally
    // pairing an operation with a source array of the wrong dtype.
    Sum(View<'a>, Vec<i64>),
    SumU64(ArrayView1<'a, u64>, ArrayView1<'a, bool>, Vec<u64>),
    SumF32(View<'a>, Vec<f32>, Vec<f32>),
    SumF64(View<'a>, Vec<f64>, Vec<f64>),
    /// Reference the set-level count-all buffer; no value array or mask is
    /// needed. The buffer is shared so repeated count-all requests are
    /// computed once and materialized per request only when results are built.
    CountAll,
    /// Count valid source positions using only a dtype-independent mask.
    CountNonNullMask(ArrayView1<'a, bool>, Vec<i64>),
    Product(View<'a>, Vec<i64>),
    ProductU64(ArrayView1<'a, u64>, ArrayView1<'a, bool>, Vec<u64>),
    ProductF32(View<'a>, Vec<f32>),
    ProductF64(View<'a>, Vec<f64>),
    Min(View<'a>, Vec<i64>),
    Max(View<'a>, Vec<i64>),
}

/// Collection of independent accumulators for one fused comparison pass.
///
/// A set preserves the caller's aggregation order. For example, if the input
/// requests are `[(values_a, mask_a, "sum"), (values_b, mask_b, "max")]`,
/// [`into_results`](Self::into_results) returns the sum array first and the
/// max-position array second. The set owns only its output buffers; input
/// arrays and masks remain borrowed views into the NumPy objects supplied by
/// the caller.
pub(crate) struct AggregationSet<'a> {
    /// One independent accumulator for every requested `(array, mask, op)`.
    states: Vec<State<'a>>,
    /// Number of positions in each result array.
    output_len: usize,
    /// Number of positions available in every source aggregation array.
    source_len: usize,
    /// Optional count-all accumulator shared by every `CountAll` request.
    /// It is allocated only when at least one count-all request is present.
    count_all: Option<Vec<i64>>,
    /// Whether each output position received at least one successful match.
    /// This is allocated only when the caller requests matched metadata.
    matched: Option<Vec<bool>>,
    /// Whether at least one valid comparison has succeeded.
    successful: bool,
}

impl<'a> AggregationSet<'a> {
    /// Build accumulator state for all requested aggregations.
    ///
    /// This function performs all shape validation before the comparison loop
    /// starts. Doing that once is important: `update` is deliberately a small
    /// hot-path function and should not repeatedly inspect Python objects or
    /// rediscover malformed input.
    ///
    /// # Arguments
    ///
    /// * `output_len` - Number of output positions. Every result array has
    ///   this length. Forward callers pass the left-side length; reverse
    ///   callers pass the right-side length.
    /// * `source_len` - Number of source positions in each aggregation value
    ///   array. Forward callers pass the right-side length; reverse callers
    ///   pass the left-side length.
    /// * `inputs` - Parsed aggregation requests. Every request contains a
    ///   typed source array, a boolean null mask, and an operation. The
    ///   comparison-path wrapper is responsible for deciding which side of
    ///   the join is the source and passing that side's length as
    ///   `source_len`.
    ///
    /// # Errors
    ///
    /// Returns a Python exception if aggregation arrays or null masks have
    /// inconsistent lengths, or if an input cannot be represented by the
    /// internal typed state.
    pub(crate) fn new(
        output_len: usize,
        source_len: usize,
        inputs: &'a [AggregationInput<'_>],
        return_matched: bool,
    ) -> PyResult<Self> {
        let mut states = Vec::with_capacity(inputs.len());
        let mut candidate_len = None;
        for input in inputs {
            let (values, nulls, op) = match input {
                AggregationInput::CountAll => {
                    states.push(State::CountAll);
                    continue;
                }
                AggregationInput::CountNonNull(mask) => {
                    let mask = mask.as_array();
                    let length = mask.len();
                    if let Some(expected) = candidate_len {
                        ensure_equal_lengths(
                            "first aggregation array",
                            expected,
                            "current aggregation mask",
                            length,
                        )?;
                    } else {
                        candidate_len = Some(length);
                    }
                    ensure_equal_lengths(
                        "comparison source array",
                        source_len,
                        "aggregation mask",
                        length,
                    )?;
                    states.push(State::CountNonNullMask(mask, vec![0; output_len]));
                    continue;
                }
                AggregationInput::I64(a, m, o) => (Values::I64(a.as_array()), m.as_array(), *o),
                AggregationInput::I32(a, m, o) => (Values::I32(a.as_array()), m.as_array(), *o),
                AggregationInput::I16(a, m, o) => (Values::I16(a.as_array()), m.as_array(), *o),
                AggregationInput::I8(a, m, o) => (Values::I8(a.as_array()), m.as_array(), *o),
                AggregationInput::U64(a, m, o) => (Values::U64(a.as_array()), m.as_array(), *o),
                AggregationInput::U32(a, m, o) => (Values::U32(a.as_array()), m.as_array(), *o),
                AggregationInput::U16(a, m, o) => (Values::U16(a.as_array()), m.as_array(), *o),
                AggregationInput::U8(a, m, o) => (Values::U8(a.as_array()), m.as_array(), *o),
                AggregationInput::F64(a, m, o) => (Values::F64(a.as_array()), m.as_array(), *o),
                AggregationInput::F32(a, m, o) => (Values::F32(a.as_array()), m.as_array(), *o),
            };
            // `Values` retains the concrete dtype, so its length must be
            // obtained by matching each variant. This does not inspect or
            // copy any element; it only asks the borrowed ndarray view for
            // its shape.
            let length = match &values {
                Values::I64(v) => v.len(),
                Values::I32(v) => v.len(),
                Values::I16(v) => v.len(),
                Values::I8(v) => v.len(),
                Values::U64(v) => v.len(),
                Values::U32(v) => v.len(),
                Values::U16(v) => v.len(),
                Values::U8(v) => v.len(),
                Values::F64(v) => v.len(),
                Values::F32(v) => v.len(),
            };
            if let Some(expected) = candidate_len {
                ensure_equal_lengths(
                    "first aggregation array",
                    expected,
                    "current aggregation array",
                    length,
                )?;
            } else {
                candidate_len = Some(length);
            }
            ensure_equal_lengths(
                "comparison source array",
                source_len,
                "aggregation array",
                length,
            )?;
            ensure_equal_lengths("aggregation array", length, "null mask", nulls.len())?;
            let state = match op {
                AggregationOp::Sum => match &values {
                    Values::U64(values) => State::SumU64(*values, nulls, vec![0; output_len]),
                    Values::F32(_) => State::SumF32(
                        View { values, nulls },
                        vec![0.; output_len],
                        vec![0.; output_len],
                    ),
                    Values::F64(_) => State::SumF64(
                        View { values, nulls },
                        vec![0.; output_len],
                        vec![0.; output_len],
                    ),
                    _ => State::Sum(View { values, nulls }, vec![0; output_len]),
                },
                AggregationOp::CountAll => State::CountAll,
                AggregationOp::CountNonNull => {
                    return Err(PyValueError::new_err(
                        "count must use the wildcard null-mask form",
                    ));
                }
                AggregationOp::Product => match &values {
                    Values::U64(values) => State::ProductU64(*values, nulls, vec![1; output_len]),
                    Values::F32(_) => {
                        State::ProductF32(View { values, nulls }, vec![1.; output_len])
                    }
                    Values::F64(_) => {
                        State::ProductF64(View { values, nulls }, vec![1.; output_len])
                    }
                    _ => State::Product(View { values, nulls }, vec![1; output_len]),
                },
                AggregationOp::Min => State::Min(View { values, nulls }, vec![-1; output_len]),
                AggregationOp::Max => State::Max(View { values, nulls }, vec![-1; output_len]),
            };
            states.push(state);
        }
        let mut has_count_all = false;
        for state in &states {
            if matches!(state, State::CountAll) {
                has_count_all = true;
                break;
            }
        }
        Ok(Self {
            states,
            output_len,
            source_len,
            count_all: if has_count_all {
                Some(vec![0; output_len])
            } else {
                None
            },
            matched: return_matched.then(|| vec![false; output_len]),
            successful: false,
        })
    }

    /// Apply one successful comparison to every requested aggregation.
    ///
    /// `source_position` identifies the value and mask to read;
    /// `output_position` identifies the result slot to update. The comparison
    /// modules call this once per successful pair, so multiple aggregations
    /// share one traversal.
    ///
    /// `CountAll` deliberately ignores null metadata and therefore counts
    /// every successful comparison. `CountNonNullMask` and other value-based
    /// operations skip a source value whose mask is `true`. A `false` mask is
    /// treated as an assertion that the value array is null-free and the value
    /// is valid; no additional null inference or sentinel filtering is
    /// performed.
    /// Both positions are checked defensively. Normal callers have already
    /// validated these bounds, but retaining the check here makes the shared
    /// state safe when a future comparison path is added.
    ///
    /// # Arguments
    ///
    /// * `source_position` - Zero-based position in the aggregation input
    ///   arrays.
    /// * `output_position` - Zero-based position in every output array.
    pub(crate) fn update(&mut self, source_position: usize, output_position: usize) {
        if source_position >= self.source_len || output_position >= self.output_len {
            return;
        }
        self.mark_matched(output_position);
        // One successful comparison is one event. Broadcast that event to
        // every requested aggregation before moving to the next candidate.
        // This is the central multi-aggregation benefit: predicates and
        // candidate discovery run once, while all requested reductions share
        // that traversal.
        for state in &mut self.states {
            match state {
                State::CountAll => {
                    // The count-all buffer is shared across all count-all
                    // requests. This branch only signals that the request
                    // exists; the actual increment happens once below.
                }
                State::CountNonNullMask(nulls, values) => {
                    if !nulls[source_position] {
                        values[output_position] += 1;
                    }
                }
                State::Sum(view, values) => {
                    if !view.nulls[source_position] {
                        values[output_position] = values[output_position]
                            .wrapping_add(as_i64(&view.values, source_position));
                    }
                }
                State::SumU64(source, nulls, values) => {
                    if !nulls[source_position] {
                        values[output_position] =
                            values[output_position].wrapping_add(source[source_position]);
                    }
                }
                State::SumF32(view, values, compensation) => {
                    if !view.nulls[source_position] {
                        kahan_add_f32(
                            &mut values[output_position],
                            &mut compensation[output_position],
                            as_f32(&view.values, source_position),
                        );
                    }
                }
                State::SumF64(view, values, compensation) => {
                    if !view.nulls[source_position] {
                        kahan_add(
                            &mut values[output_position],
                            &mut compensation[output_position],
                            as_f64(&view.values, source_position),
                        );
                    }
                }
                State::Product(view, values) => {
                    if !view.nulls[source_position] {
                        values[output_position] = values[output_position]
                            .wrapping_mul(as_i64(&view.values, source_position));
                    }
                }
                State::ProductU64(source, nulls, values) => {
                    if !nulls[source_position] {
                        values[output_position] =
                            values[output_position].wrapping_mul(source[source_position]);
                    }
                }
                State::ProductF32(view, values) => {
                    if !view.nulls[source_position] {
                        values[output_position] *= as_f32(&view.values, source_position);
                    }
                }
                State::ProductF64(view, values) => {
                    if !view.nulls[source_position] {
                        values[output_position] *= as_f64(&view.values, source_position);
                    }
                }
                State::Min(view, positions) => {
                    if !view.nulls[source_position] {
                        let current = positions[output_position];
                        if current < 0 || is_less(view, source_position, current as usize) {
                            positions[output_position] = source_position as i64;
                        }
                    }
                }
                State::Max(view, positions) => {
                    if !view.nulls[source_position] {
                        let current = positions[output_position];
                        if current < 0 || is_greater(view, source_position, current as usize) {
                            positions[output_position] = source_position as i64;
                        }
                    }
                }
            }
        }
        if let Some(values) = &mut self.count_all {
            values[output_position] += 1;
        }
    }

    /// Aggregate dual-region candidates with an insert-only rank segment tree.
    ///
    /// ELI5: each newly exposed right row is filed into the leaf for its
    /// region value. Parent nodes summarize complete groups of leaves, so a
    /// left row can merge a logarithmic number of rank blocks instead of
    /// visiting every matching right row.
    ///
    /// This path intentionally supports associative integer operations,
    /// counts, and extrema. Floating-point sums and products retain the direct
    /// traversal because changing encounter order would change their
    /// numerical behavior; floating-point extrema remain order-independent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn aggregate_dual_rank_tree(
        &mut self,
        left_region: ArrayView1<'_, i64>,
        left_order: &[usize],
        right_region: ArrayView1<'_, i64>,
        right_labels: &[i64],
        right_ranks: &[usize],
        starts: ArrayView1<'_, usize>,
        left_has_match: &[bool],
        inputs: &'a [AggregationInput<'_>],
    ) {
        if right_labels.is_empty() {
            return;
        }

        let tree_size = right_labels.len().next_power_of_two();
        let tree_output_len = tree_size.saturating_mul(2);
        if tree_output_len == 0 {
            return;
        }

        let mut tree = AggregationSet::new(tree_output_len, right_region.len(), inputs, true)
            .expect("dual rank tree inputs were validated by the caller");
        let mut previous_end = right_region.len();
        let mut left_nodes = Vec::new();
        let mut right_nodes = Vec::new();

        for row in 0..left_region.len() {
            let start = starts[row];
            if start > previous_end || start >= right_region.len() {
                continue;
            }
            if start < previous_end {
                for position in (start..previous_end).rev() {
                    let rank = right_ranks[position];
                    if rank == usize::MAX {
                        continue;
                    }
                    let leaf = tree_size + rank;
                    tree.update(position, leaf);
                    tree.rebuild_path(leaf);
                }
                previous_end = start;
            }

            if !left_has_match[row] || left_region[row] == 0 {
                continue;
            }

            let threshold = right_labels.partition_point(|value| *value < left_region[row]);
            if threshold == right_labels.len() {
                continue;
            }
            left_nodes.clear();
            right_nodes.clear();
            let mut left = tree_size + threshold;
            let mut right = tree_size + right_labels.len();
            while left < right {
                if left & 1 == 1 {
                    left_nodes.push(left);
                    left += 1;
                }
                if right & 1 == 1 {
                    right -= 1;
                    right_nodes.push(right);
                }
                left /= 2;
                right /= 2;
            }
            for &node in left_nodes.iter().chain(right_nodes.iter().rev()) {
                self.merge_output_from(&tree, node, left_order[row]);
            }
        }
    }

    /// Recompute the rank-tree ancestors after inserting one right position.
    fn rebuild_path(&mut self, leaf: usize) {
        let mut node = leaf / 2;
        while node > 0 {
            self.copy_output_within(node * 2, node);
            self.merge_output_within(node * 2 + 1, node);
            node /= 2;
        }
    }

    fn copy_output_within(&mut self, source: usize, target: usize) {
        if let Some(matched) = &mut self.matched {
            matched[target] = matched[source];
        }
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(_, values)
                | State::Sum(_, values)
                | State::Min(_, values)
                | State::Max(_, values) => values[target] = values[source],
                State::SumU64(_, _, values) => values[target] = values[source],
                State::SumF32(..)
                | State::SumF64(..)
                | State::Product(..)
                | State::ProductU64(..)
                | State::ProductF32(..)
                | State::ProductF64(..) => {
                    unreachable!("unsupported dual rank tree state")
                }
            }
        }
        if let Some(values) = &mut self.count_all {
            values[target] = values[source];
        }
    }

    fn merge_output_within(&mut self, source: usize, target: usize) {
        if let Some(matched) = &mut self.matched {
            matched[target] |= matched[source];
        }
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(_, values) | State::Sum(_, values) => {
                    values[target] = values[target].wrapping_add(values[source]);
                }
                State::SumU64(_, _, values) => {
                    values[target] = values[target].wrapping_add(values[source]);
                }
                State::Min(view, values) => {
                    values[target] = better_extreme(view, values[target], values[source], true);
                }
                State::Max(view, values) => {
                    values[target] = better_extreme(view, values[target], values[source], false);
                }
                State::SumF32(..)
                | State::SumF64(..)
                | State::Product(..)
                | State::ProductU64(..)
                | State::ProductF32(..)
                | State::ProductF64(..) => {
                    unreachable!("unsupported dual rank tree state")
                }
            }
        }
        if let Some(values) = &mut self.count_all {
            values[target] = values[target].wrapping_add(values[source]);
        }
    }

    fn merge_output_from(&mut self, source: &Self, source_position: usize, target: usize) {
        let source_matched = source
            .matched
            .as_ref()
            .is_some_and(|matched| matched[source_position]);
        if let Some(matched) = &mut self.matched {
            matched[target] |= source_matched;
        }
        for (destination, source_state) in self.states.iter_mut().zip(&source.states) {
            match (destination, source_state) {
                (State::CountAll, State::CountAll) => {}
                (State::CountNonNullMask(_, values), State::CountNonNullMask(_, source_values))
                | (State::Sum(_, values), State::Sum(_, source_values)) => {
                    values[target] = values[target].wrapping_add(source_values[source_position]);
                }
                (State::SumU64(_, _, values), State::SumU64(_, _, source_values)) => {
                    values[target] = values[target].wrapping_add(source_values[source_position]);
                }
                (State::Min(view, values), State::Min(_, source_values)) => {
                    values[target] =
                        better_extreme(view, values[target], source_values[source_position], true);
                }
                (State::Max(view, values), State::Max(_, source_values)) => {
                    values[target] =
                        better_extreme(view, values[target], source_values[source_position], false);
                }
                _ => unreachable!("dual rank tree state mismatch"),
            }
        }
        if let (Some(values), Some(source_values)) = (&mut self.count_all, &source.count_all) {
            values[target] = values[target].wrapping_add(source_values[source_position]);
        }
        self.successful |= source_matched;
    }

    /// Whether every requested operation is supported by the rank tree.
    pub(crate) fn supports_dual_rank_tree(&self) -> bool {
        self.states.iter().all(|state| {
            matches!(
                state,
                State::CountAll
                    | State::CountNonNullMask(..)
                    | State::Sum(..)
                    | State::SumU64(..)
                    | State::Min(..)
                    | State::Max(..)
            )
        })
    }

    /// Aggregate starts-only suffixes, selecting the same adaptive strategy
    /// used by the existing single-operation forward kernels.
    ///
    /// A direct implementation of `right[start..]` scans every suffix. For
    /// example, with:
    ///
    /// ```text
    /// right  = [2, 3, 4, 5]
    /// starts = [1, 3, 0]
    /// ```
    ///
    /// the direct iteration is:
    ///
    /// ```text
    /// starts[0] = 1: [3, 4, 5]     -> sum 12, product 60
    /// starts[1] = 3: [5]           -> sum 5,  product 5
    /// starts[2] = 0: [2, 3, 4, 5]  -> sum 14, product 120
    ///
    /// sums     = [12, 5, 14]
    /// products = [60, 5, 120]
    /// ```
    ///
    /// The suffix tables are built once:
    ///
    /// ```text
    /// suffix_sum     = [14, 12, 9, 5, 0]
    /// suffix_product = [120, 60, 20, 5, 1]
    /// ```
    ///
    /// Looking up indices `[1, 3, 0]` produces the same results:
    ///
    /// ```text
    /// sums     = [suffix_sum[1], suffix_sum[3], suffix_sum[0]]
    ///          = [12, 5, 14]
    /// products = [suffix_product[1], suffix_product[3], suffix_product[0]]
    ///          = [60, 5, 120]
    /// ```
    ///
    /// Min/max use the same table shape, but store source positions instead
    /// of values:
    ///
    /// ```text
    /// suffix_min_position = [0, 1, 2, 3, -1]
    /// suffix_max_position = [3, 3, 3, 3, -1]
    ///
    /// min_positions = [suffix_min_position[1], suffix_min_position[3],
    ///                  suffix_min_position[0]] = [1, 3, 0]
    /// max_positions = [suffix_max_position[1], suffix_max_position[3],
    ///                  suffix_max_position[0]] = [3, 3, 3]
    /// ```
    ///
    /// A `-1` position means that no valid, unmasked value exists in that
    /// suffix. Equal extrema may retain either tied source position.
    pub(crate) fn aggregate_starts(&mut self, starts: ArrayView1<'_, i64>) {
        let mut ranges = Vec::with_capacity(starts.len());
        for start in starts {
            let range = match usize::try_from(*start) {
                Ok(start) if start < self.source_len => Some((start, self.source_len)),
                _ => None,
            };
            ranges.push(range);
        }
        self.mark_ranges(&ranges);
        // The adaptive strategy needs an estimate of the total amount of
        // source data that a direct implementation would visit. `ranges`
        // contains `None` for invalid boundary rows, so `flatten()` skips
        // those rows and exposes only the `(start, end)` pairs that describe
        // real half-open ranges.
        let mut width = 0_usize;
        for (start, end) in ranges.iter().flatten() {
            // `end - start` is the number of source positions in this row.
            // Saturation keeps this cost estimate safe even if a future
            // caller supplies a very large batch whose summed widths exceed
            // the platform's `usize` maximum. This is only a strategy hint;
            // it must never be allowed to wrap and select a worse strategy.
            width = width.saturating_add(end - start);
        }
        // The query count and total width now describe the cost of answering
        // each suffix directly. The helper compares that cost with one full
        // suffix-table scan before deciding whether a table is worthwhile.
        let running = use_running_tables(ranges.len(), width, self.source_len);

        // Every requested operation gets the same boundary strategy, but it
        // still owns its own typed accumulator. This loop is what allows one
        // boundary traversal to produce sum, count, product, and extrema
        // together without merging their dtype-specific implementations.
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(nulls, output) => {
                    if running {
                        let mut suffix = vec![0_i64; self.source_len + 1];
                        for position in (0..self.source_len).rev() {
                            suffix[position] = suffix[position + 1] + i64::from(!nulls[position]);
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, _)) = range {
                                output[row] = suffix[*start];
                            }
                        }
                    } else {
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                output[row] = (*start..*end)
                                    .filter(|&position| !nulls[position])
                                    .count() as i64;
                            }
                        }
                    }
                }
                State::Sum(view, output) => {
                    if running {
                        let mut suffix = vec![0_i64; self.source_len + 1];
                        for position in (0..self.source_len).rev() {
                            suffix[position] = suffix[position + 1];
                            if !view.nulls[position] {
                                suffix[position] =
                                    suffix[position].wrapping_add(as_i64(&view.values, position));
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, _)) = range {
                                output[row] = suffix[*start];
                            }
                        }
                    } else {
                        update_signed_ranges(view, output, &ranges, false);
                    }
                }
                State::SumU64(values, nulls, output) => {
                    if running {
                        let mut suffix = vec![0_u64; self.source_len + 1];
                        for position in (0..self.source_len).rev() {
                            suffix[position] = suffix[position + 1];
                            if !nulls[position] {
                                suffix[position] = suffix[position].wrapping_add(values[position]);
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, _)) = range {
                                output[row] = suffix[*start];
                            }
                        }
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, false);
                    }
                }
                State::SumF32(view, output, compensation) => {
                    update_float_ranges_f32(view, output, compensation, &ranges, false);
                }
                State::SumF64(view, output, compensation) => {
                    update_float_ranges(view, output, compensation, &ranges, false);
                }
                State::Product(view, output) => {
                    if running {
                        let mut suffix = vec![1_i64; self.source_len + 1];
                        for position in (0..self.source_len).rev() {
                            suffix[position] = suffix[position + 1];
                            if !view.nulls[position] {
                                suffix[position] =
                                    suffix[position].wrapping_mul(as_i64(&view.values, position));
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, _)) = range {
                                output[row] = suffix[*start];
                            }
                        }
                    } else {
                        update_signed_ranges(view, output, &ranges, true);
                    }
                }
                State::ProductU64(values, nulls, output) => {
                    if running {
                        let mut suffix = vec![1_u64; self.source_len + 1];
                        for position in (0..self.source_len).rev() {
                            suffix[position] = suffix[position + 1];
                            if !nulls[position] {
                                suffix[position] = suffix[position].wrapping_mul(values[position]);
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, _)) = range {
                                output[row] = suffix[*start];
                            }
                        }
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, true);
                    }
                }
                State::ProductF64(view, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            let mut value = 1.;
                            for position in *start..*end {
                                if !view.nulls[position] {
                                    value *= as_f64(&view.values, position);
                                }
                            }
                            output[row] = value;
                        }
                    }
                }
                State::ProductF32(view, output) => {
                    update_float_ranges_f32_without_compensation(view, output, &ranges);
                }
                State::Min(view, output) => {
                    update_extreme_ranges(view, output, &ranges, true, running, true)
                }
                State::Max(view, output) => {
                    update_extreme_ranges(view, output, &ranges, false, running, true)
                }
            }
        }
        if let Some(count_all) = &mut self.count_all {
            for (row, range) in ranges.iter().enumerate() {
                count_all[row] = range.map_or(0, |(start, end)| (end - start) as i64);
            }
        }
    }

    /// Aggregate ends-only prefixes using adaptive prefix tables where the
    /// operation and its arithmetic permit table lookup.
    ///
    /// A direct implementation of `right[..end]` scans every prefix. For:
    ///
    /// ```text
    /// right = [2, 3, 4, 5]
    /// ends  = [3, 1, 4]
    /// ```
    ///
    /// direct iteration gives:
    ///
    /// ```text
    /// ends[0] = 3: [2, 3, 4]     -> sum 9,  product 24,  min 0, max 2
    /// ends[1] = 1: [2]           -> sum 2,  product 2,   min 0, max 0
    /// ends[2] = 4: [2, 3, 4, 5]  -> sum 14, product 120, min 0, max 3
    ///
    /// sums          = [9, 2, 14]
    /// products      = [24, 2, 120]
    /// min_positions = [0, 0, 0]
    /// max_positions = [2, 0, 3]
    /// ```
    ///
    /// One prefix pass builds:
    ///
    /// ```text
    /// prefix_sum          = [0, 2, 5, 9, 14]
    /// prefix_product      = [1, 2, 6, 24, 120]
    /// prefix_min_position = [-1, 0, 0, 0, 0]
    /// prefix_max_position = [-1, 0, 1, 2, 3]
    /// ```
    ///
    /// Looking up ends `[3, 1, 4]` produces exactly the direct results:
    ///
    /// ```text
    /// sums          = [prefix_sum[3], prefix_sum[1], prefix_sum[4]]
    ///               = [9, 2, 14]
    /// products      = [prefix_product[3], prefix_product[1], prefix_product[4]]
    ///               = [24, 2, 120]
    /// min_positions = [prefix_min_position[3], prefix_min_position[1],
    ///                  prefix_min_position[4]]
    ///               = [0, 0, 0]
    /// max_positions = [prefix_max_position[3], prefix_max_position[1],
    ///                  prefix_max_position[4]]
    ///               = [2, 0, 3]
    /// ```
    ///
    /// A `-1` prefix position means that the prefix contains no valid,
    /// unmasked value. Nulls are skipped using the authoritative mask.
    pub(crate) fn aggregate_ends(&mut self, ends: ArrayView1<'_, i64>) {
        let mut ranges = Vec::with_capacity(ends.len());
        for end in ends {
            let range = match usize::try_from(*end) {
                Ok(end) if end > 0 && end <= self.source_len => Some((0, end)),
                _ => None,
            };
            ranges.push(range);
        }
        self.mark_ranges(&ranges);
        // Estimate the direct work for all valid prefix requests. Invalid
        // rows are represented by `None` and deliberately contribute zero to
        // this estimate because they produce no aggregation work.
        let mut width = 0_usize;
        for (start, end) in ranges.iter().flatten() {
            // Each range is half-open, so this is exactly the number of
            // source positions that a direct prefix scan would inspect.
            // Saturating arithmetic prevents the strategy estimate itself
            // from overflowing for unusually large inputs.
            width = width.saturating_add(end - start);
        }
        // A prefix table is selected only when enough repeated work is saved
        // to justify its allocation and construction pass.
        let running = use_running_tables(ranges.len(), width, self.source_len);

        // Keep the operation dispatch in one place. The prefix/suffix choice
        // is shared, while each state supplies its own arithmetic and mask
        // semantics below.
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(nulls, output) => {
                    if running {
                        let mut prefix = vec![0_i64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] = prefix[position] + i64::from(!nulls[position]);
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((_, end)) = range {
                                output[row] = prefix[*end];
                            }
                        }
                    } else {
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                output[row] = (*start..*end)
                                    .filter(|&position| !nulls[position])
                                    .count() as i64;
                            }
                        }
                    }
                }
                State::Sum(view, output) => {
                    if running {
                        let mut prefix = vec![0_i64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] = prefix[position];
                            if !view.nulls[position] {
                                prefix[position + 1] = prefix[position + 1]
                                    .wrapping_add(as_i64(&view.values, position));
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((_, end)) = range {
                                output[row] = prefix[*end];
                            }
                        }
                    } else {
                        update_signed_ranges(view, output, &ranges, false);
                    }
                }
                State::SumU64(values, nulls, output) => {
                    if running {
                        let mut prefix = vec![0_u64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] = prefix[position];
                            if !nulls[position] {
                                prefix[position + 1] =
                                    prefix[position + 1].wrapping_add(values[position]);
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((_, end)) = range {
                                output[row] = prefix[*end];
                            }
                        }
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, false);
                    }
                }
                State::SumF64(view, output, compensation) => {
                    update_float_ranges(view, output, compensation, &ranges, false);
                }
                State::SumF32(view, output, compensation) => {
                    update_float_ranges_f32(view, output, compensation, &ranges, false);
                }
                State::Product(view, output) => {
                    if running {
                        let mut prefix = vec![1_i64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] = prefix[position];
                            if !view.nulls[position] {
                                prefix[position + 1] = prefix[position + 1]
                                    .wrapping_mul(as_i64(&view.values, position));
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((_, end)) = range {
                                output[row] = prefix[*end];
                            }
                        }
                    } else {
                        update_signed_ranges(view, output, &ranges, true);
                    }
                }
                State::ProductU64(values, nulls, output) => {
                    if running {
                        let mut prefix = vec![1_u64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] = prefix[position];
                            if !nulls[position] {
                                prefix[position + 1] =
                                    prefix[position + 1].wrapping_mul(values[position]);
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((_, end)) = range {
                                output[row] = prefix[*end];
                            }
                        }
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, true);
                    }
                }
                State::ProductF32(view, output) => {
                    update_float_ranges_f32_without_compensation(view, output, &ranges);
                }
                State::ProductF64(view, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            let mut value = 1.;
                            for position in *start..*end {
                                if !view.nulls[position] {
                                    value *= as_f64(&view.values, position);
                                }
                            }
                            output[row] = value;
                        }
                    }
                }
                State::Min(view, output) => {
                    update_extreme_ranges(view, output, &ranges, true, running, false)
                }
                State::Max(view, output) => {
                    update_extreme_ranges(view, output, &ranges, false, running, false)
                }
            }
        }
        if let Some(count_all) = &mut self.count_all {
            for (row, range) in ranges.iter().enumerate() {
                count_all[row] = range.map_or(0, |(start, end)| (end - start) as i64);
            }
        }
    }

    /// Mark output rows with valid, non-empty ranges before operation-specific
    /// tables are built. This keeps `matched` independent from nullness.
    fn mark_ranges(&mut self, ranges: &[Option<(usize, usize)>]) {
        for (row, range) in ranges.iter().enumerate() {
            if range.is_some() {
                self.mark_matched(row);
            }
        }
    }

    /// Mark one output slot as matched.
    ///
    /// `successful` is always updated, while the boolean vector is updated
    /// only when the caller requested matched metadata. Keeping that policy in
    /// one helper prevents comparison and range paths from drifting apart.
    #[inline]
    fn mark_matched(&mut self, output_position: usize) {
        // `successful` answers whether any pair matched at all and is needed
        // even when the caller did not request the public matched array.
        // Keeping the allocation optional avoids building an O(output_len)
        // boolean buffer for callers that only need aggregation values.
        self.successful = true;
        if let Some(matched) = &mut self.matched {
            matched[output_position] = true;
        }
    }

    /// Mark every output slot in one valid half-open range as matched.
    #[inline]
    fn mark_matched_range(&mut self, start: usize, end: usize) {
        // Reverse range kernels discover whole half-open output intervals at
        // once. Mark the interval in one fill rather than visiting each slot
        // through the per-pair update path.
        self.successful = true;
        if let Some(matched) = &mut self.matched {
            matched[start..end].fill(true);
        }
    }

    /// Aggregate explicit half-open ranges supplied by a starts/ends path.
    ///
    /// ELI5: integer sums can subtract two running totals, like finding a
    /// trip's distance from two odometer readings. Wrapping subtraction still
    /// works when the odometer rolls over. A prefix table covers only the
    /// disjoint queried spans packed end to end; each sum takes two lookups.
    /// For example, `[i64::MAX, 1, 4]` has wrapping prefix totals
    /// `[0, i64::MAX, i64::MIN, i64::MIN + 4]`. Subtracting totals at
    /// boundaries 1 and 3 wraps back to `5`, the sum of `[1, 4]`.
    /// Products and extrema use segment trees for broad batches, while float
    /// reductions retain direct source order for numerical parity.
    pub(crate) fn aggregate_starts_ends(
        &mut self,
        starts: ArrayView1<'_, i64>,
        ends: ArrayView1<'_, i64>,
    ) {
        let mut ranges = Vec::with_capacity(starts.len());
        for row in 0..starts.len() {
            let range = match (usize::try_from(starts[row]), usize::try_from(ends[row])) {
                (Ok(start), Ok(end)) if start < end && end <= self.source_len => Some((start, end)),
                _ => None,
            };
            ranges.push(range);
        }
        self.mark_ranges(&ranges);
        // Estimate direct traversal work before choosing reusable summaries.
        // Integer sums use prefix differences; products and extrema require
        // trees because their summaries cannot generally be subtracted.
        let mut width = 0_usize;
        for (start, end) in ranges.iter().flatten() {
            // `None` rows are invalid or empty and have no source positions
            // to visit. For every valid row, the half-open width is safe to
            // subtract because range construction already enforced start <
            // end. Saturation keeps this advisory cost calculation bounded.
            width = width.saturating_add(end - start);
        }
        // ELI5: each tree only needs the contiguous source slice touched by
        // its cluster of queries. Keeping original positions in the leaves
        // preserves min/max results without allocating across gaps between
        // separate equality groups.
        let queried_spans = queried_source_spans(&ranges);
        let range_clusters = assign_range_clusters(&ranges, &queried_spans);
        let packed_layout = PackedSpanLayout::new(&queried_spans, &range_clusters);
        // The tree is useful only for broad, repeated, overlapping ranges;
        // estimate its build cost using the compact span it will actually
        // materialize, not the full source array.
        let use_tree = should_use_segment_tree(ranges.len(), width, packed_layout.length);
        let use_prefix = should_use_prefix_difference(ranges.len(), width, packed_layout.length);

        // Dispatch all requested reductions over the same set of ranges. The
        // integer sums and mask counts may use prefix differences, products
        // and extrema a tree; float reductions retain direct source order.
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(nulls, output) => {
                    if use_prefix {
                        #[cfg(test)]
                        PREFIX_COUNT_HITS.with(|hits| hits.set(hits.get() + 1));
                        let mut prefix = vec![0_i64; packed_layout.length + 1];
                        for (cluster, &(span_start, span_end)) in queried_spans.iter().enumerate() {
                            let offset = packed_layout.offsets[cluster];
                            for position in span_start..span_end {
                                let packed_position = offset + position - span_start;
                                prefix[packed_position + 1] =
                                    prefix[packed_position] + i64::from(!nulls[position]);
                            }
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = packed_layout.data_range(row, *range) {
                                output[row] = prefix[end] - prefix[start];
                            }
                        }
                    } else {
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                output[row] = (*start..*end)
                                    .filter(|&position| !nulls[position])
                                    .count() as i64;
                            }
                        }
                    }
                }
                State::Sum(view, output) => {
                    if use_prefix {
                        update_signed_prefix_ranges(view, output, &ranges, &packed_layout);
                    } else {
                        update_signed_ranges(view, output, &ranges, false);
                    }
                }
                State::SumU64(values, nulls, output) => {
                    if use_prefix {
                        update_u64_prefix_ranges(values, nulls, output, &ranges, &packed_layout);
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, false);
                    }
                }
                State::SumF64(view, output, compensation) => {
                    update_float_ranges(view, output, compensation, &ranges, false);
                }
                State::SumF32(view, output, compensation) => {
                    update_float_ranges_f32(view, output, compensation, &ranges, false);
                }
                State::Product(view, output) => {
                    if use_tree {
                        update_signed_product_segment_ranges(view, output, &ranges, &packed_layout);
                    } else {
                        update_signed_ranges(view, output, &ranges, true);
                    }
                }
                State::ProductU64(values, nulls, output) => {
                    if use_tree {
                        update_u64_product_segment_ranges(
                            values,
                            nulls,
                            output,
                            &ranges,
                            &packed_layout,
                        );
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, true);
                    }
                }
                State::ProductF64(view, output) => {
                    update_float_ranges_without_compensation(view, output, &ranges);
                }
                State::ProductF32(view, output) => {
                    update_float_ranges_f32_without_compensation(view, output, &ranges);
                }
                State::Min(view, output) => {
                    if use_tree {
                        update_extreme_segment_ranges(view, output, &ranges, &packed_layout, true);
                    } else {
                        update_extreme_ranges(view, output, &ranges, true, false, false);
                    }
                }
                State::Max(view, output) => {
                    if use_tree {
                        update_extreme_segment_ranges(view, output, &ranges, &packed_layout, false);
                    } else {
                        update_extreme_ranges(view, output, &ranges, false, false, false);
                    }
                }
            }
        }
        if let Some(count_all) = &mut self.count_all {
            for (row, range) in ranges.iter().enumerate() {
                count_all[row] = range.map_or(0, |(start, end)| (end - start) as i64);
            }
        }
    }

    /// Aggregate reverse suffix ranges into dense right-side output slots.
    ///
    /// `values[row]` is the left-side value associated with `starts[row]` and
    /// `nulls[row]` is its authoritative null flag. A source row contributes
    /// to every dense right ordinal in `start[row]..right_len`; it does not
    /// contribute the right label itself. The caller has already converted
    /// the comparison result into these ordinal boundaries.
    ///
    /// Integer reductions and counts use boundary events followed by a
    /// left-to-right sweep when that is cheaper than visiting all covered
    /// suffixes directly. Floating reductions always use direct suffix
    /// updates so source-row encounter order is preserved. Min/max boundary
    /// events store source positions; ties may retain either valid position
    /// because only the extreme value matters.
    ///
    /// # Worked example
    ///
    /// With four right positions, the source arrays are:
    ///
    /// ```text
    /// values = [2, 3]                 // one value per left/source row
    /// nulls  = [false, false]         // neither source value is null
    /// starts = [1, 0]                 // inclusive right ordinals
    /// right  = [r0, r1, r2, r3]       // output slots, labels omitted here
    /// ```
    ///
    /// Source row zero contributes `2` to `right[1..4]`; source row one
    /// contributes `3` to `right[0..4]`:
    ///
    /// ```text
    /// contribution     [ -, 2, 2, 2 ]
    /// contribution     [ 3,  3, 3, 3 ]
    /// sum               [ 3,  5, 5, 5 ]
    /// product           [ 3,  6, 6, 6 ]
    /// count-all         [ 1,  2, 2, 2 ]
    /// ```
    ///
    /// For the integer sum, the implementation generates a boundary-event
    /// array and then performs a running sweep:
    ///
    /// ```text
    /// sum events by start = [3, 2, 0, 0]
    /// running sum         = [3, 5, 5, 5]
    ///
    /// product events      = [3, 2, 1, 1]   // identity is 1
    /// running product     = [3, 6, 6, 6]
    /// ```
    ///
    /// This avoids revisiting source row zero separately for positions 1, 2,
    /// and 3. If the batch is small or sparse, `use_running_tables` selects
    /// direct range updates instead: it applies `2` to the slice `1..4` and
    /// `3` to `0..4`. Both routes produce the same integer results. Floating
    /// sums do not use the event rewrite because changing addition order can
    /// change the rounded `f64`; they visit each suffix in source-row order.
    pub(crate) fn aggregate_reverse_starts(&mut self, starts: ArrayView1<'_, i64>) {
        let mut valid_starts = Vec::with_capacity(starts.len());
        for start in starts {
            let range = match usize::try_from(*start) {
                Ok(start) if start < self.output_len => Some(start),
                _ => None,
            };
            valid_starts.push(range);
        }
        self.mark_reverse_suffixes(&valid_starts);
        // A boundary sweep touches every dense output slot. For a small or
        // sparse batch, direct suffix updates touch fewer slots and are the
        // better route. This is the reverse analogue of the adaptive running
        // suffix-table choice used by the forward aggregation path.
        let mut total_width = 0_usize;
        for start in valid_starts.iter().flatten() {
            total_width = total_width.saturating_add(self.output_len - *start);
        }
        let use_sweep = use_running_tables(starts.len(), total_width, self.output_len);
        let suffix_ranges: Vec<_> = valid_starts
            .iter()
            .map(|start| start.map(|start| (start, self.output_len)))
            .collect();

        // Every state gets the same already-validated boundaries. The match
        // below is the dtype/op dispatch: after construction, no Python
        // objects or operation strings are inspected in this hot loop.
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(nulls, output) => {
                    if !use_sweep {
                        update_count_ranges_mask(nulls, output, &suffix_ranges);
                        continue;
                    }
                    let mut events = vec![0_i64; self.output_len];
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !nulls[row] {
                                events[*start] += 1;
                            }
                        }
                    }
                    let mut running = 0_i64;
                    for position in 0..self.output_len {
                        running += events[position];
                        output[position] = running;
                    }
                }
                State::Sum(view, output) => {
                    if !use_sweep {
                        update_reverse_signed_ranges(view, output, &suffix_ranges, false);
                        continue;
                    }
                    let mut events = vec![0_i64; self.output_len];
                    // Direct updates are deliberate for floating sums. The
                    // source-row order is part of the numerical contract, so
                    // an event table must not reorder additions.
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !view.nulls[row] {
                                events[*start] =
                                    events[*start].wrapping_add(as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 0_i64;
                    for position in 0..self.output_len {
                        running = running.wrapping_add(events[position]);
                        output[position] = running;
                    }
                }
                State::SumU64(values, nulls, output) => {
                    if !use_sweep {
                        update_reverse_u64_ranges(values, nulls, output, &suffix_ranges, false);
                        continue;
                    }
                    let mut events = vec![0_u64; self.output_len];
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !nulls[row] {
                                events[*start] = events[*start].wrapping_add(values[row]);
                            }
                        }
                    }
                    let mut running = 0_u64;
                    for position in 0..self.output_len {
                        running = running.wrapping_add(events[position]);
                        output[position] = running;
                    }
                }
                State::SumF64(view, output, compensation) => {
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !view.nulls[row] {
                                for position in *start..self.output_len {
                                    kahan_add(
                                        &mut output[position],
                                        &mut compensation[position],
                                        as_f64(&view.values, row),
                                    );
                                }
                            }
                        }
                    }
                }
                State::SumF32(view, output, compensation) => {
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !view.nulls[row] {
                                for position in *start..self.output_len {
                                    kahan_add_f32(
                                        &mut output[position],
                                        &mut compensation[position],
                                        as_f32(&view.values, row),
                                    );
                                }
                            }
                        }
                    }
                }
                State::Product(view, output) => {
                    if !use_sweep {
                        update_reverse_signed_ranges(view, output, &suffix_ranges, true);
                        continue;
                    }
                    let mut events = vec![1_i64; self.output_len];
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !view.nulls[row] {
                                events[*start] =
                                    events[*start].wrapping_mul(as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 1_i64;
                    for position in 0..self.output_len {
                        running = running.wrapping_mul(events[position]);
                        output[position] = running;
                    }
                }
                State::ProductU64(values, nulls, output) => {
                    if !use_sweep {
                        update_reverse_u64_ranges(values, nulls, output, &suffix_ranges, true);
                        continue;
                    }
                    let mut events = vec![1_u64; self.output_len];
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !nulls[row] {
                                events[*start] = events[*start].wrapping_mul(values[row]);
                            }
                        }
                    }
                    let mut running = 1_u64;
                    for position in 0..self.output_len {
                        running = running.wrapping_mul(events[position]);
                        output[position] = running;
                    }
                }
                State::ProductF64(view, output) => {
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !view.nulls[row] {
                                let value = as_f64(&view.values, row);
                                for slot in &mut output[*start..self.output_len] {
                                    *slot *= value;
                                }
                            }
                        }
                    }
                }
                State::ProductF32(view, output) => {
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !view.nulls[row] {
                                let value = as_f32(&view.values, row);
                                for slot in &mut output[*start..self.output_len] {
                                    *slot *= value;
                                }
                            }
                        }
                    }
                }
                State::Min(view, output) => {
                    if use_sweep {
                        update_reverse_boundary_extreme(view, output, &valid_starts, true, false);
                    } else {
                        update_reverse_range_extreme(view, output, &suffix_ranges, true);
                    }
                }
                State::Max(view, output) => {
                    if use_sweep {
                        update_reverse_boundary_extreme(view, output, &valid_starts, false, false);
                    } else {
                        update_reverse_range_extreme(view, output, &suffix_ranges, false);
                    }
                }
            }
        }
        self.write_reverse_count_all_suffixes(&valid_starts, use_sweep);
    }

    /// Aggregate reverse prefix ranges into dense right-side output slots.
    ///
    /// `values[row]` is the left-side value associated with `ends[row]` and
    /// `nulls[row]` is its authoritative null flag. Each non-null source row
    /// contributes to every dense right ordinal in `0..end[row]`; the caller
    /// has already converted comparison results into these exclusive prefix
    /// boundaries.
    ///
    /// Integer/count operations use end-boundary events and a right-to-left
    /// sweep when that is cheaper than visiting every covered prefix slot.
    /// Sparse or small batches use direct prefix updates. Float operations
    /// always update each prefix directly in source-row order.
    ///
    /// # Worked example
    ///
    /// With four right positions, the source arrays are:
    ///
    /// ```text
    /// values = [2, 3]                 // one value per left/source row
    /// nulls  = [false, false]         // neither source value is null
    /// ends   = [3, 1]                 // exclusive right ordinals
    /// right  = [r0, r1, r2, r3]       // output slots, labels omitted here
    /// ```
    ///
    /// The first source row contributes `2` to `right[0..3]`; the second
    /// contributes `3` to `right[0..1]`:
    ///
    /// ```text
    /// contribution     [ 2, 2, 2, - ]
    /// contribution     [ 3, -, -,  - ]
    /// sum               [ 5, 2, 2,  0 ]
    /// count-all         [ 2, 1, 1,  0 ]
    /// ```
    ///
    /// For integer sums, `2` is recorded at event slot `2` and `3` at event
    /// slot `0`:
    ///
    /// ```text
    /// sum events by end - 1 = [3, 0, 2, 0]
    /// right-to-left running = [5, 2, 2, 0]
    ///
    /// product events         = [3, 1, 2, 1]
    /// right-to-left running  = [6, 2, 2, 1]
    /// ```
    ///
    /// A small or sparse batch instead applies `2` to `0..3` and `3` to
    /// `0..1` directly. A zero-width or out-of-bounds end contributes no
    /// event and no direct slice, so it cannot make an output position
    /// matched. Count-all follows the same selected traversal and ignores
    /// the null mask.
    pub(crate) fn aggregate_reverse_ends(&mut self, ends: ArrayView1<'_, i64>) {
        let mut valid_ends = Vec::with_capacity(ends.len());
        for end in ends {
            let range = match usize::try_from(*end) {
                Ok(end) if end > 0 && end <= self.output_len => Some(end),
                _ => None,
            };
            valid_ends.push(range);
        }
        self.mark_reverse_prefixes(&valid_ends);
        // A prefix boundary sweep costs one pass over every dense output
        // position. Direct prefix updates are preferable when the total
        // covered width is small, using the same adaptive cost comparison as
        // the forward ends-only aggregation path.
        let mut total_width = 0_usize;
        for end in valid_ends.iter().flatten() {
            total_width = total_width.saturating_add(*end);
        }
        let use_sweep = use_running_tables(ends.len(), total_width, self.output_len);
        let prefix_ranges: Vec<_> = valid_ends
            .iter()
            .map(|end| end.map(|end| (0, end)))
            .collect();

        // Prefix traversal is separate from suffix traversal because the
        // same boundary-event idea requires the opposite sweep direction.
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(nulls, output) => {
                    if !use_sweep {
                        update_count_ranges_mask(nulls, output, &prefix_ranges);
                        continue;
                    }
                    let mut events = vec![0_i64; self.output_len];
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !nulls[row] {
                                events[*end - 1] += 1;
                            }
                        }
                    }
                    let mut running = 0_i64;
                    for position in (0..self.output_len).rev() {
                        running += events[position];
                        output[position] = running;
                    }
                }
                State::Sum(view, output) => {
                    if !use_sweep {
                        update_reverse_signed_ranges(view, output, &prefix_ranges, false);
                        continue;
                    }
                    let mut events = vec![0_i64; self.output_len];
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !view.nulls[row] {
                                events[*end - 1] =
                                    events[*end - 1].wrapping_add(as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 0_i64;
                    for position in (0..self.output_len).rev() {
                        running = running.wrapping_add(events[position]);
                        output[position] = running;
                    }
                }
                State::SumU64(values, nulls, output) => {
                    if !use_sweep {
                        update_reverse_u64_ranges(values, nulls, output, &prefix_ranges, false);
                        continue;
                    }
                    let mut events = vec![0_u64; self.output_len];
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !nulls[row] {
                                events[*end - 1] = events[*end - 1].wrapping_add(values[row]);
                            }
                        }
                    }
                    let mut running = 0_u64;
                    for position in (0..self.output_len).rev() {
                        running = running.wrapping_add(events[position]);
                        output[position] = running;
                    }
                }
                State::SumF64(view, output, compensation) => {
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !view.nulls[row] {
                                for position in 0..*end {
                                    kahan_add(
                                        &mut output[position],
                                        &mut compensation[position],
                                        as_f64(&view.values, row),
                                    );
                                }
                            }
                        }
                    }
                }
                State::SumF32(view, output, compensation) => {
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !view.nulls[row] {
                                for position in 0..*end {
                                    kahan_add_f32(
                                        &mut output[position],
                                        &mut compensation[position],
                                        as_f32(&view.values, row),
                                    );
                                }
                            }
                        }
                    }
                }
                State::Product(view, output) => {
                    if !use_sweep {
                        update_reverse_signed_ranges(view, output, &prefix_ranges, true);
                        continue;
                    }
                    let mut events = vec![1_i64; self.output_len];
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !view.nulls[row] {
                                events[*end - 1] =
                                    events[*end - 1].wrapping_mul(as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 1_i64;
                    for position in (0..self.output_len).rev() {
                        running = running.wrapping_mul(events[position]);
                        output[position] = running;
                    }
                }
                State::ProductU64(values, nulls, output) => {
                    if !use_sweep {
                        update_reverse_u64_ranges(values, nulls, output, &prefix_ranges, true);
                        continue;
                    }
                    let mut events = vec![1_u64; self.output_len];
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !nulls[row] {
                                events[*end - 1] = events[*end - 1].wrapping_mul(values[row]);
                            }
                        }
                    }
                    let mut running = 1_u64;
                    for position in (0..self.output_len).rev() {
                        running = running.wrapping_mul(events[position]);
                        output[position] = running;
                    }
                }
                State::ProductF64(view, output) => {
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !view.nulls[row] {
                                let value = as_f64(&view.values, row);
                                for slot in &mut output[..*end] {
                                    *slot *= value;
                                }
                            }
                        }
                    }
                }
                State::ProductF32(view, output) => {
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !view.nulls[row] {
                                let value = as_f32(&view.values, row);
                                for slot in &mut output[..*end] {
                                    *slot *= value;
                                }
                            }
                        }
                    }
                }
                State::Min(view, output) => {
                    if use_sweep {
                        update_reverse_boundary_extreme(view, output, &valid_ends, true, true);
                    } else {
                        update_reverse_range_extreme(view, output, &prefix_ranges, true);
                    }
                }
                State::Max(view, output) => {
                    if use_sweep {
                        update_reverse_boundary_extreme(view, output, &valid_ends, false, true);
                    } else {
                        update_reverse_range_extreme(view, output, &prefix_ranges, false);
                    }
                }
            }
        }
        self.write_reverse_count_all_prefixes(&valid_ends, use_sweep);
    }

    /// Aggregate arbitrary reverse half-open ranges using dense output slots.
    ///
    /// `values[row]` is the left-side value associated with the half-open
    /// interval `starts[row]..ends[row]`; `nulls[row]` is its authoritative
    /// null flag. The method writes to dense right ordinals, not to the label
    /// values in `right_index`.
    ///
    /// This method intentionally does not use a `HashMap`: every right ordinal
    /// has a required result slot, so direct positional writes are both simpler
    /// and compatible with the fused API's full-length output contract. The
    /// existing reverse starts/ends kernels use a dense-array route for broad
    /// batches and a map route for compact sparse outputs. The fused API cannot
    /// use that compact map route because it must return identity/sentinel
    /// values for every right slot, including untouched slots.
    ///
    /// # Worked example
    ///
    /// For four right positions, the source arrays are:
    ///
    /// ```text
    /// values = [2, 3]
    /// nulls  = [false, false]
    /// starts = [1, 0]                 // inclusive
    /// ends   = [4, 2]                 // exclusive
    /// ```
    ///
    /// Thus source row zero updates `right[1..4]`, and source row one updates
    /// `right[0..2]`:
    ///
    /// ```text
    /// contribution     [ -, 2, 2, 2 ]
    /// contribution     [ 3,  3, -, - ]
    /// sum               [ 3,  5, 2, 2 ]
    /// matched           [ T,  T, T, T ]
    /// ```
    ///
    /// The direct range loop is the natural dense reverse-update algorithm:
    /// each valid source row walks exactly the slots it covers. For example,
    /// the integer sum performs:
    ///
    /// ```text
    /// output starts at [0, 0, 0, 0]
    /// apply row 0 value 2 to slots 1..4 -> [0, 2, 2, 2]
    /// apply row 1 value 3 to slots 0..2 -> [3, 5, 2, 2]
    /// ```
    ///
    /// A forward segment tree is useful when many output queries ask for
    /// source-range summaries. This reverse path has the opposite shape:
    /// source rows apply updates to output ranges. A segment tree would need a
    /// separate lazy range-update implementation for each operation, would
    /// not help floating sums whose encounter order is significant, and would
    /// still need a full output materialization. Direct dense slices therefore
    /// preserve the existing reverse kernel's optimized dense route. Invalid,
    /// inverted, or zero-width ranges are discarded before the loop, so they
    /// affect neither `matched` nor any aggregation.
    pub(crate) fn aggregate_reverse_starts_ends(
        &mut self,
        starts: ArrayView1<'_, i64>,
        ends: ArrayView1<'_, i64>,
    ) {
        let mut ranges = Vec::with_capacity(starts.len());
        for row in 0..starts.len() {
            let range = match (usize::try_from(starts[row]), usize::try_from(ends[row])) {
                (Ok(start), Ok(end)) if start < end && end <= self.output_len => Some((start, end)),
                _ => None,
            };
            ranges.push(range);
        }
        self.mark_reverse_ranges(&ranges);

        // Unlike starts-only and ends-only, arbitrary intervals do not share a
        // single boundary sweep. The dense slice is therefore updated directly
        // for each valid source range.
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNullMask(nulls, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !nulls[row] {
                                for slot in &mut output[*start..*end] {
                                    *slot += 1;
                                }
                            }
                        }
                    }
                }
                State::Sum(view, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                let value = as_i64(&view.values, row);
                                for slot in &mut output[*start..*end] {
                                    *slot = slot.wrapping_add(value);
                                }
                            }
                        }
                    }
                }
                State::SumU64(values, nulls, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !nulls[row] {
                                for slot in &mut output[*start..*end] {
                                    *slot = slot.wrapping_add(values[row]);
                                }
                            }
                        }
                    }
                }
                State::SumF64(view, output, compensation) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                let value = as_f64(&view.values, row);
                                for position in *start..*end {
                                    kahan_add(
                                        &mut output[position],
                                        &mut compensation[position],
                                        value,
                                    );
                                }
                            }
                        }
                    }
                }
                State::SumF32(view, output, compensation) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                let value = as_f32(&view.values, row);
                                for position in *start..*end {
                                    kahan_add_f32(
                                        &mut output[position],
                                        &mut compensation[position],
                                        value,
                                    );
                                }
                            }
                        }
                    }
                }
                State::Product(view, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                let value = as_i64(&view.values, row);
                                for slot in &mut output[*start..*end] {
                                    *slot = slot.wrapping_mul(value);
                                }
                            }
                        }
                    }
                }
                State::ProductU64(values, nulls, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !nulls[row] {
                                for slot in &mut output[*start..*end] {
                                    *slot = slot.wrapping_mul(values[row]);
                                }
                            }
                        }
                    }
                }
                State::ProductF64(view, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                let value = as_f64(&view.values, row);
                                for slot in &mut output[*start..*end] {
                                    *slot *= value;
                                }
                            }
                        }
                    }
                }
                State::ProductF32(view, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                let value = as_f32(&view.values, row);
                                for slot in &mut output[*start..*end] {
                                    *slot *= value;
                                }
                            }
                        }
                    }
                }
                State::Min(view, output) => {
                    update_reverse_range_extreme(view, output, &ranges, true);
                }
                State::Max(view, output) => {
                    update_reverse_range_extreme(view, output, &ranges, false);
                }
            }
        }
        self.write_reverse_count_all_ranges(&ranges);
    }

    /// Mark every right position covered by at least one valid suffix.
    fn mark_reverse_suffixes(&mut self, starts: &[Option<usize>]) {
        for start in starts.iter().flatten() {
            self.mark_matched_range(*start, self.output_len);
        }
    }

    /// Mark every right position covered by at least one valid prefix.
    fn mark_reverse_prefixes(&mut self, ends: &[Option<usize>]) {
        for end in ends.iter().flatten() {
            self.mark_matched_range(0, *end);
        }
    }

    /// Mark every right position covered by an arbitrary valid interval.
    fn mark_reverse_ranges(&mut self, ranges: &[Option<(usize, usize)>]) {
        for (start, end) in ranges.iter().flatten() {
            self.mark_matched_range(*start, *end);
        }
    }

    /// Compute count-all for reverse suffixes.
    ///
    /// The sweep route reuses the same boundary events as integer sum/count.
    /// The direct route is selected for sparse batches and visits only the
    /// suffix slots that are actually covered.
    fn write_reverse_count_all_suffixes(&mut self, starts: &[Option<usize>], use_sweep: bool) {
        let Some(output) = &mut self.count_all else {
            return;
        };
        if !use_sweep {
            for start in starts.iter().flatten() {
                for slot in &mut output[*start..] {
                    *slot += 1;
                }
            }
            return;
        }
        let mut events = vec![0_i64; self.output_len];
        for start in starts.iter().flatten() {
            events[*start] += 1;
        }
        let mut running = 0_i64;
        for position in 0..self.output_len {
            running += events[position];
            output[position] = running;
        }
    }

    /// Compute count-all for reverse prefixes.
    ///
    /// Sparse batches use direct dense slice increments; broad batches use
    /// end-boundary events and sweep toward the beginning of the output.
    fn write_reverse_count_all_prefixes(&mut self, ends: &[Option<usize>], use_sweep: bool) {
        let Some(output) = &mut self.count_all else {
            return;
        };
        if !use_sweep {
            for end in ends.iter().flatten() {
                for slot in &mut output[..*end] {
                    *slot += 1;
                }
            }
            return;
        }
        let mut events = vec![0_i64; self.output_len];
        for end in ends.iter().flatten() {
            events[*end - 1] += 1;
        }
        let mut running = 0_i64;
        for position in (0..self.output_len).rev() {
            running += events[position];
            output[position] = running;
        }
    }

    /// Compute count-all for arbitrary reverse intervals using dense output
    /// slots. Null masks are intentionally irrelevant here.
    fn write_reverse_count_all_ranges(&mut self, ranges: &[Option<(usize, usize)>]) {
        let Some(output) = &mut self.count_all else {
            return;
        };
        for (start, end) in ranges.iter().flatten() {
            for slot in &mut output[*start..*end] {
                *slot += 1;
            }
        }
    }

    /// Return whether this comparison pass observed no successful comparison.
    ///
    /// This flag describes comparison success, not aggregation success. In
    /// particular, it remains `true` when a comparison matched but every
    /// source value was marked null. In that case value-based operations keep
    /// their identity or sentinel (`0` for sum, `1` for product, and `-1` for
    /// min/max), while `matched` still identifies the output position as a
    /// real match. Count-all and count-non-null can likewise produce different
    /// values for the same matched position.
    ///
    /// The Python wrappers call this after traversal. They return `None` only
    /// when the entire pass had no successful comparisons; otherwise they
    /// return `(matched, aggregation_arrays)`, including positions whose
    /// aggregation values remain at their identities.
    pub(crate) fn is_empty(&self) -> bool {
        !self.successful
    }

    /// Convert the match indicator and accumulator buffers into Python values.
    ///
    /// The result order is exactly the input aggregation order. This method
    /// consumes the set because the internal buffers can be moved directly
    /// into NumPy-owned arrays without cloning them.
    ///
    /// # Arguments
    ///
    /// * `py` - The active Python interpreter token required to create NumPy
    ///   objects safely.
    ///
    /// # Returns
    ///
    /// The optional boolean match indicator followed by one NumPy array per
    /// requested aggregation. The indicator is omitted when the caller did
    /// not request matched metadata.
    pub(crate) fn into_results(self, py: Python<'_>) -> (Option<Py<PyAny>>, Vec<Py<PyAny>>) {
        let matched = self.matched.map(|values| {
            Array1::from_vec(values)
                .into_pyarray(py)
                .unbind()
                .into_any()
        });
        let count_all = self.count_all;
        let mut results = Vec::with_capacity(self.states.len());
        for state in self.states {
            let result = match state {
                State::Sum(_, v) | State::Product(_, v) => {
                    Array1::from_vec(v).into_pyarray(py).unbind().into_any()
                }
                State::SumU64(_, _, v) | State::ProductU64(_, _, v) => {
                    Array1::from_vec(v).into_pyarray(py).unbind().into_any()
                }
                State::SumF64(_, v, _compensation) => {
                    // Existing forward float kernels expose the running
                    // `total`, not `total + compensation`. Keep that
                    // finalization contract so fused and legacy paths remain
                    // numerically compatible.
                    Array1::from_vec(v).into_pyarray(py).unbind().into_any()
                }
                State::SumF32(_, v, _compensation) => {
                    Array1::from_vec(v).into_pyarray(py).unbind().into_any()
                }
                State::ProductF64(_, v) => Array1::from_vec(v).into_pyarray(py).unbind().into_any(),
                State::ProductF32(_, v) => Array1::from_vec(v).into_pyarray(py).unbind().into_any(),
                State::CountAll => Array1::from_vec(
                    count_all
                        .as_ref()
                        .expect("count-all state requires a shared count-all buffer")
                        .clone(),
                )
                .into_pyarray(py)
                .unbind()
                .into_any(),
                State::CountNonNullMask(_, v) | State::Min(_, v) | State::Max(_, v) => {
                    Array1::from_vec(v).into_pyarray(py).unbind().into_any()
                }
            };
            results.push(result);
        }
        (matched, results)
    }
}

/// Decide whether the prefix/suffix materialization cost is justified.
///
/// The public adaptive helper already contains the work-factor policy. The
/// explicit query-count guard here mirrors the existing specialized kernels
/// and makes the strategy visible at the aggregation-state call site.
fn use_running_tables(query_count: usize, total_width: usize, source_len: usize) -> bool {
    query_count > MAX_DIRECT_QUERY_COUNT
        && should_use_running_aggregation(query_count, total_width, source_len)
}

/// Find disjoint contiguous source spans covering all valid requests.
fn queried_source_spans(ranges: &[Option<(usize, usize)>]) -> Vec<(usize, usize)> {
    let mut intervals: Vec<_> = ranges.iter().flatten().copied().collect();
    intervals.sort_unstable();
    let mut spans = Vec::new();
    for (start, end) in intervals {
        if let Some((_, current_end)) = spans.last_mut() {
            if start <= *current_end {
                *current_end = (*current_end).max(end);
                continue;
            }
        }
        spans.push((start, end));
    }
    spans
}

/// Assign each valid request to its disjoint source span.
fn assign_range_clusters(
    ranges: &[Option<(usize, usize)>],
    spans: &[(usize, usize)],
) -> Vec<Option<usize>> {
    ranges
        .iter()
        .map(|range| {
            let (start, end) = (*range)?;
            // Spans are sorted and disjoint. Find the first span whose end
            // is beyond this range, then check that the range starts inside
            // it; this avoids scanning every span for every row.
            let cluster = spans.partition_point(|&(_, span_end)| span_end <= start);
            spans
                .get(cluster)
                .filter(|&&(span_start, span_end)| start >= span_start && end <= span_end)
                .map(|_| cluster)
        })
        .collect()
}

/// Coordinates for prefix tables and trees over queried spans laid end to
/// end, with unqueried gaps omitted.
struct PackedSpanLayout<'a> {
    spans: &'a [(usize, usize)],
    clusters: &'a [Option<usize>],
    offsets: Vec<usize>,
    length: usize,
}

impl<'a> PackedSpanLayout<'a> {
    fn new(spans: &'a [(usize, usize)], clusters: &'a [Option<usize>]) -> Self {
        let mut offsets = Vec::with_capacity(spans.len());
        let mut length = 0;
        for &(start, end) in spans {
            offsets.push(length);
            length += end - start;
        }
        Self {
            spans,
            clusters,
            offsets,
            length,
        }
    }

    /// Convert one original source range to packed data coordinates.
    fn data_range(&self, row: usize, range: Option<(usize, usize)>) -> Option<(usize, usize)> {
        let cluster = self.clusters[row]?;
        let (start, end) = range?;
        let (span_start, _) = self.spans[cluster];
        let offset = self.offsets[cluster];
        Some((offset + start - span_start, offset + end - span_start))
    }

    /// Convert one original source range to its half-open packed-tree range.
    fn packed_range(&self, row: usize, range: Option<(usize, usize)>) -> Option<(usize, usize)> {
        let (start, end) = self.data_range(row, range)?;
        Some((self.length + start, self.length + end))
    }
}

/// Answer signed sums with wrapping differences over the packed queried spans.
///
/// ELI5: pack only the roads actually traveled into one running-distance log.
/// Earlier spans cancel when two readings from a later span are subtracted,
/// even if the accumulated total wraps around. Nulls add zero, and ndarray
/// indexing preserves logical positions for strided sources and masks.
fn update_signed_prefix_ranges(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    layout: &PackedSpanLayout<'_>,
) {
    #[cfg(test)]
    PREFIX_SUM_HITS.with(|hits| hits.set(hits.get() + 1));
    let mut prefix = vec![0_i64; layout.length + 1];
    for (cluster, &(start, end)) in layout.spans.iter().enumerate() {
        let offset = layout.offsets[cluster];
        for position in start..end {
            let packed = offset + position - start;
            prefix[packed + 1] = prefix[packed];
            if !view.nulls[position] {
                prefix[packed + 1] = prefix[packed].wrapping_add(as_i64(&view.values, position));
            }
        }
    }
    for (row, range) in ranges.iter().enumerate() {
        if let Some((start, end)) = layout.data_range(row, *range) {
            output[row] = prefix[end].wrapping_sub(prefix[start]);
        }
    }
}

/// Preserve the full unsigned width, including values above `i64::MAX`.
/// ELI5: this uses the same rolling odometer as the signed prefix helper,
/// but keeps unsigned readings throughout so no high bits are lost.
fn update_u64_prefix_ranges(
    values: &ArrayView1<'_, u64>,
    nulls: &ArrayView1<'_, bool>,
    output: &mut [u64],
    ranges: &[Option<(usize, usize)>],
    layout: &PackedSpanLayout<'_>,
) {
    #[cfg(test)]
    PREFIX_SUM_HITS.with(|hits| hits.set(hits.get() + 1));
    let mut prefix = vec![0_u64; layout.length + 1];
    for (cluster, &(start, end)) in layout.spans.iter().enumerate() {
        let offset = layout.offsets[cluster];
        for position in start..end {
            let packed = offset + position - start;
            prefix[packed + 1] = prefix[packed];
            if !nulls[position] {
                prefix[packed + 1] = prefix[packed].wrapping_add(values[position]);
            }
        }
    }
    for (row, range) in ranges.iter().enumerate() {
        if let Some((start, end)) = layout.data_range(row, *range) {
            output[row] = prefix[end].wrapping_sub(prefix[start]);
        }
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static PREFIX_SUM_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PREFIX_COUNT_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Answer arbitrary integer products with an iterative segment tree.
///
/// ELI5: products cannot undo earlier factors by division: a zero erases the
/// answer, and wrapping multiplication loses information for even factors.
/// The tree instead combines reusable blocks in logarithmic query time.
/// Leaves live at `tree[length + position]`; each parent combines its two
/// children at `tree[2 * node]` and `tree[2 * node + 1]`.
fn update_signed_product_segment_ranges(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    layout: &PackedSpanLayout<'_>,
) {
    let length = layout.length;

    // A null must not change a product, so null leaves start at one.
    let mut tree = vec![1_i64; length * 2];

    // Install valid source values in the leaf half of the tree. The boolean
    // mask is authoritative; the numeric value is never inspected for
    // sentinel or null-like content.
    for (cluster, &(span_start, span_end)) in layout.spans.iter().enumerate() {
        let offset = layout.offsets[cluster];
        for position in span_start..span_end {
            if !view.nulls[position] {
                tree[length + offset + position - span_start] = as_i64(&view.values, position);
            }
        }
    }

    // Build larger block summaries from the leaves upward. After this pass,
    // each tree node represents one contiguous half-open source block.
    for node in (1..length).rev() {
        tree[node] = tree[node * 2].wrapping_mul(tree[node * 2 + 1]);
    }

    // Answer each requested range independently. `None` means that the
    // boundary row was invalid and therefore contributes no output value.
    for (row, range) in ranges.iter().enumerate() {
        let Some((mut start, mut end)) = layout.packed_range(row, *range) else {
            continue;
        };

        // `packed_range` has already mapped the caller's source boundaries
        // into the leaf portion of the flat tree.
        let mut total = 1_i64;

        // Move both boundaries toward their common parent. Whenever a
        // boundary identifies a complete child block, consume that block and
        // move past it. This visits O(log n) blocks and never visits a source
        // element twice.
        while start < end {
            if start & 1 == 1 {
                total = total.wrapping_mul(tree[start]);
                start += 1;
            }
            if end & 1 == 1 {
                end -= 1;
                total = total.wrapping_mul(tree[end]);
            }
            start /= 2;
            end /= 2;
        }

        // All complete blocks covering this half-open range have now been
        // combined into the output slot for this row.
        output[row] = total;
    }
}

/// Answer arbitrary `u64` products with wrapping segment-tree arithmetic.
///
/// This is intentionally separate from the signed implementation. A `u64`
/// value may be larger than `i64::MAX`, so converting it would violate the
/// output dtype contract. The tree layout and query algorithm are otherwise
/// identical.
fn update_u64_product_segment_ranges(
    values: &ArrayView1<'_, u64>,
    nulls: &ArrayView1<'_, bool>,
    output: &mut [u64],
    ranges: &[Option<(usize, usize)>],
    layout: &PackedSpanLayout<'_>,
) {
    let length = layout.length;

    // Null leaves begin at the neutral element for the selected operation.
    let mut tree = vec![1_u64; length * 2];

    // Copy only values whose mask says they are valid. No null inference is
    // performed from the u64 payload.
    for (cluster, &(span_start, span_end)) in layout.spans.iter().enumerate() {
        let offset = layout.offsets[cluster];
        for position in span_start..span_end {
            if !nulls[position] {
                tree[length + offset + position - span_start] = values[position];
            }
        }
    }

    // Combine neighboring leaf blocks to build the reusable summaries.
    for node in (1..length).rev() {
        tree[node] = tree[node * 2].wrapping_mul(tree[node * 2 + 1]);
    }

    // Query every valid half-open range using the same two-boundary walk as
    // the signed implementation, but retain u64 arithmetic throughout.
    for (row, range) in ranges.iter().enumerate() {
        let Some((mut start, mut end)) = layout.packed_range(row, *range) else {
            continue;
        };

        // `packed_range` has already translated source positions to leaf
        // positions. The end remains exclusive, so the query includes
        // exactly `start..end`.
        let mut total = 1_u64;
        while start < end {
            if start & 1 == 1 {
                total = total.wrapping_mul(tree[start]);
                start += 1;
            }
            if end & 1 == 1 {
                end -= 1;
                total = total.wrapping_mul(tree[end]);
            }
            start /= 2;
            end /= 2;
        }

        // The accumulator now represents precisely this output row's range.
        output[row] = total;
    }
}

/// Compute signed integer sums or products by scanning each requested range.
///
/// This is the fallback for small batches, where allocating a running table
/// would cost more than the direct scans. Arithmetic intentionally wraps, as
/// it does in the existing integer aggregation kernels.
fn update_signed_ranges(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        let mut total = if product { 1_i64 } else { 0_i64 };
        for position in *start..*end {
            if view.nulls[position] {
                continue;
            }
            let value = as_i64(&view.values, position);
            total = if product {
                total.wrapping_mul(value)
            } else {
                total.wrapping_add(value)
            };
        }
        output[row] = total;
    }
}

/// Compute `u64` sums or products by scanning each requested range.
fn update_u64_ranges(
    values: &ArrayView1<'_, u64>,
    nulls: &ArrayView1<'_, bool>,
    output: &mut [u64],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        let mut total = if product { 1_u64 } else { 0_u64 };
        for position in *start..*end {
            if nulls[position] {
                continue;
            }
            total = if product {
                total.wrapping_mul(values[position])
            } else {
                total.wrapping_add(values[position])
            };
        }
        output[row] = total;
    }
}

/// Compute floating-point reductions in source encounter order.
///
/// Prefix/suffix lookup tables are intentionally not used for floats. A
/// compensated sum is order-dependent, and multiplying precomputed pieces
/// can likewise change the sequence of operations. Keeping the direct loop
/// makes the fused path agree with the existing forward kernels.
fn update_float_ranges(
    view: &View<'_>,
    output: &mut [f64],
    compensation: &mut [f64],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        if product {
            let mut total = 1.;
            for position in *start..*end {
                if !view.nulls[position] {
                    total *= as_f64(&view.values, position);
                }
            }
            output[row] = total;
        } else {
            let mut total = 0.;
            let mut correction = 0.;
            for position in *start..*end {
                if !view.nulls[position] {
                    kahan_add(&mut total, &mut correction, as_f64(&view.values, position));
                }
            }
            output[row] = total;
            compensation[row] = correction;
        }
    }
}

/// Compute `f32` floating-point reductions in source encounter order.
fn update_float_ranges_f32(
    view: &View<'_>,
    output: &mut [f32],
    compensation: &mut [f32],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        if product {
            let mut total = 1_f32;
            for position in *start..*end {
                if !view.nulls[position] {
                    total *= as_f32(&view.values, position);
                }
            }
            output[row] = total;
        } else {
            let mut total = 0_f32;
            let mut correction = 0_f32;
            for position in *start..*end {
                if !view.nulls[position] {
                    kahan_add_f32(&mut total, &mut correction, as_f32(&view.values, position));
                }
            }
            output[row] = total;
            compensation[row] = correction;
        }
    }
}

/// Compute `f32` products for ranges without a compensation buffer.
fn update_float_ranges_f32_without_compensation(
    view: &View<'_>,
    output: &mut [f32],
    ranges: &[Option<(usize, usize)>],
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        let mut total = 1_f32;
        for position in *start..*end {
            if !view.nulls[position] {
                total *= as_f32(&view.values, position);
            }
        }
        output[row] = total;
    }
}

/// Compute floating-point products for arbitrary ranges without a sum
/// compensation buffer. Multiplication intentionally follows source order.
fn update_float_ranges_without_compensation(
    view: &View<'_>,
    output: &mut [f64],
    ranges: &[Option<(usize, usize)>],
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        let mut total = 1.;
        for position in *start..*end {
            if !view.nulls[position] {
                total *= as_f64(&view.values, position);
            }
        }
        output[row] = total;
    }
}

/// Merge two candidate positions for an extreme segment tree.
///
/// The arguments are positions, not values. `-1` represents an empty or
/// all-null block, so the non-empty candidate wins immediately. When both
/// candidates are valid, the original typed arrays are compared through the
/// explicit min/max helpers. Equality leaves `left` unchanged, which gives a
/// deterministic winner while still satisfying the position-agnostic API.
fn better_extreme(view: &View<'_>, left: i64, right: i64, min: bool) -> i64 {
    if left < 0 {
        return right;
    }
    if right < 0 {
        return left;
    }
    let right_wins = if min {
        is_less(view, right as usize, left as usize)
    } else {
        is_greater(view, right as usize, left as usize)
    };
    if right_wins {
        right
    } else {
        left
    }
}

/// Answer arbitrary min/max ranges with an iterative position segment tree.
///
/// Nodes store source positions rather than values. This preserves the
/// public contract that min/max returns a position, while the comparator
/// reads the original typed array and mask. A null leaf is represented by
/// `-1`, so nullness remains entirely mask-driven.
///
/// The tree stores positions rather than copied values. For example, if a
/// block contains `[5, 1, 6]`, its minimum node stores position `1`, not the
/// number `1`; callers can then use that position to retrieve the original
/// source value. Dtype-specific comparisons remain in `is_less` and
/// `is_greater`.
fn update_extreme_segment_ranges(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    layout: &PackedSpanLayout<'_>,
    min: bool,
) {
    let length = layout.length;

    // `-1` means that a block has no valid candidate. `better_extreme` treats
    // it as an empty block and prefers a real position whenever one exists.
    let mut tree = vec![-1_i64; length * 2];

    // A valid leaf stores its source position. A masked leaf stays `-1`; the
    // mask, rather than a special numeric value, decides validity.
    for (cluster, &(span_start, span_end)) in layout.spans.iter().enumerate() {
        let offset = layout.offsets[cluster];
        for position in span_start..span_end {
            if !view.nulls[position] {
                // Keep the original source position in the leaf so min/max
                // results retain their existing coordinate contract.
                tree[length + offset + position - span_start] = position as i64;
            }
        }
    }

    // Build one winning position for every larger contiguous block. Strict
    // comparisons preserve the existing winner when values tie; the API only
    // requires that the returned position identify an actual extreme value.
    for node in (1..length).rev() {
        tree[node] = better_extreme(view, tree[node * 2], tree[node * 2 + 1], min);
    }
    for (row, range) in ranges.iter().enumerate() {
        let Some((mut start, mut end)) = layout.packed_range(row, *range) else {
            continue;
        };

        // `packed_range` has already moved source coordinates into the leaf
        // half of the flat tree; `end` remains exclusive throughout.
        let mut winner = -1_i64;
        while start < end {
            if start & 1 == 1 {
                winner = better_extreme(view, winner, tree[start], min);
                start += 1;
            }
            if end & 1 == 1 {
                end -= 1;
                winner = better_extreme(view, winner, tree[end], min);
            }
            start /= 2;
            end /= 2;
        }

        // `winner` is either a valid source position or -1 when the range had
        // no unmasked values.
        output[row] = winner;
    }
}

/// Build min/max positions for arbitrary ranges by direct scanning.
fn update_extreme_ranges(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    min: bool,
    running: bool,
    suffix: bool,
) {
    if running {
        // A suffix query can be answered from the winner at every position;
        // a prefix query uses the same idea while scanning left-to-right.
        // Strict comparisons preserve the first encountered position when
        // equal values tie, although callers only rely on the value being an
        // actual minimum or maximum.
        let mut winners = vec![-1_i64; view.nulls.len() + 1];
        if suffix {
            for position in (0..view.nulls.len()).rev() {
                let mut winner = winners[position + 1];
                if !view.nulls[position]
                    && (winner < 0
                        || if min {
                            is_less(view, position, winner as usize)
                        } else {
                            is_greater(view, position, winner as usize)
                        })
                {
                    winner = position as i64;
                }
                winners[position] = winner;
            }
            for (row, range) in ranges.iter().enumerate() {
                if let Some((start, _)) = range {
                    output[row] = winners[*start];
                }
            }
        } else {
            for position in 0..view.nulls.len() {
                let mut winner = winners[position];
                if !view.nulls[position]
                    && (winner < 0
                        || if min {
                            is_less(view, position, winner as usize)
                        } else {
                            is_greater(view, position, winner as usize)
                        })
                {
                    winner = position as i64;
                }
                winners[position + 1] = winner;
            }
            for (row, range) in ranges.iter().enumerate() {
                if let Some((_, end)) = range {
                    output[row] = winners[*end];
                }
            }
        }
        return;
    }

    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        let mut winner = -1_i64;
        for position in *start..*end {
            if view.nulls[position] {
                continue;
            }
            let improves = winner < 0
                || if min {
                    is_less(view, position, winner as usize)
                } else {
                    is_greater(view, position, winner as usize)
                };
            if improves {
                winner = position as i64;
            }
        }
        output[row] = winner;
    }
}

/// Apply boundary-event min/max logic to reverse suffix or prefix ranges.
///
/// `boundaries` contains either a suffix start or a prefix end for each source
/// row. A source row is first selected as the best event at its boundary, then
/// those events are swept across the dense right-side output. This avoids
/// comparing the same source row independently against every covered slot.
/// `prefix` selects the right-to-left prefix sweep; suffixes sweep left to
/// right.
///
/// The two-stage structure mirrors the sum/product event algorithm:
///
/// ```text
/// suffix starts: boundary 1 -> candidate is active at [1..]
///                boundary 0 -> candidate is active at [0..]
/// sweep:         winner[0], winner[1], winner[2], ...
///
/// prefix ends:   end 3 -> candidate is active at [..3]
///                end 1 -> candidate is active at [..1]
/// sweep:         ..., winner[2], winner[1], winner[0]
/// ```
///
/// A boundary slot keeps only the best candidate that begins or ends there;
/// the sweep then compares winners from different boundaries. Equality does
/// not replace the current winner. The API does not promise which tied source
/// position is returned, only that a returned position contains the true
/// minimum or maximum value.
fn update_reverse_boundary_extreme(
    view: &View<'_>,
    output: &mut [i64],
    boundaries: &[Option<usize>],
    min: bool,
    prefix: bool,
) {
    let mut events = vec![-1_i64; output.len()];
    for (row, boundary) in boundaries.iter().enumerate() {
        let Some(boundary) = boundary else { continue };
        if view.nulls[row] {
            continue;
        }
        let slot = if prefix { *boundary - 1 } else { *boundary };
        let current = events[slot];
        if current < 0
            || if min {
                is_less(view, row, current as usize)
            } else {
                is_greater(view, row, current as usize)
            }
        {
            events[slot] = row as i64;
        }
    }

    let mut winner = -1_i64;
    if prefix {
        for position in (0..output.len()).rev() {
            winner = merge_extreme_positions(view, winner, events[position], min);
            output[position] = winner;
        }
    } else {
        for position in 0..output.len() {
            winner = merge_extreme_positions(view, winner, events[position], min);
            output[position] = winner;
        }
    }
}

/// Merge a source-row position into an existing min/max winner. `-1` is the
/// no-winner sentinel and is ignored whenever a valid candidate exists.
fn merge_extreme_positions(view: &View<'_>, left: i64, right: i64, min: bool) -> i64 {
    if left < 0 {
        return right;
    }
    if right < 0 {
        return left;
    }
    let right_wins = if min {
        is_less(view, right as usize, left as usize)
    } else {
        is_greater(view, right as usize, left as usize)
    };
    if right_wins {
        right
    } else {
        left
    }
}

/// Count valid source rows across dense reverse ranges without a value view.
///
/// The wildcard `count` request supplies only its authoritative null mask, so
/// the aggregation never needs to extract or inspect the source dtype.
fn update_count_ranges_mask(
    nulls: &ArrayView1<'_, bool>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        if nulls[row] {
            continue;
        }
        for slot in &mut output[*start..*end] {
            *slot += 1;
        }
    }
}

/// Apply one source-row integer value to every covered dense reverse output
/// slot. This differs from `update_signed_ranges`: forward ranges select a
/// source slice for each output row, while reverse ranges use one source row
/// to update an output slice. Keeping the helpers separate prevents the
/// boundary indices from being accidentally used to index the source array.
fn update_reverse_signed_ranges(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        if view.nulls[row] {
            continue;
        }
        let value = as_i64(&view.values, row);
        for slot in &mut output[*start..*end] {
            *slot = if product {
                slot.wrapping_mul(value)
            } else {
                slot.wrapping_add(value)
            };
        }
    }
}

/// `u64` counterpart to [`update_reverse_signed_ranges`]. The separate
/// helper preserves the required `u64` accumulator without narrowing values
/// through `i64`.
fn update_reverse_u64_ranges(
    values: &ArrayView1<'_, u64>,
    nulls: &ArrayView1<'_, bool>,
    output: &mut [u64],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        if nulls[row] {
            continue;
        }
        let value = values[row];
        for slot in &mut output[*start..*end] {
            *slot = if product {
                slot.wrapping_mul(value)
            } else {
                slot.wrapping_add(value)
            };
        }
    }
}

/// Apply min/max directly to dense arbitrary reverse intervals. The output
/// position stores the source row that owns the winning value; ties may keep
/// either valid position.
///
/// Unlike the boundary cases, an arbitrary interval has two independent
/// boundaries and cannot be represented by one monotonic event sweep. The
/// dense slice update is therefore both straightforward and appropriate:
/// for each valid source row, compare its value with the current winner in
/// every covered output slot. Because output is already dense and aligned to
/// `right_index`, a map would only add hashing and a later reorder step.
fn update_reverse_range_extreme(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    min: bool,
) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        if view.nulls[row] {
            continue;
        }
        for slot in &mut output[*start..*end] {
            let current = *slot;
            if current < 0
                || if min {
                    is_less(view, row, current as usize)
                } else {
                    is_greater(view, row, current as usize)
                }
            {
                *slot = row as i64;
            }
        }
    }
}

/// Read an integer that is valid for an `i64` accumulator.
///
/// State construction guarantees that this helper is never called for `u64`
/// or floating-point input. The unreachable branch makes that invariant
/// visible to readers and prevents accidental narrowing conversions from
/// being added silently later.
fn as_i64(values: &Values<'_>, n: usize) -> i64 {
    match values {
        Values::I64(v) => v[n],
        Values::I32(v) => v[n] as i64,
        Values::I16(v) => v[n] as i64,
        Values::I8(v) => v[n] as i64,
        Values::U32(v) => v[n] as i64,
        Values::U16(v) => v[n] as i64,
        Values::U8(v) => v[n] as i64,
        Values::U64(_) | Values::F64(_) | Values::F32(_) => {
            unreachable!("signed integer state must contain non-u64 integer values")
        }
    }
}
/// Read a floating-point value for an `f64` accumulator.
///
/// Only `f32` requires widening. `f64` is already in the output type, and all
/// integer conversions are intentionally excluded from this helper.
fn as_f64(values: &Values<'_>, n: usize) -> f64 {
    match values {
        Values::F64(v) => v[n],
        Values::F32(v) => v[n] as f64,
        _ => unreachable!("floating-point state must contain f32 or f64 values"),
    }
}

/// Read a floating-point value for an `f32` accumulator.
fn as_f32(values: &Values<'_>, n: usize) -> f32 {
    match values {
        Values::F32(v) => v[n],
        _ => unreachable!("f32 floating-point state must contain f32 values"),
    }
}

/// Add one float using the Kahan-style compensation used by the existing
/// aggregation kernels.
///
/// The running total is the public result; compensation only carries rounding
/// information into the next update. IEEE-754 infinities need one special
/// safeguard: adding an infinity can make the internal compensation `NaN`
/// even though the running total is correctly infinite. Resetting only the
/// compensation preserves that valid infinity and prevents the next finite
/// value from being poisoned by stale non-finite correction state. A genuine
/// `+infinity + -infinity` still produces `NaN` in the running total, as it
/// should.
fn kahan_add(total: &mut f64, compensation: &mut f64, value: f64) {
    let difference = value - *compensation;
    let increment = *total + difference;
    *compensation = (increment - *total) - difference;
    if !compensation.is_finite() {
        *compensation = 0.;
    }
    *total = increment;
}

/// Kahan addition using `f32` storage and arithmetic.
///
/// The compensation reset mirrors [`kahan_add`]. An infinite input can make
/// the compensation `NaN` even when the running total is a valid infinity;
/// carrying that `NaN` into the next update would poison the total.
fn kahan_add_f32(total: &mut f32, compensation: &mut f32, value: f32) {
    let adjusted = value - *compensation;
    let next = *total + adjusted;
    *compensation = (next - *total) - adjusted;
    if !compensation.is_finite() {
        *compensation = 0.;
    }
    *total = next;
}
/// Return whether the candidate at `a` is strictly less than the current
/// winner at `b`.
///
/// The comparison is explicit for every supported dtype. This keeps min's
/// behavior visible and avoids routing a simple boolean decision through
/// `std::cmp::Ordering` and a fallback for values that are not totally ordered.
fn is_less(view: &View<'_>, a: usize, b: usize) -> bool {
    match &view.values {
        Values::I64(v) => v[a] < v[b],
        Values::I32(v) => v[a] < v[b],
        Values::I16(v) => v[a] < v[b],
        Values::I8(v) => v[a] < v[b],
        Values::U64(v) => v[a] < v[b],
        Values::U32(v) => v[a] < v[b],
        Values::U16(v) => v[a] < v[b],
        Values::U8(v) => v[a] < v[b],
        Values::F64(v) => v[a] < v[b],
        Values::F32(v) => v[a] < v[b],
    }
}

/// Return whether the candidate at `a` is strictly greater than the current
/// winner at `b`.
///
/// Equality is intentionally false, which preserves the first encountered
/// position when min/max values tie. Null handling and the `-1` sentinel are
/// handled by the surrounding state branches before this helper is called.
fn is_greater(view: &View<'_>, a: usize, b: usize) -> bool {
    match &view.values {
        Values::I64(v) => v[a] > v[b],
        Values::I32(v) => v[a] > v[b],
        Values::I16(v) => v[a] > v[b],
        Values::I8(v) => v[a] > v[b],
        Values::U64(v) => v[a] > v[b],
        Values::U32(v) => v[a] > v[b],
        Values::U16(v) => v[a] > v[b],
        Values::U8(v) => v[a] > v[b],
        Values::F64(v) => v[a] > v[b],
        Values::F32(v) => v[a] > v[b],
    }
}

#[cfg(test)]
mod tests {
    use super::{
        assign_range_clusters, kahan_add_f32, queried_source_spans, update_extreme_segment_ranges,
        update_signed_prefix_ranges, update_signed_product_segment_ranges,
        update_u64_prefix_ranges, update_u64_product_segment_ranges, PackedSpanLayout, Values,
        View,
    };
    use numpy::ndarray::Array1;

    #[test]
    fn kahan_add_f32_resets_non_finite_compensation() {
        let mut total = 5.0_f32;
        let mut compensation = 0.0_f32;

        kahan_add_f32(&mut total, &mut compensation, f32::INFINITY);
        assert_eq!(total, f32::INFINITY);
        assert_eq!(compensation, 0.0);

        // A finite value after infinity must preserve the valid infinite
        // total instead of becoming NaN through stale compensation.
        kahan_add_f32(&mut total, &mut compensation, 3.0);
        assert_eq!(total, f32::INFINITY);
        assert_eq!(compensation, 0.0);
    }

    #[test]
    fn compact_segment_ranges_match_direct_reductions() {
        let values = Array1::from_vec(vec![91_i64, 2, 7, 3, 11, 5, 13, 17, 19, 23, 29, 31]);
        let nulls = Array1::from_vec(vec![
            false, false, false, false, false, false, true, false, false, false, false, false,
        ]);
        let view = View {
            values: Values::I64(values.view()),
            nulls: nulls.view(),
        };
        let ranges = vec![Some((2, 5)), Some((4, 10)), None];
        let spans = queried_source_spans(&ranges);
        assert_eq!(spans, vec![(2, 10)]);
        let clusters = assign_range_clusters(&ranges, &spans);
        let layout = PackedSpanLayout::new(&spans, &clusters);

        let mut sums = vec![0_i64; ranges.len()];
        update_signed_prefix_ranges(&view, &mut sums, &ranges, &layout);
        let expected_sums: Vec<_> = ranges
            .iter()
            .map(|range| {
                range.map_or(0, |(start, end)| {
                    (start..end)
                        .filter(|&position| !nulls[position])
                        .map(|position| values[position])
                        .sum()
                })
            })
            .collect();
        assert_eq!(sums, expected_sums);

        let mut products = vec![1_i64; ranges.len()];
        update_signed_product_segment_ranges(&view, &mut products, &ranges, &layout);
        assert_eq!(products, vec![231, 408_595, 1]);

        let unsigned = Array1::from_vec(vec![91_u64, 2, 7, 3, 11, 5, 13, 17, 19, 23, 29, 31]);
        let mut unsigned_sums = vec![0_u64; ranges.len()];
        update_u64_prefix_ranges(
            &unsigned.view(),
            &nulls.view(),
            &mut unsigned_sums,
            &ranges,
            &layout,
        );
        assert_eq!(unsigned_sums, vec![21, 75, 0]);

        let mut unsigned_products = vec![1_u64; ranges.len()];
        update_u64_product_segment_ranges(
            &unsigned.view(),
            &nulls.view(),
            &mut unsigned_products,
            &ranges,
            &layout,
        );
        assert_eq!(unsigned_products, vec![231, 408_595, 1]);

        let mut minima = vec![-1_i64; ranges.len()];
        update_extreme_segment_ranges(&view, &mut minima, &ranges, &layout, true);
        assert_eq!(minima, vec![3, 5, -1]);

        let disjoint_ranges = vec![Some((2, 5)), Some((10, 12)), None];
        let disjoint_spans = queried_source_spans(&disjoint_ranges);
        assert_eq!(disjoint_spans, vec![(2, 5), (10, 12)]);
        let disjoint_clusters = assign_range_clusters(&disjoint_ranges, &disjoint_spans);
        let disjoint_layout = PackedSpanLayout::new(&disjoint_spans, &disjoint_clusters);
        let mut disjoint_sums = vec![0_i64; disjoint_ranges.len()];
        update_signed_prefix_ranges(
            &view,
            &mut disjoint_sums,
            &disjoint_ranges,
            &disjoint_layout,
        );
        assert_eq!(disjoint_sums, vec![21, 60, 0]);
    }
}

#[cfg(test)]
#[path = "aggregation_state/prefix_tests.rs"]
mod prefix_tests;
