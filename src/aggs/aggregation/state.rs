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
//! Integer `sum` and `prod` intentionally use fixed-width wrapping arithmetic
//! (`wrapping_add`/`wrapping_mul`). This is a published dtype-specific
//! deviation from pandas, which may promote integer results during
//! aggregation. Floating-point operations retain their corresponding `f32`
//! or `f64` arithmetic and returns floating results as `f64`. Position,
//! length, and allocation arithmetic remains checked and must not wrap.

use super::input::{AggregationInput, AggregationOp};
use numpy::ndarray::{Array1, ArrayView1};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::aggs::adaptive::{
    should_use_running_aggregation, should_use_segment_tree, MAX_DIRECT_QUERY_COUNT,
};
use crate::aggs::ensure_equal_lengths;

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

#[derive(Clone, Copy)]
enum IntegerWidth {
    I64,
    I32,
    I16,
    I8,
    U32,
    U16,
    U8,
}

impl Values<'_> {
    fn integer_width(&self) -> IntegerWidth {
        match self {
            Values::I64(_) => IntegerWidth::I64,
            Values::I32(_) => IntegerWidth::I32,
            Values::I16(_) => IntegerWidth::I16,
            Values::I8(_) => IntegerWidth::I8,
            Values::U32(_) => IntegerWidth::U32,
            Values::U16(_) => IntegerWidth::U16,
            Values::U8(_) => IntegerWidth::U8,
            Values::U64(_) | Values::F64(_) | Values::F32(_) => IntegerWidth::I64,
        }
    }
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

impl View<'_> {
    fn wrapping_add(&self, left: i64, right: i64) -> i64 {
        wrap_integer(left.wrapping_add(right), self.values.integer_width())
    }

    fn wrapping_mul(&self, left: i64, right: i64) -> i64 {
        wrap_integer(left.wrapping_mul(right), self.values.integer_width())
    }
}

/// Runtime state for one operation, including its output buffer.
///
/// Each variant has a fixed output type dictated by the aggregation contract:
/// signed values use `i64`, `u64` stays `u64`, floating-point values use
/// `f64`, and positions/counts use `i64`.
enum State<'a> {
    // A state variant stores both the typed source view and the output
    // buffer. Keeping them together prevents an update from accidentally
    // pairing an operation with a source array of the wrong dtype.
    Sum(View<'a>, Vec<i64>),
    SumU64(ArrayView1<'a, u64>, ArrayView1<'a, bool>, Vec<u64>),
    SumF64(View<'a>, Vec<f64>, Vec<f64>),
    /// Reference the set-level count-all buffer; no value array or mask is
    /// needed. The buffer is shared so repeated count-all requests are
    /// computed once and materialized per request only when results are built.
    CountAll,
    /// Count only successful comparisons whose source mask marks a value valid.
    CountNonNull(View<'a>, Vec<i64>),
    /// Count valid source positions using only a dtype-independent mask.
    CountNonNullMask(ArrayView1<'a, bool>, Vec<i64>),
    Product(View<'a>, Vec<i64>),
    ProductU64(ArrayView1<'a, u64>, ArrayView1<'a, bool>, Vec<u64>),
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
    /// This is independent of null masks and aggregation identities.
    matched: Vec<bool>,
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
                    Values::F64(_) | Values::F32(_) => State::SumF64(
                        View { values, nulls },
                        vec![0.; output_len],
                        vec![0.; output_len],
                    ),
                    _ => State::Sum(View { values, nulls }, vec![0; output_len]),
                },
                AggregationOp::CountAll => State::CountAll,
                AggregationOp::CountNonNull => {
                    State::CountNonNull(View { values, nulls }, vec![0; output_len])
                }
                AggregationOp::Product => match &values {
                    Values::U64(values) => State::ProductU64(*values, nulls, vec![1; output_len]),
                    Values::F64(_) | Values::F32(_) => {
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
            matched: vec![false; output_len],
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
    /// every successful comparison. `CountNonNull` and other value-based
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
        self.successful = true;
        self.matched[output_position] = true;
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
                State::CountNonNull(view, values) => {
                    if !view.nulls[source_position] {
                        values[output_position] += 1;
                    }
                }
                State::CountNonNullMask(nulls, values) => {
                    if !nulls[source_position] {
                        values[output_position] += 1;
                    }
                }
                State::Sum(view, values) => {
                    if !view.nulls[source_position] {
                        values[output_position] = view.wrapping_add(
                            values[output_position],
                            as_i64(&view.values, source_position),
                        );
                    }
                }
                State::SumU64(source, nulls, values) => {
                    if !nulls[source_position] {
                        values[output_position] =
                            values[output_position].wrapping_add(source[source_position]);
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
                        values[output_position] = view.wrapping_mul(
                            values[output_position],
                            as_i64(&view.values, source_position),
                        );
                    }
                }
                State::ProductU64(source, nulls, values) => {
                    if !nulls[source_position] {
                        values[output_position] =
                            values[output_position].wrapping_mul(source[source_position]);
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
                State::CountNonNull(view, output) => {
                    if running {
                        let mut suffix = vec![0_i64; self.source_len + 1];
                        for position in (0..self.source_len).rev() {
                            suffix[position] =
                                suffix[position + 1] + i64::from(!view.nulls[position]);
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, _)) = range {
                                output[row] = suffix[*start];
                            }
                        }
                    } else {
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                let mut count = 0_i64;
                                for position in *start..*end {
                                    if !view.nulls[position] {
                                        count += 1;
                                    }
                                }
                                output[row] = count;
                            }
                        }
                    }
                }
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
                                suffix[position] = view
                                    .wrapping_add(suffix[position], as_i64(&view.values, position));
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
                State::SumF64(view, output, compensation) => {
                    update_float_ranges(view, output, compensation, &ranges, false);
                }
                State::Product(view, output) => {
                    if running {
                        let mut suffix = vec![1_i64; self.source_len + 1];
                        for position in (0..self.source_len).rev() {
                            suffix[position] = suffix[position + 1];
                            if !view.nulls[position] {
                                suffix[position] = view
                                    .wrapping_mul(suffix[position], as_i64(&view.values, position));
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
                State::CountNonNull(view, output) => {
                    if running {
                        let mut prefix = vec![0_i64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] =
                                prefix[position] + i64::from(!view.nulls[position]);
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((_, end)) = range {
                                output[row] = prefix[*end];
                            }
                        }
                    } else {
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                let mut count = 0_i64;
                                for position in *start..*end {
                                    if !view.nulls[position] {
                                        count += 1;
                                    }
                                }
                                output[row] = count;
                            }
                        }
                    }
                }
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
                                prefix[position + 1] = view.wrapping_add(
                                    prefix[position + 1],
                                    as_i64(&view.values, position),
                                );
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
                State::Product(view, output) => {
                    if running {
                        let mut prefix = vec![1_i64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] = prefix[position];
                            if !view.nulls[position] {
                                prefix[position + 1] = view.wrapping_mul(
                                    prefix[position + 1],
                                    as_i64(&view.values, position),
                                );
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
                self.matched[row] = true;
                self.successful = true;
            }
        }
    }

    /// Aggregate explicit half-open ranges supplied by a starts/ends path.
    ///
    /// Starts-only and ends-only ranges can use one-dimensional suffix or
    /// prefix tables. An arbitrary starts/ends pair cannot generally share
    /// either table, so this method deliberately preserves the simple direct
    /// range walk. That walk is also the numerically correct choice for
    /// floating-point reductions because it retains source encounter order.
    ///
    /// For broad batches, associative integer operations and min/max use an
    /// iterative segment tree. For:
    ///
    /// ```text
    /// right = [2, 3, 4, 5, 1, 6, 7, 8]
    /// range = [1..6) = [3, 4, 5, 1, 6]
    /// ```
    ///
    /// normal iteration produces sum `19`, product `360`, minimum position
    /// `4`, and maximum position `5`.
    ///
    /// The tree first summarizes blocks:
    ///
    /// ```text
    /// block 0..2: sum 5,  product 6,   min position 0, max position 1
    /// block 2..4: sum 9,  product 20,  min position 2, max position 3
    /// block 4..6: sum 7,  product 6,   min position 4, max position 5
    /// block 6..8: sum 15, product 56,  min position 6, max position 7
    /// block 0..4: sum 14, product 120, min position 0, max position 3
    /// block 4..8: sum 22, product 336, min position 4, max position 7
    /// ```
    ///
    /// It answers `[1..6)` by combining complete pieces:
    ///
    /// ```text
    /// [1..6) = [1..2) + [2..4) + [4..6)
    ///         -> sum 3 + 9 + 7 = 19
    ///         -> product 3 * 20 * 6 = 360
    ///         -> min position 4, max position 5
    /// ```
    ///
    /// This is equivalent to normal iteration, but broad overlapping ranges
    /// reuse the summaries instead of rescanning every source element.
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
        // Starts/ends requests may describe arbitrary overlapping intervals.
        // Unlike starts-only and ends-only requests, they cannot be answered
        // by one prefix or suffix table. We therefore estimate the total
        // direct interval width before selecting the segment-tree strategy.
        let mut width = 0_usize;
        for (start, end) in ranges.iter().flatten() {
            // `None` rows are invalid or empty and have no source positions
            // to visit. For every valid row, the half-open width is safe to
            // subtract because range construction already enforced start <
            // end. Saturation keeps this advisory cost calculation bounded.
            width = width.saturating_add(end - start);
        }
        // The tree is useful only for broad, repeated, overlapping ranges;
        // the adaptive helper also rejects small batches and narrow queries.
        let use_tree = should_use_segment_tree(ranges.len(), width, self.source_len);

        // Dispatch all requested reductions over the same set of ranges. The
        // associative integer and extreme operations may use the tree; float
        // reductions intentionally retain direct source order.
        for state in &mut self.states {
            match state {
                State::CountAll => {}
                State::CountNonNull(view, output) => {
                    if use_tree {
                        let mut prefix = vec![0_i64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] =
                                prefix[position] + i64::from(!view.nulls[position]);
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                output[row] = prefix[*end] - prefix[*start];
                            }
                        }
                    } else {
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                let mut count = 0_i64;
                                for position in *start..*end {
                                    if !view.nulls[position] {
                                        count += 1;
                                    }
                                }
                                output[row] = count;
                            }
                        }
                    }
                }
                State::CountNonNullMask(nulls, output) => {
                    if use_tree {
                        let mut prefix = vec![0_i64; self.source_len + 1];
                        for position in 0..self.source_len {
                            prefix[position + 1] = prefix[position] + i64::from(!nulls[position]);
                        }
                        for (row, range) in ranges.iter().enumerate() {
                            if let Some((start, end)) = range {
                                output[row] = prefix[*end] - prefix[*start];
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
                    if use_tree {
                        update_signed_segment_ranges(view, output, &ranges, false);
                    } else {
                        update_signed_ranges(view, output, &ranges, false);
                    }
                }
                State::SumU64(values, nulls, output) => {
                    if use_tree {
                        update_u64_segment_ranges(values, nulls, output, &ranges, false);
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, false);
                    }
                }
                State::SumF64(view, output, compensation) => {
                    update_float_ranges(view, output, compensation, &ranges, false);
                }
                State::Product(view, output) => {
                    if use_tree {
                        update_signed_segment_ranges(view, output, &ranges, true);
                    } else {
                        update_signed_ranges(view, output, &ranges, true);
                    }
                }
                State::ProductU64(values, nulls, output) => {
                    if use_tree {
                        update_u64_segment_ranges(values, nulls, output, &ranges, true);
                    } else {
                        update_u64_ranges(values, nulls, output, &ranges, true);
                    }
                }
                State::ProductF64(view, output) => {
                    update_float_ranges_without_compensation(view, output, &ranges);
                }
                State::Min(view, output) => {
                    if use_tree {
                        update_extreme_segment_ranges(view, output, &ranges, true);
                    } else {
                        update_extreme_ranges(view, output, &ranges, true, false, false);
                    }
                }
                State::Max(view, output) => {
                    if use_tree {
                        update_extreme_segment_ranges(view, output, &ranges, false);
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
                State::CountNonNull(view, output) => {
                    if !use_sweep {
                        update_count_ranges(view, output, &suffix_ranges);
                        continue;
                    }
                    // `events[p]` means: add this row's contribution when the
                    // sweep reaches right position `p`. A suffix beginning
                    // at `p` affects `p` and every position to its right.
                    let mut events = vec![0_i64; self.output_len];
                    for (row, start) in valid_starts.iter().enumerate() {
                        if let Some(start) = start {
                            if !view.nulls[row] {
                                events[*start] += 1;
                            }
                        }
                    }
                    // Once the boundary event has been applied, its value is
                    // active for the rest of the suffix sweep.
                    let mut running = 0_i64;
                    for position in 0..self.output_len {
                        running += events[position];
                        output[position] = running;
                    }
                }
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
                                    view.wrapping_add(events[*start], as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 0_i64;
                    for position in 0..self.output_len {
                        running = view.wrapping_add(running, events[position]);
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
                                    view.wrapping_mul(events[*start], as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 1_i64;
                    for position in 0..self.output_len {
                        running = view.wrapping_mul(running, events[position]);
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
                State::CountNonNull(view, output) => {
                    if !use_sweep {
                        update_count_ranges(view, output, &prefix_ranges);
                        continue;
                    }
                    let mut events = vec![0_i64; self.output_len];
                    for (row, end) in valid_ends.iter().enumerate() {
                        if let Some(end) = end {
                            if !view.nulls[row] {
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
                                    view.wrapping_add(events[*end - 1], as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 0_i64;
                    for position in (0..self.output_len).rev() {
                        running = view.wrapping_add(running, events[position]);
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
                                    view.wrapping_mul(events[*end - 1], as_i64(&view.values, row));
                            }
                        }
                    }
                    let mut running = 1_i64;
                    for position in (0..self.output_len).rev() {
                        running = view.wrapping_mul(running, events[position]);
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
                State::CountNonNull(view, output) => {
                    // Count-non-null uses the mask; count-all is written once
                    // below and intentionally does not inspect any mask.
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                for slot in &mut output[*start..*end] {
                                    *slot += 1;
                                }
                            }
                        }
                    }
                }
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
                                    *slot = view.wrapping_add(*slot, value);
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
                State::Product(view, output) => {
                    for (row, range) in ranges.iter().enumerate() {
                        if let Some((start, end)) = range {
                            if !view.nulls[row] {
                                let value = as_i64(&view.values, row);
                                for slot in &mut output[*start..*end] {
                                    *slot = view.wrapping_mul(*slot, value);
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
            self.matched[*start..].fill(true);
            self.successful = true;
        }
    }

    /// Mark every right position covered by at least one valid prefix.
    fn mark_reverse_prefixes(&mut self, ends: &[Option<usize>]) {
        for end in ends.iter().flatten() {
            self.matched[..*end].fill(true);
            self.successful = true;
        }
    }

    /// Mark every right position covered by an arbitrary valid interval.
    fn mark_reverse_ranges(&mut self, ranges: &[Option<(usize, usize)>]) {
        for (start, end) in ranges.iter().flatten() {
            self.matched[*start..*end].fill(true);
            self.successful = true;
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
    /// A pair containing the boolean match indicator followed by one NumPy
    /// array per requested aggregation. The caller wraps this pair in the
    /// public `(matched, aggregation_arrays)` tuple.
    pub(crate) fn into_results(self, py: Python<'_>) -> (Py<PyAny>, Vec<Py<PyAny>>) {
        let matched = Array1::from_vec(self.matched)
            .into_pyarray(py)
            .unbind()
            .into_any();
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
                State::ProductF64(_, v) => Array1::from_vec(v).into_pyarray(py).unbind().into_any(),
                State::CountAll => Array1::from_vec(
                    count_all
                        .as_ref()
                        .expect("count-all state requires a shared count-all buffer")
                        .clone(),
                )
                .into_pyarray(py)
                .unbind()
                .into_any(),
                State::CountNonNull(_, v)
                | State::CountNonNullMask(_, v)
                | State::Min(_, v)
                | State::Max(_, v) => Array1::from_vec(v).into_pyarray(py).unbind().into_any(),
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

/// Answer arbitrary integer ranges with an iterative segment tree.
///
/// Range-only starts/ends queries cannot use one prefix or suffix table, but
/// broad batches still benefit from sharing work between overlapping ranges.
/// The tree stores the associative operation (wrapping sum or product) at
/// each internal node and answers each half-open range in logarithmic time.
///
/// The iterative layout is compact and beginner-friendly once its indexing
/// rule is understood: leaves live at `tree[length + position]`, and each
/// parent at `tree[node]` combines children `tree[2 * node]` and
/// `tree[2 * node + 1]`. This avoids a recursive allocation for every node.
fn update_signed_segment_ranges(
    view: &View<'_>,
    output: &mut [i64],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    let length = view.nulls.len();

    // Start every leaf with the operation identity. A null value must not
    // change a sum or product, so null leaves are zero for addition and one
    // for multiplication.
    let mut tree = vec![if product { 1_i64 } else { 0_i64 }; length * 2];

    // Install valid source values in the leaf half of the tree. The boolean
    // mask is authoritative; the numeric value is never inspected for
    // sentinel or null-like content.
    for position in 0..length {
        if !view.nulls[position] {
            tree[length + position] = as_i64(&view.values, position);
        }
    }

    // Build larger block summaries from the leaves upward. After this pass,
    // each tree node represents one contiguous half-open source block.
    for node in (1..length).rev() {
        tree[node] = if product {
            view.wrapping_mul(tree[node * 2], tree[node * 2 + 1])
        } else {
            view.wrapping_add(tree[node * 2], tree[node * 2 + 1])
        };
    }

    // Answer each requested range independently. `None` means that the
    // boundary row was invalid and therefore contributes no output value.
    for (row, range) in ranges.iter().enumerate() {
        let Some((mut start, mut end)) = range else {
            continue;
        };

        // The caller's source boundaries are half-open. Add `length` to map
        // them into the leaf portion of the flat tree without changing the
        // inclusive/exclusive meaning of either boundary.
        start += length;
        end += length;
        let mut total = if product { 1_i64 } else { 0_i64 };

        // Move both boundaries toward their common parent. Whenever a
        // boundary identifies a complete child block, consume that block and
        // move past it. This visits O(log n) blocks and never visits a source
        // element twice.
        while start < end {
            if start & 1 == 1 {
                total = if product {
                    view.wrapping_mul(total, tree[start])
                } else {
                    view.wrapping_add(total, tree[start])
                };
                start += 1;
            }
            if end & 1 == 1 {
                end -= 1;
                total = if product {
                    view.wrapping_mul(total, tree[end])
                } else {
                    view.wrapping_add(total, tree[end])
                };
            }
            start /= 2;
            end /= 2;
        }

        // All complete blocks covering this half-open range have now been
        // combined into the output slot for this row.
        output[row] = total;
    }
}

/// Answer arbitrary `u64` ranges with wrapping segment-tree arithmetic.
///
/// This is intentionally separate from the signed implementation. A `u64`
/// value may be larger than `i64::MAX`, so converting it would violate the
/// output dtype contract. The tree layout and query algorithm are otherwise
/// identical.
fn update_u64_segment_ranges(
    values: &ArrayView1<'_, u64>,
    nulls: &ArrayView1<'_, bool>,
    output: &mut [u64],
    ranges: &[Option<(usize, usize)>],
    product: bool,
) {
    let length = values.len();

    // Null leaves begin at the neutral element for the selected operation.
    let mut tree = vec![if product { 1_u64 } else { 0_u64 }; length * 2];

    // Copy only values whose mask says they are valid. No null inference is
    // performed from the u64 payload.
    for position in 0..length {
        if !nulls[position] {
            tree[length + position] = values[position];
        }
    }

    // Combine neighboring leaf blocks to build the reusable summaries.
    for node in (1..length).rev() {
        tree[node] = if product {
            tree[node * 2].wrapping_mul(tree[node * 2 + 1])
        } else {
            tree[node * 2].wrapping_add(tree[node * 2 + 1])
        };
    }

    // Query every valid half-open range using the same two-boundary walk as
    // the signed implementation, but retain u64 arithmetic throughout.
    for (row, range) in ranges.iter().enumerate() {
        let Some((mut start, mut end)) = range else {
            continue;
        };

        // Translate source positions to leaf positions. The end remains
        // exclusive, so the query includes exactly `start..end`.
        start += length;
        end += length;
        let mut total = if product { 1_u64 } else { 0_u64 };
        while start < end {
            if start & 1 == 1 {
                total = if product {
                    total.wrapping_mul(tree[start])
                } else {
                    total.wrapping_add(tree[start])
                };
                start += 1;
            }
            if end & 1 == 1 {
                end -= 1;
                total = if product {
                    total.wrapping_mul(tree[end])
                } else {
                    total.wrapping_add(tree[end])
                };
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
                view.wrapping_mul(total, value)
            } else {
                view.wrapping_add(total, value)
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
    min: bool,
) {
    let length = view.nulls.len();

    // `-1` means that a block has no valid candidate. `better_extreme` treats
    // it as an empty block and prefers a real position whenever one exists.
    let mut tree = vec![-1_i64; length * 2];

    // A valid leaf stores its source position. A masked leaf stays `-1`; the
    // mask, rather than a special numeric value, decides validity.
    for position in 0..length {
        if !view.nulls[position] {
            tree[length + position] = position as i64;
        }
    }

    // Build one winning position for every larger contiguous block. Strict
    // comparisons preserve the existing winner when values tie; the API only
    // requires that the returned position identify an actual extreme value.
    for node in (1..length).rev() {
        tree[node] = better_extreme(view, tree[node * 2], tree[node * 2 + 1], min);
    }
    for (row, range) in ranges.iter().enumerate() {
        let Some((mut start, mut end)) = range else {
            continue;
        };

        // Move from source coordinates into the leaf half of the flat tree;
        // `end` remains exclusive throughout the query.
        start += length;
        end += length;
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

/// Count valid values across dense reverse ranges.
///
/// This is the direct fallback for sparse starts-only and ends-only batches.
/// The mask is authoritative: a `true` entry marks the source value null and
/// contributes nothing; a `false` entry contributes one to every output slot
/// covered by that source row. Count-all is intentionally handled by its
/// separate writer because it must ignore this mask.
fn update_count_ranges(view: &View<'_>, output: &mut [i64], ranges: &[Option<(usize, usize)>]) {
    for (row, range) in ranges.iter().enumerate() {
        let Some((start, end)) = range else { continue };
        if view.nulls[row] {
            continue;
        }
        for slot in &mut output[*start..*end] {
            *slot += 1;
        }
    }
}

/// Count valid source rows across dense reverse ranges without a value view.
///
/// This is the dtype-independent counterpart to [`update_count_ranges`]. The
/// wildcard `count` request supplies only its authoritative null mask, so the
/// aggregation never needs to extract or inspect the source dtype.
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
                view.wrapping_mul(*slot, value)
            } else {
                view.wrapping_add(*slot, value)
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

/// Reduce an `i64` value to the source integer width before storing it.
///
/// The accumulator remains represented as `i64` for the existing output
/// contract, but every update is reduced to the source width so narrow
/// integer overflow wraps where the input dtype would wrap.
fn wrap_integer(value: i64, width: IntegerWidth) -> i64 {
    match width {
        IntegerWidth::I64 => value,
        IntegerWidth::I32 => value as i32 as i64,
        IntegerWidth::I16 => value as i16 as i64,
        IntegerWidth::I8 => value as i8 as i64,
        IntegerWidth::U32 => (value as u32) as i64,
        IntegerWidth::U16 => (value as u16) as i64,
        IntegerWidth::U8 => (value as u8) as i64,
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
