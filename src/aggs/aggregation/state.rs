//! Shared forward aggregation state for fused conditional-join comparisons.
//!
//! The comparison kernels own candidate traversal.  They report successful
//! `(output_row, candidate_position)` pairs to [`AggregationSet`], which keeps
//! one accumulator per requested operation and emits one result per output
//! row.

use super::input::{AggregationInput, AggregationOp};
use numpy::ndarray::{Array1, ArrayView1};
use numpy::IntoPyArray;
use pyo3::prelude::*;

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

/// The source values and their null metadata for one requested aggregation.
///
/// The value array is expected to be null-free; null tracking is supplied
/// separately by `nulls`. A `true` entry means the corresponding value is
/// null, while `false` means it is valid. The mask is authoritative: this code
/// does not inspect a value or infer nullness from sentinels or special values.
/// Null metadata is used by value-based operations (`sum`, `product`, `min`,
/// and `max`), while `count` intentionally counts the comparison event
/// regardless of this mask.
struct View<'a> {
    values: Values<'a>,
    nulls: ArrayView1<'a, bool>,
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
    Count(Vec<i64>),
    Product(View<'a>, Vec<i64>),
    ProductU64(ArrayView1<'a, u64>, ArrayView1<'a, bool>, Vec<u64>),
    ProductF64(View<'a>, Vec<f64>),
    Min(View<'a>, Vec<i64>),
    Max(View<'a>, Vec<i64>),
}

pub(crate) struct AggregationSet<'a> {
    /// One independent accumulator for every requested `(array, mask, op)`.
    states: Vec<State<'a>>,
    /// Number of rows produced by the comparison side of the join.
    output_len: usize,
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
    /// * `output_len` - Number of left/comparison rows. Every output array has
    ///   this length.
    /// * `expected_candidate_len` - Number of candidate positions in each
    ///   aggregation value array.
    /// * `inputs` - Parsed aggregation requests. Every request contains a
    ///   typed candidate array, a boolean null mask, and an operation.
    ///
    /// # Errors
    ///
    /// Returns a Python exception if aggregation arrays or null masks have
    /// inconsistent lengths, or if an input cannot be represented by the
    /// internal typed state.
    pub(crate) fn new(
        output_len: usize,
        expected_candidate_len: usize,
        inputs: &'a [AggregationInput<'_>],
    ) -> PyResult<Self> {
        let mut states = Vec::with_capacity(inputs.len());
        let mut candidate_len = None;
        for input in inputs {
            let (values, nulls, op) = match input {
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
                "comparison right array",
                expected_candidate_len,
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
                AggregationOp::Count => State::Count(vec![0; output_len]),
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
        Ok(Self {
            states,
            output_len,
            successful: false,
        })
    }

    /// Apply one successful comparison to every requested aggregation.
    ///
    /// `output_row` identifies the left row whose result should be updated;
    /// `candidate` identifies the matching position in each right-side
    /// aggregation array. The comparison modules call this once per
    /// successful candidate, so multiple aggregations share one traversal.
    ///
    /// Count deliberately ignores null metadata and therefore counts every
    /// successful comparison. Other operations skip a candidate whose mask is
    /// `true`. A `false` mask is treated as an assertion that the value array is
    /// null-free and the value is valid; no additional null inference or
    /// sentinel filtering is performed.
    /// Out-of-range output rows are ignored defensively; candidate bounds are
    /// guaranteed by construction and validation in the comparison callers.
    ///
    /// # Arguments
    ///
    /// * `output_row` - Zero-based output row to update.
    /// * `candidate` - Zero-based candidate position in the source arrays.
    pub(crate) fn update(&mut self, output_row: usize, candidate: usize) {
        if output_row >= self.output_len {
            return;
        }
        self.successful = true;
        // One successful comparison is one event. Broadcast that event to
        // every requested aggregation before moving to the next candidate.
        // This is the central multi-aggregation benefit: predicates and
        // candidate discovery run once, while all requested reductions share
        // that traversal.
        for state in &mut self.states {
            match state {
                State::Count(values) => values[output_row] += 1,
                State::Sum(view, values) => {
                    if !view.nulls[candidate] {
                        values[output_row] =
                            values[output_row].wrapping_add(as_i64(&view.values, candidate));
                    }
                }
                State::SumU64(source, nulls, values) => {
                    if !nulls[candidate] {
                        values[output_row] = values[output_row].wrapping_add(source[candidate]);
                    }
                }
                State::SumF64(view, values, compensation) => {
                    if !view.nulls[candidate] {
                        kahan_add(
                            &mut values[output_row],
                            &mut compensation[output_row],
                            as_f64(&view.values, candidate),
                        );
                    }
                }
                State::Product(view, values) => {
                    if !view.nulls[candidate] {
                        values[output_row] =
                            values[output_row].wrapping_mul(as_i64(&view.values, candidate));
                    }
                }
                State::ProductU64(source, nulls, values) => {
                    if !nulls[candidate] {
                        values[output_row] = values[output_row].wrapping_mul(source[candidate]);
                    }
                }
                State::ProductF64(view, values) => {
                    if !view.nulls[candidate] {
                        values[output_row] *= as_f64(&view.values, candidate);
                    }
                }
                State::Min(view, positions) => {
                    if !view.nulls[candidate] {
                        let current = positions[output_row];
                        if current < 0
                            || compare(view, candidate, current as usize)
                                == std::cmp::Ordering::Less
                        {
                            positions[output_row] = candidate as i64;
                        }
                    }
                }
                State::Max(view, positions) => {
                    if !view.nulls[candidate] {
                        let current = positions[output_row];
                        if current < 0
                            || compare(view, candidate, current as usize)
                                == std::cmp::Ordering::Greater
                        {
                            positions[output_row] = candidate as i64;
                        }
                    }
                }
            }
        }
    }

    /// Return whether no successful comparison has been observed.
    ///
    /// The comparison wrappers use this to return Python `None` instead of a
    /// collection of identity-filled arrays when no candidate matched.
    pub(crate) fn is_empty(&self) -> bool {
        !self.successful
    }

    /// Convert all accumulator buffers into NumPy arrays.
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
    /// A vector containing one NumPy array per requested aggregation.
    pub(crate) fn into_results(self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.states
            .into_iter()
            .map(|state| match state {
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
                State::Count(v) | State::Min(_, v) | State::Max(_, v) => {
                    Array1::from_vec(v).into_pyarray(py).unbind().into_any()
                }
            })
            .collect()
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

/// Add one float using the Kahan-style compensation used by existing forward
/// kernels. The running total is the public result; compensation only carries
/// rounding information into the next update.
fn kahan_add(total: &mut f64, compensation: &mut f64, value: f64) {
    let difference = value - *compensation;
    let increment = *total + difference;
    *compensation = (increment - *total) - difference;
    *total = increment;
}
fn compare(view: &View<'_>, a: usize, b: usize) -> std::cmp::Ordering {
    match &view.values {
        Values::I64(v) => v[a].partial_cmp(&v[b]),
        Values::I32(v) => v[a].partial_cmp(&v[b]),
        Values::I16(v) => v[a].partial_cmp(&v[b]),
        Values::I8(v) => v[a].partial_cmp(&v[b]),
        Values::U64(v) => v[a].partial_cmp(&v[b]),
        Values::U32(v) => v[a].partial_cmp(&v[b]),
        Values::U16(v) => v[a].partial_cmp(&v[b]),
        Values::U8(v) => v[a].partial_cmp(&v[b]),
        Values::F64(v) => v[a].partial_cmp(&v[b]),
        Values::F32(v) => v[a].partial_cmp(&v[b]),
    }
    .unwrap_or(std::cmp::Ordering::Greater)
}
