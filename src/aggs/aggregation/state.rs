//! Shared forward aggregation state for fused conditional-join comparisons.
//!
//! The comparison kernels own candidate traversal.  They report successful
//! `(output_row, candidate_position)` pairs to [`AggregationSet`], which keeps
//! one accumulator per requested operation and emits one result per output
//! row.

use super::input::AggregationInput;
use super::op::AggregationOp;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::aggs::ensure_equal_lengths;

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

struct View<'a> {
    values: Values<'a>,
    nulls: ArrayView1<'a, bool>,
}

enum State<'a> {
    // A state variant stores both the typed source view and the output
    // buffer. Keeping them together prevents an update from accidentally
    // pairing an operation with a source array of the wrong dtype.
    Sum(View<'a>, Vec<i64>),
    SumU64(View<'a>, Vec<u64>),
    SumF64(View<'a>, Vec<f64>, Vec<f64>),
    Count(Vec<i64>),
    Product(View<'a>, Vec<i64>),
    ProductU64(View<'a>, Vec<u64>),
    ProductF64(View<'a>, Vec<f64>),
    Min(View<'a>, Vec<i64>),
    Max(View<'a>, Vec<i64>),
}

pub(crate) struct AggregationSet<'a> {
    states: Vec<State<'a>>,
    output_len: usize,
    successful: bool,
}

impl<'a> AggregationSet<'a> {
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
            let view = View { values, nulls };
            let state = match op {
                AggregationOp::Sum => match &view.values {
                    Values::U64(_) => State::SumU64(view, vec![0; output_len]),
                    Values::F64(_) | Values::F32(_) => {
                        State::SumF64(view, vec![0.; output_len], vec![0.; output_len])
                    }
                    _ => State::Sum(view, vec![0; output_len]),
                },
                AggregationOp::Count => State::Count(vec![0; output_len]),
                AggregationOp::Product => match &view.values {
                    Values::U64(_) => State::ProductU64(view, vec![1; output_len]),
                    Values::F64(_) | Values::F32(_) => {
                        State::ProductF64(view, vec![1.; output_len])
                    }
                    _ => State::Product(view, vec![1; output_len]),
                },
                AggregationOp::Min => State::Min(view, vec![-1; output_len]),
                AggregationOp::Max => State::Max(view, vec![-1; output_len]),
            };
            states.push(state);
        }
        Ok(Self {
            states,
            output_len,
            successful: false,
        })
    }

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
                State::SumU64(view, values) => {
                    if !view.nulls[candidate] {
                        values[output_row] =
                            values[output_row].wrapping_add(as_u64(&view.values, candidate));
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
                State::ProductU64(view, values) => {
                    if !view.nulls[candidate] {
                        values[output_row] =
                            values[output_row].wrapping_mul(as_u64(&view.values, candidate));
                    }
                }
                State::ProductF64(view, values) => {
                    if !view.nulls[candidate] {
                        values[output_row] *= as_f64(&view.values, candidate);
                    }
                }
                State::Min(view, positions) => {
                    update_extreme(view, positions, output_row, candidate, true)
                }
                State::Max(view, positions) => {
                    update_extreme(view, positions, output_row, candidate, false)
                }
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.successful
    }
    pub(crate) fn into_results(self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.states
            .into_iter()
            .map(|state| match state {
                State::Sum(_, v) | State::Product(_, v) => {
                    Array1::from_vec(v).into_pyarray(py).unbind().into_any()
                }
                State::SumU64(_, v) | State::ProductU64(_, v) => {
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

/// Add one value using the Kahan-style compensation used by the existing
/// forward float-sum kernels.
///
/// ELI5: `compensation` remembers the tiny rounding error lost by the last
/// addition and feeds it into the next addition. The public result remains
/// `total`, matching the established aggregation contract.
fn kahan_add(total: &mut f64, compensation: &mut f64, value: f64) {
    let difference = value - *compensation;
    let increment = *total + difference;
    *compensation = (increment - *total) - difference;
    *total = increment;
}

fn as_i64(values: &Values<'_>, n: usize) -> i64 {
    match values {
        Values::I64(v) => v[n],
        Values::I32(v) => v[n] as i64,
        Values::I16(v) => v[n] as i64,
        Values::I8(v) => v[n] as i64,
        Values::U64(v) => v[n] as i64,
        Values::U32(v) => v[n] as i64,
        Values::U16(v) => v[n] as i64,
        Values::U8(v) => v[n] as i64,
        Values::F64(v) => v[n] as i64,
        Values::F32(v) => v[n] as i64,
    }
}
fn as_u64(values: &Values<'_>, n: usize) -> u64 {
    match values {
        Values::U64(v) => v[n],
        Values::I64(v) => v[n] as u64,
        Values::I32(v) => v[n] as u64,
        Values::I16(v) => v[n] as u64,
        Values::I8(v) => v[n] as u64,
        Values::U32(v) => v[n] as u64,
        Values::U16(v) => v[n] as u64,
        Values::U8(v) => v[n] as u64,
        Values::F64(v) => v[n] as u64,
        Values::F32(v) => v[n] as u64,
    }
}
fn as_f64(values: &Values<'_>, n: usize) -> f64 {
    match values {
        Values::F64(v) => v[n],
        Values::F32(v) => v[n] as f64,
        Values::I64(v) => v[n] as f64,
        Values::I32(v) => v[n] as f64,
        Values::I16(v) => v[n] as f64,
        Values::I8(v) => v[n] as f64,
        Values::U64(v) => v[n] as f64,
        Values::U32(v) => v[n] as f64,
        Values::U16(v) => v[n] as f64,
        Values::U8(v) => v[n] as f64,
    }
}
fn update_extreme(
    view: &View<'_>,
    positions: &mut [i64],
    row: usize,
    candidate: usize,
    minimum: bool,
) {
    if view.nulls[candidate] {
        return;
    }
    let current = positions[row];
    if current < 0
        || if minimum {
            compare(view, candidate, current as usize) == std::cmp::Ordering::Less
        } else {
            compare(view, candidate, current as usize) == std::cmp::Ordering::Greater
        }
    {
        positions[row] = candidate as i64;
    }
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
