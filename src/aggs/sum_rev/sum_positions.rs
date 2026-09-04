use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_index, checked_range, ensure_equal_lengths_core, ensure_nonempty_core,
    should_use_dense_match_storage, WrapAdd,
};
use std::collections::{hash_map::Entry, HashMap};

/// Accumulate integer values for the indirect ranges in `positions`.
/// `index` is expected to contain unique labels, as guaranteed by the
/// pyjanitor producer for this path.
///
/// ELI5: `positions` already gives us a validated ordinal into `index`, so the
/// HashMap can use that ordinal directly. We only look up the original label
/// while emitting the result. Returned label/total pairs are aligned, but their
/// ordering is unspecified and follows HashMap iteration order in sparse mode.
/// Integer totals use wrapping arithmetic, so overflow has the same
/// deterministic result in debug and release builds. `A` is the accumulator
/// type: every integer dtype instantiates this with `A = i64`, except
/// `uint64`, which instantiates it with `A = u64` so values `>= 2**63`
/// don't get sign-flipped by a forced `i64` cast (see `WrapAdd`).
///
/// # Arguments
///
/// * `arr` - Left-side values; must not be empty.
/// * `starts` - Inclusive start of each half-open tape range.
/// * `ends` - Exclusive end of each half-open tape range.
/// * `index` - Right-side labels addressed by ordinal; must not be empty.
/// * `positions` - Candidate tape of ordinals into `index`; must not be empty.
/// * `booleans` - Null mask for `arr`; true rows do not contribute sums.
/// * `to_acc` - Converts each left value to the accumulator type.
#[allow(clippy::too_many_arguments)]
pub fn sum_positions_int_core<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_acc: F,
) -> Result<(Vec<i64>, Vec<A>), String>
where
    T: Copy,
    A: WrapAdd,
    F: Fn(T) -> A,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let dense = should_use_dense_match_storage(index.len(), positions.len());
    Ok(sum_positions_int_core_with_storage_unchecked(
        arr, starts, ends, index, positions, booleans, to_acc, dense,
    ))
}

/// Run integer positional summation with an explicit storage mode.
/// This is a Rust-only benchmark entry point; production callers should use
/// [`sum_positions_int_core`] for automatic dispatch.
///
/// The array arguments and `to_acc` have the same meanings as
/// [`sum_positions_int_core`]. `dense` selects vector storage when true and
/// HashMap storage when false.
#[allow(clippy::too_many_arguments)]
pub fn sum_positions_int_core_with_storage<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_acc: F,
    dense: bool,
) -> Result<(Vec<i64>, Vec<A>), String>
where
    T: Copy,
    A: WrapAdd,
    F: Fn(T) -> A,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    Ok(sum_positions_int_core_with_storage_unchecked(
        arr, starts, ends, index, positions, booleans, to_acc, dense,
    ))
}

#[allow(clippy::too_many_arguments)]
fn sum_positions_int_core_with_storage_unchecked<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_acc: F,
    dense: bool,
) -> (Vec<i64>, Vec<A>)
where
    T: Copy,
    A: WrapAdd,
    F: Fn(T) -> A,
{
    // Dense storage trades memory for direct ordinal indexing. `positions.len()`
    // is only a cheap tape-size heuristic, not a distinct-ordinal count; a
    // highly repeated tape can therefore overselect dense storage. The sparse
    // fallback handles shorter tapes, and adaptive promotion is a future
    // refinement.
    if dense {
        let mut seen = vec![false; index.len()];
        let mut totals = vec![A::ZERO; index.len()];
        for (current, start, end, boolean) in izip!(
            arr.into_iter(),
            starts.into_iter(),
            ends.into_iter(),
            booleans.into_iter()
        ) {
            let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                continue;
            };
            let current_ = to_acc(*current);
            for nn in start_..end_ {
                let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                    continue;
                };
                seen[indexer_] = true;
                // Insert state before checking the mask: null-only labels
                // must still be emitted with the additive identity.
                if !*boolean {
                    totals[indexer_] = totals[indexer_].wrap_add(current_);
                }
            }
        }
        let mut labels = Vec::new();
        let mut values = Vec::new();
        for (ordinal, was_seen) in seen.into_iter().enumerate() {
            if was_seen {
                labels.push(index[ordinal]);
                values.push(totals[ordinal]);
            }
        }
        return (labels, values);
    }

    let mut totals: HashMap<usize, A> = HashMap::new();

    for (current, start, end, boolean) in izip!(
        arr.into_iter(),
        starts.into_iter(),
        ends.into_iter(),
        booleans.into_iter()
    ) {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let current_ = to_acc(*current);
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                continue;
            };
            let total = match totals.entry(indexer_) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(A::ZERO),
            };
            // Insert state before checking the mask: null-only labels must
            // still be emitted with the additive identity.
            if !*boolean {
                // ELI5: once a label's bucket is found, add in the same
                // wraparound style as the forward kernel. This keeps a very
                // large integer from panicking only in debug/test builds.
                *total = total.wrap_add(current_);
            }
        }
    }

    let mut labels = Vec::with_capacity(totals.len());
    let mut values = Vec::with_capacity(totals.len());
    for (ordinal, value) in totals {
        labels.push(index[ordinal]);
        values.push(value);
    }
    (labels, values)
}

/// Accumulate floating-point values while retaining pandas-style compensated
/// summation. The ordinal map stores both pieces of state together.
/// `index` is expected to contain unique labels, as guaranteed by the
/// pyjanitor producer for this path.
///
/// ELI5: the ordinal is the state key, so one map lookup finds both the total
/// and its rounding-error correction without a label-to-slot translation.
///
/// # Arguments
///
/// * `arr` - Left-side floating-point values; must not be empty.
/// * `starts` - Inclusive start of each half-open tape range.
/// * `ends` - Exclusive end of each half-open tape range.
/// * `index` - Right-side labels addressed by ordinal; must not be empty.
/// * `positions` - Candidate tape of ordinals into `index`; must not be empty.
/// * `booleans` - Null mask for `arr`; true rows do not contribute sums.
/// * `to_f64` - Converts each left value to `f64`.
#[allow(clippy::too_many_arguments)]
pub fn sum_positions_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_f64: F,
) -> Result<(Vec<i64>, Vec<f64>), String>
where
    T: Copy,
    F: Fn(T) -> f64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let dense = should_use_dense_match_storage(index.len(), positions.len());
    Ok(sum_positions_float_core_with_storage_unchecked(
        arr, starts, ends, index, positions, booleans, to_f64, dense,
    ))
}

/// Run floating-point positional summation with an explicit storage mode.
/// This is a Rust-only benchmark entry point; production callers should use
/// [`sum_positions_float_core`] for automatic dispatch.
///
/// The array arguments and `to_f64` have the same meanings as
/// [`sum_positions_float_core`]. `dense` selects vector storage when true and
/// HashMap storage when false.
#[allow(clippy::too_many_arguments)]
pub fn sum_positions_float_core_with_storage<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_f64: F,
    dense: bool,
) -> Result<(Vec<i64>, Vec<f64>), String>
where
    T: Copy,
    F: Fn(T) -> f64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    Ok(sum_positions_float_core_with_storage_unchecked(
        arr, starts, ends, index, positions, booleans, to_f64, dense,
    ))
}

#[allow(clippy::too_many_arguments)]
fn sum_positions_float_core_with_storage_unchecked<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_f64: F,
    dense: bool,
) -> (Vec<i64>, Vec<f64>)
where
    T: Copy,
    F: Fn(T) -> f64,
{
    if dense {
        let mut seen = vec![false; index.len()];
        let mut totals = vec![(0., 0.); index.len()];
        for (current, start, end, boolean) in izip!(
            arr.into_iter(),
            starts.into_iter(),
            ends.into_iter(),
            booleans.into_iter()
        ) {
            let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                continue;
            };
            let current_ = to_f64(*current);
            for nn in start_..end_ {
                let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                    continue;
                };
                seen[indexer_] = true;
                // Insert state before checking the mask: null-only labels
                // must still be emitted with the additive identity.
                if *boolean {
                    continue;
                }
                let (total, compensation) = &mut totals[indexer_];
                let difference = current_ - *compensation;
                let increment = *total + difference;
                *compensation = (increment - *total) - difference;
                if !compensation.is_finite() {
                    *compensation = 0.;
                }
                *total = increment;
            }
        }
        let mut labels = Vec::new();
        let mut values = Vec::new();
        for (ordinal, was_seen) in seen.into_iter().enumerate() {
            if was_seen {
                labels.push(index[ordinal]);
                let (total, _compensation) = totals[ordinal];
                // Preserve the established reverse-sum result contract. The
                // compensation is internal correction state, not output.
                values.push(total);
            }
        }
        return (labels, values);
    }

    let mut totals: HashMap<usize, (f64, f64)> = HashMap::new();

    for (current, start, end, boolean) in izip!(
        arr.into_iter(),
        starts.into_iter(),
        ends.into_iter(),
        booleans.into_iter()
    ) {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let current_ = to_f64(*current);
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                continue;
            };
            let (total, compensation) = match totals.entry(indexer_) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert((0., 0.)),
            };
            // Insert state before checking the mask: null-only labels must
            // still be emitted with the additive identity.
            if *boolean {
                continue;
            }
            let difference = current_ - *compensation;
            let increment = *total + difference;
            *compensation = (increment - *total) - difference;
            // Adapted from pandas' cython code. Infinite values should not
            // turn the compensation term into NaN and poison later sums.
            if !compensation.is_finite() {
                *compensation = 0.;
            }
            *total = increment;
        }
    }

    let mut labels = Vec::with_capacity(totals.len());
    let mut values = Vec::with_capacity(totals.len());
    for (ordinal, (total, _compensation)) in totals {
        labels.push(index[ordinal]);
        // Preserve the established reverse-sum result contract. The
        // compensation is internal correction state, not output.
        values.push(total);
    }
    (labels, values)
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        /// Sum values for labels reached through the positional candidate
        /// tape. Integer accumulation wraps on overflow.
        ///
        /// # Arguments
        /// * `arr` - Left-side values; must not be empty.
        /// * `starts` - Inclusive positional range starts.
        /// * `ends` - Exclusive positional range ends.
        /// * `index` - Right-side labels addressed by `positions`.
        /// * `positions` - Positional candidate tape.
        /// * `booleans` - Null mask; `True` rows are ignored.
        ///
        /// `arr`, `index`, and `positions` must not be empty. `starts`, `ends`,
        /// and `booleans` must match `arr` in length. Returned label/value
        /// pairs are aligned but unordered.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let (labels, totals) =
                sum_positions_int_core(arr, starts, ends, index, positions, booleans, |value| {
                    value as $acc
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(totals);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapAdd`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_sum_rev_positions_int64, i64, i64);
compute_ints!(compute_sum_rev_positions_int32, i32, i64);
compute_ints!(compute_sum_rev_positions_int16, i16, i64);
compute_ints!(compute_sum_rev_positions_int8, i8, i64);
compute_ints!(compute_sum_rev_positions_uint64, u64, u64);
compute_ints!(compute_sum_rev_positions_uint32, u32, i64);
compute_ints!(compute_sum_rev_positions_uint16, u16, i64);
compute_ints!(compute_sum_rev_positions_uint8, u8, i64);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// Sum floating-point values for labels reached through the positional
        /// candidate tape using compensated accumulation.
        ///
        /// # Arguments
        /// * `arr` - Left-side values; must not be empty.
        /// * `starts` - Inclusive positional range starts.
        /// * `ends` - Exclusive positional range ends.
        /// * `index` - Right-side labels addressed by `positions`.
        /// * `positions` - Positional candidate tape.
        /// * `booleans` - Null mask; `True` rows are ignored.
        ///
        /// `arr`, `index`, and `positions` must not be empty. `starts`, `ends`,
        /// and `booleans` must match `arr` in length. Returned label/value
        /// pairs are aligned but unordered.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let (labels, totals) =
                sum_positions_float_core(arr, starts, ends, index, positions, booleans, |value| {
                    value as f64
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(totals);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_sum_rev_positions_f64, f64);
compute_floats!(compute_sum_rev_positions_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        sum_positions_float_core, sum_positions_float_core_with_storage, sum_positions_int_core,
    };
    use numpy::ndarray::array;

    #[test]
    fn integer_positions_keep_labels_and_skip_invalid_entries() {
        let arr = array![10_i64, 20, 30];
        let starts = array![0_i64, 2, 4];
        let ends = array![2_i64, 4, 5];
        let index = array![100_i64, 200, 300];
        let positions = array![1_i64, 0, 1, 2, -1];
        let booleans = array![false, true, false];

        let (labels, totals) = sum_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        let mut got: Vec<_> = labels.into_iter().zip(totals).collect();
        got.sort_unstable();
        // The null row still creates slots, but contributes no value.
        assert_eq!(got, vec![(100, 10), (200, 10), (300, 0)]);
    }

    #[test]
    fn integer_positions_skip_empty_and_inverted_ranges() {
        let arr = array![7_i64, 11];
        let starts = array![1_i64, 3];
        let ends = array![1_i64, 2];
        let index = array![42_i64];
        let positions = array![0_i64];
        let booleans = array![false, false];

        let (labels, totals) = sum_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert!(labels.is_empty());
        assert!(totals.is_empty());
    }

    #[test]
    fn floating_positions_use_compensated_sum_for_duplicate_labels() {
        let arr = array![0.1_f64, 0.2, 0.3];
        let starts = array![0_i64, 1, 2];
        let ends = array![1_i64, 2, 3];
        let index = array![5_i64];
        let positions = array![0_i64, 0, 0];
        let booleans = array![false, false, false];

        let (labels, totals) = sum_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert_eq!(labels, vec![5]);
        assert!((totals[0] - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn floating_positions_preserve_cancellation_result_in_both_storage_modes() {
        let arr = array![1e16_f64, 1.0, -1e16];
        let starts = array![0_i64, 1, 2];
        let ends = array![1_i64, 2, 3];
        let index = array![7_i64];
        let positions = array![0_i64, 0, 0];
        let booleans = array![false, false, false];

        for dense in [false, true] {
            let (labels, totals) = sum_positions_float_core_with_storage(
                arr.view(),
                starts.view(),
                ends.view(),
                index.view(),
                positions.view(),
                booleans.view(),
                |value| value,
                dense,
            )
            .unwrap();
            assert_eq!(labels, vec![7]);
            assert_eq!(totals, vec![0.0]);
        }
    }

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let arr = array![value];
        let starts = array![0_i64];
        let ends = array![1_i64];
        let index = array![5_i64];
        let positions = array![0_i64];
        let booleans = array![false];
        let (labels, totals) = sum_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |v: u64| v,
        )
        .unwrap();
        assert_eq!(labels, vec![5]);
        assert_eq!(totals, vec![value]);
    }

    #[test]
    fn integer_positions_wrap_on_overflow() {
        let arr = array![i64::MAX, 1];
        let starts = array![0_i64, 1];
        let ends = array![1_i64, 2];
        let index = array![5_i64];
        let positions = array![0_i64, 0];
        let booleans = array![false, false];

        let (labels, totals) = sum_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert_eq!(labels, vec![5]);
        assert_eq!(totals, vec![i64::MIN]);
    }
}
