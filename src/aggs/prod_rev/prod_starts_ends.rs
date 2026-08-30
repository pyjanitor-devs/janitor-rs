use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    cached_row_value, ensure_equal_lengths, materialize_labels, range_reduce, WrapMul,
};

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if starts.len() != ends.len() {
        return Err("starts and ends must have equal lengths");
    }
    if arr.len() != starts.len() {
        return Err("arr, starts, and ends must have equal lengths");
    }
    if arr.len() != booleans.len() {
        return Err("arr and booleans must have equal lengths");
    }
    if arr.is_empty() || index.is_empty() {
        return Err("arr, starts, booleans, and index cannot be empty");
    }
    Ok(())
}

/// Multiply values into one state slot per right-row ordinal.
///
/// janitor-rs is primarily called by pyjanitor. Its conditional-join path
/// resets the right DataFrame index to unique row labels before sorting or
/// filtering, so labels can be reordered or gapped but are not duplicated.
/// `item`, the ordinal position in `index`, is the state slot; `index[item]`
/// is the output label. `starts` and `ends` describe half-open ranges
/// `[start, end)`. Invalid or zero-width ranges are skipped by `checked_range`,
/// and empty input arrays are rejected.
///
/// # Preconditions
///
/// `index` must contain unique labels in positional order. pyjanitor provides
/// this by normalizing the right side to `range(len(right))`; direct callers
/// must provide it themselves. Duplicate labels are unsupported and are not
/// merged by the positional accumulator.
///
/// A null row leaves a touched slot at the multiplicative identity, `1`.
/// Integer accumulators use `wrapping_mul` through `WrapMul`, matching the
/// project's overflow contract. `A` is `i64` for ordinary integer dtypes and
/// `u64` for `uint64`, preserving values at and above `2**63`.
///
/// ELI5: every right-row position gets a numbered drawer containing its
/// product. The printed row identity may be 42, 7, or 100; the drawer number
/// is its position on the shelf, not the printed identity.
pub fn prod_rev_start_end_int_core<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut convert: F,
) -> Result<(Array1<i64>, Array1<A>), &'static str>
where
    T: Copy,
    A: WrapMul,
    F: FnMut(T) -> A,
{
    validate_inputs(arr, starts, ends, index, booleans)?;
    let (touched, products) =
        range_reduce(starts, ends, index.len(), A::ONE, |row, _item, product| {
            if !booleans[row] {
                *product = product.wrap_mul(convert(arr[row]));
            }
        });
    let labels = materialize_labels(&touched, index);
    Ok((labels, Array1::from_vec(products)))
}

pub fn prod_rev_start_end_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    validate_inputs(arr, starts, ends, index, booleans)?;
    // Cache only the current row's conversion. The reducer visits one row's
    // range contiguously, so this avoids both repeated conversion and an
    // O(arr.len()) memoization allocation.
    let mut cached_row = usize::MAX;
    let mut cached_value = 0.0_f64;
    // Integer products use `WrapMul` because integer overflow is intentionally
    // modular in the reverse kernels. Floating-point multiplication follows
    // IEEE-754 instead: overflow yields signed infinity, and invalid cases
    // such as zero times infinity yield NaN. There is no float analogue of
    // integer `wrapping_mul`, so applying the integer rule here would change
    // the established float semantics.
    //
    // ELI5: integer arithmetic goes around a fixed-size loop; float arithmetic
    // can leave the loop and say “infinity” or “not a number.”
    let (touched, products) =
        range_reduce(starts, ends, index.len(), 1.0_f64, |row, _item, product| {
            if !booleans[row] {
                let current =
                    cached_row_value(row, arr, &mut cached_row, &mut cached_value, &mut to_f64);
                *product *= current;
            }
        });
    let labels = materialize_labels(&touched, index);
    Ok((labels, Array1::from_vec(products)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        /// `index` must be a unique positional domain. pyjanitor supplies a
        /// normalized `range(len(right))`; direct callers must provide it.
        /// This correctness precondition is unchecked to avoid an extra pass.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)> {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let result = prod_rev_start_end_int_core(arr, starts, ends, index, booleans, |value| {
                value as $acc
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((result.0.into_pyarray(py), result.1.into_pyarray(py)))
        }
    };
}

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// `index` must be a unique positional domain. pyjanitor supplies a
        /// normalized `range(len(right))`; direct callers must provide it.
        /// This correctness precondition is unchecked to avoid an extra pass.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let result =
                prod_rev_start_end_float_core(arr, starts, ends, index, booleans, |value| {
                    value as f64
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((result.0.into_pyarray(py), result.1.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapMul`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_prod_rev_start_end_int64, i64, i64);
compute_ints!(compute_prod_rev_start_end_int32, i32, i64);
compute_ints!(compute_prod_rev_start_end_int16, i16, i64);
compute_ints!(compute_prod_rev_start_end_int8, i8, i64);
compute_ints!(compute_prod_rev_start_end_uint64, u64, u64);
compute_ints!(compute_prod_rev_start_end_uint32, u32, i64);
compute_ints!(compute_prod_rev_start_end_uint16, u16, i64);
compute_ints!(compute_prod_rev_start_end_uint8, u8, i64);
compute_floats!(compute_prod_rev_start_end_f64, f64);
compute_floats!(compute_prod_rev_start_end_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let got = prod_rev_start_end_int_core(
            array![value].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![10_i64].view(),
            array![false].view(),
            |v: u64| v,
        );
        assert_eq!(got, Ok((array![10], array![value])));
    }

    #[test]
    fn dense_slots_multiply_unique_gapped_labels() {
        let got = prod_rev_start_end_int_core(
            array![2_i64, 3, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![42_i64, 7, 100].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![42, 7, 100], array![8, 6, 3])));
    }

    #[test]
    fn duplicate_index_labels_are_explicitly_unsupported() {
        let got = prod_rev_start_end_int_core(
            array![2_i64, 3, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![10_i64, 20, 10].view(),
            array![false, false, false].view(),
            |value| value,
        );
        // Ordinal slots are independent; duplicate labels are not merged.
        assert_eq!(got, Ok((array![10, 20, 10], array![8, 6, 3])));
    }

    #[test]
    fn null_rows_emit_labels_with_multiplicative_identity() {
        let got = prod_rev_start_end_int_core(
            array![5_i64].view(),
            array![0_i64].view(),
            array![2_i64].view(),
            array![10_i64, 20].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10, 20], array![1, 1])));
    }

    #[test]
    fn invalid_or_zero_width_ranges_are_skipped() {
        let got = prod_rev_start_end_int_core(
            array![1_i64, 2, 3].view(),
            array![2_i64, -1, 1].view(),
            array![2_i64, 1, 4].view(),
            array![10_i64, 20].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![], array![])));
    }

    #[test]
    fn validation_rejects_shape_mismatches_and_empty_inputs() {
        assert!(prod_rev_start_end_int_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
        assert!(prod_rev_start_end_int_core(
            (array![] as Array1<i64>).view(),
            array![].view(),
            array![].view(),
            array![10_i64].view(),
            array![].view(),
            |value: i64| value,
        )
        .is_err());
    }
}
