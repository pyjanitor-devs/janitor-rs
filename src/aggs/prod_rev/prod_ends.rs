use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    ends_domain, ends_labels, ensure_equal_lengths_core, into_starts_ends_result, sweep_reduce,
    WrapMul,
};

/// `A` is the accumulator type: every integer dtype instantiates this with
/// `A = i64`, except `uint64`, which instantiates it with `A = u64` so
/// values `>= 2**63` don't get sign-flipped by a forced `i64` cast (see
/// `WrapMul`).
#[allow(private_bounds)]
pub fn prod_rev_ends_int_core<T: Copy, A: WrapMul, F: FnMut(T) -> A>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut convert: F,
) -> Result<(Array1<i64>, Array1<A>), &'static str> {
    ensure_equal_lengths_core(
        arr.len(),
        ends.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    ensure_equal_lengths_core(
        arr.len(),
        booleans.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    let max_end = ends_domain(ends, index.len())?;
    // ELI5: a prefix row joins the running product just to the left of its
    // end. Combine rows at the same end into one product event, then sweep
    // right-to-left and apply each event once. Wrapping multiplication is
    // associative, so no per-row `next` metadata is needed; `end == 0` has
    // no emitted slot and therefore contributes nothing.
    let events = izip!(arr, ends, booleans)
        .filter(|(_, end, boolean)| !**boolean && **end > 0)
        .map(|(current, end, _)| (*end as usize - 1, convert(*current)));
    let values = sweep_reduce(
        max_end,
        A::ONE,
        events,
        (0..max_end).rev(),
        |left, right| left.wrap_mul(right),
    )?;
    Ok((ends_labels(max_end, index), values))
}

pub fn prod_rev_ends_float_core<T: Copy, F: FnMut(T) -> f64>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str> {
    ensure_equal_lengths_core(
        arr.len(),
        ends.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    ensure_equal_lengths_core(
        arr.len(),
        booleans.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    let max_end = ends_domain(ends, index.len())?;
    // Keep floats on the direct nested loop. Unlike integer wrapping
    // multiplication, IEEE-754 multiplication is not safely regroupable:
    // changing the order can change rounding and the propagation of NaN or
    // infinity. ELI5: integer products are exact on a wrapping number wheel,
    // but decimal-like floats can give a different answer when we rearrange
    // the multiplication, so this path preserves the old row/position order.
    let mut values = vec![1_f64; max_end];
    for (current, end, boolean) in izip!(arr, ends, booleans) {
        if *boolean {
            continue;
        }
        let current = to_f64(*current);
        // Example: `end = 3` means the row contributes to slots 0, 1, and 2;
        // `take(3)` selects exactly that prefix for multiplication.
        for value in values.iter_mut().take(*end as usize) {
            *value *= current;
        }
    }
    Ok((ends_labels(max_end, index), Array1::from_vec(values)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)> {
            into_starts_ends_result(
                py,
                prod_rev_ends_int_core(
                    arr.as_array(),
                    ends.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                    |value| value as $acc,
                ),
            )
        }
    };
}
macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
            into_starts_ends_result(
                py,
                prod_rev_ends_float_core(
                    arr.as_array(),
                    ends.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                    |value| value as f64,
                ),
            )
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapMul`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_prod_rev_end_int64, i64, i64);
compute_ints!(compute_prod_rev_end_int32, i32, i64);
compute_ints!(compute_prod_rev_end_int16, i16, i64);
compute_ints!(compute_prod_rev_end_int8, i8, i64);
compute_ints!(compute_prod_rev_end_uint64, u64, u64);
compute_ints!(compute_prod_rev_end_uint32, u32, i64);
compute_ints!(compute_prod_rev_end_uint16, u16, i64);
compute_ints!(compute_prod_rev_end_uint8, u8, i64);
compute_floats!(compute_prod_rev_end_f64, f64);
compute_floats!(compute_prod_rev_end_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn integer_core_groups_prefixes_and_preserves_identity() {
        let got = prod_rev_ends_int_core(
            array![2_i64, 3].view(),
            array![2_i64, 3].view(),
            array![20_i64, 10, 90].view(),
            array![false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![20, 10, 90], array![6, 6, 3])));

        let got = prod_rev_ends_int_core(
            array![2_i64].view(),
            array![1_i64].view(),
            array![10_i64].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10], array![1])));
    }

    #[test]
    fn integer_core_wraps_and_float_core_handles_infinity() {
        let got = prod_rev_ends_int_core(
            array![i64::MAX, 2].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10], array![-2])));

        let got = prod_rev_ends_float_core(
            array![f64::INFINITY, 2.0].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false, false].view(),
            |value| value,
        );
        assert!(got.unwrap().1[0].is_infinite());
    }

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let got = prod_rev_ends_int_core(
            array![value].view(),
            array![1_i64].view(),
            array![20_i64].view(),
            array![false].view(),
            |v: u64| v,
        );
        assert_eq!(got, Ok((array![20], array![value])));
    }

    #[test]
    fn rejects_invalid_bounds() {
        assert_eq!(
            prod_rev_ends_int_core(
                array![1_i64].view(),
                array![0_i64].view(),
                array![10_i64].view(),
                array![false].view(),
                |value| value,
            ),
            Ok((array![], array![]))
        );
    }

    #[test]
    fn integer_sweep_preserves_products_and_zero_width_rows() {
        let arr = Array1::from_elem(100, 2_i64);
        let mut ends = Array1::from_elem(100, 1000_i64);
        ends[99] = 0;
        let index = Array1::from_iter(0_i64..1000);
        let booleans = Array1::from_elem(100, false);
        let got = prod_rev_ends_int_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got.unwrap().1, Array1::from_elem(1000, 0_i64));
    }
}
