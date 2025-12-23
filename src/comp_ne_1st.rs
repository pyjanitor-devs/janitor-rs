/// compare rows where starts and ends exist - for !=
/// but no matches exist yet
use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn binary_compare<T: std::cmp::PartialOrd>(left: &T, right: &T, op: i8) -> bool {
    match op {
        0 => left > right,
        1 => left >= right,
        2 => left < right,
        3 => left <= right,
        4 => left == right,
        _ => left != right,
    }
}

/// keep original for debugging
// fn array_compare_int64(
//     left: ArrayView1<'_, i64>,
//     right: ArrayView1<'_, i64>,
//     starts: ArrayView1<'_, i64>,
//     ends: ArrayView1<'_, i64>,
//     left_booleans: ArrayView1<'_, bool>,
//     right_booleans: ArrayView1<'_, bool>,
//
//     is_extension_array: bool,
//     op: i8,
// ) -> (Array1<i8>, Array1<i64>, i64) {
//     let mut result = Array1::<i8>::zeros(length as usize);
//     let mut counts_array = Array1::<i64>::zeros(left.len());
//     let mut total: i64 = 0;
//     let mut n: usize = 0;
//     let zipped = izip!(
//         left.into_iter(),
//         left_booleans.into_iter(),
//         starts.into_iter(),
//         ends.into_iter(),
//     );
//     for (position, (left_val, left_bool, start, end)) in zipped.enumerate() {
//         let start_ = *start as usize;
//         let end_ = *end as usize;
//         let mut counter: i64 = 0;
//         for nn in start_..end_ {
//             let right_bool_ = right_booleans[nn];
//             if (*left_bool || right_bool_) && !is_extension_array {
//                 n += 1;
//                 continue;
//             }
//             if (*left_bool || right_bool_) && is_extension_array {
//                 result[n] = 1;
//                 n += 1;
//                 continue;
//             }

//             let right_val = right[nn];
//             let compare = binary_compare(left_val, &right_val, op);
//             counter += compare as i64;
//             total += compare as i64;
//             result[n] = compare as i8;
//             n += 1;
//         }

//         counts_array[position] = counter;
//     }
//     (result, counts_array, total)
// }

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        fn $fname(
            left: ArrayView1<'_, $type>,
            right: ArrayView1<'_, $type>,
            starts: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
            left_booleans: ArrayView1<'_, bool>,
            right_booleans: ArrayView1<'_, bool>,

            is_extension_array: bool,
            op: i8,
        ) -> (Array1<i8>, Array1<i64>, i64)
        // The macro will expand into the contents of this block.
        {
            let length: i64 = ends.iter().zip(starts.iter()).map(|(e, s)| e - s).sum();
            let mut result = Array1::<i8>::zeros(length as usize);
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(
                left.into_iter(),
                left_booleans.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
            );
            for (position, (left_val, left_bool, start, end)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    let right_bool_ = right_booleans[nn];
                    //pd.NA != pd.NA returns pd.NA, which defaults to False
                    // pd.NA != anything returns pd.NA, which defaults to False
                    // whereas np.nan != np.nan returns True
                    // np.nan != anything returns True
                    if (*left_bool || right_bool_) && is_extension_array {
                        n += 1;
                        continue;
                    }
                    if (*left_bool || right_bool_) && !is_extension_array {
                        result[n] = 1;
                        n += 1;
                        counter += 1;
                        total += 1;
                        continue;
                    }

                    let right_val = right[nn];
                    let compare = binary_compare(left_val, &right_val, op);
                    counter += compare as i64;
                    total += compare as i64;
                    result[n] = compare as i8;
                    n += 1;
                }

                counts_array[position] = counter;
            }
            (result, counts_array, total)
        }
    };
}

generic_compare!(array_compare_int64, i64);
generic_compare!(array_compare_int32, i32);
generic_compare!(array_compare_int16, i16);
generic_compare!(array_compare_int8, i8);
generic_compare!(array_compare_uint64, u64);
generic_compare!(array_compare_uint32, u32);
generic_compare!(array_compare_uint16, u16);
generic_compare!(array_compare_uint8, u8);
generic_compare!(array_compare_float64, f64);
generic_compare!(array_compare_float32, f32);

#[pyfunction(name = "compare_start_end_ne_1st_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int64(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int32(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int16(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i8>,
    right: PyReadonlyArray1<'py, i8>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int8(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_float32(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_float64(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint64(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint32(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint16(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_start_end_ne_1st_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();

    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint8(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_compare() {
        assert!(binary_compare(&5, &3, 0)); // >
        assert!(binary_compare(&5, &5, 1)); // >=
        assert!(binary_compare(&3, &5, 2)); // <
        assert!(binary_compare(&3, &3, 3)); // <=
        assert!(binary_compare(&5, &5, 4)); // ==
        assert!(binary_compare(&5, &3, 5)); // !=
    }

    #[test]
    fn test_array_compare_int64_basic() {
        let left = Array1::from(vec![1i64, 2, 3]);
        let right = Array1::from(vec![1i64, 2, 3, 4, 5]);
        let starts = Array1::from(vec![0i64, 1, 3]);
        let ends = Array1::from(vec![2i64, 3, 5]);
        let left_booleans = Array1::from(vec![false, false, false]);
        let right_booleans = Array1::from(vec![false, false, false, false, false]);

        let (result, counts, total) = array_compare_int64(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            5, // !=
        );

        assert_eq!(result.len(), 6);
        assert_eq!(counts.len(), 3);
        assert_eq!(total, 4);
    }

    #[test]
    fn test_array_compare_with_nulls_extension_array() {
        let left = Array1::from(vec![1i64, 2, 3]);
        let right = Array1::from(vec![1i64, 2, 3, 4, 5]);
        let starts = Array1::from(vec![0i64, 1, 3]);
        let ends = Array1::from(vec![2i64, 3, 5]);
        let left_booleans = Array1::from(vec![true, false, false]);
        let right_booleans = Array1::from(vec![false, true, false, false, false]);

        let (result, counts, total) = array_compare_int64(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            left_booleans.view(),
            right_booleans.view(),
            true,
            5, // !=
        );

        assert_eq!(result[0], 0); // null handling for extension array
        assert_eq!(result[1], 0);
        assert_eq!(counts[0], 0);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_array_compare_with_nulls_non_extension() {
        let left = Array1::from(vec![1i64, 2]);
        let right = Array1::from(vec![1i64, 2, 3]);
        let starts = Array1::from(vec![0i64, 1]);
        let ends = Array1::from(vec![2i64, 3]);
        let left_booleans = Array1::from(vec![true, false]);
        let right_booleans = Array1::from(vec![false, true, false]);

        let (result, counts, total) = array_compare_int64(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            5,
        );

        assert_eq!(result[0], 1); // NaN != NaN is true
        assert_eq!(result[1], 1);
        assert_eq!(counts[0], 2);
        assert_eq!(total, 4);
    }

    #[test]
    fn test_array_compare_float64() {
        let left = Array1::from(vec![1.5f64, 2.5, 3.5]);
        let right = Array1::from(vec![1.0f64, 2.0, 3.0, 4.0]);
        let starts = Array1::from(vec![0i64, 1, 2]);
        let ends = Array1::from(vec![2i64, 3, 4]);
        let left_booleans = Array1::from(vec![false, false, false]);
        let right_booleans = Array1::from(vec![false, false, false, false]);

        let (result, counts, total) = array_compare_float64(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            5,
        );

        assert!(total > 0);
        assert_eq!(result.len(), 6);
        assert_eq!(counts.len(), 3);
        assert_eq!(total, 6);
    }

    #[test]
    fn test_compare_int32_equality() {
        let left = Array1::from(vec![5i32, 5, 5]);
        let right = Array1::from(vec![5i32, 5, 5, 5]);
        let starts = Array1::from(vec![0i64, 1, 2]);
        let ends = Array1::from(vec![2i64, 3, 4]);
        let left_booleans = Array1::from(vec![false, false, false]);
        let right_booleans = Array1::from(vec![false, false, false, false]);

        let (result, _counts, total) = array_compare_int32(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            4, // ==
        );

        assert_eq!(total, 6);
        assert!(result.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_compare_uint8() {
        let left = Array1::from(vec![1u8, 2]);
        let right = Array1::from(vec![1u8, 2, 3]);
        let starts = Array1::from(vec![0i64, 1]);
        let ends = Array1::from(vec![2i64, 3]);
        let left_booleans = Array1::from(vec![false, false]);
        let right_booleans = Array1::from(vec![false, false, false]);

        let (result, counts, _total) = array_compare_uint8(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            0, // >
        );

        assert_eq!(result.len(), 4);
        assert_eq!(counts.len(), 2);
    }
}
