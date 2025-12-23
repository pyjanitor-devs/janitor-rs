/// compare rows where only ends exist (usually a >/>= join)
/// and matches does not exist yet
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

/// keep original - for debugging
// fn array_compare_int64(
//     left: ArrayView1<'_, i64>,
//     right: ArrayView1<'_, i64>,
//     ends: ArrayView1<'_, i64>,
//
//     op: i8,
// ) -> (Array1<i8>, Array1<i64>, i64) {
//     let mut result = Array1::<i8>::zeros(length as usize);
//     let mut counts_array = Array1::<i64>::zeros(left.len());
//     let mut total: i64 = 0;
//     let start: usize = 0;
//     let mut n: usize = 0;
//     let zipped = left.into_iter().zip(ends.into_iter());
//     for (position, (left_val, end)) in zipped.enumerate() {
//         let end_ = *end as usize;
//         let mut counter: i64 = 0;
//         for nn in start..end_ {
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
            ends: ArrayView1<'_, i64>,
            op: i8,
        ) -> (Array1<i8>, Array1<i64>, i64)
        // The macro will expand into the contents of this block.
        {
            let length = ends.sum();
            let mut result = Array1::<i8>::zeros(length as usize);
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let start: usize = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(ends.into_iter());
            for (position, (left_val, end)) in zipped.enumerate() {
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start..end_ {
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

#[pyfunction(name = "compare_first_end_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int64(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int32(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int16(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i8>,
    right: PyReadonlyArray1<'py, i8>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int8(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_float32(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_float64(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint64(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint32(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint16(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_end_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    ends: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint8(left, right, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;
    #[test]
    fn test_binary_compare_greater_than() {
        assert!(binary_compare(&5, &3, 0));
        assert!(!binary_compare(&3, &5, 0));
        assert!(!binary_compare(&5, &5, 0));
    }

    #[test]
    fn test_binary_compare_greater_equal() {
        assert!(binary_compare(&5, &3, 1));
        assert!(!binary_compare(&3, &5, 1));
        assert!(binary_compare(&5, &5, 1));
    }

    #[test]
    fn test_binary_compare_less_than() {
        assert!(!binary_compare(&5, &3, 2));
        assert!(binary_compare(&3, &5, 2));
        assert!(!binary_compare(&5, &5, 2));
    }

    #[test]
    fn test_binary_compare_less_equal() {
        assert!(!binary_compare(&5, &3, 3));
        assert!(binary_compare(&3, &5, 3));
        assert!(binary_compare(&5, &5, 3));
    }

    #[test]
    fn test_binary_compare_equal() {
        assert!(!binary_compare(&5, &3, 4));
        assert!(binary_compare(&5, &5, 4));
    }

    #[test]
    fn test_binary_compare_not_equal() {
        assert!(binary_compare(&5, &3, 5));
        assert!(!binary_compare(&5, &5, 5));
    }

    #[test]
    fn test_array_compare_int64_basic() {
        let left = array![10i64, 20, 30];
        let right = array![5i64, 15, 25, 35, 45];
        let ends = array![2i64, 4, 5];

        let (result, counts, total) =
            array_compare_int64(left.view(), right.view(), ends.view(), 0);

        assert_eq!(result.len(), 11);
        assert_eq!(counts.len(), 3);
        assert_eq!(total, 6);
    }

    #[test]
    fn test_array_compare_int32_greater_equal() {
        let left = array![10i32, 20];
        let right = array![10i32, 15, 20];
        let ends = array![2i64, 3];

        let (result, counts, total) =
            array_compare_int32(left.view(), right.view(), ends.view(), 1);

        assert_eq!(result[0], 1); // 10 >= 10
        assert_eq!(result[1], 0); // 10 >= 15
        assert_eq!(result[2], 1); // 20 >= 10
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 3);
        assert_eq!(total, 4);
    }

    #[test]
    fn test_array_compare_float64_less_than() {
        let left = array![1.5, 2.5];
        let right = array![2.0, 3.0, 4.0];
        let ends = array![2, 3];

        let (result, counts, total) =
            array_compare_float64(left.view(), right.view(), ends.view(), 2);

        assert_eq!(result[0], 1); // 1.5 < 2.0
        assert_eq!(result[1], 1); // 1.5 < 3.0
        assert_eq!(result[2], 0); // 2.5 < 2
        assert_eq!(counts[0], 2);
        assert_eq!(counts[1], 2);
        assert_eq!(total, 4);
    }

    #[test]
    fn test_array_compare_uint64_equal() {
        let left = array![5u64, 10];
        let right = array![5u64, 10, 15];
        let ends = array![2, 3];

        let (result, counts, total) =
            array_compare_uint64(left.view(), right.view(), ends.view(), 4);

        assert_eq!(result[0], 1); // 5 == 5
        assert_eq!(result[1], 0); // 5 == 10
        assert_eq!(result[2], 0); // 10 == 5
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 1);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_array_compare_int8_empty_result() {
        let left = array![1i8];
        let right = array![2i8];
        let ends = array![0i64];

        let (result, counts, total) = array_compare_int8(left.view(), right.view(), ends.view(), 0);

        assert_eq!(result.len(), 0);
        assert_eq!(counts[0], 0);
        assert_eq!(total, 0);
    }

    #[test]
    fn test_array_compare_uint16_all_matches() {
        let left = array![100u16];
        let right = array![50u16, 75, 90];
        let ends = array![3i64];

        let (result, counts, total) =
            array_compare_uint16(left.view(), right.view(), ends.view(), 0);

        assert_eq!(result.len(), 3);
        assert_eq!(counts[0], 3);
        assert_eq!(total, 3);
        assert!(result.iter().all(|&x| x == 1));
    }
}
