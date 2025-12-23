// comparisons based on positions
/// handles comparisions where nulls do not exist
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

/// keep original - useful for debugging
// fn array_compare_uint8(
//     left: ArrayView1<'_, u8>,
//     right: ArrayView1<'_, u8>,
//     starts: ArrayView1<'_, i64>,
//     positions: ArrayView1<'_, i64>,
//     ends: ArrayView1<'_, i64>,
//     op: i8,
// ) -> (Array1<i64>, Array1<i64>, i64) {
//     let mut result = Array1::<i64>::zeros(positions.len());
//     let mut counts_array = Array1::<i64>::zeros(ends.len());
//     let mut total: i64 = 0;
//     let mut n: usize = 0;
//     let zipped = izip!(
//         left.into_iter(),
//         starts.into_iter(),
//         ends.into_iter(),
//     );
//     for (position, (left_val, start, count_)) in zipped.enumerate() {
//         let size: i64 = *count_;
//         let start_ = *start as usize;
//         let end = *start + size;
//         let end_ = end as usize;
//         let mut counter: i64 = 0;
//         for nn in start_..end_ {
//             let mut indexer = positions[nn];
//             if indexer == -1 {
//                 result[n] = -1;
//                 n += 1;
//                 continue;
//             }
//             let indexer_ = indexer as usize;
//             let right_val = right[indexer_];
//             let compare = binary_compare(left_val, &right_val, op);
//             counter += compare as i64;
//             total += compare as i64;
//             indexer = if compare { indexer } else { -1 };
//             result[n] = indexer;
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
            positions: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
            op: i8,
        ) -> (Array1<i64>, Array1<i64>, i64)
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut counts_array = Array1::<i64>::zeros(ends.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter(),);
            for (position, (left_val, start, end)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    let mut indexer = positions[nn];
                    if indexer == -1 {
                        result[n] = -1;
                        n += 1;
                        continue;
                    }
                    let indexer_ = indexer as usize;
                    let right_val = right[indexer_];
                    let compare = binary_compare(left_val, &right_val, op);
                    counter += compare as i64;
                    total += compare as i64;
                    indexer = if compare { indexer } else { -1 };
                    result[n] = indexer;
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

#[pyfunction(name = "compare_posns_int64")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_int64(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_int32")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_int32(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_int16")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_int16(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_int8")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i8>,
    right: PyReadonlyArray1<'py, i8>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_int8(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_float32")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_float32(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_float64")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_float64(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_uint64")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_uint64(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_uint32")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_uint32(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_uint16")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_uint16(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_posns_uint8")]
#[pyo3(signature = (*, left, right, starts, positions, ends, op))]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    op: i8,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let positions = positions.as_array();
    let ends = ends.as_array();
    let (result, counts_array, total) =
        array_compare_uint8(left, right, starts, positions, ends, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    #[test]
    fn test_array_compare_posns_int64() {
        let left = Array1::from_vec(vec![10, 20]);
        let right = Array1::from_vec(vec![5, 15, 25, 30]);
        let starts = Array1::from_vec(vec![0, 2]);
        let ends = Array1::from_vec(vec![2, 4]);
        let positions = Array1::from_vec(vec![0, 1, 2, 3]);

        let (result, counts_array, total) = array_compare_int64(
            left.view(),
            right.view(),
            starts.view(),
            positions.view(),
            ends.view(),
            0, // greater than
        );

        assert_eq!(result, Array1::from_vec(vec![0, -1, -1, -1]));
        assert_eq!(counts_array, Array1::from_vec(vec![1, 0]));
        assert_eq!(total, 1);
    }

    #[test]
    fn test_array_compare_posns_with_minus_one() {
        let left = Array1::from_vec(vec![10, 20]);
        let right = Array1::from_vec(vec![5, 15, 25, 30]);
        let starts = Array1::from_vec(vec![0, 2]);
        let ends = Array1::from_vec(vec![2, 4]);
        let positions = Array1::from_vec(vec![0, -1, 2, 3]);

        let (result, counts_array, total) = array_compare_int64(
            left.view(),
            right.view(),
            starts.view(),
            positions.view(),
            ends.view(),
            0, // greater than
        );

        assert_eq!(result, Array1::from_vec(vec![0, -1, -1, -1]));
        assert_eq!(counts_array, Array1::from_vec(vec![1, 0]));
        assert_eq!(total, 1);
    }

    #[test]
    fn test_array_compare_posns_float64() {
        let left = Array1::from_vec(vec![10.5, 20.3]);
        let right = Array1::from_vec(vec![5.2, 15.7, 25.1]);
        let starts = Array1::from_vec(vec![0, 1]);
        let ends = Array1::from_vec(vec![1, 3]);
        let positions = Array1::from_vec(vec![0, 1, 2]);

        let (result, counts_array, total) = array_compare_float64(
            left.view(),
            right.view(),
            starts.view(),
            positions.view(),
            ends.view(),
            2, // less than
        );

        assert_eq!(result, Array1::from_vec(vec![-1, -1, 2]));
        assert_eq!(counts_array, Array1::from_vec(vec![0, 1]));
        assert_eq!(total, 1);
    }
}
