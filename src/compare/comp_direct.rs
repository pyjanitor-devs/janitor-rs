//! Direct first/last selection over a candidate range.

use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::op::CompareOp;

/// Return one complete match per left row without building a matches tape.
///
/// ELI5: the matches-tape kernels write one yes/no ticket for every candidate.
/// This kernel checks the whole predicate list for a candidate before moving
/// on, then keeps the smallest or largest original right-row index. The
/// candidate range may be sorted by a join column, so its scan order is not
/// necessarily the user's original right-row order.
pub fn select_start_end_core<T: PartialOrd + Copy>(
    left: &[numpy::ndarray::ArrayView1<'_, T>],
    right: &[numpy::ndarray::ArrayView1<'_, T>],
    left_index: numpy::ndarray::ArrayView1<'_, i64>,
    right_index: numpy::ndarray::ArrayView1<'_, i64>,
    starts: numpy::ndarray::ArrayView1<'_, i64>,
    ends: numpy::ndarray::ArrayView1<'_, i64>,
    ops: &[CompareOp],
    first: bool,
) -> (Vec<i64>, Vec<i64>) {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    let right_len = right[0].len();

    for (left_position, (start, end)) in izip!(starts.iter(), ends.iter()).enumerate() {
        if *start < 0 || *end == -1 || *start >= *end || *end > right_len as i64 {
            continue;
        }
        let mut selected: Option<i64> = None;
        let mut consider = |right_position: usize| {
            let matches = left.iter().zip(right.iter()).zip(ops.iter()).all(
                |((left_column, right_column), op)| {
                    op.apply(&left_column[left_position], &right_column[right_position])
                },
            );
            if matches {
                let candidate = right_index[right_position];
                let replace = selected.is_none()
                    || (first && candidate < selected.unwrap())
                    || (!first && candidate > selected.unwrap());
                if replace {
                    selected = Some(candidate);
                }
            }
            matches
        };
        let range = (*start as usize)..(*end as usize);
        for right_position in range {
            consider(right_position);
        }
        if let Some(candidate) = selected {
            left_indices.push(left_index[left_position]);
            right_indices.push(candidate);
        }
    }
    (left_indices, right_indices)
}

macro_rules! generic_direct {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: Vec<PyReadonlyArray1<'py, $type>>,
            right: Vec<PyReadonlyArray1<'py, $type>>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            ops: Vec<i8>,
            first: bool,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            if left.is_empty() || left.len() != right.len() || left.len() != ops.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "left, right, and ops must have the same non-zero length",
                ));
            }
            let left_views: Vec<_> = left.iter().map(|array| array.as_array()).collect();
            let right_views: Vec<_> = right.iter().map(|array| array.as_array()).collect();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            let left_len = left_views[0].len();
            let right_len = right_views[0].len();
            if starts.as_array().len() != left_len
                || ends.as_array().len() != left_len
                || left_index.len() != left_len
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "starts and ends must have one entry per left row",
                ));
            }
            if left_views.iter().any(|array| array.len() != left_len)
                || right_views.iter().any(|array| array.len() != right_len)
                || right_index.len() != right_len
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "all left and all right columns must have matching lengths",
                ));
            }
            let ops: Vec<_> = ops
                .into_iter()
                .map(CompareOp::try_from_code)
                .collect::<PyResult<Vec<_>>>()?;
            let (left_indices, right_indices) = select_start_end_core(
                &left_views,
                &right_views,
                left_index,
                right_index,
                starts.as_array(),
                ends.as_array(),
                &ops,
                first,
            );
            Ok((
                left_indices.into_pyarray(py),
                right_indices.into_pyarray(py),
            ))
        }
    };
}

generic_direct!(select_start_end_direct_int64, i64);
generic_direct!(select_start_end_direct_int32, i32);
generic_direct!(select_start_end_direct_int16, i16);
generic_direct!(select_start_end_direct_int8, i8);
generic_direct!(select_start_end_direct_uint64, u64);
generic_direct!(select_start_end_direct_uint32, u32);
generic_direct!(select_start_end_direct_uint16, u16);
generic_direct!(select_start_end_direct_uint8, u8);
generic_direct!(select_start_end_direct_f64, f64);
generic_direct!(select_start_end_direct_f32, f32);

/// Register the direct-selection exports.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(select_start_end_direct_int64, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_int32, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_int16, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_int8, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_f64, m)?)?;
    m.add_function(wrap_pyfunction!(select_start_end_direct_f32, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn selects_first_and_last_complete_match() {
        let left_1 = array![2_i64, 3];
        let left_2 = array![2_i64, 3];
        let right_1 = array![1_i64, 1, 1, 1];
        let right_2 = array![3_i64, 3, 3, 3];
        let starts_array = array![0_i64, 0];
        let ends_array = array![4_i64, 4];
        let left = [left_1.view(), left_2.view()];
        let right = [right_1.view(), right_2.view()];
        let left_index_array = array![10_i64, 20];
        let right_index_array = array![24_i64, 58, 2, 13];
        let left_index = left_index_array.view();
        let right_index = right_index_array.view();
        let starts = starts_array.view();
        let ends = ends_array.view();
        let ops = [CompareOp::Ge, CompareOp::Le];

        let (first_left, first_right) = select_start_end_core(
            &left,
            &right,
            left_index,
            right_index,
            starts,
            ends,
            &ops,
            true,
        );
        assert_eq!(first_left, vec![10, 20]);
        assert_eq!(first_right, vec![2, 2]);

        let (last_left, last_right) = select_start_end_core(
            &left,
            &right,
            left_index,
            right_index,
            starts,
            ends,
            &ops,
            false,
        );
        assert_eq!(last_left, vec![10, 20]);
        assert_eq!(last_right, vec![58, 58]);
    }

    #[test]
    fn skips_invalid_ranges_and_preserves_left_positions() {
        let left_array = array![2_i64, 2, 2];
        let right_array = array![1_i64, 2, 3];
        let starts_array = array![-1_i64, 0, 2];
        let ends_array = array![2_i64, 2, 4];
        let left = [left_array.view()];
        let right = [right_array.view()];
        let left_index_array = array![10_i64, 20, 30];
        let right_index_array = array![0_i64, 1, 2];
        let left_index = left_index_array.view();
        let right_index = right_index_array.view();
        let starts = starts_array.view();
        let ends = ends_array.view();
        let ops = [CompareOp::Eq];

        let (left_indices, right_indices) = select_start_end_core(
            &left,
            &right,
            left_index,
            right_index,
            starts,
            ends,
            &ops,
            true,
        );
        assert_eq!(left_indices, vec![20]);
        assert_eq!(right_indices, vec![1]);
    }
}
