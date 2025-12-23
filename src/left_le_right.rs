/// find positions where left region <= right region
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

fn left_le_right(
    left: ArrayView1<'_, i64>,
    right: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    max_right: i64,
) -> (Array1<i64>, Array1<i64>, i64) {
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut right_val: i64;
    let mut total: i64 = 0;
    let mut end = right.len();
    let mut counts = BTreeMap::new();
    let zipped = left.iter().zip(starts.iter());
    let zipped = zipped.enumerate();
    // step1: get the counts
    for (position, (left_val, start)) in zipped {
        let start_ = *start as usize;
        for nn in start_..end {
            right_val = right[nn];
            *counts.entry(right_val).or_insert(0) += 1;
        }
        end = start_;
        let mut counter: i64 = 0;
        for (_, size) in counts.range(left_val..=&max_right) {
            counter += size;
            total += size;
        }
        counts_array[position] = counter;
    }
    let total_counts: i64 = counts.values().sum();
    // keep track of the very first position of values from right
    // in the lookup array
    let mut dictionary: HashMap<i64, i64> = HashMap::new();
    let (k, val) = counts.pop_last().unwrap();
    let mut position = total_counts - val;
    dictionary.insert(k, position);
    // ensure iteration starts from the largest value in right
    // for proper alignment in the lookup_array
    for (key, count) in counts.into_iter().rev() {
        position -= count;
        dictionary.insert(key, position);
    }

    let mut lookup_array = Array1::<i64>::zeros(total_counts as usize);
    let mut result = Array1::<i64>::zeros(total as usize);
    let mut right_val: i64;
    let mut end = right.len();
    let mut counts = BTreeMap::new();
    let zipped = left.iter().zip(starts.iter());
    let mut n: usize = 0;
    // step2: build the actual positions
    for (left_val, start) in zipped {
        let start_ = *start as usize;
        for position in start_..end {
            right_val = right[position];
            *counts.entry(right_val).or_insert(0) += 1;
            let lookup_position = *dictionary.get(&right_val).unwrap() as usize;
            let size = counts.get(&right_val).unwrap();
            // zero indexing ,hence the value-1
            let indexer = lookup_position + (size - 1);
            lookup_array[indexer] = position as i64;
        }
        end = start_;
        for (key, size) in counts.range(left_val..=&max_right) {
            let lookup_position = *dictionary.get(&key).unwrap() as usize;
            for indexer in lookup_position..lookup_position + size {
                let position = lookup_array[indexer];
                result[n] = position;
                n += 1;
            }
        }
    }

    (result, counts_array, total)
}

#[pyfunction(name = "get_positions_where_left_le_right")]
pub fn region_positions<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    max_right: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let (result, counts_array, total) = left_le_right(left, right, starts, max_right);
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
    fn test_left_le_right_simple() {
        let left = Array1::from_vec(vec![10, 20]);
        let right = Array1::from_vec(vec![5, 15, 25]);
        let starts = Array1::from_vec(vec![0, 1]);
        let max_right = 30;

        let (result, counts_array, total) =
            left_le_right(left.view(), right.view(), starts.view(), max_right);

        assert_eq!(counts_array, Array1::from_vec(vec![2, 1]));
        assert_eq!(total, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_left_le_right_all_match() {
        let left = Array1::from_vec(vec![5, 10]);
        let right = Array1::from_vec(vec![10, 20, 30]);
        // starts should be in descending order
        let starts = Array1::from_vec(vec![1, 0]);
        // max_right should be equal to right.max()
        let max_right = 30;

        let (result, counts_array, total) =
            left_le_right(left.view(), right.view(), starts.view(), max_right);

        assert_eq!(counts_array, Array1::from_vec(vec![2, 3]));
        assert_eq!(total, 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_left_le_right_with_duplicates() {
        let left = Array1::from_vec(vec![10, 15]);
        let right = Array1::from_vec(vec![10, 10, 20]);
        // starts should be in descending order
        let starts = Array1::from_vec(vec![2, 0]);
        // max_right should be equal to right.max()
        let max_right = 20;

        let (result, counts_array, total) =
            left_le_right(left.view(), right.view(), starts.view(), max_right);

        assert_eq!(counts_array, Array1::from_vec(vec![1, 1]));
        assert_eq!(total, 2);
        assert_eq!(result.len(), 2);
    }
}
