/// find positions where left region <= right region
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

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
    // keep track of counts per left region
    let mut counts_array = Array1::<i64>::zeros(left.len());
    // keep track of positions for each entry in the right region
    let mut positions = Array1::<i64>::zeros(right.len());
    // supports in updating the positions array above
    let mut trackers = Array1::<i64>::zeros((max_right + 1) as usize);
    let mut right_val: i64;
    let mut total: i64 = 0;
    let mut end = right.len();
    // keep count of each unique entry in the right region
    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    let zipped = left.iter().zip(starts.iter());
    let zipped = zipped.enumerate();
    // step1: get the counts
    for (position, (left_val, start)) in zipped {
        let start_ = *start as usize;
        // get counts per right region
        for nn in (start_..end).rev() {
            right_val = right[nn];
            let tracker = counts.entry(right_val).or_insert(0);
            *tracker += 1;
            // build the next positions per right_val
            // say right_val is 3, the first position is 6
            // nothing is marked
            // the next time 3 is encountered, the next position is 4
            // store 4 in the 6th position in `positions`
            // the next time 3 is encountered, the next position is 2
            // store 2 in the 4th position in `positions`
            // this will come handy in phase 2
            // when building the matched positions
            if *tracker > 1 {
                let ind = trackers[right_val as usize];
                positions[ind as usize] = nn as i64;
            }
            // trackers will always change
            // to hold the latest position per entry in the right region
            trackers[right_val as usize] = nn as i64;
        }
        // this ensures we dont go through already visited positions
        end = start_;
        let mut counter: i64 = 0;
        // get counts of left_val <= right_region
        if let Some((&last_key, &_)) = counts.last_key_value() {
            // left_val has to be <= right
            // therefore the last_key/maximum key
            // has to be greater than left_val
            if *left_val > last_key {
                continue;
            }
            for (_, size) in counts.range(left_val..=&last_key) {
                counter += size;
                total += size;
            }
        }
        counts_array[position] = counter;
    }

    // step2: build the actual positions
    // use dictionary to store the largest possible positions per unique entry
    // in the right region
    // this serves as the starting point when looping to build the positions
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(counts.len());
    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    let mut result = Array1::<i64>::zeros(total as usize);
    let zipped = left.into_iter().zip(starts.into_iter());
    let mut end = right.len();
    let mut n: usize = 0;
    for (left_val, start) in zipped {
        let start_ = *start as usize;
        // get counts per right region
        for nn in (start_..end).rev() {
            right_val = right[nn];
            let _ = *dictionary.entry(right_val).or_insert(nn as i64);
            let tracker = counts.entry(right_val).or_insert(0);
            *tracker += 1;
        }
        end = start_;
        // get positions where left_val <= right_region
        if let Some((&last_key, &_)) = counts.last_key_value() {
            // left_val has to be <= right
            // therefore the last_key/maximum key
            // has to be greater than left_val
            if *left_val > last_key {
                continue;
            }
            // build the actual positions
            // by jumping through the already stored indices
            for (key, size) in counts.range(left_val..=&last_key) {
                let size_ = *size as usize;
                if let Some(&posn) = dictionary.get(key) {
                    result[n] = posn;
                    n += 1;
                    let mut next_posn = posn;
                    for _ in 1..size_ {
                        next_posn = positions[next_posn as usize];
                        result[n] = next_posn;
                        n += 1;
                    }
                }
            }
        }
    }

    (
        counts_array.into_pyarray(py),
        result.into_pyarray(py),
        total,
    )
}
