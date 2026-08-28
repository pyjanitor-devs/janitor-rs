use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ends_labels, into_starts_ends_result};

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if arr.len() != ends.len() || arr.len() != booleans.len() {
        return Err("arr, ends, and booleans must have equal lengths");
    }
    Ok(())
}

/// Choose the sweep only when it has enough work to repay its row metadata.
///
/// The old loop performs approximately `rows * width` winner checks. The
/// sweep performs approximately `rows + width` event/output work, but also
/// allocates a flat bucket head array and one link per row. The `8x` margin is
/// deliberately conservative: it avoids paying that metadata and setup cost
/// when the ranges are narrow, while still selecting the sweep for the wide
/// inputs where the nested loop repeats the same work many times. The value
/// was selected from the tiny/large/very-large/narrow benchmark matrix rather
/// than from an input-size assumption. Saturating arithmetic keeps this
/// estimate safe even for dimensions near `usize::MAX`.
///
/// ELI5: use the shortcut only when there are enough repeated chores for the
/// shortcut to be worth setting up; for a tiny job, just do the chores.
fn should_sweep(rows: usize, width: usize) -> bool {
    let repeated_work = rows.saturating_mul(width);
    let sweep_work = rows.saturating_add(width);
    repeated_work > sweep_work.saturating_mul(8)
}

/// Groups reverse-maximum ends by compact candidate ordinal.
/// ELI5: every prefix starts at zero, so the largest end is the exact slot count.
pub fn max_rev_ends_core<T: PartialOrd + PartialEq + Copy>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, ends, booleans)?;
    let max_end = ends.iter().copied().max().unwrap() as usize;

    if !should_sweep(arr.len(), max_end) {
        let mut values = vec![arr[0]; max_end];
        let mut positions = vec![-1_i64; max_end];
        for (row, ((current, end), boolean)) in
            arr.iter().zip(ends.iter()).zip(booleans.iter()).enumerate()
        {
            for (position, value) in positions
                .iter_mut()
                .zip(values.iter_mut())
                .take(*end as usize)
            {
                if *boolean {
                    continue;
                }
                if *position == -1 || *current > *value {
                    *position = row as i64;
                    *value = *current;
                }
            }
        }
        let indexers = (0..max_end).map(|item| index[item]).collect();
        return Ok((indexers, Array1::from_vec(positions)));
    }

    // ELI5: a prefix row is eligible while the sweep is left of its end.
    // Bucket each row by its end, then activate it once at `end - 1` while
    // sweeping right-to-left. The linked buckets preserve input-row order,
    // retaining the old first-row tie behavior. `end == 0` has no activation
    // bucket in the emitted domain and therefore remains an empty prefix.
    let mut head = vec![usize::MAX; max_end + 1];
    let mut next = vec![usize::MAX; arr.len()];
    for (row, end) in ends.iter().enumerate().rev() {
        next[row] = head[*end as usize];
        head[*end as usize] = row;
    }

    let mut positions = vec![-1_i64; max_end];
    let mut current_winner: Option<(T, i64)> = None;
    for position in (0..max_end).rev() {
        let mut row = head[position + 1];
        while row != usize::MAX {
            let current = arr[row];
            if !booleans[row]
                && (current_winner.is_none()
                    || current > current_winner.as_ref().unwrap().0
                    || (current == current_winner.as_ref().unwrap().0
                        && (row as i64) < current_winner.as_ref().unwrap().1))
            {
                current_winner = Some((current, row as i64));
            }
            row = next[row];
        }
        if let Some((_, row)) = current_winner {
            positions[position] = row;
        }
    }
    Ok((ends_labels(max_end, index), Array1::from_vec(positions)))
}

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            into_starts_ends_result(
                py,
                max_rev_ends_core(
                    arr.as_array(),
                    ends.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                ),
            )
        }
    };
}

compute!(compute_max_rev_end_int64, i64);
compute!(compute_max_rev_end_int32, i32);
compute!(compute_max_rev_end_int16, i16);
compute!(compute_max_rev_end_int8, i8);
compute!(compute_max_rev_end_uint64, u64);
compute!(compute_max_rev_end_uint32, u32);
compute!(compute_max_rev_end_uint16, u16);
compute!(compute_max_rev_end_uint8, u8);
compute!(compute_max_rev_end_f64, f64);
compute!(compute_max_rev_end_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;
    #[test]
    fn finds_max_positions_and_labels() {
        let got = max_rev_ends_core(
            array![5_i64, 2, 4].view(),
            array![2_i64, 3, 1].view(),
            array![50_i64, 10, 90].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((array![50, 10, 90], array![0, 0, 1])));
    }
    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            max_rev_ends_core(
                array![1_i64].view(),
                array![0_i64].view(),
                array![1_i64].view(),
                array![false].view()
            ),
            Ok((array![], array![]))
        );
    }

    #[test]
    fn sweep_preserves_first_tie_and_skips_zero_width_rows() {
        let mut arr = Array1::from_elem(100, 0_i64);
        arr[0] = 5;
        arr[1] = 7;
        let mut ends = Array1::from_elem(100, 1000_i64);
        ends[1] = 500;
        ends[99] = 0;
        let index = Array1::from_iter(0_i64..1000);
        let mut booleans = Array1::from_elem(100, false);
        booleans[99] = true;
        let got = max_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view());
        let (labels, positions) = got.unwrap();
        assert_eq!(labels, index);
        assert_eq!(positions[0], 1);
        assert_eq!(positions[499], 1);
        assert_eq!(positions[500], 0);
    }

    #[test]
    fn sweep_preserves_smallest_row_on_equal_maximum() {
        let mut arr = Array1::from_elem(20, 99_i64);
        arr[2] = 7;
        arr[18] = 7;
        let mut ends = Array1::zeros(20);
        ends[2] = 20;
        ends[18] = 1;
        let index = Array1::from_iter(0..20_i64);
        let mut booleans = Array1::from_elem(20, true);
        booleans[2] = false;
        booleans[18] = false;
        let got = max_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view());
        assert_eq!(got, Ok((index, Array1::from_elem(20, 2_i64))));
    }
}
