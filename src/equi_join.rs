//! Standalone issue #218 equi positional core.
//!
//! PyJanitor owns scalar/MultiIndex construction, `get_indexer`, factorization,
//! null semantics, sorting, and the mapping from sorted physical positions to
//! original right positions. This file owns only duplicate-right equi matching.

use crate::aggs::ensure_equal_lengths_core;
use crate::join_common::Keep;
use numpy::{ndarray::ArrayView1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug)]
struct DenseRightMetadata {
    any: Vec<usize>,
    first: Vec<usize>,
    last: Vec<usize>,
    groups: Vec<Vec<usize>>,
}

fn build_dense_right_metadata(
    right_index: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
) -> Result<DenseRightMetadata, String> {
    let code_count = right_codes
        .iter()
        .copied()
        .filter(|&code| code >= 0)
        .max()
        .map(|code| {
            usize::try_from(code)
                .map_err(|_| "right code exceeds the positional contract")?
                .checked_add(1)
                .ok_or("right code exceeds the positional contract")
        })
        .transpose()
        .map_err(|_| "right code exceeds the positional contract")?
        .unwrap_or(0);
    let mut any = vec![usize::MAX; code_count];
    let mut first = vec![usize::MAX; code_count];
    let mut last = vec![usize::MAX; code_count];
    let mut groups = vec![Vec::new(); code_count];
    for (position, &code) in right_codes.iter().enumerate() {
        if code < -1 {
            return Err("right codes must be greater than or equal to -1".to_owned());
        }
        if code == -1 {
            continue;
        }
        let code = usize::try_from(code).map_err(|_| "invalid right code")?;
        groups[code].push(position);
        if any[code] == usize::MAX {
            any[code] = position;
        }
        if first[code] == usize::MAX || right_index[position] < right_index[first[code]] {
            first[code] = position;
        }
        if last[code] == usize::MAX || right_index[position] > right_index[last[code]] {
            last[code] = position;
        }
    }
    debug_assert_eq!(any.len(), code_count);
    debug_assert_eq!(first.len(), code_count);
    debug_assert_eq!(last.len(), code_count);
    debug_assert_eq!(groups.len(), code_count);
    Ok(DenseRightMetadata {
        any,
        first,
        last,
        groups,
    })
}

fn build_duplicate_equi_pairs_core(
    left_index: ArrayView1<'_, i64>,
    left_indexer: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
    keep: Keep,
) -> Result<Option<(Vec<i64>, Vec<i64>)>, String> {
    ensure_equal_lengths_core(
        "left index",
        left_index.len(),
        "equi indexer",
        left_indexer.len(),
    )?;
    ensure_equal_lengths_core(
        "right index",
        right_index.len(),
        "duplicate codes",
        right_codes.len(),
    )?;
    let metadata = build_dense_right_metadata(right_index, right_codes)?;

    let mut output_len = 0_usize;
    for row in 0..left_indexer.len() {
        let code = left_indexer[row];
        if code < -1 {
            return Err("left codes must be greater than or equal to -1".to_owned());
        }
        if code == -1 {
            continue;
        }
        let code = usize::try_from(code).map_err(|_| "invalid left code")?;
        if code >= metadata.groups.len() || metadata.groups[code].is_empty() {
            continue;
        }
        let count = match keep {
            Keep::All => metadata.groups[code].len(),
            Keep::Any | Keep::First | Keep::Last => 1,
        };
        output_len = output_len
            .checked_add(count)
            .ok_or("equi join output is too large")?;
    }
    if output_len == 0 {
        return Ok(None);
    }

    let mut left_output = Vec::with_capacity(output_len);
    let mut right_output = Vec::with_capacity(output_len);
    for row in 0..left_indexer.len() {
        let code = left_indexer[row];
        if code < 0 {
            continue;
        }
        let code = usize::try_from(code).map_err(|_| "invalid left code")?;
        if code >= metadata.groups.len() || metadata.groups[code].is_empty() {
            continue;
        }
        let positions = match keep {
            Keep::Any => std::slice::from_ref(&metadata.any[code]),
            Keep::First => std::slice::from_ref(&metadata.first[code]),
            Keep::Last => std::slice::from_ref(&metadata.last[code]),
            Keep::All => metadata.groups[code].as_slice(),
        };
        for &position in positions {
            left_output.push(left_index[row]);
            right_output.push(right_index[position]);
        }
    }
    debug_assert_eq!(left_output.len(), output_len);
    debug_assert_eq!(right_output.len(), output_len);
    Ok(Some((left_output, right_output)))
}

/// Build indices for a pure equi join with duplicate right keys.
///
/// PyJanitor handles the unique-right case directly. Here
/// `original_right_positions` contains one dense factorization code per right
/// row, aligned with `right_index`; `-1` entries are ignored.
#[pyfunction]
pub fn equi_join_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    left_indexer: PyReadonlyArray1<'py, i64>,
    original_right_positions: PyReadonlyArray1<'py, i64>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    let pairs = build_duplicate_equi_pairs_core(
        left_index.as_array(),
        left_indexer.as_array(),
        right_index.as_array(),
        original_right_positions.as_array(),
        keep,
    )
    .map_err(PyValueError::new_err)?;
    pairs
        .map(|(left_output, right_output)| {
            crate::join_common::result_dict(py, left_output, right_output, None, None)
        })
        .transpose()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(equi_join_indices, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn dense_metadata_tracks_codes_and_right_length() {
        let metadata = build_dense_right_metadata(
            array![12_i64, 5, 9, 7].view(),
            array![3_i64, 1, 3, -1].view(),
        )
        .unwrap();
        assert_eq!(metadata.groups.len(), 4);
        assert_eq!(metadata.groups[3], vec![0, 2]);
        assert_eq!(metadata.groups[1], vec![1]);
        assert_eq!(metadata.any[3], 0);
        assert_eq!(metadata.first[3], 2);
        assert_eq!(metadata.last[3], 0);
    }

    #[test]
    fn duplicate_equi_keep_modes_use_dense_codes() {
        let left_index = array![10_i64, 11, 12];
        let left_indexer = array![3_i64, 1, -1];
        let right_index = array![12_i64, 5, 9, 7];
        let right_codes = array![3_i64, 1, 3, -1];

        let any = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::Any,
        )
        .unwrap();
        assert_eq!(any.unwrap().1, vec![12, 5]);

        let first = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::First,
        )
        .unwrap();
        assert_eq!(first.unwrap().1, vec![9, 5]);

        let last = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::Last,
        )
        .unwrap();
        assert_eq!(last.unwrap().1, vec![12, 5]);

        let all = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::All,
        )
        .unwrap();
        assert_eq!(all.unwrap().1, vec![12, 9, 5]);
    }

    #[test]
    fn duplicate_equi_rejects_codes_below_minus_one() {
        let error = build_duplicate_equi_pairs_core(
            array![10_i64].view(),
            array![-2_i64].view(),
            array![100_i64].view(),
            array![0_i64].view(),
            Keep::All,
        )
        .unwrap_err();
        assert_eq!(error, "left codes must be greater than or equal to -1");

        let error = build_duplicate_equi_pairs_core(
            array![10_i64].view(),
            array![0_i64].view(),
            array![100_i64].view(),
            array![-2_i64].view(),
            Keep::All,
        )
        .unwrap_err();
        assert_eq!(error, "right codes must be greater than or equal to -1");
    }
}
