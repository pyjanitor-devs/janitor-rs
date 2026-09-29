//! Standalone issue #218 equi positional core.
//!
//! PyJanitor owns scalar/MultiIndex construction, `get_indexer`, factorization,
//! null semantics, sorting, and the mapping from sorted physical positions to
//! original right positions. This file owns only duplicate-right equi matching.
//!
//! The Python side passes two related coordinate systems into this module:
//!
//! * `right_codes` contains one dense factorization code for each physical
//!   right row. The code identifies the equi-key group for that row.
//! * `right_index` contains the right labels to emit. It has the same physical
//!   row order as `right_codes`, but it is not itself a factorization code.
//!
//! `left_indexer` contains one code per left row. `-1` means that
//! `get_indexer` found no equi-key match and produces no output for that row.
//! Nonnegative codes are looked up in the dense right-side metadata below.
//!
//! The `Keep::All` representation uses a flat positions array rather than a
//! `Vec<Vec<usize>>`. `offsets[code]..offsets[code + 1]` identifies the slice
//! in that flat array belonging to one code. This avoids one heap allocation
//! per distinct equi key.

use crate::aggs::ensure_equal_lengths_core;
use crate::join_common::Keep;
use numpy::{ndarray::ArrayView1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug)]
struct DenseRightMetadata {
    /// One physical right position for each code when `keep == Any`.
    any: Vec<usize>,
    /// The physical right position with the smallest right label for each
    /// code when `keep == First`.
    first: Vec<usize>,
    /// The physical right position with the largest right label for each code
    /// when `keep == Last`.
    last: Vec<usize>,
    /// Number of physical right rows for each code when `keep == All`.
    counts: Vec<usize>,
    /// Boundaries into `positions` when `keep == All`.
    offsets: Vec<usize>,
    /// Flattened physical right positions grouped by code when `keep == All`.
    positions: Vec<usize>,
}

/// Build the right-side lookup metadata needed by one `Keep` mode.
///
/// The factorization codes are expected to be dense: if the largest
/// nonnegative code is `3`, the metadata has slots for codes `0` through `3`.
/// A code slot may be empty if the caller supplies a sparse or malformed code
/// sequence; the matching loop treats such a slot as having no match.
///
/// The selected mode controls which metadata is allocated:
///
/// * `Any` stores one physical position per code;
/// * `First` stores the position with the smallest emitted right label;
/// * `Last` stores the position with the largest emitted right label;
/// * `All` stores counts, offsets, and one flat physical-position array.
///
/// # Arguments
///
/// * `right_index` - Emitted right labels in sorted physical-row order.
/// * `right_codes` - One factorization code for each physical right row;
///   `-1` rows are ignored.
/// * `keep` - The output-selection mode that determines which metadata is
///   necessary.
///
/// # Errors
///
/// Returns an error if a code is below `-1`, cannot be represented as a
/// physical position, or if metadata sizes would overflow `usize`.
fn build_dense_right_metadata(
    right_index: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
    keep: Keep,
) -> Result<DenseRightMetadata, String> {
    // Codes are zero-based, so the number of slots is the largest code plus
    // one. `-1` is the no-match sentinel and does not create a slot.
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
    // Allocate only the selected mode's single-position metadata. Keeping the
    // other vectors empty avoids storing three copies of equivalent lookups.
    let mut any = if keep == Keep::Any {
        vec![usize::MAX; code_count]
    } else {
        Vec::new()
    };
    let mut first = if keep == Keep::First {
        vec![usize::MAX; code_count]
    } else {
        Vec::new()
    };
    let mut last = if keep == Keep::Last {
        vec![usize::MAX; code_count]
    } else {
        Vec::new()
    };
    // Counts are needed only by `All`, where they determine both the flat
    // storage size and each code's slice boundaries.
    let mut counts: Vec<usize> = if keep == Keep::All {
        vec![0; code_count]
    } else {
        Vec::new()
    };
    // First pass: validate codes and collect the mode-specific summary.
    for (position, &code) in right_codes.iter().enumerate() {
        if code < -1 {
            return Err("right codes must be greater than or equal to -1".to_owned());
        }
        if code == -1 {
            continue;
        }
        let code = usize::try_from(code).map_err(|_| "invalid right code")?;
        if keep == Keep::All {
            counts[code] = counts[code]
                .checked_add(1)
                .ok_or("equi join right metadata is too large")?;
        }
        if keep == Keep::Any && any[code] == usize::MAX {
            any[code] = position;
        }
        if keep == Keep::First
            && (first[code] == usize::MAX || right_index[position] < right_index[first[code]])
        {
            first[code] = position;
        }
        if keep == Keep::Last
            && (last[code] == usize::MAX || right_index[position] > right_index[last[code]])
        {
            last[code] = position;
        }
    }

    let (offsets, positions) = if keep == Keep::All {
        // `offsets` has one more element than the number of codes because it
        // stores both the start and the final end boundary. For code `c`, its
        // positions live in `positions[offsets[c]..offsets[c + 1]]`.
        let mut offsets: Vec<usize> = Vec::with_capacity(code_count + 1);
        offsets.push(0);
        for &count in &counts {
            let next = offsets
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(count)
                .ok_or("equi join right metadata is too large")?;
            offsets.push(next);
        }
        // Allocate one contiguous buffer for all matching physical positions.
        let mut positions = vec![0; *offsets.last().unwrap_or(&0)];
        // Cursors begin at each code's start boundary and advance as positions
        // are copied into the flat buffer. Keep `offsets` unchanged because it
        // is the permanent lookup table.
        let mut cursors = offsets[..code_count].to_vec();
        // Second pass: preserve physical right-row order inside each code's
        // flat slice. `-1` rows have no group and are skipped.
        for (position, &code) in right_codes.iter().enumerate() {
            if code == -1 {
                continue;
            }
            let code = usize::try_from(code).map_err(|_| "invalid right code")?;
            positions[cursors[code]] = position;
            cursors[code] += 1;
        }
        (offsets, positions)
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(DenseRightMetadata {
        any,
        first,
        last,
        counts,
        offsets,
        positions,
    })
}

/// Return how many output pairs one left code contributes.
///
/// `None` means that the code is outside the metadata or has no right-side
/// match. The result is `1` for the single-match modes and the group size for
/// `Keep::All`. Keeping this logic in one helper makes the sizing pass and the
/// output pass use exactly the same match-existence rules.
fn matching_count(metadata: &DenseRightMetadata, code: usize, keep: Keep) -> Option<usize> {
    match keep {
        Keep::Any => metadata
            .any
            .get(code)
            .filter(|&&position| position != usize::MAX)
            .map(|_| 1),
        Keep::First => metadata
            .first
            .get(code)
            .filter(|&&position| position != usize::MAX)
            .map(|_| 1),
        Keep::Last => metadata
            .last
            .get(code)
            .filter(|&&position| position != usize::MAX)
            .map(|_| 1),
        Keep::All => metadata
            .counts
            .get(code)
            .copied()
            .filter(|&count| count > 0),
    }
}

fn build_duplicate_equi_pairs_core(
    left_index: ArrayView1<'_, i64>,
    left_indexer: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
    keep: Keep,
) -> Result<Option<(Vec<i64>, Vec<i64>)>, String> {
    // Validate all parallel arrays before building metadata or entering either
    // output pass. This makes the physical-row alignment explicit: every
    // right code must describe exactly one right index entry, and every left
    // indexer code must describe exactly one left index entry.
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
    let metadata = build_dense_right_metadata(right_index, right_codes, keep)?;

    // First pass: calculate the exact number of output pairs. This lets the
    // second pass allocate both result vectors once, with no growth or
    // reallocation during matching.
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
        let Some(count) = matching_count(&metadata, code, keep) else {
            continue;
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
    // Second pass: materialize the labels selected by `keep`. The metadata
    // stores physical right positions; `right_index[position]` converts each
    // selected physical position to the label returned to Python.
    for row in 0..left_indexer.len() {
        let code = left_indexer[row];
        if code < 0 {
            continue;
        }
        let code = usize::try_from(code).map_err(|_| "invalid left code")?;
        if matching_count(&metadata, code, keep).is_none() {
            continue;
        }
        let positions = match keep {
            Keep::Any => std::slice::from_ref(&metadata.any[code]),
            Keep::First => std::slice::from_ref(&metadata.first[code]),
            Keep::Last => std::slice::from_ref(&metadata.last[code]),
            Keep::All => &metadata.positions[metadata.offsets[code]..metadata.offsets[code + 1]],
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
/// PyJanitor handles the unique-right case directly. This function is for the
/// duplicate-right case, where Python has already factorized the right equi
/// values and computed the left indexer.
///
/// `original_right_positions` is the established Python-side name for the
/// right code array. Despite that name, its entries are dense factorization
/// codes, not emitted right labels. The emitted labels come from `right_index`.
/// Both right arrays use the same physical row order.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to build the result dictionary.
/// * `left_index` - Original left labels, one per left row.
/// * `right_index` - Original right labels in physical right-row order.
/// * `left_indexer` - Dense right-key codes returned by the left lookup;
///   `-1` means no equi match.
/// * `original_right_positions` - Dense right factorization codes, aligned
///   one-for-one with `right_index`; `-1` entries are ignored.
/// * `keep` - String form of the shared `Keep` mode: `"any"`, `"first"`,
///   `"last"`, or `"all"`.
///
/// # Returns
///
/// `None` when no left row has a matching right key. Otherwise, returns the
/// standard PyJanitor dictionary containing `left_index` and `right_index`
/// output arrays.
///
/// # Errors
///
/// Raises `ValueError` for an invalid `keep` value, misaligned input lengths,
/// codes below `-1`, or positional metadata that cannot be represented.
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
            Keep::All,
        )
        .unwrap();
        assert_eq!(metadata.counts, vec![0, 1, 0, 2]);
        assert_eq!(metadata.offsets, vec![0, 0, 1, 1, 3]);
        assert_eq!(metadata.positions, vec![1, 0, 2]);
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
