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
use crate::join_common::{Keep, SingleJoinResult};
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings,
    predicates_match_dispatch, Predicate,
};
use crate::range_predicate::{parse_any_range_predicate, AnyParsedRangePredicate};
use numpy::{ndarray::ArrayView1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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

/// Find the first element in a sorted physical-position slice that is at
/// least `target`.
fn lower_bound(values: &[usize], target: usize) -> usize {
    values.partition_point(|&value| value < target)
}

/// Build one or two range windows without dropping empty left rows.
///
/// The existing dual-range materializer removes empty windows because that is
/// convenient for ordinary range joins. Equi matching cannot do that: the
/// `left_indexer` remains aligned to the original left rows. This helper keeps
/// one `[start, end)` pair per left row and intersects the second range in
/// place when both ranges share the same physical right layout.
fn build_equi_range_windows<'py>(
    ranges: &[AnyParsedRangePredicate<'py>],
    left_len: usize,
    right_len: usize,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if ranges.len() > 2 {
        return Err("equi range path accepts at most two range predicates".to_owned());
    }
    if ranges.is_empty() {
        return Ok((vec![0; left_len], vec![right_len; left_len]));
    }

    let build = |range: &AnyParsedRangePredicate<'py>| -> Result<SingleJoinResult, String> {
        range.validate_range_operator()?;
        range.validate_lengths()?;
        ensure_equal_lengths_core("range left", range.left_len(), "left indexer", left_len)?;
        ensure_equal_lengths_core("range right", range.right_len(), "right index", right_len)?;
        if right_len == 0 {
            return Ok(SingleJoinResult {
                left_positions: (0..left_len).collect(),
                left_index: Vec::new(),
                right_index: Vec::new(),
                starts: vec![0; left_len],
                ends: vec![0; left_len],
            });
        }
        let windows = range.windows(true)?;
        ensure_equal_lengths_core(
            "range windows",
            windows.starts.len(),
            "left indexer",
            left_len,
        )?;
        Ok(windows)
    };

    let first = build(&ranges[0])?;
    let mut starts = first.starts;
    let mut ends = first.ends;
    if let Some(second) = ranges.get(1) {
        let second = build(second)?;
        for row in 0..left_len {
            starts[row] = starts[row].max(second.starts[row]);
            ends[row] = ends[row].min(second.ends[row]);
        }
    }
    Ok((starts, ends))
}

/// Visit duplicate-right equi candidates that survive the range window and
/// residual predicates for one left row.
///
/// The right positions for one code are sorted physical positions. Two lower
/// bounds therefore reduce the code slice to the intersection with the
/// row's half-open range window before residual predicates are evaluated.
fn visit_filtered_equi_candidates<F>(
    row: usize,
    left_code: i64,
    metadata: &DenseRightMetadata,
    starts: &[usize],
    ends: &[usize],
    right_len: usize,
    predicates: &[crate::predicate::PredicateView<'_>],
    null_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    mut visit: F,
) -> Result<usize, String>
where
    F: FnMut(usize),
{
    if left_code < -1 {
        return Err("left codes must be greater than or equal to -1".to_owned());
    }
    if left_code == -1 || starts[row] >= ends[row] {
        return Ok(0);
    }
    if ends[row] > right_len {
        return Err("equi range window is out of bounds".to_owned());
    }
    let code = usize::try_from(left_code).map_err(|_| "invalid left code")?;
    if metadata.counts.get(code).copied().unwrap_or(0) == 0 {
        return Ok(0);
    }
    let Some(&group_start) = metadata.offsets.get(code) else {
        return Ok(0);
    };
    let group_end = metadata.offsets[code + 1];
    let group = &metadata.positions[group_start..group_end];
    let first = lower_bound(group, starts[row]);
    let last = lower_bound(group, ends[row]);
    let mut count = 0_usize;
    for &right_position in &group[first..last] {
        if !predicates_match_dispatch(predicates, null_metadata, row, right_position) {
            continue;
        }
        count = count
            .checked_add(1)
            .ok_or("equi join result size exceeds platform capacity")?;
        visit(right_position);
    }
    Ok(count)
}

/// Materialize duplicate-right equi candidates after range and residual
/// filtering.
fn build_filtered_duplicate_equi_pairs_core(
    left_index: ArrayView1<'_, i64>,
    left_indexer: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    metadata: &DenseRightMetadata,
    starts: &[usize],
    ends: &[usize],
    predicates: &[crate::predicate::PredicateView<'_>],
    null_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    keep: Keep,
) -> Result<Option<(Vec<i64>, Vec<i64>)>, String> {
    ensure_equal_lengths_core(
        "equi starts",
        starts.len(),
        "left indexer",
        left_indexer.len(),
    )?;
    ensure_equal_lengths_core("equi ends", ends.len(), "left indexer", left_indexer.len())?;

    let mut output_len = 0_usize;
    for row in 0..left_indexer.len() {
        let mut row_count = 0;
        visit_filtered_equi_candidates(
            row,
            left_indexer[row],
            metadata,
            starts,
            ends,
            right_index.len(),
            predicates,
            null_metadata,
            |_| row_count += 1,
        )?;
        let emitted = if keep == Keep::All {
            row_count
        } else {
            usize::from(row_count > 0)
        };
        output_len = output_len
            .checked_add(emitted)
            .ok_or("equi join result size exceeds platform capacity")?;
    }
    if output_len == 0 {
        return Ok(None);
    }

    let mut left_output = Vec::with_capacity(output_len);
    let mut right_output = Vec::with_capacity(output_len);
    for row in 0..left_indexer.len() {
        let mut selected = None;
        visit_filtered_equi_candidates(
            row,
            left_indexer[row],
            metadata,
            starts,
            ends,
            right_index.len(),
            predicates,
            null_metadata,
            |right_position| match keep {
                Keep::All => {
                    left_output.push(left_index[row]);
                    right_output.push(right_index[right_position]);
                }
                Keep::Any => {
                    if selected.is_none() {
                        selected = Some(right_position);
                    }
                }
                Keep::First => {
                    if selected.is_none()
                        || right_index[right_position] < right_index[selected.unwrap()]
                    {
                        selected = Some(right_position);
                    }
                }
                Keep::Last => {
                    if selected.is_none()
                        || right_index[right_position] > right_index[selected.unwrap()]
                    {
                        selected = Some(right_position);
                    }
                }
            },
        )?;
        if let Some(right_position) = selected {
            left_output.push(left_index[row]);
            right_output.push(right_index[right_position]);
        }
    }
    debug_assert_eq!(left_output.len(), output_len);
    debug_assert_eq!(right_output.len(), output_len);
    Ok(Some((left_output, right_output)))
}

/// Materialize the unique-right path after residual filtering.
///
/// A unique right equi indexer contains at most one physical right candidate
/// per left row. Range predicates are supplied as residual predicates on this
/// path, so no range windows or duplicate-position metadata are required.
fn build_filtered_unique_equi_pairs_core(
    left_index: ArrayView1<'_, i64>,
    left_indexer: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    predicates: &[crate::predicate::PredicateView<'_>],
    null_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
) -> Result<Option<(Vec<i64>, Vec<i64>)>, String> {
    ensure_equal_lengths_core(
        "left index",
        left_index.len(),
        "equi indexer",
        left_indexer.len(),
    )?;
    let mut output_len = 0_usize;
    for row in 0..left_indexer.len() {
        let code = left_indexer[row];
        if code < -1 {
            return Err("left codes must be greater than or equal to -1".to_owned());
        }
        if code == -1 {
            continue;
        }
        let right_position = usize::try_from(code).map_err(|_| "invalid unique-right position")?;
        if right_position >= right_index.len()
            || !predicates_match_dispatch(predicates, null_metadata, row, right_position)
        {
            continue;
        }
        output_len = output_len
            .checked_add(1)
            .ok_or("equi join result size exceeds platform capacity")?;
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
        let right_position = usize::try_from(code).map_err(|_| "invalid unique-right position")?;
        if right_position >= right_index.len()
            || !predicates_match_dispatch(predicates, null_metadata, row, right_position)
        {
            continue;
        }
        left_output.push(left_index[row]);
        right_output.push(right_index[right_position]);
    }
    debug_assert_eq!(left_output.len(), output_len);
    debug_assert_eq!(right_output.len(), output_len);
    Ok(Some((left_output, right_output)))
}

/// Convert five-field range tuples into ordinary residual tuples.
///
/// The unique-right path has one equi candidate per left row, so a range
/// predicate is simply another filter. The range tuple additionally carries
/// aligned index arrays for window construction; residual matching needs only
/// its value arrays and operator.
fn append_range_residuals<'py>(
    py: Python<'py>,
    ranges: &Bound<'py, PyList>,
    residuals: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let combined = PyList::empty(py);
    for item in ranges.iter() {
        let tuple = item.cast::<PyTuple>()?;
        if tuple.len() != 5 {
            return Err(PyValueError::new_err(
                "equi range predicates must contain 5 elements",
            ));
        }
        combined.append(PyTuple::new(
            py,
            [tuple.get_item(0)?, tuple.get_item(2)?, tuple.get_item(4)?],
        )?)?;
    }
    for item in residuals.iter() {
        combined.append(item)?;
    }
    Ok(combined)
}

/// Build equi-join indices with optional range and residual predicates.
///
/// Range tuples use the existing five-field representation:
///
/// ```text
/// (left_values, left_index, right_values, right_index, operator)
/// ```
///
/// Residual tuples use the existing three- or six-field representation parsed
/// by [`parse_predicates_with_nulls_strings`]. The six-field form is reserved
/// for null-aware `!=` and retains the repository's existing null semantics.
///
/// Unique right keys use one direct candidate per left row. Duplicate right
/// keys build dense physical-position metadata; range windows are binary
/// searched inside the matching equi-code slice before residual predicates and
/// `keep` are applied.
#[pyfunction]
pub fn equi_join_filtered_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    left_indexer: PyReadonlyArray1<'py, i64>,
    original_right_positions: Option<PyReadonlyArray1<'py, i64>>,
    range_predicates: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    if range_predicates.len() > 2 {
        return Err(PyValueError::new_err(
            "equi range path accepts at most two range predicates",
        ));
    }

    let (parsed, metadata) = if original_right_positions.is_none() {
        let combined = append_range_residuals(py, range_predicates, residual_predicates)?;
        parse_predicates_with_nulls_strings(py, &combined)?
    } else {
        parse_predicates_with_nulls_strings(py, residual_predicates)?
    };
    check_predicate_lengths(
        &parsed,
        left_index.as_array().len(),
        right_index.as_array().len(),
    )?;
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);

    let pairs = if let Some(right_codes) = original_right_positions {
        ensure_equal_lengths_core(
            "right index",
            right_index.as_array().len(),
            "duplicate codes",
            right_codes.as_array().len(),
        )
        .map_err(PyValueError::new_err)?;
        let ranges = range_predicates
            .iter()
            .map(|item| {
                let tuple = item
                    .cast::<PyTuple>()
                    .map_err(|_| PyValueError::new_err("each equi range must be a tuple"))?;
                if tuple.len() != 5 {
                    return Err(PyValueError::new_err(
                        "equi range predicates must contain 5 elements",
                    ));
                }
                parse_any_range_predicate(tuple, true)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (starts, ends) = build_equi_range_windows(
            &ranges,
            left_index.as_array().len(),
            right_index.as_array().len(),
        )
        .map_err(PyValueError::new_err)?;
        let metadata =
            build_dense_right_metadata(right_index.as_array(), right_codes.as_array(), Keep::All)
                .map_err(PyValueError::new_err)?;
        build_filtered_duplicate_equi_pairs_core(
            left_index.as_array(),
            left_indexer.as_array(),
            right_index.as_array(),
            &metadata,
            &starts,
            &ends,
            &views,
            metadata_views.as_deref(),
            keep,
        )
        .map_err(PyValueError::new_err)?
    } else {
        build_filtered_unique_equi_pairs_core(
            left_index.as_array(),
            left_indexer.as_array(),
            right_index.as_array(),
            &views,
            metadata_views.as_deref(),
        )
        .map_err(PyValueError::new_err)?
    };

    pairs
        .map(|(left_output, right_output)| {
            crate::join_common::result_dict(py, left_output, right_output, None, None)
        })
        .transpose()
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
    m.add_function(wrap_pyfunction!(equi_join_filtered_indices, m)?)?;
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
    fn duplicate_equi_range_uses_binary_searched_code_slice() {
        let right_index = array![12_i64, 5, 9, 7, 3];
        let right_codes = array![3_i64, 1, 3, 3, 1];
        let metadata =
            build_dense_right_metadata(right_index.view(), right_codes.view(), Keep::All).unwrap();
        let result = build_filtered_duplicate_equi_pairs_core(
            array![10_i64].view(),
            array![3_i64].view(),
            right_index.view(),
            &metadata,
            &[2],
            &[4],
            &[],
            None,
            Keep::All,
        )
        .unwrap();
        assert_eq!(result.unwrap().1, vec![9, 7]);
    }

    #[test]
    fn duplicate_equi_range_applies_keep_after_window() {
        let right_index = array![100_i64, 50, 90, 70];
        let right_codes = array![2_i64, 2, 2, 2];
        let metadata =
            build_dense_right_metadata(right_index.view(), right_codes.view(), Keep::All).unwrap();
        let first = build_filtered_duplicate_equi_pairs_core(
            array![10_i64].view(),
            array![2_i64].view(),
            right_index.view(),
            &metadata,
            &[1],
            &[4],
            &[],
            None,
            Keep::First,
        )
        .unwrap();
        assert_eq!(first.unwrap().1, vec![50]);

        let last = build_filtered_duplicate_equi_pairs_core(
            array![10_i64].view(),
            array![2_i64].view(),
            right_index.view(),
            &metadata,
            &[1],
            &[3],
            &[],
            None,
            Keep::Last,
        )
        .unwrap();
        assert_eq!(last.unwrap().1, vec![90]);
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
