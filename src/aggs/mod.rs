//! Reverse aggregation input contracts.
//!
//! The match-based kernels consume a flat candidate tape. `matches.len()` is
//! the tape width (the sum of each row's candidate range), while
//! `counts_array.sum()` describes how many candidates matched and therefore
//! normally equals `matches.sum()`, not `matches.len()`. The producer,
//! pyjanitor, owns the invariant that match values are 0 or 1; Rust validates
//! the tape shape but deliberately does not scan every value.
//!
//! An empty `matches` tape is rejected by the Rust API. Pyjanitor handles the
//! legitimate all-zero-width batch before crossing this boundary and returns
//! the corresponding empty result. Individual zero-width rows remain valid
//! when other rows make the overall tape non-empty.
//!
//! ELI5: the tape is a roll of tickets shared by all rows. Rust checks that
//! the roll has the expected number of tickets, but pyjanitor decides which
//! tickets say yes (1) or no (0). A completely empty roll is not sent to Rust.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{Element, IntoPyArray, PyArray1};
use pyo3::prelude::*;

pub mod min;

pub mod prod;

pub mod max;
pub mod max_rev;
pub mod min_rev;
pub mod prod_rev;
pub mod size_rev;
pub mod sum;
pub mod sum_rev;

use pyo3::exceptions::PyValueError;

/// Reject parallel arrays that cannot describe the same number of rows.
///
/// ELI5: `zip` stops when either list runs out, so unequal lists can silently
/// leave work undone. Check the two ticket books once before the hot loop and
/// give Python a normal `ValueError` instead of plausible partial results.
///
/// # Arguments
///
/// * `left_name` - Name of the first parallel input, used in the error.
/// * `left_len` - Length of the first parallel input.
/// * `right_name` - Name of the second parallel input, used in the error.
/// * `right_len` - Length of the second parallel input.
///
/// # Errors
///
/// Returns a `ValueError` when the two lengths differ.
pub(crate) fn ensure_equal_lengths(
    left_name: &str,
    left_len: usize,
    right_name: &str,
    right_len: usize,
) -> PyResult<()> {
    if left_len == right_len {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{left_name} and {right_name} must have equal lengths; got {left_len} and {right_len}"
    )))
}

// Shared domain contract for the plain reverse `*_rev_starts` and
// `*_rev_ends` aggregation shapes (min/max/prod/sum/size).
//
// ELI5: every row's suffix `[start, right_len)` or prefix `[0, end)`
// touches a contiguous slice of the right side. The union of all those
// slices is one compact domain -- `min_start..right_len` for starts,
// `0..max_end` for ends -- so one accumulator slot per ordinal in that
// domain replaces per-row HashMaps. `starts_domain`/`ends_domain` validate
// the boundary array and derive that domain; `starts_labels`/`ends_labels`
// turn a domain back into the original right-side labels the caller
// expects in the output. Aggregate-specific accumulation (winning
// positions for min/max, running products for prod, counts for size)
// stays with each site -- only the domain and its labels are shared here.
// Validating equal lengths against `arr`/`booleans` also stays with each
// site: `size` doesn't have those arrays, so it isn't a shared concern.

/// Validate a non-empty `starts` boundary array against `right_len` and
/// derive the compact accumulator domain it implies.
///
/// Returns `(min_start, width)` where `width = right_len - min_start` is
/// the exact number of accumulator slots the compact scheme needs.
///
/// # Arguments
///
/// * `starts` - One exclusive-suffix start boundary per pyjanitor row.
/// * `right_len` - Number of positional rows in the right-side index.
///
/// # Errors
///
/// Returns an error when `starts` or the right index is empty, or when a
/// boundary is negative or larger than `right_len`.
pub(crate) fn starts_domain(
    starts: ArrayView1<'_, i64>,
    right_len: usize,
) -> Result<(usize, usize), &'static str> {
    if starts.is_empty() || right_len == 0 {
        return Err("starts and index cannot be empty");
    }
    if starts.iter().any(|start| {
        usize::try_from(*start)
            .map(|start| start > right_len)
            .unwrap_or(true)
    }) {
        return Err("starts must satisfy 0 <= start <= right_len");
    }
    let min_start = starts.iter().copied().min().unwrap() as usize;
    Ok((min_start, right_len - min_start))
}

/// Validate a non-empty `ends` boundary array against `right_len` and
/// derive the compact accumulator domain it implies.
///
/// Returns `max_end`, the exact number of accumulator slots the compact
/// scheme needs.
///
/// # Arguments
///
/// * `ends` - One exclusive-prefix end boundary per pyjanitor row.
/// * `right_len` - Number of positional rows in the right-side index.
///
/// # Errors
///
/// Returns an error when `ends` or the right index is empty, or when a
/// boundary is negative or larger than `right_len`.
pub(crate) fn ends_domain(
    ends: ArrayView1<'_, i64>,
    right_len: usize,
) -> Result<usize, &'static str> {
    if ends.is_empty() || right_len == 0 {
        return Err("ends and index cannot be empty");
    }
    if ends.iter().any(|end| {
        usize::try_from(*end)
            .map(|end| end > right_len)
            .unwrap_or(true)
    }) {
        return Err("ends must satisfy 0 <= end <= right_len");
    }
    Ok(ends.iter().copied().max().unwrap() as usize)
}

/// Materialize output labels for a `starts_domain` result.
///
/// # Arguments
///
/// * `min_start` - First absolute right-side position in the compact domain.
/// * `index` - Original right-side labels, indexed by positional ordinal.
///
/// # Returns
///
/// Labels from `min_start` through the final right-side position.
pub(crate) fn starts_labels(min_start: usize, index: ArrayView1<'_, i64>) -> Array1<i64> {
    (min_start..index.len()).map(|item| index[item]).collect()
}

/// Materialize output labels for an `ends_domain` result.
///
/// # Arguments
///
/// * `max_end` - Exclusive end of the compact prefix domain.
/// * `index` - Original right-side labels, indexed by positional ordinal.
///
/// # Returns
///
/// Labels from the first right-side position through `max_end - 1`.
pub(crate) fn ends_labels(max_end: usize, index: ArrayView1<'_, i64>) -> Array1<i64> {
    (0..max_end).map(|item| index[item]).collect()
}

// These knobs deliberately live beside the gate: they describe the policy for
// choosing an implementation, not the correctness of either implementation.
const SWEEP_WORK_RATIO: usize = 8;
const SWEEP_MEMORY_MULTIPLIER: usize = 8;

/// Choose the boundary sweep only when its one-time setup should repay the
/// repeated work in the direct nested loop without an excessive memory cost.
///
/// The direct implementation does roughly `rows * width` work. The sweep does
/// roughly `rows + width` work, but it also needs row-to-bucket metadata and a
/// bucket array. We estimate the direct path's retained storage as one value
/// and one `i64` label per output slot, then allow sweep metadata up to
/// `SWEEP_MEMORY_MULTIPLIER` times that estimate. This is intentionally a
/// conservative, allocation-independent estimate: it omits allocator headers,
/// shared label storage, and small scalar state, and therefore is a dispatch
/// guard rather than a memory guarantee.
///
/// `SWEEP_WORK_RATIO` prevents building metadata when the sweep would save only
/// a small amount of repeated work. The memory multiplier prevents a very large
/// row batch with a narrow output domain from turning the optimization into a
/// memory spike. Both are policy knobs that can be benchmark-tuned without
/// changing aggregation semantics.
///
/// ELI5: build the shortcut only when there are enough repeated chores to
/// make the setup worthwhile, and do not trade a tiny job for a huge bucket
/// of row links.
///
/// # Arguments
///
/// * `rows` - Number of pyjanitor input rows contributing boundaries.
/// * `width` - Number of positional slots in the compact right-side domain.
/// * `value_size` - Number of bytes used by one aggregate value.
///
/// # Returns
///
/// `true` when the sweep's estimated work reduction and memory use meet the
/// policy thresholds; otherwise `false` selects the direct implementation.
pub(crate) fn should_sweep(rows: usize, width: usize, value_size: usize) -> bool {
    let repeated_work = rows.saturating_mul(width);
    let sweep_work = rows.saturating_add(width);
    // The direct path keeps one value and one row position per output slot.
    // The labels are common to both paths, so they are intentionally omitted.
    // This is a conservative approximation: it models the direct path's
    // value and position buffers and the sweep's row-link and bucket arrays,
    // but excludes allocator headers and the small scalar winner state.
    let direct_bytes = width.saturating_mul(value_size + std::mem::size_of::<i64>());
    let sweep_metadata = rows
        .saturating_add(width.saturating_add(1))
        .saturating_mul(std::mem::size_of::<usize>());
    // These are policy knobs, not correctness conditions: the work ratio
    // avoids building metadata for tiny jobs, while the memory multiplier
    // limits the extra row-link/bucket storage relative to the direct path.
    let memory_budget = direct_bytes.saturating_mul(SWEEP_MEMORY_MULTIPLIER);
    repeated_work > sweep_work.saturating_mul(SWEEP_WORK_RATIO) && sweep_metadata <= memory_budget
}

/// Reduce boundary events into one value per compact output position.
///
/// ELI5: each row drops its contribution into the bucket where it becomes
/// active. We combine values sharing a bucket, then walk the buckets in the
/// caller's direction while carrying a running result. Starts and ends use
/// different bucket numbering, so callers provide those mappings; allocation
/// and sweep mechanics live here once.
///
/// Event and output buckets outside `0..width` are rejected instead of being
/// allowed to panic through an indexing operation. Domain helpers normally
/// establish this precondition before the reducer is called, but keeping the
/// guard here makes the shared helper safe for future callers too.
///
/// # Arguments
///
/// * `width` - Number of compact buckets in the output domain.
/// * `identity` - Empty aggregate value, such as `0` for sum or `1` for product.
/// * `events` - Iterator of `(bucket, value)` boundary contributions.
/// * `output_positions` - Iterator giving the scan order, forward or reverse.
/// * `combine` - Operation used to combine bucket and running values.
///
/// # Type Parameters
///
/// * `T` - Aggregate value type.
/// * `Events` - Type that produces boundary events.
/// * `OutputPositions` - Type that produces output bucket positions.
/// * `Combine` - Callable type implementing the aggregation operation.
///
/// # Returns
///
/// An array containing one running aggregate per output position.
///
/// # Errors
///
/// Returns an error if an event or requested output position is outside the
/// compact domain `0..width`.
pub(crate) fn sweep_reduce<T, Events, OutputPositions, Combine>(
    width: usize,
    identity: T,
    events: Events,
    output_positions: OutputPositions,
    combine: Combine,
) -> Result<Array1<T>, &'static str>
where
    T: Copy,
    Events: IntoIterator<Item = (usize, T)>,
    OutputPositions: IntoIterator<Item = usize>,
    Combine: Fn(T, T) -> T,
{
    let mut result = vec![identity; width];
    for (bucket, value) in events {
        let slot = result
            .get_mut(bucket)
            .ok_or("sweep event bucket is outside the output width")?;
        *slot = combine(*slot, value);
    }

    let mut running = identity;
    for position in output_positions {
        if position >= width {
            return Err("sweep output bucket is outside the output width");
        }
        running = combine(running, result[position]);
        result[position] = running;
    }
    Ok(Array1::from_vec(result))
}

/// Run a boundary sweep that returns the winning row for each output slot.
///
/// ELI5: put every row into the bucket where it becomes eligible, then walk
/// the output buckets in order while carrying the best row seen so far. The
/// boundary-specific mapping is supplied by the caller; null handling and
/// smallest-row tie-breaking live here exactly once.
///
/// `booleans` is the null mask supplied by pyjanitor. In particular, pyjanitor
/// marks floating-point NaN values as null before calling this backend, so the
/// kernel relies on that mask rather than implementing a second NaN policy for
/// every supported dtype.
///
/// # Arguments
///
/// * `arr` - Aggregate input values, one value per pyjanitor input row.
/// * `booleans` - Null mask with the same length as `arr`; `true` rows are
///   skipped. The caller must uphold this length invariant.
/// * `width` - Number of compact output buckets.
/// * `row_bucket` - Maps an input row to the bucket where it becomes active;
///   the returned bucket may be in `0..=width` because the sentinel bucket at
///   `width` represents a boundary just beyond the compact output domain.
/// * `output_bucket` - Maps an output position to its active bucket, also in
///   `0..=width`.
/// * `output_positions` - Iterator giving the scan order; each position must
///   be in `0..width` because it indexes the output array.
/// * `better` - Returns whether the current value beats the stored winner.
///
/// # Returns
///
/// An array of input-row positions, using `-1` where no non-null winner
/// exists. Equal values use the smallest input-row position as a tie-breaker.
///
/// # Panics
///
/// Panics if `booleans` is shorter than `arr`, if a caller-supplied bucket
/// mapping produces a value outside `0..=width`, or if `output_positions`
/// contains a position outside `0..width`. The production wrappers establish
/// these invariants before calling the shared reducer.
pub(crate) fn sweep_winner<T, RowBucket, OutputBucket, OutputPositions, Better>(
    arr: ArrayView1<'_, T>,
    booleans: ArrayView1<'_, bool>,
    width: usize,
    row_bucket: RowBucket,
    output_bucket: OutputBucket,
    output_positions: OutputPositions,
    better: Better,
) -> Array1<i64>
where
    T: PartialOrd + Copy,
    RowBucket: Fn(usize) -> usize,
    OutputBucket: Fn(usize) -> usize,
    OutputPositions: IntoIterator<Item = usize>,
    Better: Fn(T, T) -> bool,
{
    let mut head = vec![usize::MAX; width + 1];
    let mut next = vec![usize::MAX; arr.len()];
    for row in (0..arr.len()).rev() {
        let bucket = row_bucket(row);
        next[row] = head[bucket];
        head[bucket] = row;
    }

    let mut positions = vec![-1_i64; width];
    let mut current_winner: Option<(T, i64)> = None;
    for position in output_positions {
        let mut row = head[output_bucket(position)];
        while row != usize::MAX {
            if booleans[row] {
                row = next[row];
                continue;
            }
            let current = arr[row];
            let replaces_winner = match current_winner.as_ref() {
                None => true,
                Some((winner_value, winner_row)) => {
                    better(current, *winner_value)
                        || (current == *winner_value && (row as i64) < *winner_row)
                }
            };
            if replaces_winner {
                current_winner = Some((current, row as i64));
            }
            row = next[row];
        }
        if let Some((_, row)) = current_winner {
            positions[position] = row;
        }
    }
    Array1::from_vec(positions)
}

#[cfg(test)]
mod sweep_tests {
    use numpy::ndarray::array;

    use super::{should_sweep, sweep_reduce, sweep_winner};

    #[test]
    fn sweep_gate_accounts_for_row_link_memory_and_dtype() {
        assert!(!should_sweep(1_000_000, 9, std::mem::size_of::<i64>()));
        assert!(!should_sweep(1_000_000, 9, std::mem::size_of::<u8>()));
        assert!(should_sweep(1_000, 10_000, std::mem::size_of::<i64>()));
    }

    #[test]
    fn sweep_winner_covers_max_ties_nulls_and_both_directions() {
        let arr = array![7_i64, 7, 7];
        let booleans = array![true, false, false];

        let forward = sweep_winner(
            arr.view(),
            booleans.view(),
            3,
            |row| [0, 2, 1][row],
            |position| position,
            0..3,
            |current, winner| current > winner,
        );
        assert_eq!(forward, array![-1, 2, 1]);

        let reverse = sweep_winner(
            arr.view(),
            booleans.view(),
            3,
            |row| [3, 1, 2][row],
            |position| position + 1,
            (0..3).rev(),
            |current, winner| current > winner,
        );
        assert_eq!(reverse, array![1, 2, -1]);
    }

    #[test]
    fn sweep_reduce_supports_forward_and_reverse_event_orders() {
        let forward = sweep_reduce(3, 0_i64, [(0, 2), (1, 3), (0, 4)], 0..3, |left, right| {
            left + right
        });
        assert_eq!(forward, Ok(array![6, 9, 9]));

        let reverse = sweep_reduce(3, 1_i64, [(0, 2), (2, 3)], (0..3).rev(), |left, right| {
            left * right
        });
        assert_eq!(reverse, Ok(array![6, 3, 3]));

        let invalid = sweep_reduce(2, 0_i64, [(2, 1)], 0..2, |left, right| left + right);
        assert!(invalid.is_err());
    }
}

/// Shared return shape for every `*_rev_starts`/`*_rev_ends` `#[pyfunction]`
/// wrapper: a pair of numpy arrays, generic over the value array's element
/// type `U` so it fits min/max/size (`i64`) and prod/sum's int (`i64`) and
/// float (`f64`) variants alike.
///
/// The first array contains the original right-side labels; the second
/// contains the aggregate values for those labels.
pub(crate) type StartsEndsResult<'py, U> =
    PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<U>>)>;

/// Convert a plain reverse aggregation core's `Result` into the `PyResult`
/// pair every `*_rev_starts`/`*_rev_ends` `#[pyfunction]` wrapper returns.
///
/// ELI5: every wrapper in this family maps a core error to a `ValueError`
/// and then converts both output arrays to numpy the same way.
///
/// # Arguments
///
/// * `py` - Active Python interpreter used to create NumPy array views.
/// * `core_result` - Core output containing labels and aggregate values, or an
///   internal error message.
///
/// # Type Parameters
///
/// * `U` - Element type of the aggregate-value array.
///
/// # Errors
///
/// Converts a core error into a Python `ValueError`.
pub(crate) fn into_starts_ends_result<'py, U: Element>(
    py: Python<'py>,
    core_result: Result<(Array1<i64>, Array1<U>), &'static str>,
) -> StartsEndsResult<'py, U> {
    let (indexers, result) = core_result.map_err(PyValueError::new_err)?;
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

/// A wrapping-add accumulator identity, generic over `i64` (every integer
/// dtype except `uint64`) and `u64` (`uint64` itself).
///
/// ELI5: `sum`'s reverse kernels share one accumulation loop across every
/// integer dtype, funneling values through an `i64` accumulator -- harmless
/// for every dtype except `uint64`, where a value `>= 2**63` doesn't fit in
/// `i64` and the cast silently sign-flips it (issue #90). Parameterizing
/// the accumulator type over this trait lets the *same* loop run with an
/// `i64` accumulator for every dtype but `uint64`, and a `u64` accumulator
/// for `uint64` -- one core function instead of a second hand-copied one
/// per shape, so the `uint64` path can't quietly fall out of sync with its
/// sibling the way it did between issue #90's fix and every other shape.
pub(crate) trait WrapAdd: Copy {
    /// Additive identity used to initialize an accumulator.
    const ZERO: Self;

    /// Add two values with deliberate wrapping semantics.
    fn wrap_add(self, other: Self) -> Self;
}

impl WrapAdd for i64 {
    const ZERO: Self = 0;
    fn wrap_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl WrapAdd for u64 {
    const ZERO: Self = 0;
    fn wrap_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

/// A wrapping-multiply accumulator identity, generic over `i64` and `u64`.
///
/// ELI5: `prod`'s counterpart to `WrapAdd` -- same reasoning, multiplication
/// instead of addition, identity `1` instead of `0`.
pub(crate) trait WrapMul: Copy {
    /// Multiplicative identity used to initialize an accumulator.
    const ONE: Self;

    /// Multiply two values with deliberate wrapping semantics.
    fn wrap_mul(self, other: Self) -> Self;
}

impl WrapMul for i64 {
    const ONE: Self = 1;
    fn wrap_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }
}

impl WrapMul for u64 {
    const ONE: Self = 1;
    fn wrap_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }
}

/// Convert a signed position only when it names a real element.
///
/// ELI5: negative sentinels and positions past the end never become huge
/// `usize` values; they are rejected while they are still signed. Uses `<`
/// (not `<=`) because this is an *index* -- `index == len` is out of
/// bounds, unlike an exclusive end bound (see `checked_end`).
///
/// # Arguments
///
/// * `index` - Signed positional index supplied by a kernel or pyjanitor.
/// * `len` - Number of elements available for indexing.
///
/// # Returns
///
/// `Some` with a safe `usize` index, or `None` for negative/out-of-bounds
/// values.
pub(crate) fn checked_index(index: i64, len: usize) -> Option<usize> {
    usize::try_from(index).ok().filter(|&index| index < len)
}

/// Convert an exclusive signed end bound in the inclusive range `0..=len`.
///
/// ELI5: uses `<=` (not `<`) because `end` is a slice bound, not an index --
/// `end == len` legitimately means "up to and including the last element".
///
/// # Arguments
///
/// * `end` - Signed exclusive slice boundary.
/// * `len` - Number of elements available to the slice.
///
/// # Returns
///
/// `Some` with a safe `usize` boundary, or `None` for negative/oversized
/// values.
pub(crate) fn checked_end(end: i64, len: usize) -> Option<usize> {
    usize::try_from(end).ok().filter(|&end| end <= len)
}

/// Convert a non-empty signed half-open range contained in `0..len`.
///
/// ELI5: `start` only needs `usize::try_from` (no upper-bound check of its
/// own) because it's compared against the already-validated `end` next;
/// `start < end <= len` proves `start < len` for free.
///
/// # Arguments
///
/// * `start` - Signed inclusive start boundary.
/// * `end` - Signed exclusive end boundary.
/// * `len` - Number of elements available to the range.
///
/// # Returns
///
/// `Some((start, end))` for a non-empty in-bounds range, otherwise `None`.
pub(crate) fn checked_range(start: i64, end: i64, len: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(start).ok()?;
    let end = checked_end(end, len)?;
    (start < end).then_some((start, end))
}

/// Reject a flat `matches` tape too short for the candidate positions every
/// row's `(start, end)` range implies it must cover.
///
/// ELI5: unlike `ensure_equal_lengths`, `matches.len()` is compared against
/// the sum of all row widths. Existing callers use this as a lower bound.
///
/// # Arguments
///
/// * `expected_width` - Candidate positions required by the row ranges.
/// * `matches_len` - Number of entries in the flat candidate tape.
///
/// # Errors
///
/// Returns a `ValueError` when the tape is shorter than `expected_width`.
pub(crate) fn ensure_tape_width(expected_width: usize, matches_len: usize) -> PyResult<()> {
    if expected_width <= matches_len {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "matches must have length at least {expected_width} to cover every candidate position; got {matches_len}"
    )))
}

/// Reject an empty flat `matches` tape.
///
/// ELI5: the tape must contain at least one flag before a reverse aggregation
/// starts consuming it. The exact candidate-width check is performed
/// separately by `ensure_exact_tape_width`.
///
/// # Arguments
///
/// * `matches_len` - Number of entries in the flat candidate tape.
///
/// # Errors
///
/// Returns a `ValueError` when the tape is empty.
pub(crate) fn ensure_nonempty_matches(matches_len: usize) -> PyResult<()> {
    // Keep this check separate from the width check: an all-zero-width batch
    // is handled by pyjanitor, while a direct Rust caller must provide a real
    // candidate tape. This makes the boundary contract explicit and cheap.
    if matches_len == 0 {
        return Err(PyValueError::new_err("matches cannot be empty"));
    }
    Ok(())
}

/// Reject a flat `matches` tape whose length differs from the candidate width.
///
/// ELI5: `matches.len()` is the number of candidate positions represented by
/// the tape. `counts_array.sum()` is the number of candidates that survived
/// the comparison, so it generally equals `matches.sum()`, not
/// `matches.len()`. The producer (pyjanitor) is responsible for ensuring that
/// every `matches` value is either 0 or 1; this helper intentionally does not
/// scan the tape to enforce that value-level contract.
///
/// # Arguments
///
/// * `expected_width` - Candidate positions implied by the row ranges.
/// * `matches_len` - Number of entries in the flat candidate tape.
///
/// # Errors
///
/// Returns a `ValueError` when the two widths differ.
pub(crate) fn ensure_exact_tape_width(expected_width: usize, matches_len: usize) -> PyResult<()> {
    if expected_width == matches_len {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "matches must have length {expected_width}; got {matches_len}"
    )))
}

/// Shared function-pointer shape for the `_positions` family's `#[cfg(test)]`
/// dtype-signature checks, parameterized over the input element type `T`
/// and the (already-fixed-per-macro) result element type `R`.
///
/// ELI5: a regression test assigns a generated wrapper to
/// `PositionsFn<i8, i64>` (etc.); that only compiles if the macro was
/// instantiated with the input type the function's name promises, so a
/// regression back to the wrong type is a compile error, not a runtime
/// surprise. One shared alias instead of a copy in each `_positions.rs`
/// file, so a future signature change (e.g. a new parameter) only needs
/// updating here.
///
/// The alias is test-only and is not part of the Python module's public API.
#[cfg(test)]
pub(crate) type PositionsFn<T, R> =
    for<'py> fn(
        pyo3::Python<'py>,
        numpy::PyReadonlyArray1<'py, T>,
        numpy::PyReadonlyArray1<'py, i64>,
        numpy::PyReadonlyArray1<'py, i64>,
        numpy::PyReadonlyArray1<'py, i64>,
        numpy::PyReadonlyArray1<'py, bool>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<R>>>;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
///
/// # Arguments
///
/// * `m` - Python module receiving all aggregation exports.
///
/// # Errors
///
/// Propagates registration errors from any child aggregation module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    min::register(m)?;
    prod::register(m)?;
    max::register(m)?;
    max_rev::register(m)?;
    min_rev::register(m)?;
    prod_rev::register(m)?;
    size_rev::register(m)?;
    sum::register(m)?;
    sum_rev::register(m)?;
    Ok(())
}

#[cfg(test)]
mod adversarial_bounds_tests {
    use numpy::ndarray::array;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::Python;

    use super::ensure_equal_lengths;
    use super::max::max_ends::max_end_core;
    use super::max::max_ends_matches::max_end_match_core;
    use super::max::max_positions::max_positions_core;
    use super::max::max_starts::max_start_core;
    use super::max::max_starts_ends::max_start_end_core;
    use super::max::max_starts_ends_matches::max_start_end_match_core;
    use super::max::max_starts_matches::max_start_match_core;
    use super::min::min_ends::min_end_core;
    use super::min::min_ends_matches::min_end_match_core;
    use super::min::min_positions::min_positions_core;
    use super::min::min_starts::min_start_core;
    use super::min::min_starts_ends::min_start_end_core;
    use super::min::min_starts_ends_matches::min_start_end_match_core;
    use super::min::min_starts_matches::min_start_match_core;
    use super::{ensure_exact_tape_width, ensure_nonempty_matches, ensure_tape_width};

    #[test]
    fn equal_length_validation_accepts_empty_and_non_empty_pairs() {
        assert!(ensure_equal_lengths("starts", 0, "ends", 0).is_ok());
        assert!(ensure_equal_lengths("starts", 3, "ends", 3).is_ok());
    }

    #[test]
    fn equal_length_validation_rejects_both_mismatch_directions() {
        Python::initialize();
        for (starts_len, ends_len) in [(2, 1), (1, 2)] {
            let error = ensure_equal_lengths("starts", starts_len, "ends", ends_len)
                .expect_err("unequal parallel arrays must be rejected");
            Python::attach(|py| {
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(
                    error.value(py).to_string(),
                    format!(
                        "starts and ends must have equal lengths; got {starts_len} and {ends_len}"
                    )
                );
            });
        }
    }

    #[test]
    fn tape_width_validation_accepts_exact_and_longer_tapes() {
        assert!(ensure_tape_width(0, 0).is_ok());
        assert!(ensure_tape_width(5, 5).is_ok());
        assert!(ensure_tape_width(5, 8).is_ok());
    }

    #[test]
    fn tape_width_validation_rejects_a_too_short_tape() {
        Python::initialize();
        let error = ensure_tape_width(5, 4)
            .expect_err("a matches tape shorter than the total candidate width must be rejected");
        Python::attach(|py| {
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 4"
            );
        });
    }

    #[test]
    fn exact_tape_width_validation_rejects_short_and_long_tapes() {
        assert!(ensure_exact_tape_width(5, 5).is_ok());
        assert!(ensure_exact_tape_width(5, 4).is_err());
        assert!(ensure_exact_tape_width(5, 6).is_err());
    }

    #[test]
    fn matches_validation_rejects_empty_tapes() {
        assert!(ensure_nonempty_matches(1).is_ok());
        assert!(ensure_nonempty_matches(0).is_err());
    }

    #[test]
    fn representative_python_wrappers_reject_mismatched_lengths() {
        Python::initialize();
        Python::attach(|py| {
            // The ordinary Rust test job does not install Python's NumPy
            // module. Run this boundary test when NumPy is available, while
            // keeping the core-only test suite usable in that lean setup.
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            for (starts_values, ends_values) in
                [(vec![0_i64, 0], vec![1_i64]), (vec![0_i64], vec![1_i64, 1])]
            {
                let expected = format!(
                    "starts and ends must have equal lengths; got {} and {}",
                    starts_values.len(),
                    ends_values.len()
                );
                let starts = PyArray1::from_vec(py, starts_values);
                let ends = PyArray1::from_vec(py, ends_values);
                let arr = PyArray1::from_vec(py, vec![1_i64, 2]);
                let booleans = PyArray1::from_vec(py, vec![false, false]);
                let index = PyArray1::from_vec(py, vec![0_i64, 1]);

                let error = super::sum::sum_starts_ends::compute_sum_start_end_int64(
                    py,
                    arr.readonly(),
                    starts.readonly(),
                    ends.readonly(),
                    booleans.readonly(),
                )
                .expect_err("forward wrapper must reject unequal lengths");
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(error.value(py).to_string(), expected);

                let error = super::min_rev::min_starts_ends::compute_min_rev_start_end_int64(
                    py,
                    arr.readonly(),
                    starts.readonly(),
                    ends.readonly(),
                    index.readonly(),
                    booleans.readonly(),
                )
                .expect_err("reverse wrapper must reject unequal lengths");
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(error.value(py).to_string(), expected);

                let error = super::size_rev::computes::compute_size_rev_start_end(
                    py,
                    starts.readonly(),
                    ends.readonly(),
                    index.readonly(),
                )
                .expect_err("reverse-size wrapper must reject unequal lengths");
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(error.value(py).to_string(), expected);
            }
        });
    }

    #[test]
    fn index_builder_starts_ends_functions_reject_mismatched_lengths() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // `index_starts_and_ends*`/`build_positional_index_first/last`
            // zipped `starts` against `ends` (and `counts`) with no length
            // check, silently truncating to the shorter array on a
            // mismatch instead of raising -- unlike the `ensure_equal_lengths`
            // guard this same PR added for the analogous `right`/
            // `right_booleans` case in `comp_no_range_ne.rs`.
            let index = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![3_i64]); // mismatched: len 1 vs starts' len 2
            let counts = PyArray1::from_vec(py, vec![1_i64, 1]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            let positions = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let expected = "starts and ends must have equal lengths; got 2 and 1";

            let error = crate::index_builder::index_starts_and_ends(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                matches.readonly(),
                3,
            )
            .expect_err("mismatched starts/ends must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), expected);

            let error = crate::index_builder::index_starts_and_ends_keep_first(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                3,
            )
            .expect_err("mismatched starts/ends must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), expected);

            let error = crate::index_builder::index_starts_and_ends_keep_last(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                3,
            )
            .expect_err("mismatched starts/ends must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), expected);

            let error = crate::index_builder::build_positional_index_first(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                positions.readonly(),
                3,
            )
            .expect_err("mismatched starts/ends must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), expected);

            let error = crate::index_builder::build_positional_index_last(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                positions.readonly(),
                3,
            )
            .expect_err("mismatched starts/ends must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), expected);

            // `starts`/`counts` mismatch, `_keep_first`/`_keep_last` and
            // `build_positional_index_first/last` only.
            let starts2 = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends2 = PyArray1::from_vec(py, vec![3_i64, 3]);
            let counts2 = PyArray1::from_vec(py, vec![1_i64]); // mismatched: len 1 vs starts' len 2
            let expected_counts = "starts and counts must have equal lengths; got 2 and 1";

            let error = crate::index_builder::index_starts_and_ends_keep_first(
                py,
                index.readonly(),
                starts2.readonly(),
                ends2.readonly(),
                counts2.readonly(),
                matches.readonly(),
                3,
            )
            .expect_err("mismatched starts/counts must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), expected_counts);

            let error = crate::index_builder::build_positional_index_first(
                py,
                index.readonly(),
                starts2.readonly(),
                ends2.readonly(),
                counts2.readonly(),
                positions.readonly(),
                3,
            )
            .expect_err("mismatched starts/counts must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), expected_counts);
        });
    }

    #[test]
    fn representative_python_wrappers_reject_a_too_short_matches_tape() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // issue #41's own repro: a single row whose candidate range is
            // [0, 5), against a `matches` tape only 1 wide. Before the fix
            // this panicked (`pyo3_runtime.PanicException: index out of
            // bounds`) instead of raising a catchable Python exception.
            let arr = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4, 5]);
            let ends = PyArray1::from_vec(py, vec![5_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false; 5]);
            let error = super::max::max_ends_matches::compute_max_end_match_int64(
                py,
                arr.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect_err("a matches tape shorter than the candidate range must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 1"
            );

            // Same shape of bug, unguarded single-bound family (no
            // `checked_range` at all -- `sum`/`prod` forward `_ends_matches`
            // never validated `end` against anything before this fix).
            let arr = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4, 5]);
            let ends = PyArray1::from_vec(py, vec![5_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false; 5]);
            let error = super::sum::sum_ends_matches::compute_sum_end_match_int64(
                py,
                arr.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect_err("a matches tape shorter than the candidate range must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 1"
            );

            // `index_builder.rs`'s 9 functions had no `matches` length
            // check at all -- not even one comparable to `ensure_equal_lengths`.
            let index = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![5_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let error = crate::index_builder::index_starts_and_ends(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                matches.readonly(),
                5,
            )
            .expect_err("a matches tape shorter than the candidate range must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 1"
            );

            // A tape at least as long as the total candidate width must
            // still succeed -- this fix only rejects too-short tapes.
            let arr = PyArray1::from_vec(py, vec![1_i64, 9, 4]);
            let ends = PyArray1::from_vec(py, vec![3_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            let booleans = PyArray1::from_vec(py, vec![false; 3]);
            super::max::max_ends_matches::compute_max_end_match_int64(
                py,
                arr.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect("an exactly-sized matches tape must not be rejected");
        });
    }

    #[test]
    fn tape_width_precheck_does_not_underflow_on_a_sentinel_or_inverted_row() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // Regression for a bug in the initial #40/#41 fix: several
            // `expected_matches_width` pre-passes summed `end - start`
            // directly. The `-1` "no match" sentinel (or any `start` past
            // `end`) casts to a huge `usize`, and unlike the main loop's
            // `start_..end` range -- which is simply empty when
            // `start_ >= end`, contributing zero tape entries -- plain
            // subtraction on `usize` either panics (debug) or wraps to a
            // huge, wrong width (release), which then made a perfectly
            // valid call to `index_starts_only` spuriously reject a
            // correctly-sized `matches` tape.
            let index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let starts = PyArray1::from_vec(py, vec![-1_i64, 1]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let result = crate::index_builder::index_starts_only(
                py,
                index.readonly(),
                starts.readonly(),
                matches.readonly(),
                1,
            )
            .expect("a sentinel-start row must contribute zero width, not underflow");
            let got: Vec<i64> = result.readonly().as_array().to_vec();
            assert_eq!(got, vec![20]);

            // Same underflow shape, dual-bound family: an inverted
            // `(start, end)` row (`start > end`, no `-1` sentinel
            // involved) must also contribute zero, not a wrapped width.
            let index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let starts = PyArray1::from_vec(py, vec![2_i64, 0]);
            let ends = PyArray1::from_vec(py, vec![1_i64, 2]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1]);
            let result = crate::index_builder::index_starts_and_ends(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                matches.readonly(),
                2,
            )
            .expect("an inverted start>end row must contribute zero width, not underflow");
            let got: Vec<i64> = result.readonly().as_array().to_vec();
            assert_eq!(got, vec![10, 20]);
        });
    }

    #[test]
    fn no_range_and_positional_functions_reject_out_of_bounds_indices_without_panicking() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // issue #38/#79: `arr[*index_left as usize]`/`booleans[...]` in
            // the four `_rev/*_no_range.rs` files were indexed with no bound
            // check at all. A `left_index` entry past `arr.len()` must now
            // be rejected instead of panicking or silently disappearing.
            let arr = PyArray1::from_vec(py, vec![5_i64, 9]);
            let left_index = PyArray1::from_vec(py, vec![0_i64, 99]); // 99 is out of bounds
            let right_index = PyArray1::from_vec(py, vec![0_i64, 0]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);
            let result = super::max_rev::max_no_range::compute_max_rev_no_range_int64(
                py,
                arr.readonly(),
                left_index.readonly(),
                right_index.readonly(),
                booleans.readonly(),
            );
            assert!(result.is_err(), "invalid left_index must be rejected");

            // Adversarial-review finding folded into #38: `comp_no_range.rs`
            // only guarded the `-1` sentinel, never the upper bound, before
            // indexing `right[*right_pos as usize]`.
            let left = PyArray1::from_vec(py, vec![3_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2]);
            let positions = PyArray1::from_vec(py, vec![99_i64]); // out of bounds, not -1
            let (result, total) = super::super::compare::comp_no_range::compare_no_range_int64(
                py,
                left.readonly(),
                right.readonly(),
                positions.readonly(),
                0, // op: >
            )
            .expect("a valid op code with an out-of-bounds position must still succeed");
            assert_eq!(result.readonly().as_array().to_vec(), vec![-1_i64]);
            assert_eq!(total, 0);

            // Same finding, `comp_no_range_ne.rs`: also gains an
            // `ensure_equal_lengths("right", ..., "right_booleans", ...)`
            // whole-call check, since both are indexed by the same
            // `right_pos`.
            let left = PyArray1::from_vec(py, vec![3_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2]);
            let right_booleans = PyArray1::from_vec(py, vec![false]); // mismatched length
            let positions = PyArray1::from_vec(py, vec![0_i64]);
            let left_booleans = PyArray1::from_vec(py, vec![false]);
            let error = super::super::compare::comp_no_range_ne::compare_no_range_ne_int64(
                py,
                left.readonly(),
                right.readonly(),
                positions.readonly(),
                left_booleans.readonly(),
                right_booleans.readonly(),
                false,
                0,
            )
            .expect_err("mismatched right/right_booleans lengths must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));

            // Adversarial-review finding folded into #38:
            // `index_builder::build_positional_index` only guarded
            // `position < 0`, never the upper bound.
            let index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let positions = PyArray1::from_vec(py, vec![99_i64]); // out of bounds
            let result = crate::index_builder::build_positional_index(
                py,
                index.readonly(),
                positions.readonly(),
                1,
            );
            assert_eq!(result.readonly().as_array().to_vec(), vec![0_i64]);

            // Adversarial-review finding folded into #38: the write side
            // (`result[n] = val`) was unguarded against `n >= result.len()`
            // -- a `length` smaller than the number of in-bounds entries
            // `positions` actually yields must break cleanly, not panic.
            let index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let positions = PyArray1::from_vec(py, vec![0_i64, 1, 2]); // 3 valid entries
            let result = crate::index_builder::build_positional_index(
                py,
                index.readonly(),
                positions.readonly(),
                1, // capacity for only 1
            );
            assert_eq!(result.readonly().as_array().to_vec(), vec![10_i64]);

            // Adversarial review of PR #45 (P1): `index_builder::
            // reorder_index` used to leave a rejected mapping as the
            // crate's usual `-1` sentinel and return `Ok`, but pyjanitor's
            // only caller does an unfiltered `right.iloc[reordered_
            // positions]` on this output -- pandas treats `-1` as the
            // *last* row, not "no match", so a malformed mapping silently
            // duplicated a row into the result instead of surfacing as an
            // error. A bucket id past `starts.len()` must now reject the
            // whole call with `ValueError`, not return a `-1`-containing
            // array.
            let positions = PyArray1::from_vec(py, vec![99_i64]); // out of bounds bucket id
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let error =
                crate::index_builder::reorder_index(py, positions.readonly(), starts.readonly())
                    .expect_err(
                        "an out-of-range positions[i] bucket id must be rejected, not \
                         silently mapped to -1",
                    );
            assert!(error.is_instance_of::<PyValueError>(py));

            // Adversarial review of PR #45 (P2): `starts[bucket] +
            // counts[bucket]` used plain `+=`, so a `starts` value near
            // `i64::MAX` overflowed on a later row sharing that bucket --
            // panicking in debug builds, silently wrapping in release
            // builds. `checked_add` must turn this into the same
            // `ValueError` in every build profile, never a panic.
            let positions = PyArray1::from_vec(py, vec![0_i64, 0_i64]);
            let starts = PyArray1::from_vec(py, vec![i64::MAX]);
            let error =
                crate::index_builder::reorder_index(py, positions.readonly(), starts.readonly())
                    .expect_err(
                        "an overflowing starts[bucket] + counts[bucket] must be rejected, \
                         not panic or wrap",
                    );
            assert!(error.is_instance_of::<PyValueError>(py));

            // A valid bucket id and a non-overflowing position are not
            // sufficient: duplicate starts can make two rows target the
            // same output slot while leaving another slot untouched. That
            // would leave a `-1` for pandas to reinterpret as its last row.
            let positions = PyArray1::from_vec(py, vec![0_i64, 1_i64]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0_i64]);
            let error =
                crate::index_builder::reorder_index(py, positions.readonly(), starts.readonly())
                    .expect_err("overlapping reorder output positions must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));

            // The positional first/last variants also receive raw indexer
            // values from the caller. A positive out-of-bounds indexer must
            // be skipped just like the existing -1 sentinel, not panic.
            let index = PyArray1::from_vec(py, vec![10_i64, 20_i64]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![1_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let positions = PyArray1::from_vec(py, vec![99_i64]);
            let first = crate::index_builder::build_positional_index_first(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                positions.readonly(),
                1,
            )
            .expect("an invalid first-position indexer must be skipped");
            assert_eq!(first.readonly().as_array().to_vec(), vec![-1_i64]);
            let last = crate::index_builder::build_positional_index_last(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                positions.readonly(),
                1,
            )
            .expect("an invalid last-position indexer must be skipped");
            assert_eq!(last.readonly().as_array().to_vec(), vec![-1_i64]);

            // The starts/ends family must validate the range used to index
            // `index` before touching it; the three variants share this
            // same unchecked loop shape.
            let index = PyArray1::from_vec(py, vec![10_i64, 20_i64]);
            let starts = PyArray1::from_vec(py, vec![99_i64]);
            let ends = PyArray1::from_vec(py, vec![100_i64]);
            let matches = PyArray1::from_vec(py, Vec::<i8>::new());
            let result = crate::index_builder::index_starts_and_ends(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                matches.readonly(),
                1,
            )
            .expect("an invalid starts/ends row must be skipped");
            assert_eq!(result.readonly().as_array().to_vec(), vec![0_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let first = crate::index_builder::index_starts_and_ends_keep_first(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                1,
            )
            .expect("an invalid keep-first starts/ends row must be skipped");
            assert_eq!(first.readonly().as_array().to_vec(), vec![-1_i64]);
            let last = crate::index_builder::index_starts_and_ends_keep_last(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                1,
            )
            .expect("an invalid keep-last starts/ends row must be skipped");
            assert_eq!(last.readonly().as_array().to_vec(), vec![-1_i64]);

            // The ends-only siblings must reject an end beyond `index.len()`
            // before their `index[nn]` loops can walk past the value array.
            let index = PyArray1::from_vec(py, vec![10_i64, 20_i64]);
            let ends = PyArray1::from_vec(py, vec![3_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            let result = crate::index_builder::index_ends_only(
                py,
                index.readonly(),
                ends.readonly(),
                matches.readonly(),
                1,
            )
            .expect("an oversized end must be skipped");
            assert_eq!(result.readonly().as_array().to_vec(), vec![0_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let first = crate::index_builder::index_ends_only_keep_first(
                py,
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                1,
            )
            .expect("an oversized keep-first end must be skipped");
            assert_eq!(first.readonly().as_array().to_vec(), vec![-1_i64]);
            let last = crate::index_builder::index_ends_only_keep_last(
                py,
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                1,
            )
            .expect("an oversized keep-last end must be skipped");
            assert_eq!(last.readonly().as_array().to_vec(), vec![-1_i64]);

            // A -1 start with a zero-count row must not cast to a huge
            // usize and underflow while computing the skipped tape width.
            let starts = PyArray1::from_vec(py, vec![-1_i64]);
            let counts = PyArray1::from_vec(py, vec![0_i64]);
            let matches = PyArray1::from_vec(py, Vec::<i8>::new());
            let first = crate::index_builder::index_starts_only_keep_first(
                py,
                index.readonly(),
                starts.readonly(),
                counts.readonly(),
                matches.readonly(),
                1,
            )
            .expect("a sentinel zero-count start must not underflow");
            assert_eq!(first.readonly().as_array().to_vec(), vec![0_i64]);
            let last = crate::index_builder::index_starts_only_keep_last(
                py,
                index.readonly(),
                starts.readonly(),
                counts.readonly(),
                matches.readonly(),
                1,
            )
            .expect("a sentinel zero-count start must not underflow");
            assert_eq!(last.readonly().as_array().to_vec(), vec![0_i64]);
        });
    }

    #[test]
    fn every_forward_core_rejects_signed_and_one_past_bounds() {
        let arr = array![5_i64, 1, 4];
        let booleans = array![false, false, false];
        let invalid_starts = array![-1_i64, 3, 4];
        let invalid_ends = array![-1_i64, 4];
        let zero_counts = array![0_i64, 0, 0];
        let empty_matches = array![];

        assert_eq!(
            min_start_core(arr.view(), invalid_starts.view(), booleans.view()),
            array![-1, -1, -1]
        );
        assert_eq!(
            max_start_core(arr.view(), invalid_starts.view(), booleans.view()),
            array![-1, -1, -1]
        );
        assert_eq!(
            min_end_core(arr.view(), invalid_ends.view(), booleans.view()),
            array![-1, -1]
        );
        assert_eq!(
            max_end_core(arr.view(), invalid_ends.view(), booleans.view()),
            array![-1, -1]
        );

        let starts = array![0_i64, 0];
        assert_eq!(
            min_start_end_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_start_end_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );

        let positions = array![0_i64, 1, 2];
        let position_starts = array![-1_i64, 0];
        let position_ends = array![1_i64, 4];
        assert_eq!(
            min_positions_core(
                arr.view(),
                position_starts.view(),
                position_ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_positions_core(
                arr.view(),
                position_starts.view(),
                position_ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );

        assert_eq!(
            min_start_match_core(
                arr.view(),
                invalid_starts.view(),
                zero_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1, -1]
        );
        assert_eq!(
            max_start_match_core(
                arr.view(),
                invalid_starts.view(),
                zero_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1, -1]
        );

        let invalid_end_counts = array![0_i64, 0];
        assert_eq!(
            min_end_match_core(
                arr.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_end_match_core(
                arr.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            min_start_end_match_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_start_end_match_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
    }

    #[test]
    fn zero_count_rows_return_minus_one_without_shifting_the_match_tape() {
        let arr = array![5_i64, 1, 4];
        let booleans = array![false, false, false];
        let counts = array![0_i64, 1];

        let starts = array![0_i64, 1];
        let start_matches = array![0_i8, 0, 0, 1, 0];
        assert_eq!(
            min_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let ends = array![3_i64, 2];
        let end_matches = array![0_i8, 0, 0, 0, 1];
        assert_eq!(
            min_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let interval_ends = array![3_i64, 3];
        assert_eq!(
            min_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
    }

    #[test]
    fn invalid_rows_contribute_zero_slots_to_the_match_tape() {
        let arr = array![5_i64, 1, 4];
        let booleans = array![false, false, false];
        let counts = array![0_i64, 1];

        let starts = array![-1_i64, 1];
        let start_matches = array![1_i8, 0];
        assert_eq!(
            min_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let ends = array![-1_i64, 2];
        let end_matches = array![0_i8, 1];
        assert_eq!(
            min_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let interval_ends = array![2_i64, 3];
        assert_eq!(
            min_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
    }

    #[test]
    fn positions_outside_the_value_array_are_skipped() {
        let arr = array![5_i64, 1, 4];
        let starts = array![0_i64];
        let ends = array![2_i64];
        let positions = array![-2_i64, 3];
        let booleans = array![false, false, false];

        assert_eq!(
            min_positions_core(
                arr.view(),
                starts.view(),
                ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1]
        );
        assert_eq!(
            max_positions_core(
                arr.view(),
                starts.view(),
                ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1]
        );
    }
}
