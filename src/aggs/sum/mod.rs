use pyo3::prelude::*;

pub mod sum_ends;
pub mod sum_ends_matches;
pub mod sum_positions;
pub mod sum_starts;
pub mod sum_starts_ends;
pub mod sum_starts_ends_matches;
pub mod sum_starts_matches;

// Three is the deliberately conservative crossover used by the adaptive
// forward-sum paths: building a running-sum buffer costs about one full scan
// of `arr`, plus O(arr.len()) temporary memory, so repeated direct scans must
// be meaningfully more expensive before we allocate that buffer. This value
// matches the measured cutoff used by the corresponding pyjanitor approach
// and the forward-sum benchmarks.
//
// ELI5: if only a few people ask for sums, answer each person by walking the
// shelf they asked about. If the questions together would walk the shelf more
// than roughly three times, walk it once while writing down every running sum,
// then answer the questions from those notes.
const RUNNING_SUM_WORK_FACTOR: usize = 3;

/// Chooses a materialized prefix/suffix scan when repeated range work is
/// expected to exceed the cost of building one full-array running total.
///
/// The query-count guard avoids paying for a prepass and buffer for a handful
/// of queries. The work comparison is conservative and overflow-safe.
pub(crate) fn should_use_running_sum(
    query_count: usize,
    total_width: usize,
    array_len: usize,
) -> bool {
    query_count > RUNNING_SUM_WORK_FACTOR
        && total_width > array_len.saturating_mul(RUNNING_SUM_WORK_FACTOR)
}

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sum_ends::register(m)?;
    sum_ends_matches::register(m)?;
    sum_positions::register(m)?;
    sum_starts::register(m)?;
    sum_starts_ends::register(m)?;
    sum_starts_ends_matches::register(m)?;
    sum_starts_matches::register(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::should_use_running_sum;

    #[test]
    fn prefix_sum_cutoff_avoids_small_query_batches() {
        assert!(!should_use_running_sum(3, usize::MAX, 10));
    }

    #[test]
    fn prefix_sum_cutoff_requires_more_than_three_scans() {
        assert!(!should_use_running_sum(4, 30, 10));
        assert!(should_use_running_sum(4, 31, 10));
    }

    #[test]
    fn prefix_sum_cutoff_is_overflow_safe() {
        assert!(!should_use_running_sum(4, usize::MAX, usize::MAX));
        assert!(should_use_running_sum(4, usize::MAX, 1));
    }
}
