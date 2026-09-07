// Three is the deliberately conservative crossover used by the adaptive
// forward aggregation paths: building a running buffer costs about one full
// scan of `arr`, plus O(arr.len()) temporary memory, so repeated direct scans
// must be meaningfully more expensive before we allocate that buffer.
//
// ELI5: if only a few people ask questions about a shelf, answer each person
// by walking the part they asked about. If the questions together would walk
// the shelf more than roughly three times, walk it once while writing down
// running answers, then answer from those notes.
const RUNNING_AGGREGATION_WORK_FACTOR: usize = 3;

/// Choose a materialized running scan when repeated range work is expected to
/// exceed the cost of building one full-array aggregation buffer.
///
/// The query-count guard avoids paying for a prepass and buffer for only a
/// handful of queries. The work comparison is conservative and overflow-safe.
pub(crate) fn should_use_running_aggregation(
    query_count: usize,
    total_width: usize,
    array_len: usize,
) -> bool {
    query_count > RUNNING_AGGREGATION_WORK_FACTOR
        && total_width > array_len.saturating_mul(RUNNING_AGGREGATION_WORK_FACTOR)
}

#[cfg(test)]
mod tests {
    use super::should_use_running_aggregation;

    #[test]
    fn cutoff_avoids_small_query_batches() {
        assert!(!should_use_running_aggregation(3, usize::MAX, 10));
    }

    #[test]
    fn cutoff_requires_more_than_three_scans() {
        assert!(!should_use_running_aggregation(4, 30, 10));
        assert!(should_use_running_aggregation(4, 31, 10));
    }

    #[test]
    fn cutoff_is_overflow_safe() {
        assert!(!should_use_running_aggregation(4, usize::MAX, usize::MAX));
        assert!(should_use_running_aggregation(4, usize::MAX, 1));
    }
}
