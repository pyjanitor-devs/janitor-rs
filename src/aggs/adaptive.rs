// Three is the deliberately conservative crossover used by the adaptive
// prefix/suffix aggregation paths: building a running buffer costs about one
// full scan of `arr`, plus O(arr.len()) temporary memory, so repeated direct
// scans must be meaningfully more expensive before we allocate that buffer.
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

// A segment tree does not answer a query in O(1), unlike a prefix or suffix
// buffer. Keep a separate cost model so many narrow ranges do not trigger a
// tree merely because their total width exceeds the running-buffer cutoff.
const SEGMENT_TREE_BUILD_WORK_FACTOR: usize = 5;
const SEGMENT_TREE_QUERY_WORK_FACTOR: usize = 2;

/// Choose a segment tree only when direct range scans are estimated to cost
/// more than building the tree and visiting its nodes for every query.
///
/// ELI5: a segment tree is a filing cabinet whose drawers summarize blocks of
/// values. Opening the cabinet costs a full pass over the shelf, and answering
/// a question opens several drawers. Do that only when walking each requested
/// section directly would take longer than opening the cabinet and drawers.
pub(crate) fn should_use_segment_tree(
    query_count: usize,
    total_width: usize,
    array_len: usize,
) -> bool {
    if query_count <= 3 || array_len == 0 {
        return false;
    }
    // The iterative range walk works with any number of leaves; padding to a
    // power of two only wastes memory. `array_len - 1` is safe after the zero
    // length guard and gives the height of the smallest covering tree.
    let tree_height = (usize::BITS - (array_len - 1).leading_zeros()) as usize;
    let build_cost = array_len.saturating_mul(SEGMENT_TREE_BUILD_WORK_FACTOR);
    let query_cost = query_count
        .saturating_mul(tree_height)
        .saturating_mul(SEGMENT_TREE_QUERY_WORK_FACTOR);
    total_width > build_cost.saturating_add(query_cost)
}

#[cfg(test)]
mod tests {
    use super::{should_use_running_aggregation, should_use_segment_tree};

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

    #[test]
    fn segment_tree_cutoff_rejects_many_narrow_ranges() {
        assert!(!should_use_segment_tree(100_000, 400_000, 100_000));
    }

    #[test]
    fn segment_tree_cutoff_accepts_many_broad_ranges() {
        assert!(should_use_segment_tree(100_000, 5_000_000_000, 100_000));
    }

    #[test]
    fn segment_tree_cutoff_is_overflow_safe() {
        assert!(!should_use_segment_tree(usize::MAX, usize::MAX, usize::MAX));
    }
}
