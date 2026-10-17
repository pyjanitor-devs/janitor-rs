//! Shared coordinates for trees over queried source spans.
//!
//! A packed layout concatenates the disjoint source spans touched by a batch
//! of half-open ranges. Queries keep their original source coordinates at the
//! API boundary, but tree leaves use the compact coordinates so unqueried gaps
//! do not consume build or query storage.

/// Find disjoint contiguous source spans covering all valid requests.
pub(crate) fn queried_source_spans(ranges: &[Option<(usize, usize)>]) -> Vec<(usize, usize)> {
    let mut intervals: Vec<_> = ranges.iter().flatten().copied().collect();
    intervals.sort_unstable();
    let mut spans = Vec::new();
    for (start, end) in intervals {
        if let Some((_, current_end)) = spans.last_mut() {
            if start <= *current_end {
                *current_end = (*current_end).max(end);
                continue;
            }
        }
        spans.push((start, end));
    }
    spans
}

/// Assign each valid request to its disjoint source span.
pub(crate) fn assign_range_clusters(
    ranges: &[Option<(usize, usize)>],
    spans: &[(usize, usize)],
) -> Vec<Option<usize>> {
    ranges
        .iter()
        .map(|range| {
            let (start, end) = (*range)?;
            // Spans are sorted and disjoint. Find the first span whose end is
            // beyond this range, then check that the range starts inside it.
            let cluster = spans.partition_point(|&(_, span_end)| span_end <= start);
            spans
                .get(cluster)
                .filter(|&&(span_start, span_end)| start >= span_start && end <= span_end)
                .map(|_| cluster)
        })
        .collect()
}

/// Coordinates for trees over queried spans laid end to end.
pub(crate) struct PackedSpanLayout<'a> {
    pub(crate) spans: &'a [(usize, usize)],
    pub(crate) clusters: &'a [Option<usize>],
    pub(crate) offsets: Vec<usize>,
    pub(crate) length: usize,
}

impl<'a> PackedSpanLayout<'a> {
    pub(crate) fn new(spans: &'a [(usize, usize)], clusters: &'a [Option<usize>]) -> Self {
        let mut offsets = Vec::with_capacity(spans.len());
        let mut length = 0;
        for &(start, end) in spans {
            offsets.push(length);
            length += end - start;
        }
        Self {
            spans,
            clusters,
            offsets,
            length,
        }
    }

    /// Convert one original source range to packed data coordinates.
    pub(crate) fn data_range(
        &self,
        row: usize,
        range: Option<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        let cluster = self.clusters[row]?;
        let (start, end) = range?;
        let (span_start, _) = self.spans[cluster];
        let offset = self.offsets[cluster];
        Some((offset + start - span_start, offset + end - span_start))
    }

    /// Convert one original source range to its half-open packed-tree range.
    pub(crate) fn packed_range(
        &self,
        row: usize,
        range: Option<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        let (start, end) = self.data_range(row, range)?;
        Some((self.length + start, self.length + end))
    }
}
