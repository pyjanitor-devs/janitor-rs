//! Skip right rows that fail an unordered range comparison.
//!
//! Range joins narrow each left row to a window of right rows using sorted
//! anchors. A range predicate whose right values are *not* sorted in that
//! order -- the equi path's second range, or the default range path's
//! residual behind a cumulative envelope -- can only be checked row by row,
//! and one extreme right value (a very long interval) can keep every later
//! window wide while almost nothing in it matches. [`SecondRangeSkipScan`]
//! summarizes the right values in blocks so a scan can walk past runs of
//! rows that cannot match; [`WindowResidualScan`] applies it to the residual
//! predicates of a window-based join.

use numpy::ndarray::ArrayView1;

use super::compare_op::CompareOp;
use super::join_types::SingleJoinResult;
use super::predicate::{
    predicates_match_dispatch, NullMetadata, NullMetadataView, Predicate, PredicateView,
};

#[cfg(test)]
thread_local! {
    pub(crate) static SECOND_RANGE_SKIP_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static SECOND_RANGE_SUPER_SKIPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static WINDOW_RESIDUAL_SKIP_PLANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static WINDOW_RESIDUAL_SKIP_DENSE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Rows summarized by one block of [`SecondRangeSkipScan`].
const SECOND_RANGE_BLOCK: usize = 32;

/// Blocks summarized by one super-block (`32 * 32 = 1,024` rows).
const SECOND_RANGE_SUPER: usize = 32;

/// Windows narrower than this are scanned directly; the block bookkeeping
/// would cost more than reading a couple of blocks.
const SECOND_RANGE_DIRECT_WIDTH: usize = 2 * SECOND_RANGE_BLOCK;

/// Two-level block summaries that skip right rows failing an unordered range.
///
/// The cumulative envelope can only trim a window's prefix or suffix. One
/// early right row with an extreme value (a very long interval, say) keeps
/// every later window wide, and the exact scan then reads many candidates per
/// match. This index stores the most permissive value of every
/// `SECOND_RANGE_BLOCK`-row block -- the maximum for `<`/`<=`, the minimum for
/// `>`/`>=` -- and of every `SECOND_RANGE_SUPER`-block super-block. A query
/// walks its window left to right, skipping a whole super-block or block
/// whose summary already fails the predicate, and checks rows exactly only
/// inside surviving blocks. Positions are therefore reported in ascending
/// order, exactly like the direct scan.
///
/// The cost is about one summary check per 1,024 rows of a sparse window and
/// one per 32 rows of a dense one, plus the rows of the surviving blocks.
/// A segment tree prunes sparse windows only marginally better but pays a
/// per-node descent on dense windows, where nothing can be skipped; measured
/// on AIList, that made dense joins up to 1.7x slower in the kernel.
///
/// ELI5: each box of 32 cards has a sticky note with its best card, and each
/// crate of 32 boxes has a note with the best card in the crate. Walking down
/// the aisle, we skip any crate or box whose best card cannot win, and only
/// open the boxes that might hold a winner.
///
/// The summaries cover the whole prepared right layout, so they are
/// independent of equality windows. A window that starts or ends inside a
/// block or super-block still reads that summary: it covers a superset of the
/// window's rows, which can only make skipping less aggressive, never wrong.
pub(crate) struct SecondRangeSkipScan<'a, T> {
    values: ArrayView1<'a, T>,
    /// One summary per block; empty when the scanner reads windows directly.
    blocks: Vec<T>,
    /// One summary per super-block of `SECOND_RANGE_SUPER` blocks.
    supers: Vec<T>,
    op: CompareOp,
}

impl<'a, T: PartialOrd + Copy> SecondRangeSkipScan<'a, T> {
    /// A scanner without summaries: every query reads its window directly.
    ///
    /// Used when the candidate windows are too narrow in total to repay the
    /// summary build (see [`second_range_skip_pays_off`]).
    pub(crate) fn direct(values: ArrayView1<'a, T>, op: CompareOp) -> Self {
        Self {
            values,
            blocks: Vec::new(),
            supers: Vec::new(),
            op,
        }
    }

    /// Build block and super-block summaries for `left op right`.
    ///
    /// `op` must be a range comparator.
    pub(crate) fn new(values: ArrayView1<'a, T>, op: CompareOp) -> Self {
        debug_assert!(!matches!(op, CompareOp::Eq | CompareOp::Ne));
        // `left < right` needs some right value above `left`, so a block's
        // most permissive value is its maximum; `left > right` needs one
        // below, so it is the minimum.
        let use_max = matches!(op, CompareOp::Lt | CompareOp::Le);
        let most_permissive = |chunk: &mut dyn Iterator<Item = T>| {
            let mut current = chunk.next().expect("summary chunks are non-empty");
            for value in chunk {
                let better = if use_max {
                    value > current
                } else {
                    value < current
                };
                if better {
                    current = value;
                }
            }
            current
        };
        let block_count = values.len().div_ceil(SECOND_RANGE_BLOCK);
        let blocks: Vec<T> = (0..block_count)
            .map(|block| {
                let start = block * SECOND_RANGE_BLOCK;
                let end = (start + SECOND_RANGE_BLOCK).min(values.len());
                most_permissive(&mut (start..end).map(|position| values[position]))
            })
            .collect();
        let supers = blocks
            .chunks(SECOND_RANGE_SUPER)
            .map(|chunk| most_permissive(&mut chunk.iter().copied()))
            .collect();
        Self {
            values,
            blocks,
            supers,
            op,
        }
    }

    /// Call `emit` for every position in `[start, end)` whose right value
    /// satisfies `left op right`, in ascending position order.
    ///
    /// `emit` returns `false` to stop the query early (used by `keep="any"`
    /// and the `first`/`last` selection helper).
    /// The caller guarantees `start <= end <= values.len()`.
    #[inline]
    pub(crate) fn for_each_match(
        &self,
        left: T,
        start: usize,
        end: usize,
        mut emit: impl FnMut(usize) -> bool,
    ) {
        debug_assert!(start <= end && end <= self.values.len());
        if self.blocks.is_empty() || end - start < SECOND_RANGE_DIRECT_WIDTH {
            for position in start..end {
                if self.op.apply(&left, &self.values[position]) && !emit(position) {
                    return;
                }
            }
            return;
        }
        #[cfg(test)]
        SECOND_RANGE_SKIP_HITS.with(|hits| hits.set(hits.get() + 1));
        let mut block = start / SECOND_RANGE_BLOCK;
        let last_block = (end - 1) / SECOND_RANGE_BLOCK;
        while block <= last_block {
            // ELI5: at the front door of a crate, read the crate's note. If
            // even its best card loses, walk past all 32 boxes at once. A
            // crate the window only partly covers is checked the same way:
            // its note covers more cards than we need, never fewer.
            if block.is_multiple_of(SECOND_RANGE_SUPER)
                && !self
                    .op
                    .apply(&left, &self.supers[block / SECOND_RANGE_SUPER])
            {
                #[cfg(test)]
                SECOND_RANGE_SUPER_SKIPS.with(|skips| skips.set(skips.get() + 1));
                block += SECOND_RANGE_SUPER;
                continue;
            }
            if self.op.apply(&left, &self.blocks[block]) {
                let block_start = (block * SECOND_RANGE_BLOCK).max(start);
                let block_end = ((block + 1) * SECOND_RANGE_BLOCK).min(end);
                for position in block_start..block_end {
                    if self.op.apply(&left, &self.values[position]) && !emit(position) {
                        return;
                    }
                }
            }
            block += 1;
        }
    }

    /// Rows of `[start, end)` inside blocks the walk would open.
    ///
    /// This is the work [`Self::for_each_match`] does after skipping, without
    /// reading any row values. A window too narrow for summaries reports its
    /// full width.
    pub(crate) fn surviving_rows(&self, left: T, start: usize, end: usize) -> usize {
        debug_assert!(start <= end && end <= self.values.len());
        if self.blocks.is_empty() || end - start < SECOND_RANGE_DIRECT_WIDTH {
            return end - start;
        }
        let mut rows = 0;
        let mut block = start / SECOND_RANGE_BLOCK;
        let last_block = (end - 1) / SECOND_RANGE_BLOCK;
        while block <= last_block {
            if block.is_multiple_of(SECOND_RANGE_SUPER)
                && !self
                    .op
                    .apply(&left, &self.supers[block / SECOND_RANGE_SUPER])
            {
                block += SECOND_RANGE_SUPER;
                continue;
            }
            if self.op.apply(&left, &self.blocks[block]) {
                rows += ((block + 1) * SECOND_RANGE_BLOCK).min(end)
                    - (block * SECOND_RANGE_BLOCK).max(start);
            }
            block += 1;
        }
        rows
    }

    /// Count the positions in `[start, end)` that `for_each_match` would
    /// report, without a per-match callback.
    ///
    /// ELI5: when we only need how many cards win, tallying a whole open box
    /// in one sweep is faster than handing over the cards one at a time; the
    /// tally loop has no early exit, so the compiler can vectorize it.
    #[inline]
    pub(crate) fn count_matches(&self, left: T, start: usize, end: usize) -> usize {
        debug_assert!(start <= end && end <= self.values.len());
        let count_range = |from: usize, to: usize| {
            (from..to)
                .filter(|&position| self.op.apply(&left, &self.values[position]))
                .count()
        };
        if self.blocks.is_empty() || end - start < SECOND_RANGE_DIRECT_WIDTH {
            return count_range(start, end);
        }
        #[cfg(test)]
        SECOND_RANGE_SKIP_HITS.with(|hits| hits.set(hits.get() + 1));
        let mut count = 0;
        let mut block = start / SECOND_RANGE_BLOCK;
        let last_block = (end - 1) / SECOND_RANGE_BLOCK;
        while block <= last_block {
            if block.is_multiple_of(SECOND_RANGE_SUPER)
                && !self
                    .op
                    .apply(&left, &self.supers[block / SECOND_RANGE_SUPER])
            {
                block += SECOND_RANGE_SUPER;
                continue;
            }
            if self.op.apply(&left, &self.blocks[block]) {
                count += count_range(
                    (block * SECOND_RANGE_BLOCK).max(start),
                    ((block + 1) * SECOND_RANGE_BLOCK).min(end),
                );
            }
            block += 1;
        }
        count
    }
}

/// Report whether a [`SecondRangeSkipScan`] should scan the narrowed windows.
///
/// Building the summaries reads every right value once, so it pays off only
/// when the windows still to be scanned are much wider in total than the
/// right layout. Below twice its length, the direct scan costs at most about
/// one extra pass over the right values, which is what the build itself
/// costs; on AIList exons x chainXenTro3Link (53M of window over 51M right
/// rows) the build was pure overhead.
///
/// ELI5: labeling every box costs one look at every card. That is worth it
/// only when we were going to look at more cards than that anyway.
pub(crate) fn second_range_skip_pays_off(
    right_len: usize,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
) -> bool {
    // ELI5: the windows were validated before narrowing; saturating keeps
    // an unexpected inverted window at zero width, matching the empty
    // `start..end` loop that would scan it.
    skip_pays_off(
        right_len,
        starts
            .iter()
            .zip(ends.iter())
            .filter(|(&start, &end)| start >= 0 && end >= 0)
            .map(|(&start, &end)| (end as usize).saturating_sub(start as usize)),
    )
}

/// [`second_range_skip_pays_off`] over plain window widths.
///
/// Stops reading widths as soon as the total passes twice `right_len`.
pub(crate) fn skip_pays_off(right_len: usize, widths: impl Iterator<Item = usize>) -> bool {
    let mut total_width = 0_usize;
    for width in widths {
        total_width = total_width.saturating_add(width);
        if total_width / 2 > right_len {
            return true;
        }
    }
    false
}

/// Windows sampled by [`WindowResidualScan::plan`] to estimate how much the
/// summaries can skip.
const DENSITY_SAMPLE_ROWS: usize = 256;

/// One typed residual promoted to a [`SecondRangeSkipScan`], with its left
/// values.
enum TypedSkipScan<'a> {
    I64(ArrayView1<'a, i64>, SecondRangeSkipScan<'a, i64>),
    I32(ArrayView1<'a, i32>, SecondRangeSkipScan<'a, i32>),
    I16(ArrayView1<'a, i16>, SecondRangeSkipScan<'a, i16>),
    I8(ArrayView1<'a, i8>, SecondRangeSkipScan<'a, i8>),
    U64(ArrayView1<'a, u64>, SecondRangeSkipScan<'a, u64>),
    U32(ArrayView1<'a, u32>, SecondRangeSkipScan<'a, u32>),
    U16(ArrayView1<'a, u16>, SecondRangeSkipScan<'a, u16>),
    U8(ArrayView1<'a, u8>, SecondRangeSkipScan<'a, u8>),
    F64(ArrayView1<'a, f64>, SecondRangeSkipScan<'a, f64>),
    F32(ArrayView1<'a, f32>, SecondRangeSkipScan<'a, f32>),
}

/// Run `$body` with the typed left values and scanner bound to `$left` and
/// `$scan`.
macro_rules! with_typed_scan {
    ($typed:expr, |$left:ident, $scan:ident| $body:expr) => {
        match $typed {
            TypedSkipScan::I64($left, $scan) => $body,
            TypedSkipScan::I32($left, $scan) => $body,
            TypedSkipScan::I16($left, $scan) => $body,
            TypedSkipScan::I8($left, $scan) => $body,
            TypedSkipScan::U64($left, $scan) => $body,
            TypedSkipScan::U32($left, $scan) => $body,
            TypedSkipScan::U16($left, $scan) => $body,
            TypedSkipScan::U8($left, $scan) => $body,
            TypedSkipScan::F64($left, $scan) => $body,
            TypedSkipScan::F32($left, $scan) => $body,
        }
    };
}

/// Residual predicates of a window join, with one range residual answered by
/// a [`SecondRangeSkipScan`].
///
/// The default range path keeps a non-monotonic second range as a residual
/// behind its cumulative envelope (and the single-range path keeps every
/// later range as a residual). The window loops then check every candidate
/// in every window. When one early right row has an extreme value, the
/// envelope barely narrows anything: on AIList ex-anno x chainRn4 the
/// windows hold 15.1 billion candidates for 2.8 million matches. Scanning
/// the windows with the residual's block summaries visits only blocks that
/// can hold a match, then checks the remaining residuals on those
/// candidates. Matches are still reported in ascending position order, so
/// `keep` selection and aggregation update order are unchanged.
///
/// ELI5: one of the rules is "the card must be at least this big". Instead
/// of reading every card in the aisle against every rule, use the sticky
/// notes from [`SecondRangeSkipScan`] to skip boxes that fail the size rule,
/// and read the other rules only on cards that pass it.
pub(crate) struct WindowResidualScan<'a> {
    scan: TypedSkipScan<'a>,
    rest: Vec<PredicateView<'a>>,
    rest_metadata: Option<Vec<NullMetadataView<'a>>>,
}

impl<'a> WindowResidualScan<'a> {
    /// Promote the first eligible range residual when the windows are wide
    /// enough to repay the summary build (see [`skip_pays_off`]) and the
    /// summaries can actually skip most of them.
    ///
    /// Wide windows alone are not enough. On a dense join such as Arun's
    /// dominance counts nearly every candidate matches, nothing can be
    /// skipped, and the per-candidate scan is about 15% slower than the
    /// caller's vectorized residual mask. So after building the summaries,
    /// up to [`DENSITY_SAMPLE_ROWS`] evenly spaced windows are walked at
    /// block level only; when more than half of their candidates sit in
    /// blocks the scan would open, the plan is dropped.
    ///
    /// A residual is eligible when its operator is `<`, `<=`, `>` or `>=`
    /// and it carries no null masks: a masked row's stored value is
    /// arbitrary, so value summaries could skip or admit it wrongly. Returns
    /// `None` when no residual is eligible or the windows are narrow; the
    /// caller then keeps its existing window loop.
    pub(crate) fn plan(
        predicates: &'a [Predicate<'_>],
        metadata: Option<&'a [NullMetadata<'_>]>,
        windows: &SingleJoinResult,
    ) -> Option<Self> {
        let unmasked = |position: usize| {
            metadata.is_none_or(|metadata| {
                metadata[position].left.is_none() && metadata[position].right.is_none()
            })
        };
        let chosen = predicates
            .iter()
            .enumerate()
            .position(|(position, predicate)| {
                !matches!(predicate_op(predicate), CompareOp::Eq | CompareOp::Ne)
                    && unmasked(position)
            })?;
        let widths = windows
            .starts
            .iter()
            .zip(&windows.ends)
            .map(|(&start, &end)| end.saturating_sub(start));
        if !skip_pays_off(predicates[chosen].right_len(), widths) {
            return None;
        }
        macro_rules! typed {
            ($variant:ident, $left:expr, $right:expr, $op:expr) => {
                TypedSkipScan::$variant($left, SecondRangeSkipScan::new($right, $op))
            };
        }
        let scan = match predicates[chosen].view() {
            PredicateView::I64(left, right, op) => typed!(I64, left, right, op),
            PredicateView::I32(left, right, op) => typed!(I32, left, right, op),
            PredicateView::I16(left, right, op) => typed!(I16, left, right, op),
            PredicateView::I8(left, right, op) => typed!(I8, left, right, op),
            PredicateView::U64(left, right, op) => typed!(U64, left, right, op),
            PredicateView::U32(left, right, op) => typed!(U32, left, right, op),
            PredicateView::U16(left, right, op) => typed!(U16, left, right, op),
            PredicateView::U8(left, right, op) => typed!(U8, left, right, op),
            PredicateView::F64(left, right, op) => typed!(F64, left, right, op),
            PredicateView::F32(left, right, op) => typed!(F32, left, right, op),
        };
        // ELI5: before trusting the sticky notes, peek at a few aisles. If
        // most boxes in them would be opened anyway, the notes are not
        // saving any work.
        let rows = windows.starts.len();
        let step = rows.div_ceil(DENSITY_SAMPLE_ROWS).max(1);
        let (mut sampled, mut surviving) = (0_usize, 0_usize);
        for row in (0..rows).step_by(step) {
            let (start, end) = (windows.starts[row], windows.ends[row]);
            if start >= end {
                continue;
            }
            let left_position = windows.left_positions[row];
            sampled += end - start;
            surviving += with_typed_scan!(&scan, |left, scan| {
                scan.surviving_rows(left[left_position], start, end)
            });
        }
        if surviving.saturating_mul(2) > sampled {
            #[cfg(test)]
            WINDOW_RESIDUAL_SKIP_DENSE.with(|dense| dense.set(dense.get() + 1));
            return None;
        }
        #[cfg(test)]
        WINDOW_RESIDUAL_SKIP_PLANS.with(|plans| plans.set(plans.get() + 1));
        let rest = predicates
            .iter()
            .enumerate()
            .filter(|&(position, _)| position != chosen)
            .map(|(_, predicate)| predicate.view())
            .collect();
        let rest_metadata = metadata.map(|metadata| {
            metadata
                .iter()
                .enumerate()
                .filter(|&(position, _)| position != chosen)
                .map(|(_, metadata)| metadata.view())
                .collect()
        });
        Some(Self {
            scan,
            rest,
            rest_metadata,
        })
    }

    /// Call `emit` for every position in `[start, end)` that passes every
    /// residual for `left_position`, in ascending order. `emit` returns
    /// `false` to stop early.
    #[inline]
    pub(crate) fn for_each_match(
        &self,
        left_position: usize,
        start: usize,
        end: usize,
        mut emit: impl FnMut(usize) -> bool,
    ) {
        let rest = &self.rest;
        let rest_metadata = self.rest_metadata.as_deref();
        with_typed_scan!(&self.scan, |left, scan| {
            scan.for_each_match(left[left_position], start, end, |right_position| {
                if rest.is_empty()
                    || predicates_match_dispatch(rest, rest_metadata, left_position, right_position)
                {
                    emit(right_position)
                } else {
                    true
                }
            })
        })
    }

    /// Count the positions [`Self::for_each_match`] would report.
    #[inline]
    pub(crate) fn count_matches(&self, left_position: usize, start: usize, end: usize) -> usize {
        if self.rest.is_empty() {
            return with_typed_scan!(&self.scan, |left, scan| {
                scan.count_matches(left[left_position], start, end)
            });
        }
        let mut count = 0_usize;
        self.for_each_match(left_position, start, end, |_| {
            count += 1;
            true
        });
        count
    }
}

/// The comparison operator of a parsed predicate.
fn predicate_op(predicate: &Predicate<'_>) -> CompareOp {
    match predicate {
        Predicate::I64(_, _, op)
        | Predicate::I32(_, _, op)
        | Predicate::I16(_, _, op)
        | Predicate::I8(_, _, op)
        | Predicate::U64(_, _, op)
        | Predicate::U32(_, _, op)
        | Predicate::U16(_, _, op)
        | Predicate::U8(_, _, op)
        | Predicate::F64(_, _, op)
        | Predicate::F32(_, _, op) => *op,
    }
}
