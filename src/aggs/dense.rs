use std::collections::HashMap;

use numpy::ndarray::Array1;

/// Above this many slots, growing the dense array further switches to a
/// sorted `HashMap` fallback instead of continuing to resize -- see issue
/// #70. Bounds the dense path's worst-case allocation to roughly
/// `DENSE_CAPACITY_CAP * size_of::<T>()` bytes (a few tens of MB even for
/// the largest accumulator shape in this crate, `(f64, f64)`) regardless
/// of how far out a touched row position lands, while staying two orders
/// of magnitude above every `length` this crate's own benchmarks and
/// tests exercise (the `n=100_000` fixtures top out at `length ~ 10_000`),
/// so the fast path is untouched for realistic join sizes.
const DENSE_CAPACITY_CAP: usize = 1 << 20;

/// The two storage strategies `DenseSlots` switches between. Kept private
/// -- callers only ever see `DenseSlots`'s `touch`/`to_arrays` surface,
/// never which strategy backs a particular call.
enum Storage<T> {
    /// Slot `i` holds row `i`'s value directly -- see `DenseSlots`'s own
    /// doc comment for the mailbox-rack ELI5. Cheap and cache-friendly as
    /// long as the touched-key domain stays under `DENSE_CAPACITY_CAP`.
    Dense { values: Vec<T>, seen: Vec<bool> },
    /// The pre-#23 `HashMap<i64, T>` strategy, used once the domain would
    /// make a dense `Vec` too large relative to any plausible touched-key
    /// count -- e.g. a single match at row position 10,000,000. Sorted at
    /// emission time (`to_arrays`) to keep the same ascending-order
    /// contract the dense path gives for free.
    Sparse(HashMap<i64, T>),
}

/// A row-position-indexed accumulator for reverse aggregations, replacing a
/// `HashMap<i64, T>` keyed by conditional-join row positions.
///
/// ELI5: the old code wrote each answer on an envelope labeled with its
/// apartment number and tossed it into a shuffled mail bin, then read the
/// bin back out in whatever order the envelopes happened to land in --
/// that's how `HashMap` iteration works. So instead we use a mailbox rack
/// -- slot `i` for row `i` -- and read the rack back out in slot order. No
/// hashing, no shuffling, and the same output every time for the same
/// input. See issue #23.
///
/// Correctness note (see issue #69): the `length` passed to `new` is only
/// a *capacity hint*, exactly like the old code's `HashMap::with_capacity
/// (length)` -- it is never a promise that every key touched will be
/// `< length`. pyjanitor's equi-join caller, for one, passes
/// `right_index.size` (the match *count*) as `length`, while `right_index`
/// itself holds actual right-dataframe row positions that can run far
/// past that count on a sparse join. `touch` grows the backing storage on
/// demand instead of trusting the hint, so an out-of-range key resizes
/// rather than panics.
///
/// Memory-bound note (see issue #70): growing the mailbox rack to fit a
/// single far-out key is *also* wrong -- one match at row 10,000,000 must
/// not allocate ten million mailboxes for it. `DenseSlots` is an adaptive
/// hybrid, not a pure `Vec`: it stays on the mailbox rack (`Storage::
/// Dense`) as long as that's cheap, and falls back to a `HashMap`
/// (`Storage::Sparse`, sorted before it's read back out) once a key would
/// push the rack past `DENSE_CAPACITY_CAP`. Callers never see the switch
/// -- `touch` and `to_arrays` behave identically either way.
pub(crate) struct DenseSlots<T> {
    storage: Storage<T>,
}

impl<T: Copy + Default> DenseSlots<T> {
    /// `length` is a capacity hint (sized to avoid reallocation in the
    /// common case where every key really does land `< length`), not a
    /// bound -- `touch` grows past it safely if a key doesn't fit, and
    /// falls back to `Storage::Sparse` if `length` itself is already
    /// past `DENSE_CAPACITY_CAP` (defense in depth: a caller-supplied hint
    /// that large is itself a signal the hint isn't trustworthy, per
    /// issue #69's finding that `length` is sometimes far smaller than the
    /// true domain -- there's no reason to assume it's never far larger).
    pub(crate) fn new(length: usize) -> Self {
        if length > DENSE_CAPACITY_CAP {
            return DenseSlots {
                storage: Storage::Sparse(HashMap::new()),
            };
        }
        DenseSlots {
            storage: Storage::Dense {
                values: vec![T::default(); length],
                seen: vec![false; length],
            },
        }
    }

    /// Mirrors `HashMap::entry(key).or_insert(default)`: the first touch of
    /// a slot seeds it with `default` and marks it seen (so it survives
    /// into the final output even if the caller never actually updates it
    /// past this point -- the old `HashMap` code inserted its entry before
    /// checking any row filter, and callers here preserve that by calling
    /// `touch` unconditionally too); later touches just hand back the
    /// slot's current value.
    ///
    /// `key` is the *raw*, signed row position -- not yet cast or bounds
    /// checked -- so this can reject a negative key (returning `None`, the
    /// same "skip this row" signal `checked_index` gives its callers)
    /// instead of letting `as usize` wrap it into a huge positive index.
    /// A non-negative key beyond current capacity grows the backing
    /// storage to fit, unless that growth would cross
    /// `DENSE_CAPACITY_CAP`, in which case this converts to `Storage::
    /// Sparse` first (copying over whatever was already touched) and
    /// inserts into that instead -- see the struct-level doc comment.
    #[inline]
    pub(crate) fn touch(&mut self, key: i64, default: T) -> Option<&mut T> {
        let key = usize::try_from(key).ok()?;
        // A key that would grow the dense arrays past the cap converts
        // storage first, in its own borrow, so the match below always
        // sees the storage it needs to touch -- can't reassign
        // `self.storage` while still holding a `&mut` into its old value.
        if let Storage::Dense { values, seen } = &self.storage {
            if key >= values.len() && key + 1 > DENSE_CAPACITY_CAP {
                self.storage = Storage::Sparse(Self::dense_to_sparse(values, seen));
            }
        }
        match &mut self.storage {
            Storage::Dense { values, seen } => {
                if key >= values.len() {
                    let new_len = key + 1;
                    values.resize(new_len, T::default());
                    seen.resize(new_len, false);
                }
                if !seen[key] {
                    seen[key] = true;
                    values[key] = default;
                }
                Some(&mut values[key])
            }
            Storage::Sparse(map) => Some(map.entry(key as i64).or_insert(default)),
        }
    }

    /// One-time conversion of whatever's already been touched in the dense
    /// arrays into a fresh `HashMap`, used by `touch` the moment a key
    /// would grow the dense arrays past `DENSE_CAPACITY_CAP`. Bounded by
    /// the *current* (pre-growth) dense length, not by the triggering key,
    /// so this is cheap even when the triggering key is enormous.
    fn dense_to_sparse(values: &[T], seen: &[bool]) -> HashMap<i64, T> {
        let touched = seen.iter().filter(|&&s| s).count();
        let mut map = HashMap::with_capacity(touched + 1);
        for (key, (&s, &value)) in seen.iter().zip(values.iter()).enumerate() {
            if s {
                map.insert(key as i64, value);
            }
        }
        map
    }

    /// Ascending row-position order over slots touched at least once --
    /// the same *set* `HashMap::iter` would have produced, just
    /// deterministically ordered instead of shuffled. `Storage::Sparse`
    /// sorts its entries to match; `Storage::Dense` is already ascending
    /// by construction (slot `i` is row `i`).
    pub(crate) fn iter_touched(&self) -> Box<dyn Iterator<Item = (i64, &T)> + '_> {
        match &self.storage {
            Storage::Dense { values, seen } => Box::new(
                seen.iter()
                    .zip(values.iter())
                    .enumerate()
                    .filter_map(|(key, (&seen, value))| seen.then_some((key as i64, value))),
            ),
            Storage::Sparse(map) => {
                let mut pairs: Vec<(i64, &T)> =
                    map.iter().map(|(&key, value)| (key, value)).collect();
                pairs.sort_unstable_by_key(|(key, _)| *key);
                Box::new(pairs.into_iter())
            }
        }
    }

    /// Emit the `(indexers, result)` pair pyjanitor expects: ascending row
    /// positions paired with a value projected out of each touched slot.
    pub(crate) fn to_arrays<R>(
        &self,
        mut project: impl FnMut(&T) -> R,
    ) -> (Array1<i64>, Array1<R>) {
        let capacity_hint = match &self.storage {
            Storage::Dense { values, .. } => values.len(),
            Storage::Sparse(map) => map.len(),
        };
        let mut indexers = Vec::with_capacity(capacity_hint);
        let mut result = Vec::with_capacity(capacity_hint);
        for (key, value) in self.iter_touched() {
            indexers.push(key);
            result.push(project(value));
        }
        (Array1::from_vec(indexers), Array1::from_vec(result))
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseSlots, DENSE_CAPACITY_CAP};

    #[test]
    fn touch_seeds_default_once_and_keeps_later_updates() {
        let mut slots: DenseSlots<i64> = DenseSlots::new(3);
        *slots.touch(1, 0).unwrap() += 5;
        *slots.touch(1, 0).unwrap() += 2;
        assert_eq!(slots.iter_touched().collect::<Vec<_>>(), vec![(1, &7)]);
    }

    #[test]
    fn a_touch_that_never_updates_still_survives_to_output() {
        // Mirrors the old `HashMap::entry(...).or_insert(...)` call
        // happening before a row filter is checked: even a "no-op" touch
        // must leave the slot in the emitted output, at its default.
        let mut slots: DenseSlots<i64> = DenseSlots::new(2);
        slots.touch(0, -1);
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert_eq!(indexers.to_vec(), vec![0]);
        assert_eq!(result.to_vec(), vec![-1]);
    }

    #[test]
    fn untouched_slots_are_absent_and_order_is_ascending_by_row_position() {
        let mut slots: DenseSlots<i64> = DenseSlots::new(5);
        for key in [3, 0, 4] {
            *slots.touch(key, 0).unwrap() += 1;
        }
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert_eq!(indexers.to_vec(), vec![0, 3, 4]);
        assert_eq!(result.to_vec(), vec![1, 1, 1]);
    }

    #[test]
    fn projection_extracts_one_field_of_a_tuple_accumulator() {
        // Models the sum_rev float shape: (total, compensation) stored
        // together, only `total` emitted.
        let mut slots: DenseSlots<(f64, f64)> = DenseSlots::new(2);
        *slots.touch(0, (0., 0.)).unwrap() = (1.5, 0.25);
        let (indexers, result) = slots.to_arrays(|(total, _compensation)| *total);
        assert_eq!(indexers.to_vec(), vec![0]);
        assert_eq!(result.to_vec(), vec![1.5]);
    }

    // --- issue #69 regression coverage ---------------------------------

    #[test]
    fn a_key_far_beyond_the_capacity_hint_grows_instead_of_panicking() {
        // `length` passed to `new` is only a capacity hint (see the
        // struct-level doc comment) -- pyjanitor's equi-join caller passes
        // the match *count*, not the right dataframe's row count, so a
        // key well past the hint is an expected, common case, not a bug
        // in the caller.
        let mut slots: DenseSlots<i64> = DenseSlots::new(1);
        *slots.touch(10, 0).unwrap() += 7;
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert_eq!(indexers.to_vec(), vec![10]);
        assert_eq!(result.to_vec(), vec![7]);
    }

    #[test]
    fn a_negative_key_is_rejected_not_wrapped_into_a_huge_index() {
        // `-1 as usize` is `usize::MAX`, not a small out-of-range index --
        // resizing to fit that would abort the process, not just panic.
        // Negative keys must be rejected up front instead.
        let mut slots: DenseSlots<i64> = DenseSlots::new(4);
        assert!(slots.touch(-1, 0).is_none());
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert!(indexers.to_vec().is_empty());
        assert!(result.to_vec().is_empty());
    }

    #[test]
    fn growth_does_not_disturb_already_touched_low_keys() {
        let mut slots: DenseSlots<i64> = DenseSlots::new(2);
        *slots.touch(0, 0).unwrap() += 1;
        *slots.touch(1, 0).unwrap() += 2;
        *slots.touch(9, 0).unwrap() += 3;
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert_eq!(indexers.to_vec(), vec![0, 1, 9]);
        assert_eq!(result.to_vec(), vec![1, 2, 3]);
    }

    // --- issue #70 regression coverage ----------------------------------

    #[test]
    fn a_single_match_far_past_the_cap_does_not_allocate_the_whole_domain() {
        // The exact class of input issue #70 flagged: one match at an
        // enormous row position (modelling a match at row 10,000,000 of a
        // ten-million-row right dataframe). This must not allocate a
        // multi-million-element Vec -- Storage::Sparse handles it instead.
        // (This test asserts correctness, not the allocation count itself
        // -- see bench_dense_sparse_high_position_old_vs_new for the
        // allocation-size evidence.)
        let mut slots: DenseSlots<i64> = DenseSlots::new(1);
        *slots.touch(10_000_000, 0).unwrap() += 7;
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert_eq!(indexers.to_vec(), vec![10_000_000]);
        assert_eq!(result.to_vec(), vec![7]);
    }

    #[test]
    fn several_sparse_high_position_groups_are_emitted_sorted_ascending() {
        // Multiple groups scattered far apart, all past the cap -- the
        // Storage::Sparse path must still emit them in ascending row-
        // position order, matching the dense path's contract.
        let mut slots: DenseSlots<i64> = DenseSlots::new(1);
        for (key, value) in [(9_000_000_i64, 3_i64), (1_000_000, 5), (5_000_000, 7)] {
            *slots.touch(key, 0).unwrap() += value;
        }
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert_eq!(indexers.to_vec(), vec![1_000_000, 5_000_000, 9_000_000]);
        assert_eq!(result.to_vec(), vec![5, 7, 3]);
    }

    #[test]
    fn a_key_just_under_the_cap_stays_dense() {
        // Sanity check on the boundary itself: a key that lands exactly at
        // the cap must still take the (faster) dense path, not fall back
        // early.
        let mut slots: DenseSlots<i64> = DenseSlots::new(1);
        *slots.touch((DENSE_CAPACITY_CAP - 1) as i64, 0).unwrap() += 1;
        assert!(matches!(slots.storage, super::Storage::Dense { .. }));
    }

    #[test]
    fn a_key_that_would_cross_the_cap_converts_to_sparse_and_keeps_prior_touches() {
        let mut slots: DenseSlots<i64> = DenseSlots::new(1);
        *slots.touch(3, 0).unwrap() += 1;
        assert!(matches!(slots.storage, super::Storage::Dense { .. }));
        *slots.touch((DENSE_CAPACITY_CAP + 1) as i64, 0).unwrap() += 2;
        assert!(matches!(slots.storage, super::Storage::Sparse(_)));
        let (indexers, result) = slots.to_arrays(|v| *v);
        assert_eq!(indexers.to_vec(), vec![3, (DENSE_CAPACITY_CAP + 1) as i64]);
        assert_eq!(result.to_vec(), vec![1, 2]);
    }

    #[test]
    fn a_length_hint_already_past_the_cap_starts_sparse() {
        let slots: DenseSlots<i64> = DenseSlots::new(DENSE_CAPACITY_CAP + 1);
        assert!(matches!(slots.storage, super::Storage::Sparse(_)));
    }
}
