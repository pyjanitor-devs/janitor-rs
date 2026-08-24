use numpy::ndarray::Array1;

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
// ELI5 (`Vec`, not `Array1`, for these two fields): `Array1`/`ArrayView1`
// are the shipping containers this crate uses to hand data to and from
// Python/numpy -- built to match exactly what numpy expects on the other
// side, and used right at the door (function parameters in, `(indexers,
// result)` out). `values`/`seen` never leave this struct, though; they're
// purely internal scratch bookkeeping nobody outside ever sees, so there's
// no reason to carry `ndarray`'s extra shape/stride machinery for them --
// a plain `Vec` (indexed writes, nothing fancier) is the simpler box for
// work done entirely inside the house. Repacking into `Array1` at the end
// (`to_arrays`, via `Array1::from_vec`) is free, not a tradeoff: `ndarray`
// stores an owned array's data in a `Vec` internally, so that's a move,
// not a copy.
pub(crate) struct DenseSlots<T> {
    values: Vec<T>,
    seen: Vec<bool>,
}

impl<T: Copy + Default> DenseSlots<T> {
    /// `length` is a capacity hint (sized to avoid reallocation in the
    /// common case where every key really does land `< length`), not a
    /// bound -- `touch` grows past it safely if a key doesn't fit.
    pub(crate) fn new(length: usize) -> Self {
        DenseSlots {
            values: vec![T::default(); length],
            seen: vec![false; length],
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
    /// storage to fit rather than panicking; see the struct-level doc
    /// comment for why `length` in `new` can't be trusted as a hard bound.
    #[inline]
    pub(crate) fn touch(&mut self, key: i64, default: T) -> Option<&mut T> {
        let key = usize::try_from(key).ok()?;
        if key >= self.values.len() {
            let new_len = key + 1;
            self.values.resize(new_len, T::default());
            self.seen.resize(new_len, false);
        }
        if !self.seen[key] {
            self.seen[key] = true;
            self.values[key] = default;
        }
        Some(&mut self.values[key])
    }

    /// Ascending row-position order over slots touched at least once --
    /// the same *set* `HashMap::iter` would have produced, just
    /// deterministically ordered instead of shuffled.
    pub(crate) fn iter_touched(&self) -> impl Iterator<Item = (usize, &T)> {
        self.seen
            .iter()
            .zip(self.values.iter())
            .enumerate()
            .filter_map(|(key, (&seen, value))| seen.then_some((key, value)))
    }

    /// Emit the `(indexers, result)` pair pyjanitor expects: ascending row
    /// positions paired with a value projected out of each touched slot.
    pub(crate) fn to_arrays<R>(
        &self,
        mut project: impl FnMut(&T) -> R,
    ) -> (Array1<i64>, Array1<R>) {
        let mut indexers = Vec::with_capacity(self.values.len());
        let mut result = Vec::with_capacity(self.values.len());
        for (key, value) in self.iter_touched() {
            indexers.push(key as i64);
            result.push(project(value));
        }
        (Array1::from_vec(indexers), Array1::from_vec(result))
    }
}

#[cfg(test)]
mod tests {
    use super::DenseSlots;

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
}
