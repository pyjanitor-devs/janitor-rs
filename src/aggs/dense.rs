use numpy::ndarray::Array1;

/// A row-position-indexed accumulator for reverse aggregations, replacing a
/// `HashMap<i64, T>` keyed by conditional-join row positions.
///
/// ELI5: the old code wrote each answer on an envelope labeled with its
/// apartment number and tossed it into a shuffled mail bin, then read the
/// bin back out in whatever order the envelopes happened to land in that's
/// how `HashMap` iteration works. Apartment numbers here are never
/// arbitrary, though: they're dense row positions in `0..length`, hedge to
/// hedge. So instead we use a mailbox rack -- slot `i` for row `i` -- and
/// read the rack back out in slot order. No hashing, no shuffling, and the
/// same output every time for the same input. See issue #23.
pub(crate) struct DenseSlots<T> {
    values: Vec<T>,
    seen: Vec<bool>,
}

impl<T: Copy> DenseSlots<T> {
    pub(crate) fn new(length: usize) -> Self
    where
        T: Default,
    {
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
    #[inline]
    pub(crate) fn touch(&mut self, key: usize, default: T) -> &mut T {
        if !self.seen[key] {
            self.seen[key] = true;
            self.values[key] = default;
        }
        &mut self.values[key]
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
        *slots.touch(1, 0) += 5;
        *slots.touch(1, 0) += 2;
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
            *slots.touch(key, 0) += 1;
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
        *slots.touch(0, (0., 0.)) = (1.5, 0.25);
        let (indexers, result) = slots.to_arrays(|(total, _compensation)| *total);
        assert_eq!(indexers.to_vec(), vec![0]);
        assert_eq!(result.to_vec(), vec![1.5]);
    }
}
