//! HashMap-per-candidate versus one-time ordinal-to-slot mapping.

use criterion::{criterion_group, criterion_main, Criterion};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static TOTAL: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

type Input<'a> = (&'a [i64], &'a [usize], &'a [i64], &'a [i8]);

fn hash_sum((arr, starts, index, matches): Input<'_>) -> Vec<i64> {
    let mut map = HashMap::with_capacity(index.len());
    let mut tape_offset = 0;
    for (row, &start) in starts.iter().enumerate() {
        for label in index.iter().skip(start) {
            if matches[tape_offset] != 0 {
                *map.entry(*label).or_insert(0) += arr[row];
            }
            tape_offset += 1;
        }
    }
    let mut output = map.into_values().collect::<Vec<_>>();
    output.sort_unstable();
    output
}

fn ordinal_slots_sum((arr, starts, index, matches): Input<'_>) -> Vec<i64> {
    let mut slots = HashMap::with_capacity(index.len());
    let mut ordinal_to_slot = Vec::with_capacity(index.len());
    let mut labels = Vec::new();
    for &label in index {
        let slot = if let Some(slot) = slots.get(&label) {
            *slot
        } else {
            let slot = labels.len();
            slots.insert(label, slot);
            labels.push(label);
            slot
        };
        ordinal_to_slot.push(slot);
    }
    let mut totals = vec![0_i64; labels.len()];
    let mut seen = vec![false; labels.len()];
    let mut tape_offset = 0;
    for (row, &start) in starts.iter().enumerate() {
        for item in start..index.len() {
            if matches[tape_offset] != 0 {
                let slot = ordinal_to_slot[item];
                seen[slot] = true;
                totals[slot] += arr[row];
            }
            tape_offset += 1;
        }
    }
    let mut output = totals
        .into_iter()
        .zip(seen)
        .filter_map(|(total, seen)| seen.then_some(total))
        .collect::<Vec<_>>();
    output.sort_unstable();
    output
}

fn allocations<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    TOTAL.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    std::mem::forget(f());
    (TOTAL.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed))
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_starts_matches_ordinal_slots_sum");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    for (name, rows, right_len, start, duplicate) in [
        ("tiny_unique", 32_usize, 32_usize, 16_usize, false),
        ("large_unique", 1_000, 10_000, 5_000, false),
        ("very_large_duplicate", 10_000, 100_000, 99_990, true),
        ("super_large_duplicate", 1_000_000, 1_000_000, 999_992, true),
    ] {
        let arr = (0..rows)
            .map(|row| (row % 7 + 1) as i64)
            .collect::<Vec<_>>();
        let starts = vec![start; rows];
        let index = (0..right_len)
            .map(|item| {
                if duplicate {
                    (item % 16) as i64
                } else {
                    (item as i64) * 10 + 7
                }
            })
            .collect::<Vec<_>>();
        let row_width = right_len - start;
        let matches = vec![1_i8; rows * row_width];
        let input = (
            arr.as_slice(),
            starts.as_slice(),
            index.as_slice(),
            matches.as_slice(),
        );
        assert_eq!(hash_sum(input), ordinal_slots_sum(input));
        eprintln!(
            "{name}: hash memory {:?}, ordinal-slot memory {:?}",
            allocations(|| hash_sum(input)),
            allocations(|| ordinal_slots_sum(input))
        );
        group.bench_function(format!("hash_map/{name}"), |b| {
            b.iter(|| hash_sum(black_box(input)))
        });
        group.bench_function(format!("ordinal_slots/{name}"), |b| {
            b.iter(|| ordinal_slots_sum(black_box(input)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
