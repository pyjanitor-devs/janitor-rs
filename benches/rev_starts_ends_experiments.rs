//! Algorithm-level experiments for reverse `_starts_ends`.
//!
//! These deliberately use the same `[start, end)` semantics as the production
//! kernels, but keep the alternatives local until their runtime/memory tradeoffs
//! justify a production rewrite.

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

struct Input<'a> {
    values: &'a [i64],
    starts: &'a [usize],
    ends: &'a [usize],
    index: &'a [i64],
}

fn hash_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut map = HashMap::with_capacity(input.index.len());
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            *map.entry(input.index[item]).or_insert(0) += input.values[row];
        }
    }
    let mut out = map.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut totals = vec![0_i64; input.index.len()];
    let mut seen = vec![false; input.index.len()];
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            seen[item] = true;
            totals[item] += input.values[row];
        }
    }
    let mut out = input
        .index
        .iter()
        .enumerate()
        .filter(|(item, _)| seen[*item])
        .map(|(item, &label)| (label, totals[item]))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn ordinal_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut slots = HashMap::with_capacity(input.index.len());
    let mut ordinal_to_slot = Vec::with_capacity(input.index.len());
    let mut labels = Vec::new();
    for &label in input.index {
        let slot = match slots.get(&label) {
            Some(&slot) => slot,
            None => {
                let slot = labels.len();
                slots.insert(label, slot);
                labels.push(label);
                slot
            }
        };
        ordinal_to_slot.push(slot);
    }
    let mut totals = vec![0_i64; labels.len()];
    let mut seen = vec![false; labels.len()];
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            let slot = ordinal_to_slot[item];
            seen[slot] = true;
            totals[slot] += input.values[row];
        }
    }
    let mut out = labels
        .into_iter()
        .enumerate()
        .filter(|(slot, _)| seen[*slot])
        .map(|(slot, label)| (label, totals[slot]))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_size(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut map = HashMap::with_capacity(input.index.len());
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            *map.entry(input.index[item]).or_insert(0) += 1;
        }
    }
    let mut out = map.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn sweep_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    let mut active_events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += input.values[row];
        events[input.ends[row]] -= input.values[row];
        active_events[input.starts[row]] += 1;
        active_events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut active = 0_i64;
    let mut out = Vec::new();
    for item in 0..input.index.len() {
        running += events[item];
        active += active_events[item];
        if active > 0 {
            out.push((input.index[item], running));
        }
    }
    out.sort_unstable();
    out
}

fn sweep_sum_compact(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    let mut active_events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += input.values[row];
        events[input.ends[row]] -= input.values[row];
        active_events[input.starts[row]] += 1;
        active_events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut active = 0_i64;
    let mut compacted = HashMap::with_capacity(input.index.len());
    for item in 0..input.index.len() {
        running += events[item];
        active += active_events[item];
        if active > 0 {
            *compacted.entry(input.index[item]).or_insert(0) += running;
        }
    }
    let mut out = compacted.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn sweep_size(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += 1;
        events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut out = Vec::new();
    for item in 0..input.index.len() {
        running += events[item];
        if running != 0 {
            out.push((input.index[item], running));
        }
    }
    out.sort_unstable();
    out
}

fn sweep_size_compact(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += 1;
        events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut compacted = HashMap::with_capacity(input.index.len());
    for item in 0..input.index.len() {
        running += events[item];
        if running != 0 {
            *compacted.entry(input.index[item]).or_insert(0) += running;
        }
    }
    let mut out = compacted.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn allocations<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    TOTAL.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    black_box(f());
    (TOTAL.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed))
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_starts_ends_experiments");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    for (name, rows, right_len, width, duplicate) in [
        ("tiny_narrow_unique", 32, 32, 4, false),
        ("tiny_narrow_unique_reversed", 32, 32, 4, false),
        ("large_broad_unique", 1_000, 10_000, 8_000, false),
        ("large_broad_unique_reversed", 1_000, 10_000, 8_000, false),
        ("large_narrow_unique", 1_000, 10_000, 8, false),
        ("large_narrow_unique_reversed", 1_000, 10_000, 8, false),
        ("very_large_broad_unique", 1_000, 100_000, 80_000, false),
        (
            "very_large_broad_unique_reversed",
            1_000,
            100_000,
            80_000,
            false,
        ),
        (
            "super_large_narrow_duplicate",
            1_000_000,
            1_000_000,
            8,
            true,
        ),
    ] {
        let values = (0..rows)
            .map(|row| (row % 7 + 1) as i64)
            .collect::<Vec<_>>();
        let starts = (0..rows)
            .map(|row| row % (right_len - width + 1))
            .collect::<Vec<_>>();
        let ends = starts
            .iter()
            .map(|&start| start + width)
            .collect::<Vec<_>>();
        let reversed = name.ends_with("reversed");
        let index = (0..right_len)
            .map(|item| {
                if duplicate {
                    (item % 32) as i64
                } else if reversed {
                    (right_len - 1 - item) as i64
                } else {
                    item as i64
                }
            })
            .collect::<Vec<_>>();
        let input = Input {
            values: &values,
            starts: &starts,
            ends: &ends,
            index: &index,
        };

        let hash = hash_sum(&input);
        if !duplicate {
            assert_eq!(hash, dense_sum(&input));
        }
        assert_eq!(hash, sweep_sum_compact(&input));
        assert_eq!(hash_size(&input), sweep_size_compact(&input));
        eprintln!(
            "{name}: hash_sum {:?}, dense_sum {:?}, ordinal_sum {:?}, sweep_sum {:?}, sweep_sum_compact {:?}, hash_size {:?}, sweep_size {:?}, sweep_size_compact {:?}",
            allocations(|| hash_sum(&input)),
            if duplicate { (0, 0) } else { allocations(|| dense_sum(&input)) },
            allocations(|| ordinal_sum(&input)),
            allocations(|| sweep_sum(&input)),
            allocations(|| sweep_sum_compact(&input)),
            allocations(|| hash_size(&input)),
            allocations(|| sweep_size(&input)),
            allocations(|| sweep_size_compact(&input)),
        );
        group.bench_function(format!("sum/hash/{name}"), |b| {
            b.iter(|| hash_sum(black_box(&input)))
        });
        group.bench_function(format!("sum/ordinal/{name}"), |b| {
            b.iter(|| ordinal_sum(black_box(&input)))
        });
        group.bench_function(format!("sum/sweep/{name}"), |b| {
            b.iter(|| sweep_sum(black_box(&input)))
        });
        group.bench_function(format!("sum/sweep_compact/{name}"), |b| {
            b.iter(|| sweep_sum_compact(black_box(&input)))
        });
        group.bench_function(format!("size/hash/{name}"), |b| {
            b.iter(|| hash_size(black_box(&input)))
        });
        group.bench_function(format!("size/sweep/{name}"), |b| {
            b.iter(|| sweep_size(black_box(&input)))
        });
        group.bench_function(format!("size/sweep_compact/{name}"), |b| {
            b.iter(|| sweep_size_compact(black_box(&input)))
        });
        if !duplicate {
            group.bench_function(format!("sum/dense/{name}"), |b| {
                b.iter(|| dense_sum(black_box(&input)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
