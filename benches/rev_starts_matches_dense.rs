//! HashMap versus dense-ordinal benchmark for reverse starts-matches sum.

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

fn hash_map_sum(arr: &[i64], starts: &[usize], index: &[i64], matches: &[i8]) -> Vec<(i64, i64)> {
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
    let mut output = map.into_iter().collect::<Vec<_>>();
    output.sort_unstable_by_key(|(label, _)| *label);
    output
}

fn dense_sum(arr: &[i64], starts: &[usize], index: &[i64], matches: &[i8]) -> Vec<(i64, i64)> {
    let mut totals = vec![0_i64; index.len()];
    let mut seen = vec![false; index.len()];
    let mut labels = Vec::new();
    let mut tape_offset = 0;
    for (row, &start) in starts.iter().enumerate() {
        for item in start..index.len() {
            if matches[tape_offset] != 0 {
                if !seen[item] {
                    seen[item] = true;
                    labels.push(index[item]);
                }
                totals[item] += arr[row];
            }
            tape_offset += 1;
        }
    }
    labels
        .into_iter()
        .map(|label| (label, totals[label as usize]))
        .collect()
}

fn allocations<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    TOTAL.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    std::mem::forget(f());
    (TOTAL.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed))
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_starts_matches_dense_sum");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));
    for (name, rows, right_len, start, duplicate, sparse) in [
        (
            "tiny_unique_dense",
            32_usize,
            32_usize,
            16_usize,
            false,
            false,
        ),
        ("large_unique_dense", 1_000, 10_000, 5_000, false, false),
        (
            "very_large_unique_sparse",
            10_000,
            100_000,
            99_990,
            false,
            true,
        ),
        (
            "super_large_duplicate_dense",
            1_000_000,
            1_000_000,
            999_992,
            true,
            false,
        ),
    ] {
        let arr = vec![1_i64; rows];
        let starts = vec![start; rows];
        let index = (0..right_len)
            .map(|item| {
                if duplicate {
                    (item % 16) as i64
                } else {
                    item as i64
                }
            })
            .collect::<Vec<_>>();
        let row_width = right_len - start;
        let matches = (0..rows * row_width)
            .map(|item| if sparse && item % 100 != 0 { 0 } else { 1 })
            .collect::<Vec<_>>();
        let old = hash_map_sum(&arr, &starts, &index, &matches);
        if !duplicate {
            let mut new = dense_sum(&arr, &starts, &index, &matches);
            new.sort_unstable_by_key(|(label, _)| *label);
            assert_eq!(old, new);
        }
        eprintln!(
            "{name}: old memory {:?}, dense memory {:?}",
            allocations(|| hash_map_sum(&arr, &starts, &index, &matches)),
            if duplicate {
                (0, 0)
            } else {
                allocations(|| dense_sum(&arr, &starts, &index, &matches))
            }
        );
        group.bench_function(format!("hash_map/{name}"), |b| {
            b.iter(|| {
                hash_map_sum(
                    black_box(&arr),
                    black_box(&starts),
                    black_box(&index),
                    black_box(&matches),
                )
            })
        });
        if !duplicate {
            group.bench_function(format!("dense/{name}"), |b| {
                b.iter(|| {
                    dense_sum(
                        black_box(&arr),
                        black_box(&starts),
                        black_box(&index),
                        black_box(&matches),
                    )
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
