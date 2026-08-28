//! HashMap versus dense-ordinal benchmarks for all starts-matches aggregations.

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
    let mut n = 0;
    for (row, &start) in starts.iter().enumerate() {
        for label in index.iter().skip(start) {
            if matches[n] != 0 {
                *map.entry(*label).or_insert(0) += arr[row];
            }
            n += 1;
        }
    }
    let mut out = map.into_values().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_sum((arr, starts, index, matches): Input<'_>) -> Vec<i64> {
    let mut values = vec![0_i64; index.len()];
    let mut seen = vec![false; index.len()];
    let mut labels = Vec::new();
    let mut n = 0;
    for (row, &start) in starts.iter().enumerate() {
        for item in start..index.len() {
            if matches[n] != 0 {
                if !seen[item] {
                    seen[item] = true;
                    labels.push(item);
                }
                values[item] += arr[row];
            }
            n += 1;
        }
    }
    let mut out = labels
        .into_iter()
        .map(|item| values[item])
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_size((_, starts, index, matches): Input<'_>) -> Vec<i64> {
    let mut map = HashMap::with_capacity(index.len());
    let mut n = 0;
    for &start in starts {
        for label in index.iter().skip(start) {
            if matches[n] != 0 {
                *map.entry(*label).or_insert(0) += 1;
            }
            n += 1;
        }
    }
    let mut out = map.into_values().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_size((_, starts, index, matches): Input<'_>) -> Vec<i64> {
    let mut values = vec![0_i64; index.len()];
    let mut seen = vec![false; index.len()];
    let mut labels = Vec::new();
    let mut n = 0;
    for &start in starts {
        for item in start..index.len() {
            if matches[n] != 0 {
                if !seen[item] {
                    seen[item] = true;
                    labels.push(item);
                }
                values[item] += 1;
            }
            n += 1;
        }
    }
    let mut out = labels
        .into_iter()
        .map(|item| values[item])
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_prod((arr, starts, index, matches): Input<'_>) -> Vec<i64> {
    let mut map = HashMap::with_capacity(index.len());
    let mut n = 0;
    for (row, &start) in starts.iter().enumerate() {
        for label in index.iter().skip(start) {
            if matches[n] != 0 {
                *map.entry(*label).or_insert(1) *= arr[row];
            }
            n += 1;
        }
    }
    let mut out = map.into_values().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_prod((arr, starts, index, matches): Input<'_>) -> Vec<i64> {
    let mut values = vec![1_i64; index.len()];
    let mut seen = vec![false; index.len()];
    let mut labels = Vec::new();
    let mut n = 0;
    for (row, &start) in starts.iter().enumerate() {
        for item in start..index.len() {
            if matches[n] != 0 {
                if !seen[item] {
                    seen[item] = true;
                    labels.push(item);
                }
                values[item] *= arr[row];
            }
            n += 1;
        }
    }
    let mut out = labels
        .into_iter()
        .map(|item| values[item])
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_extreme((arr, starts, index, matches): Input<'_>, maximum: bool) -> Vec<i64> {
    let mut map = HashMap::with_capacity(index.len());
    let mut n = 0;
    for (row, &start) in starts.iter().enumerate() {
        for label in index.iter().skip(start) {
            if matches[n] != 0 {
                let entry = map.entry(*label).or_insert((arr[row], row as i64));
                if (maximum && arr[row] > entry.0) || (!maximum && arr[row] < entry.0) {
                    *entry = (arr[row], row as i64);
                }
            }
            n += 1;
        }
    }
    let mut out = map
        .into_values()
        .map(|(value, _)| value)
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_extreme((arr, starts, index, matches): Input<'_>, maximum: bool) -> Vec<i64> {
    let mut values = vec![0_i64; index.len()];
    let mut seen = vec![false; index.len()];
    let mut labels = Vec::new();
    let mut n = 0;
    for (row, &start) in starts.iter().enumerate() {
        for item in start..index.len() {
            if matches[n] != 0 {
                if !seen[item] {
                    seen[item] = true;
                    labels.push(item);
                    values[item] = arr[row];
                } else if (maximum && arr[row] > values[item])
                    || (!maximum && arr[row] < values[item])
                {
                    values[item] = arr[row];
                }
            }
            n += 1;
        }
    }
    let mut out = labels
        .into_iter()
        .map(|item| values[item])
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn allocations<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    TOTAL.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    std::mem::forget(f());
    (TOTAL.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed))
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_starts_matches_dense_all");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
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
        let arr = (0..rows)
            .map(|row| (row % 7 + 1) as i64)
            .collect::<Vec<_>>();
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
        let input = (
            arr.as_slice(),
            starts.as_slice(),
            index.as_slice(),
            matches.as_slice(),
        );
        if !duplicate {
            assert_eq!(hash_sum(input), dense_sum(input));
            assert_eq!(hash_size(input), dense_size(input));
            assert_eq!(hash_prod(input), dense_prod(input));
            assert_eq!(hash_extreme(input, false), dense_extreme(input, false));
            assert_eq!(hash_extreme(input, true), dense_extreme(input, true));
        }
        eprintln!(
            "{name}: sum {:?} -> {:?}, size {:?} -> {:?}, prod {:?} -> {:?}, min/max {:?}",
            allocations(|| hash_sum(input)),
            if duplicate {
                (0, 0)
            } else {
                allocations(|| dense_sum(input))
            },
            allocations(|| hash_size(input)),
            if duplicate {
                (0, 0)
            } else {
                allocations(|| dense_size(input))
            },
            allocations(|| hash_prod(input)),
            if duplicate {
                (0, 0)
            } else {
                allocations(|| dense_prod(input))
            },
            allocations(|| hash_extreme(input, false)),
        );
        for (label, hash, dense) in [
            (
                "sum",
                hash_sum as fn(Input<'_>) -> Vec<i64>,
                dense_sum as fn(Input<'_>) -> Vec<i64>,
            ),
            ("size", hash_size, dense_size),
            ("prod", hash_prod, dense_prod),
        ] {
            group.bench_function(format!("{label}/hash_map/{name}"), |b| {
                b.iter(|| hash(black_box(input)))
            });
            if !duplicate {
                group.bench_function(format!("{label}/dense/{name}"), |b| {
                    b.iter(|| dense(black_box(input)))
                });
            }
        }
        if !duplicate {
            for (label, hash, dense) in [
                (
                    "min",
                    hash_extreme as fn(Input<'_>, bool) -> Vec<i64>,
                    dense_extreme as fn(Input<'_>, bool) -> Vec<i64>,
                ),
                ("max", hash_extreme, dense_extreme),
            ] {
                let maximum = label == "max";
                group.bench_function(format!("{label}/hash_map/{name}"), |b| {
                    b.iter(|| hash(black_box(input), maximum))
                });
                group.bench_function(format!("{label}/dense/{name}"), |b| {
                    b.iter(|| dense(black_box(input), maximum))
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
