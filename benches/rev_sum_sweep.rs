//! Compare the pre-sweep reverse sum loops with the monotone activation sweep.
//!
//! This benchmark is intentionally old-vs-new: it keeps a small copy of the
//! previous row-by-row implementation so the asymptotic improvement is
//! measured against the code it replaces.

use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::{sum_rev_ends_int_core, sum_rev_starts_int_core};
use numpy::ndarray::{Array1, ArrayView1};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        let live = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocation_report<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    let before_allocated = ALLOCATED.load(Ordering::Relaxed);
    let before_current = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before_current, Ordering::Relaxed);
    black_box(f());
    (
        ALLOCATED.load(Ordering::Relaxed) - before_allocated,
        PEAK.load(Ordering::Relaxed).saturating_sub(before_current),
    )
}

fn old_starts(
    arr: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> (Array1<i64>, Array1<i64>) {
    assert_eq!(arr.len(), starts.len());
    assert_eq!(arr.len(), booleans.len());
    assert!(!arr.is_empty() && !index.is_empty());
    assert!(starts.iter().all(|start| usize::try_from(*start)
        .map(|start| start <= index.len())
        .unwrap_or(false)));
    let min_start = starts.iter().copied().min().unwrap() as usize;
    let width = index.len() - min_start;
    let mut values = vec![0_i64; width];
    for ((current, start), boolean) in arr.iter().zip(starts.iter()).zip(booleans.iter()) {
        if *boolean {
            continue;
        }
        for value in values.iter_mut().skip(*start as usize - min_start) {
            *value = value.wrapping_add(*current);
        }
    }
    let indexers = (min_start..index.len()).map(|item| index[item]).collect();
    (Array1::from_vec(indexers), Array1::from_vec(values))
}

fn old_ends(
    arr: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> (Array1<i64>, Array1<i64>) {
    assert_eq!(arr.len(), ends.len());
    assert_eq!(arr.len(), booleans.len());
    assert!(!arr.is_empty() && !index.is_empty());
    assert!(ends.iter().all(|end| usize::try_from(*end)
        .map(|end| end <= index.len())
        .unwrap_or(false)));
    let max_end = ends.iter().copied().max().unwrap() as usize;
    let mut values = vec![0_i64; max_end];
    for ((current, end), boolean) in arr.iter().zip(ends.iter()).zip(booleans.iter()) {
        if *boolean {
            continue;
        }
        for value in values.iter_mut().take(*end as usize) {
            *value = value.wrapping_add(*current);
        }
    }
    let indexers = (0..max_end).map(|item| index[item]).collect();
    (Array1::from_vec(indexers), Array1::from_vec(values))
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_sum_sweep_old_vs_new");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for (name, rows, right_len) in [
        ("tiny_dense", 32_usize, 32_usize),
        ("large_dense", 1_000, 10_000),
        ("very_large_dense", 10_000, 100_000),
        ("super_large_narrow", 1_000_000, 1_000_000),
    ] {
        let arr = Array1::from_iter((0..rows).map(|i| (i % 97) as i64));
        let index = Array1::from_iter(0..right_len as i64);
        let starts = if name.ends_with("narrow") {
            Array1::from_elem(rows, (right_len - 8) as i64)
        } else {
            Array1::zeros(rows)
        };
        let ends = if name.ends_with("narrow") {
            Array1::from_elem(rows, 8_i64)
        } else {
            Array1::from_elem(rows, right_len as i64)
        };
        let booleans = Array1::from_elem(rows, false);

        let old_start = old_starts(arr.view(), starts.view(), index.view(), booleans.view());
        let new_start = sum_rev_starts_int_core(
            arr.view(),
            starts.mapv(|value| value).view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(old_start, new_start, "starts mismatch for {name}");

        let old_end = old_ends(arr.view(), ends.view(), index.view(), booleans.view());
        let new_end = sum_rev_ends_int_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(old_end, new_end, "ends mismatch for {name}");

        let old_start_memory = allocation_report(|| {
            old_starts(arr.view(), starts.view(), index.view(), booleans.view())
        });
        let new_start_memory = allocation_report(|| {
            sum_rev_starts_int_core(
                arr.view(),
                starts.view(),
                index.view(),
                booleans.view(),
                |value| value,
            )
        });
        let old_end_memory =
            allocation_report(|| old_ends(arr.view(), ends.view(), index.view(), booleans.view()));
        let new_end_memory = allocation_report(|| {
            sum_rev_ends_int_core(
                arr.view(),
                ends.view(),
                index.view(),
                booleans.view(),
                |value| value,
            )
        });
        eprintln!(
            "{name}: starts old {}B/{}B new {}B/{}B; ends old {}B/{}B new {}B/{}B",
            old_start_memory.0,
            old_start_memory.1,
            new_start_memory.0,
            new_start_memory.1,
            old_end_memory.0,
            old_end_memory.1,
            new_end_memory.0,
            new_end_memory.1
        );

        group.bench_function(format!("starts/old/{name}"), |b| {
            b.iter(|| {
                old_starts(
                    black_box(arr.view()),
                    black_box(starts.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("starts/sweep/{name}"), |b| {
            b.iter(|| {
                sum_rev_starts_int_core(
                    black_box(arr.view()),
                    black_box(starts.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                    |value| value,
                )
            })
        });
        group.bench_function(format!("ends/old/{name}"), |b| {
            b.iter(|| {
                old_ends(
                    black_box(arr.view()),
                    black_box(ends.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("ends/sweep/{name}"), |b| {
            b.iter(|| {
                sum_rev_ends_int_core(
                    black_box(arr.view()),
                    black_box(ends.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                    |value| value,
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
