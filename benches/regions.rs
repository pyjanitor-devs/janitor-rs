use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::{dual_regions_aggregate, dual_regions_aggregate_legacy};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::hint::black_box;

struct CountingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static BASE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            record_alloc(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
        record_free(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new_pointer = System.realloc(pointer, layout, size);
        if !new_pointer.is_null() {
            if size >= layout.size() {
                record_alloc(size - layout.size());
            } else {
                record_free(layout.size() - size);
            }
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_alloc(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(current) => peak = current,
        }
    }
}

fn record_free(bytes: usize) {
    LIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

fn reset_peak() {
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    BASE_BYTES.store(live, Ordering::Relaxed);
    PEAK_BYTES.store(live, Ordering::Relaxed);
}

fn peak_delta() -> usize {
    PEAK_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(BASE_BYTES.load(Ordering::Relaxed))
}

struct Prepared<'py> {
    left: Bound<'py, PyArray1<i64>>,
    right: Bound<'py, PyArray1<i64>>,
    first: Bound<'py, PyTuple>,
    second: Bound<'py, PyTuple>,
    aligner: Bound<'py, PyArray1<i64>>,
    aggregations: Bound<'py, PyList>,
}

fn prepare<'py>(py: Python<'py>, size: usize) -> Prepared<'py> {
    // Keep the candidate window fixed while the input grows. This makes the
    // legacy traversal measurable at large sizes instead of accidentally
    // turning the benchmark into an unbounded quadratic stress test.
    const WINDOW: usize = 64;
    assert!(size >= WINDOW);
    let left_values = vec![(size - WINDOW) as i64; size];
    let right_values: Vec<i64> = (0..size as i64).collect();
    let left = PyArray1::from_vec(py, (0..size as i64).collect());
    let right = PyArray1::from_vec(py, (0..size as i64).collect());
    let first = PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, left_values.clone()).into_any(),
            PyArray1::from_vec(py, right_values.clone()).into_any(),
            "<=".into_pyobject(py).unwrap().into_any(),
        ],
    )
    .unwrap();
    let second = PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, left_values).into_any(),
            PyArray1::from_vec(py, right_values.clone()).into_any(),
            "<=".into_pyobject(py).unwrap().into_any(),
        ],
    )
    .unwrap();
    let aligner = PyArray1::from_vec(py, (0..size as i64).collect());
    let mask = PyArray1::from_vec(py, vec![false; size]);
    let values = PyArray1::from_vec(py, right_values);
    let aggregations = PyList::new(
        py,
        [
            PyTuple::new(
                py,
                [
                    values.clone().into_any(),
                    mask.clone().into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap()
            .into_any(),
            PyTuple::new(
                py,
                [
                    values.clone().into_any(),
                    mask.clone().into_any(),
                    "min".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap()
            .into_any(),
            PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "max".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap()
            .into_any(),
        ],
    )
    .unwrap();
    Prepared {
        left,
        right,
        first,
        second,
        aligner,
        aggregations,
    }
}

fn call(prepared: &Prepared<'_>, py: Python<'_>, optimized: bool) {
    let result = if optimized {
        dual_regions_aggregate(
            py,
            prepared.left.as_any(),
            prepared.right.as_any(),
            &prepared.first,
            &prepared.second,
            prepared.aligner.readonly(),
            &prepared.aggregations,
            false,
        )
    } else {
        dual_regions_aggregate_legacy(
            py,
            prepared.left.as_any(),
            prepared.right.as_any(),
            &prepared.first,
            &prepared.second,
            prepared.aligner.readonly(),
            &prepared.aggregations,
            false,
        )
    };
    black_box(result.unwrap().is_some());
}

fn benchmark_regions(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("dual_region_aggregation");
    // Large fixtures make the asymptotic difference visible, while a smaller
    // sample count keeps the deliberately retained legacy baseline practical.
    group.sample_size(20);
    for size in [1_024, 8_192, 65_536, 262_144, 1_048_576] {
        Python::attach(|py| {
            let prepared = prepare(py, size);
            for (name, optimized) in [("legacy", false), ("rank_tree", true)] {
                group.bench_with_input(BenchmarkId::new(name, size), &size, |b, _| {
                    b.iter(|| call(&prepared, py, optimized));
                });
                reset_peak();
                call(&prepared, py, optimized);
                eprintln!(
                    "dual_region_aggregation/{name}/{size}: peak additional bytes {}",
                    peak_delta()
                );
            }
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_regions);
criterion_main!(benches);
