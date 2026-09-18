//! Benchmark for the retained fused no-range positional path.

use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::compare_batch_no_range;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("batch_no_range_indices");
    group.sample_size(10);

    for &length in &[8_usize, 1_024, 16_384, 262_144] {
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, (0..length as i64).collect());
            let right = PyArray1::from_vec(py, (0..length as i64).collect());
            let positions = PyArray1::from_vec(py, (0..length as i64).collect());
            let left_index = PyArray1::from_vec(py, (0..length as i64).collect());
            let right_index = PyArray1::from_vec(py, (0..length as i64).collect());
            let predicates = PyList::empty(py);
            predicates
                .append(
                    PyTuple::new(
                        py,
                        [
                            left.clone().into_any(),
                            right.clone().into_any(),
                            4_i8.into_pyobject(py).unwrap().into_any(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();

            group.bench_function(format!("n={length}"), |b| {
                b.iter(|| {
                    compare_batch_no_range(
                        py,
                        black_box(&predicates),
                        black_box(left_index.readonly()),
                        black_box(right_index.readonly()),
                        black_box(positions.readonly()),
                    )
                })
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
