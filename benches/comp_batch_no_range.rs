//! Compare the fused no-range positional path with the existing sequential path.
//!
//! Each row already has at most one candidate right position. The benchmark
//! therefore measures the complete wrapper call through output-label
//! generation, not just predicate matching. Allocation counters are reported
//! separately because Criterion reports time, not allocation behaviour.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::{
    compare_batch_no_range, compare_no_range_f64, compare_no_range_int64, compare_no_range_ne_int64,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::hint::black_box;

#[path = "support/mod.rs"]
mod support;

#[derive(Clone, Copy)]
enum Shape {
    All,
    Sparse,
    Mixed,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Self::All => "all-survive",
            Self::Sparse => "sparse-candidates",
            Self::Mixed => "mixed-survival",
        }
    }
}

struct Fixture {
    left: Vec<i64>,
    right: Vec<i64>,
    left_float: Vec<f64>,
    right_float: Vec<f64>,
    positions: Vec<i64>,
    left_index: Vec<i64>,
    right_index: Vec<i64>,
    left_mask: Vec<bool>,
    right_mask: Vec<bool>,
}

struct PyFixture<'py> {
    left: Bound<'py, PyArray1<i64>>,
    right: Bound<'py, PyArray1<i64>>,
    left_float: Bound<'py, PyArray1<f64>>,
    right_float: Bound<'py, PyArray1<f64>>,
    positions: Bound<'py, PyArray1<i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: Bound<'py, PyArray1<i64>>,
    left_mask: Bound<'py, PyArray1<bool>>,
    right_mask: Bound<'py, PyArray1<bool>>,
    plain_predicates: Bound<'py, PyList>,
    mixed_predicates: Bound<'py, PyList>,
    null_predicates_extension: Bound<'py, PyList>,
    null_predicates_non_extension: Bound<'py, PyList>,
}

type Output = Option<(Vec<i64>, Vec<i64>)>;

fn fixture(length: usize, shape: Shape) -> Fixture {
    let left: Vec<i64> = (0..length).map(|value| value as i64).collect();
    let right = left.clone();
    let left_float: Vec<f64> = left.iter().map(|value| *value as f64 + 0.5).collect();
    let right_float: Vec<f64> = right
        .iter()
        .map(|value| *value as f64 + if *value % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let positions = (0..length)
        .map(|row| match shape {
            Shape::All => (row % length) as i64,
            Shape::Sparse => {
                if row % 10 == 0 {
                    (row % length) as i64
                } else {
                    -1
                }
            }
            Shape::Mixed => (row % length) as i64,
        })
        .collect();
    Fixture {
        left_index: (0..length).map(|value| value as i64 + 10_000).collect(),
        right_index: (0..length).map(|value| value as i64 + 20_000).collect(),
        left_mask: (0..length).map(|row| row % 3 == 0).collect(),
        right_mask: (0..length).map(|row| row % 5 == 0).collect(),
        left,
        right,
        left_float,
        right_float,
        positions,
    }
}

fn predicates<'py>(py: Python<'py>, fixture: &Fixture) -> PyResult<PyFixture<'py>> {
    let left = PyArray1::from_vec(py, fixture.left.clone());
    let right = PyArray1::from_vec(py, fixture.right.clone());
    let left_float = PyArray1::from_vec(py, fixture.left_float.clone());
    let right_float = PyArray1::from_vec(py, fixture.right_float.clone());
    let positions = PyArray1::from_vec(py, fixture.positions.clone());
    let left_index = PyArray1::from_vec(py, fixture.left_index.clone());
    let right_index = PyArray1::from_vec(py, fixture.right_index.clone());
    let left_mask = PyArray1::from_vec(py, fixture.left_mask.clone());
    let right_mask = PyArray1::from_vec(py, fixture.right_mask.clone());

    let plain_predicates = PyList::empty(py);
    plain_predicates.append(PyTuple::new(
        py,
        [
            left.clone().into_any(),
            right.clone().into_any(),
            4_i8.into_pyobject(py)?.into_any(),
        ],
    )?)?;

    let mixed_predicates = PyList::empty(py);
    mixed_predicates.append(PyTuple::new(
        py,
        [
            left.clone().into_any(),
            right.clone().into_any(),
            4_i8.into_pyobject(py)?.into_any(),
        ],
    )?)?;
    mixed_predicates.append(PyTuple::new(
        py,
        [
            left_float.clone().into_any(),
            right_float.clone().into_any(),
            3_i8.into_pyobject(py)?.into_any(),
        ],
    )?)?;

    let null_predicates = |extension_flag: i8| -> PyResult<Bound<'py, PyList>> {
        let predicates = PyList::empty(py);
        predicates.append(PyTuple::new(
            py,
            [
                left.clone().into_any(),
                right.clone().into_any(),
                5_i8.into_pyobject(py)?.into_any(),
                left_mask.clone().into_any(),
                right_mask.clone().into_any(),
                extension_flag.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        Ok(predicates)
    };
    let null_predicates_extension = null_predicates(1)?;
    let null_predicates_non_extension = null_predicates(0)?;

    Ok(PyFixture {
        left,
        right,
        left_float,
        right_float,
        positions,
        left_index,
        right_index,
        left_mask,
        right_mask,
        plain_predicates,
        mixed_predicates,
        null_predicates_extension,
        null_predicates_non_extension,
    })
}

fn compact_old<'py>(
    left_index: &Bound<'py, PyArray1<i64>>,
    right_index: &Bound<'py, PyArray1<i64>>,
    positions: &Bound<'py, PyArray1<i64>>,
) -> Output {
    let positions_readonly = positions.readonly();
    let left_index_readonly = left_index.readonly();
    let right_index_readonly = right_index.readonly();
    let positions = positions_readonly.as_array();
    let left_index = left_index_readonly.as_array();
    let right_index = right_index_readonly.as_array();
    let mut left_output = Vec::new();
    let mut right_output = Vec::new();
    for (row, position) in positions.iter().enumerate() {
        if *position >= 0 && (*position as usize) < right_index.len() {
            left_output.push(left_index[row]);
            right_output.push(right_index[*position as usize]);
        }
    }
    if left_output.is_empty() {
        None
    } else {
        Some((left_output, right_output))
    }
}

fn old_plain(py: Python<'_>, fixture: &PyFixture<'_>, mixed: bool) -> Output {
    let first = compare_no_range_int64(
        py,
        fixture.left.readonly(),
        fixture.right.readonly(),
        fixture.positions.readonly(),
        4,
    )
    .unwrap()
    .0;
    if !mixed {
        return compact_old(&fixture.left_index, &fixture.right_index, &first);
    }
    let second = compare_no_range_f64(
        py,
        fixture.left_float.readonly(),
        fixture.right_float.readonly(),
        first.readonly(),
        3,
    )
    .unwrap()
    .0;
    compact_old(&fixture.left_index, &fixture.right_index, &second)
}

fn old_null(py: Python<'_>, fixture: &PyFixture<'_>, is_extension_array: bool) -> Output {
    let result = compare_no_range_ne_int64(
        py,
        fixture.left.readonly(),
        fixture.right.readonly(),
        fixture.positions.readonly(),
        fixture.left_mask.readonly(),
        fixture.right_mask.readonly(),
        is_extension_array,
        5,
    )
    .unwrap()
    .0;
    compact_old(&fixture.left_index, &fixture.right_index, &result)
}

fn new_path(py: Python<'_>, fixture: &PyFixture<'_>, predicates: &Bound<'_, PyList>) -> Output {
    compare_batch_no_range(
        py,
        predicates,
        fixture.left_index.readonly(),
        fixture.right_index.readonly(),
        fixture.positions.readonly(),
    )
    .unwrap()
    .map(|(left, right)| {
        (
            left.readonly().as_array().to_vec(),
            right.readonly().as_array().to_vec(),
        )
    })
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("comp_batch_no_range");
    group.sample_size(10);
    for &length in &[8, 1_024, 16_384, 262_144, 1_048_576] {
        for shape in [Shape::All, Shape::Sparse, Shape::Mixed] {
            let label = format!("{}/n={length}", shape.name());
            Python::attach(|py| {
                let fixture = fixture(length, shape);
                let py_fixture = predicates(py, &fixture).unwrap();
                let predicates = if matches!(shape, Shape::Mixed) {
                    &py_fixture.mixed_predicates
                } else {
                    &py_fixture.plain_predicates
                };
                let old = if matches!(shape, Shape::Mixed) {
                    old_plain(py, &py_fixture, true)
                } else {
                    old_plain(py, &py_fixture, false)
                };
                let new = new_path(py, &py_fixture, predicates);
                assert_eq!(old, new, "old and new outputs differ for {label}");

                let (old_bytes, old_allocs, old_peak) = support::count_allocations(|| {
                    if matches!(shape, Shape::Mixed) {
                        old_plain(py, &py_fixture, true)
                    } else {
                        old_plain(py, &py_fixture, false)
                    }
                });
                let (new_bytes, new_allocs, new_peak) =
                    support::count_allocations(|| new_path(py, &py_fixture, predicates));
                eprintln!(
                "{label}: old {old_bytes} bytes/{old_allocs} allocs/{old_peak} peak; new {new_bytes} bytes/{new_allocs} allocs/{new_peak} peak"
            );

                group.bench_function(BenchmarkId::new("old", &label), |b| {
                    b.iter(|| {
                        black_box(if matches!(shape, Shape::Mixed) {
                            old_plain(py, &py_fixture, true)
                        } else {
                            old_plain(py, &py_fixture, false)
                        })
                    })
                });
                group.bench_function(BenchmarkId::new("new", &label), |b| {
                    b.iter(|| black_box(new_path(py, &py_fixture, predicates)))
                });
            });
        }
    }
    for &length in &[8, 1_024, 16_384, 262_144, 1_048_576] {
        for (label, is_extension_array) in [("null-extension", true), ("null-non-extension", false)]
        {
            let label = format!("{label}/n={length}");
            Python::attach(|py| {
                let fixture = fixture(length, Shape::All);
                let py_fixture = predicates(py, &fixture).unwrap();
                let predicates = if is_extension_array {
                    &py_fixture.null_predicates_extension
                } else {
                    &py_fixture.null_predicates_non_extension
                };
                let old = old_null(py, &py_fixture, is_extension_array);
                let new = new_path(py, &py_fixture, predicates);
                assert_eq!(old, new, "old and new outputs differ for {label}");

                let (old_bytes, old_allocs, old_peak) =
                    support::count_allocations(|| old_null(py, &py_fixture, is_extension_array));
                let (new_bytes, new_allocs, new_peak) =
                    support::count_allocations(|| new_path(py, &py_fixture, predicates));
                eprintln!(
                    "{label}: old {old_bytes} bytes/{old_allocs} allocs/{old_peak} peak; new {new_bytes} bytes/{new_allocs} allocs/{new_peak} peak"
                );

                group.bench_function(BenchmarkId::new("old", &label), |b| {
                    b.iter(|| black_box(old_null(py, &py_fixture, is_extension_array)))
                });
                group.bench_function(BenchmarkId::new("new", &label), |b| {
                    b.iter(|| black_box(new_path(py, &py_fixture, predicates)))
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
