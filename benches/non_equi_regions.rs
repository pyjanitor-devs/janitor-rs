//! Compare direct dual/multi-region index construction with simple references.
//!
//! The wrappers are called through Python so timings include NumPy/PyO3
//! extraction, validation, and output conversion. Allocation reports are one
//! call per fixture and are printed separately from Criterion timings.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::{
    build_dual_region_indices_all, build_dual_region_indices_any, build_dual_region_indices_first,
    build_dual_region_indices_last, compare_multi_region_indices_all,
    compare_multi_region_indices_any, compare_multi_region_indices_first,
    compare_multi_region_indices_last, region_positions,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::hint::black_box;

#[path = "support/mod.rs"]
mod support;

#[derive(Clone, Copy)]
enum Selection {
    First,
    Last,
    Any,
    All,
}

struct Fixture {
    left: Vec<i64>,
    right: Vec<i64>,
    left_region: Vec<i64>,
    right_region: Vec<i64>,
    starts: Vec<i64>,
    left_index: Vec<i64>,
    right_index: Vec<i64>,
    left_extra: Vec<i64>,
    right_extra: Vec<i64>,
}

struct PyFixture<'py> {
    left: Bound<'py, PyArray1<i64>>,
    right: Bound<'py, PyArray1<i64>>,
    left_region: Bound<'py, PyArray1<i64>>,
    right_region: Bound<'py, PyArray1<i64>>,
    starts: Bound<'py, PyArray1<i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: Bound<'py, PyArray1<i64>>,
    predicates: Bound<'py, PyList>,
    left_extra: Bound<'py, PyArray1<i64>>,
    right_extra: Bound<'py, PyArray1<i64>>,
    left_len: usize,
    right_len: usize,
}

impl<'py> PyFixture<'py> {
    fn new(py: Python<'py>, fixture: &Fixture) -> PyResult<Self> {
        Ok(Self {
            left: PyArray1::from_vec(py, fixture.left.clone()),
            right: PyArray1::from_vec(py, fixture.right.clone()),
            left_region: PyArray1::from_vec(py, fixture.left_region.clone()),
            right_region: PyArray1::from_vec(py, fixture.right_region.clone()),
            starts: PyArray1::from_vec(py, fixture.starts.clone()),
            left_index: PyArray1::from_vec(py, fixture.left_index.clone()),
            right_index: PyArray1::from_vec(py, fixture.right_index.clone()),
            predicates: predicates(py, fixture)?,
            left_extra: PyArray1::from_vec(py, fixture.left_extra.clone()),
            right_extra: PyArray1::from_vec(py, fixture.right_extra.clone()),
            left_len: fixture.left.len(),
            right_len: fixture.right.len(),
        })
    }
}

fn run_dual(py: Python<'_>, fixture: &PyFixture<'_>, selection: Selection) {
    let output = match selection {
        Selection::First => build_dual_region_indices_first(
            py,
            fixture.left.readonly(),
            fixture.right.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.readonly(),
            fixture.right_index.readonly(),
        ),
        Selection::Last => build_dual_region_indices_last(
            py,
            fixture.left.readonly(),
            fixture.right.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.readonly(),
            fixture.right_index.readonly(),
        ),
        Selection::Any => build_dual_region_indices_any(
            py,
            fixture.left.readonly(),
            fixture.right.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.readonly(),
            fixture.right_index.readonly(),
        ),
        Selection::All => build_dual_region_indices_all(
            py,
            fixture.left.readonly(),
            fixture.right.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.readonly(),
            fixture.right_index.readonly(),
        ),
    };
    black_box(output.unwrap());
}

fn run_multi(py: Python<'_>, fixture: &PyFixture<'_>, selection: Selection) {
    let output = match selection {
        Selection::First => compare_multi_region_indices_first(
            py,
            &fixture.predicates,
            fixture.left_region.readonly(),
            fixture.right_region.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.clone(),
            fixture.right_index.readonly(),
        ),
        Selection::Last => compare_multi_region_indices_last(
            py,
            &fixture.predicates,
            fixture.left_region.readonly(),
            fixture.right_region.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.clone(),
            fixture.right_index.readonly(),
        ),
        Selection::Any => compare_multi_region_indices_any(
            py,
            &fixture.predicates,
            fixture.left_region.readonly(),
            fixture.right_region.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.clone(),
            fixture.right_index.readonly(),
        ),
        Selection::All => compare_multi_region_indices_all(
            py,
            &fixture.predicates,
            fixture.left_region.readonly(),
            fixture.right_region.readonly(),
            fixture.starts.readonly(),
            fixture.left_index.clone(),
            fixture.right_index.readonly(),
        ),
    };
    black_box(output.unwrap());
}

fn run_current_regions(
    py: Python<'_>,
    fixture: &PyFixture<'_>,
    selection: Selection,
    with_extra_predicate: bool,
) {
    // Legacy regions flow: first materialize the flattened candidate positions
    // and per-left-row counts, then walk those candidates again to apply the
    // selection and (for multi) the residual equality predicate.
    let left_region = if with_extra_predicate {
        fixture.left_region.readonly()
    } else {
        fixture.left.readonly()
    };
    let right_region = if with_extra_predicate {
        fixture.right_region.readonly()
    } else {
        fixture.right.readonly()
    };
    let (counts, positions, _) = region_positions(
        py,
        left_region,
        right_region,
        fixture.starts.readonly(),
        fixture.right_len as i64,
    );
    let counts = counts.readonly();
    let positions = positions.readonly();
    let counts = counts.as_array();
    let positions = positions.as_array();
    let left_index = fixture.left_index.readonly();
    let left_index = left_index.as_array();
    let right_index = fixture.right_index.readonly();
    let right_index = right_index.as_array();
    let mut left_output = Vec::new();
    let mut right_output = Vec::new();
    let mut flat_position = 0_usize;

    let left_extra = fixture.left_extra.readonly();
    let left_extra = left_extra.as_array();
    let right_extra = fixture.right_extra.readonly();
    let right_extra = right_extra.as_array();
    for row in 0..fixture.left_len {
        let row_count = counts[row].max(0) as usize;
        let mut selected = None;
        for candidate in positions.iter().skip(flat_position).take(row_count) {
            let right_position = *candidate as usize;
            if with_extra_predicate && left_extra[row] != right_extra[right_position] {
                continue;
            }
            match selection {
                Selection::Any => {
                    selected = Some(right_position);
                    break;
                }
                Selection::First => {
                    if selected
                        .is_none_or(|current| right_index[right_position] < right_index[current])
                    {
                        selected = Some(right_position);
                    }
                }
                Selection::Last => {
                    if selected
                        .is_none_or(|current| right_index[right_position] > right_index[current])
                    {
                        selected = Some(right_position);
                    }
                }
                Selection::All => {
                    left_output.push(left_index[row]);
                    right_output.push(right_index[right_position]);
                }
            }
        }
        if let Some(right_position) = selected {
            left_output.push(left_index[row]);
            right_output.push(right_index[right_position]);
        }
        flat_position += row_count;
    }
    black_box((left_output, right_output));
}

fn fixture(left_len: usize, right_len: usize, shape: usize) -> Fixture {
    // Make exactly one right position satisfy the region condition for each
    // valid row. This keeps the `All` output bounded while still exercising
    // narrow, broad, and mixed candidate-region scans.
    let left = vec![right_len.saturating_sub(1) as i64; left_len];
    let right = (0..right_len).map(|position| position as i64).collect();
    let left_region = vec![right_len.saturating_sub(1) as i64; left_len];
    let right_region = (0..right_len).map(|position| position as i64).collect();
    let starts = (0..left_len)
        .map(|row| match shape {
            0 => (right_len.saturating_sub(1)) as i64,
            1 => 0,
            _ => right_len.saturating_sub(row.min(right_len)) as i64,
        })
        .collect();
    let left_index = (0..left_len).map(|row| row as i64).collect();
    let right_index = (0..right_len).map(|row| row as i64).collect();
    let left_extra = vec![0_i64; left_len];
    let right_extra = vec![0_i64; right_len];
    Fixture {
        left,
        right,
        left_region,
        right_region,
        starts,
        left_index,
        right_index,
        left_extra,
        right_extra,
    }
}

fn dual_reference(f: &Fixture, selection: Selection) -> Option<(Vec<i64>, Vec<i64>)> {
    let mut left_output = Vec::new();
    let mut right_output = Vec::new();
    for row in 0..f.left.len() {
        let start = f.starts[row] as usize;
        let mut selected = None;
        for right_position in start..f.right.len() {
            if f.left[row] > f.right[right_position] {
                continue;
            }
            match selection {
                Selection::Any => {
                    selected = Some(right_position);
                    break;
                }
                Selection::First => {
                    if selected.is_none_or(|current| {
                        f.right_index[right_position] < f.right_index[current]
                    }) {
                        selected = Some(right_position);
                    }
                }
                Selection::Last => {
                    if selected.is_none_or(|current| {
                        f.right_index[right_position] > f.right_index[current]
                    }) {
                        selected = Some(right_position);
                    }
                }
                Selection::All => {
                    left_output.push(f.left_index[row]);
                    right_output.push(f.right_index[right_position]);
                }
            }
        }
        if let Some(right_position) = selected {
            left_output.push(f.left_index[row]);
            right_output.push(f.right_index[right_position]);
        }
    }
    (!left_output.is_empty()).then_some((left_output, right_output))
}

fn multi_reference(f: &Fixture, selection: Selection) -> Option<(Vec<i64>, Vec<i64>)> {
    let mut left_output = Vec::new();
    let mut right_output = Vec::new();
    for row in 0..f.left.len() {
        let start = f.starts[row] as usize;
        let mut selected = None;
        for right_position in start..f.right_region.len() {
            if f.left_region[row] > f.right_region[right_position]
                || f.left_extra[row] != f.right_extra[right_position]
            {
                continue;
            }
            match selection {
                Selection::Any => {
                    selected = Some(right_position);
                    break;
                }
                Selection::First => {
                    if selected.is_none_or(|current| {
                        f.right_index[right_position] < f.right_index[current]
                    }) {
                        selected = Some(right_position);
                    }
                }
                Selection::Last => {
                    if selected.is_none_or(|current| {
                        f.right_index[right_position] > f.right_index[current]
                    }) {
                        selected = Some(right_position);
                    }
                }
                Selection::All => {
                    left_output.push(f.left_index[row]);
                    right_output.push(f.right_index[right_position]);
                }
            }
        }
        if let Some(right_position) = selected {
            left_output.push(f.left_index[row]);
            right_output.push(f.right_index[right_position]);
        }
    }
    (!left_output.is_empty()).then_some((left_output, right_output))
}

fn predicates<'py>(py: Python<'py>, f: &Fixture) -> PyResult<Bound<'py, PyList>> {
    PyList::new(
        py,
        [PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, f.left_extra.clone()).into_any(),
                PyArray1::from_vec(py, f.right_extra.clone()).into_any(),
                4_i8.into_pyobject(py)?.into_any(),
            ],
        )?],
    )
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("non_equi_regions");
    let shapes = [("narrow", 0_usize), ("broad", 1), ("mixed", 2)];
    let sizes = [
        (8, 16, "tiny"),
        (128, 256, "small"),
        (2_048, 4_096, "large"),
        (16_384, 32_768, "very_large"),
        (65_536, 131_072, "super_large"),
        (262_144, 524_288, "extreme"),
        (1_048_576, 2_097_152, "super_extreme"),
    ];

    for &(left_len, right_len, size) in &sizes {
        for &(shape, shape_id) in &shapes {
            let f = fixture(left_len, right_len, shape_id);
            let label = format!("{size}/{shape}/left={left_len}/right={right_len}");
            Python::attach(|py| {
                let py_fixture = PyFixture::new(py, &f).unwrap();
                for selection in [
                    Selection::First,
                    Selection::Last,
                    Selection::Any,
                    Selection::All,
                ] {
                    let selection_name = match selection {
                        Selection::First => "first",
                        Selection::Last => "last",
                        Selection::Any => "any",
                        Selection::All => "all",
                    };
                    let dual_allocation =
                        support::count_allocations(|| run_dual(py, &py_fixture, selection));
                    let multi_allocation =
                        support::count_allocations(|| run_multi(py, &py_fixture, selection));
                    let current_dual_allocation = support::count_allocations(|| {
                        run_current_regions(py, &py_fixture, selection, false)
                    });
                    let current_multi_allocation = support::count_allocations(|| {
                        run_current_regions(py, &py_fixture, selection, true)
                    });
                    eprintln!(
                        "{label} {selection_name}: proposed dual {} bytes/{} allocs/{} peak; proposed multi {} bytes/{} allocs/{} peak; current dual {} bytes/{} allocs/{} peak; current multi {} bytes/{} allocs/{} peak",
                        dual_allocation.0,
                        dual_allocation.1,
                        dual_allocation.2,
                        multi_allocation.0,
                        multi_allocation.1,
                        multi_allocation.2,
                        current_dual_allocation.0,
                        current_dual_allocation.1,
                        current_dual_allocation.2,
                        current_multi_allocation.0,
                        current_multi_allocation.1,
                        current_multi_allocation.2,
                    );
                }
            });

            for selection in [
                Selection::First,
                Selection::Last,
                Selection::Any,
                Selection::All,
            ] {
                let selection_name = match selection {
                    Selection::First => "first",
                    Selection::Last => "last",
                    Selection::Any => "any",
                    Selection::All => "all",
                };
                let id = BenchmarkId::new(format!("dual/{selection_name}"), &label);
                group.bench_function(id, |b| {
                    b.iter(|| {
                        black_box(dual_reference(black_box(&f), selection));
                    })
                });
                let id = BenchmarkId::new(format!("wrapper/dual/{selection_name}"), &label);
                Python::attach(|py| {
                    let py_fixture = PyFixture::new(py, &f).unwrap();
                    group.bench_function(id, |b| {
                        b.iter(|| run_dual(py, &py_fixture, selection));
                    });
                });
                let id = BenchmarkId::new(format!("current/regions/dual/{selection_name}"), &label);
                Python::attach(|py| {
                    let py_fixture = PyFixture::new(py, &f).unwrap();
                    group.bench_function(id, |b| {
                        b.iter(|| run_current_regions(py, &py_fixture, selection, false));
                    });
                });
                let id = BenchmarkId::new(format!("multi/{selection_name}"), &label);
                group.bench_function(id, |b| {
                    b.iter(|| {
                        black_box(multi_reference(black_box(&f), selection));
                    })
                });
                let id = BenchmarkId::new(format!("wrapper/multi/{selection_name}"), &label);
                Python::attach(|py| {
                    let py_fixture = PyFixture::new(py, &f).unwrap();
                    group.bench_function(id, |b| {
                        b.iter(|| run_multi(py, &py_fixture, selection));
                    });
                });
                let id =
                    BenchmarkId::new(format!("current/regions/multi/{selection_name}"), &label);
                Python::attach(|py| {
                    let py_fixture = PyFixture::new(py, &f).unwrap();
                    group.bench_function(id, |b| {
                        b.iter(|| run_current_regions(py, &py_fixture, selection, true));
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
