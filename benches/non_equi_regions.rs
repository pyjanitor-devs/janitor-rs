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
    compare_multi_region_indices_last,
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
    left_float: Vec<f64>,
    right_float: Vec<f64>,
    left_ne: Vec<i32>,
    right_ne: Vec<i32>,
    left_ne_mask: Vec<bool>,
    right_ne_mask: Vec<bool>,
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
        })
    }
}

type Output = (Vec<i64>, Vec<i64>);
type PyOutput<'py> = Option<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

fn output_vectors<'py>(output: PyResult<PyOutput<'py>>) -> PyResult<Option<Output>> {
    output.map(|output| {
        output.map(|(left, right)| {
            (
                left.readonly().as_array().to_vec(),
                right.readonly().as_array().to_vec(),
            )
        })
    })
}

fn dual_output(
    py: Python<'_>,
    fixture: &PyFixture<'_>,
    selection: Selection,
) -> PyResult<Option<Output>> {
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
    output_vectors(output)
}

fn run_dual(py: Python<'_>, fixture: &PyFixture<'_>, selection: Selection) {
    black_box(dual_output(py, fixture, selection).unwrap());
}

fn multi_output(
    py: Python<'_>,
    fixture: &PyFixture<'_>,
    selection: Selection,
) -> PyResult<Option<Output>> {
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
    output_vectors(output)
}

fn run_multi(py: Python<'_>, fixture: &PyFixture<'_>, selection: Selection) {
    black_box(multi_output(py, fixture, selection).unwrap());
}

fn multi_predicates_match(fixture: &Fixture, row: usize, right: usize) -> bool {
    // Keep this deliberately simple: it is the benchmark's old-path oracle,
    // not production code. It mirrors three heterogeneous residual
    // predicates used by the proposed multi-region wrapper.
    fixture.left_extra[row] == fixture.right_extra[right]
        && fixture.left_float[row] <= fixture.right_float[right]
        && {
            let left_null = fixture.left_ne_mask[row];
            let right_null = fixture.right_ne_mask[right];
            if left_null || right_null {
                false
            } else {
                fixture.left_ne[row] != fixture.right_ne[right]
            }
        }
}

fn fixture(left_len: usize, right_len: usize, shape: usize) -> Fixture {
    // The first three shapes keep exactly one right position satisfying the
    // residual predicates for each valid row. `multi_groups` adds one large
    // duplicate-heavy low-value group and up to 63 singleton upper groups.
    // Most rows qualify only the top four groups, so this exercises large
    // BTreeMap/duplicate-chain construction and bounded multi-group traversal
    // without making every row scan the complete right frame. The first row
    // deliberately qualifies from value 0 so the long duplicate chain is
    // traversed at scale; keeping this to one row avoids a quadratic fixture.
    let group_count = right_len.min(64);
    let tail_start = right_len.saturating_sub(group_count);
    let multi_group_threshold = group_count.saturating_sub(4) as i64;
    let right_region: Vec<i64> = (0..right_len)
        .map(|position| {
            if shape == 3 {
                if position < tail_start {
                    0
                } else {
                    (position - tail_start) as i64
                }
            } else {
                position as i64
            }
        })
        .collect();
    let left = vec![right_len.saturating_sub(1) as i64; left_len];
    // Keep the dual condition's value array unique so its legacy reference
    // remains comparable. The repeated region values above specifically
    // exercise the multi-region path; dual duplicate-value coverage is a
    // separate benchmark gap because the two implementations are independent.
    let right = (0..right_len).map(|position| position as i64).collect();
    let mut left_region = if shape == 3 {
        vec![multi_group_threshold; left_len]
    } else {
        vec![right_len.saturating_sub(1) as i64; left_len]
    };
    if shape == 3 {
        left_region[0] = 0;
    }
    let starts = (0..left_len)
        .map(|row| match shape {
            0 => (right_len.saturating_sub(1)) as i64,
            1 => 0,
            2 => right_len.saturating_sub(row.min(right_len)) as i64,
            _ => 0,
        })
        .collect();
    let left_index = (0..left_len).map(|row| row as i64).collect();
    let right_index = (0..right_len).map(|row| row as i64).collect();
    let left_extra = vec![0_i64; left_len];
    let right_extra = vec![0_i64; right_len];
    let left_float = vec![right_len.saturating_sub(1) as f64; left_len];
    let right_float = (0..right_len).map(|position| position as f64).collect();
    let left_ne = (0..left_len).map(|row| row as i32).collect();
    let right_ne = (0..right_len).map(|row| (row + 1) as i32).collect();
    let left_ne_mask = vec![false; left_len];
    let right_ne_mask = vec![false; right_len];
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
        left_float,
        right_float,
        left_ne,
        right_ne,
        left_ne_mask,
        right_ne_mask,
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
                || !multi_predicates_match(f, row, right_position)
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

fn multi_reference_all_two_pass(f: &Fixture, narrow_second_pass: bool) -> Option<Output> {
    let mut total = 0_usize;
    let mut bounds = vec![None; f.left.len()];
    for (row, row_bounds) in bounds.iter_mut().enumerate() {
        let start = f.starts[row] as usize;
        for right_position in start..f.right_region.len() {
            if f.left_region[row] > f.right_region[right_position]
                || !multi_predicates_match(f, row, right_position)
            {
                continue;
            }
            let (min_position, max_position) =
                row_bounds.get_or_insert((right_position, right_position));
            *min_position = (*min_position).min(right_position);
            *max_position = (*max_position).max(right_position);
            total += 1;
        }
    }
    if total == 0 {
        return None;
    }

    let mut left_output = Vec::with_capacity(total);
    let mut right_output = Vec::with_capacity(total);
    for (row, row_bounds) in bounds.iter().enumerate() {
        let start = f.starts[row] as usize;
        for right_position in start..f.right_region.len() {
            if narrow_second_pass
                && row_bounds.is_some_and(|(min_position, max_position)| {
                    right_position < min_position || right_position > max_position
                })
            {
                continue;
            }
            if f.left_region[row] <= f.right_region[right_position]
                && multi_predicates_match(f, row, right_position)
            {
                left_output.push(f.left_index[row]);
                right_output.push(f.right_index[right_position]);
            }
        }
    }
    debug_assert_eq!(left_output.len(), total);
    Some((left_output, right_output))
}

fn predicates<'py>(py: Python<'py>, f: &Fixture) -> PyResult<Bound<'py, PyList>> {
    let predicates = PyList::empty(py);
    predicates.append(PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, f.left_extra.clone()).into_any(),
            PyArray1::from_vec(py, f.right_extra.clone()).into_any(),
            4_i8.into_pyobject(py)?.into_any(),
        ],
    )?)?;
    predicates.append(PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, f.left_float.clone()).into_any(),
            PyArray1::from_vec(py, f.right_float.clone()).into_any(),
            3_i8.into_pyobject(py)?.into_any(),
        ],
    )?)?;
    predicates.append(PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, f.left_ne.clone()).into_any(),
            PyArray1::from_vec(py, f.right_ne.clone()).into_any(),
            5_i8.into_pyobject(py)?.into_any(),
            PyArray1::from_vec(py, f.left_ne_mask.clone()).into_any(),
            PyArray1::from_vec(py, f.right_ne_mask.clone()).into_any(),
            0_i8.into_pyobject(py)?.into_any(),
        ],
    )?)?;
    Ok(predicates)
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("non_equi_regions");
    let shapes = [
        ("narrow", 0_usize),
        ("broad", 1),
        ("mixed", 2),
        ("multi_groups", 3),
    ];
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
            let with_bounds_allocation =
                support::count_allocations(|| multi_reference_all_two_pass(&f, true));
            let without_bounds_allocation =
                support::count_allocations(|| multi_reference_all_two_pass(&f, false));
            eprintln!(
                "{label} reference multi all with_bounds {} bytes/{} allocs/{} peak; without_bounds {} bytes/{} allocs/{} peak",
                with_bounds_allocation.0,
                with_bounds_allocation.1,
                with_bounds_allocation.2,
                without_bounds_allocation.0,
                without_bounds_allocation.1,
                without_bounds_allocation.2,
            );
            Python::attach(|py| {
                let py_fixture = PyFixture::new(py, &f).unwrap();
                for selection in [
                    Selection::First,
                    Selection::Last,
                    Selection::Any,
                    Selection::All,
                ] {
                    // Validate semantics before entering Criterion's timed
                    // closures. A fast result is not useful if it differs
                    // from the direct reference implementation.
                    let expected_dual = dual_reference(&f, selection);
                    let proposed_dual = dual_output(py, &py_fixture, selection).unwrap();
                    assert_eq!(proposed_dual, expected_dual);

                    let expected_multi = multi_reference(&f, selection);
                    let proposed_multi = multi_output(py, &py_fixture, selection).unwrap();
                    assert_eq!(proposed_multi, expected_multi);
                    if matches!(selection, Selection::All) {
                        assert_eq!(multi_reference_all_two_pass(&f, true), expected_multi);
                        assert_eq!(multi_reference_all_two_pass(&f, false), expected_multi);
                    }
                }
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
                    eprintln!(
                        "{label} {selection_name}: dual {} bytes/{} allocs/{} peak; multi {} bytes/{} allocs/{} peak",
                        dual_allocation.0,
                        dual_allocation.1,
                        dual_allocation.2,
                        multi_allocation.0,
                        multi_allocation.1,
                        multi_allocation.2,
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
                let id = BenchmarkId::new(format!("multi/{selection_name}"), &label);
                group.bench_function(id, |b| {
                    b.iter(|| {
                        black_box(multi_reference(black_box(&f), selection));
                    })
                });
                if matches!(selection, Selection::All) {
                    for (name, narrow_second_pass) in
                        [("with_bounds", true), ("without_bounds", false)]
                    {
                        let id = BenchmarkId::new(format!("reference/multi/all/{name}"), &label);
                        group.bench_function(id, |b| {
                            b.iter(|| {
                                black_box(multi_reference_all_two_pass(
                                    black_box(&f),
                                    narrow_second_pass,
                                ));
                            })
                        });
                    }
                }
                let id = BenchmarkId::new(format!("wrapper/multi/{selection_name}"), &label);
                Python::attach(|py| {
                    let py_fixture = PyFixture::new(py, &f).unwrap();
                    group.bench_function(id, |b| {
                        b.iter(|| run_multi(py, &py_fixture, selection));
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
