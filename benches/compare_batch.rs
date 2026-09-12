//! Benchmark the fused comparison-to-index kernel.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::{
    compare_batch_indices_all, compare_batch_indices_any, compare_batch_indices_first,
    compare_batch_indices_last,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

#[path = "support/mod.rs"]
mod support;

fn baseline_indices(
    left: &[i64],
    right: &[i64],
    right_labels: &[i64],
    first: bool,
) -> Option<(Vec<i64>, Vec<i64>)> {
    let mut matches = vec![false; left.len() * right.len()];
    for (left_position, left_value) in left.iter().enumerate() {
        for (right_position, right_value) in right.iter().enumerate() {
            // Keep this baseline deliberately close to the existing
            // multi-condition tape approach: evaluate every candidate,
            // retain the full match tape, then reduce it to one result per
            // left row and finally build the output indices.
            matches[left_position * right.len() + right_position] =
                *left_value > *right_value && *left_value != *right_value;
        }
    }
    let mut output_left = Vec::new();
    let mut output_right = Vec::new();
    for (left_position, _) in left.iter().enumerate() {
        let row = &matches[left_position * right.len()..(left_position + 1) * right.len()];
        let mut selected = None;
        for (right_position, matched) in row.iter().enumerate() {
            if *matched
                && selected.is_none_or(|current| {
                    (first && right_labels[right_position] < right_labels[current])
                        || (!first && right_labels[right_position] > right_labels[current])
                })
            {
                selected = Some(right_position);
            }
        }
        if let Some(right_position) = selected {
            output_left.push(left_position as i64);
            output_right.push(right_labels[right_position]);
        }
    }
    (!output_left.is_empty()).then_some((output_left, output_right))
}

fn baseline_all(left: &[i64], right: &[i64], right_labels: &[i64]) -> Option<(Vec<i64>, Vec<i64>)> {
    let mut matches = vec![false; left.len() * right.len()];
    for (left_position, left_value) in left.iter().enumerate() {
        for (right_position, right_value) in right.iter().enumerate() {
            matches[left_position * right.len() + right_position] =
                *left_value > *right_value && *left_value != *right_value;
        }
    }
    let mut output_left = Vec::new();
    let mut output_right = Vec::new();
    for (position, matched) in matches.into_iter().enumerate() {
        if matched {
            output_left.push((position / right.len()) as i64);
            output_right.push(right_labels[position % right.len()]);
        }
    }
    (!output_left.is_empty()).then_some((output_left, output_right))
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("compare_batch_indices");
    for &(left_len, right_len) in &[
        (8, 16),
        (64, 256),
        (256, 1_024),
        (512, 16_384),
        (2_048, 8_192),
        (4_096, 32_768),
        (8_192, 32_768),
    ] {
        let label = format!("left={left_len}/right={right_len}");
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, (0..left_len).map(|value| value as i64).collect());
            let right = PyArray1::from_vec(py, (0..right_len).map(|value| value as i64).collect());
            let left_float =
                PyArray1::from_vec(py, (0..left_len).map(|value| value as f64 + 0.5).collect());
            let right_float =
                PyArray1::from_vec(py, (0..right_len).map(|value| value as f64).collect());
            let left_index =
                PyArray1::from_vec(py, (0..left_len).map(|value| value as i64).collect());
            let right_index =
                PyArray1::from_vec(py, (0..right_len).rev().map(|value| value as i64).collect());
            let left_values: Vec<i64> = (0..left_len).map(|value| value as i64).collect();
            let right_values: Vec<i64> = (0..right_len).map(|value| value as i64).collect();
            let right_labels: Vec<i64> = (0..right_len).rev().map(|value| value as i64).collect();
            let predicates = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            left.clone().into_any(),
                            right.clone().into_any(),
                            0_i8.into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left_float.into_any(),
                            right_float.into_any(),
                            0_i8.into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let false_left = PyArray1::from_vec(py, vec![false; left_len]);
            let false_right = PyArray1::from_vec(py, vec![false; right_len]);
            let mixed_predicates = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            left.clone().into_any(),
                            right.clone().into_any(),
                            0_i8.into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left.clone().into_any(),
                            right.clone().into_any(),
                            5_i8.into_pyobject(py)?.into_any(),
                            false_left.into_any(),
                            false_right.into_any(),
                            0_i8.into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;

            let (baseline_bytes, _, baseline_peak) = support::count_allocations(|| {
                baseline_indices(&left_values, &right_values, &right_labels, true)
            });
            let (baseline_all_bytes, _, baseline_all_peak) = support::count_allocations(|| {
                baseline_all(&left_values, &right_values, &right_labels)
            });
            let (current_all_bytes, _, current_all_peak) = support::count_allocations(|| {
                let starts = PyArray1::from_vec(py, vec![0_i64; left_len]);
                let ends = PyArray1::from_vec(py, vec![right_len as i64; left_len]);
                compare_batch_indices_all(
                    py,
                    &predicates,
                    Some(starts.readonly()),
                    Some(ends.readonly()),
                    left_index.clone(),
                    right_index.readonly(),
                )
                .unwrap()
            });
            let (last_bytes, _, last_peak) = support::count_allocations(|| {
                let starts = PyArray1::from_vec(py, vec![0_i64; left_len]);
                let ends = PyArray1::from_vec(py, vec![right_len as i64; left_len]);
                compare_batch_indices_last(
                    py,
                    &predicates,
                    Some(starts.readonly()),
                    Some(ends.readonly()),
                    left_index.clone(),
                    right_index.readonly(),
                )
                .unwrap()
            });
            eprintln!(
                "compare_batch {label}: current-all {current_all_bytes} bytes/{current_all_peak} peak; vector-baseline-all {baseline_all_bytes} bytes/{baseline_all_peak} peak; first baseline {baseline_bytes} bytes/{baseline_peak} peak; last {last_bytes} bytes/{last_peak} peak"
            );

            group.bench_with_input(BenchmarkId::new("ordinary", &label), &label, |b, _| {
                b.iter(|| {
                    let starts = PyArray1::from_vec(py, vec![0_i64; left_len]);
                    let ends = PyArray1::from_vec(py, vec![right_len as i64; left_len]);
                    compare_batch_indices_first(
                        py,
                        &predicates,
                        Some(starts.readonly()),
                        Some(ends.readonly()),
                        left_index.clone(),
                        right_index.readonly(),
                    )
                    .unwrap();
                })
            });
            group.bench_with_input(BenchmarkId::new("mixed", &label), &label, |b, _| {
                b.iter(|| {
                    let starts = PyArray1::from_vec(py, vec![0_i64; left_len]);
                    let ends = PyArray1::from_vec(py, vec![right_len as i64; left_len]);
                    compare_batch_indices_first(
                        py,
                        &mixed_predicates,
                        Some(starts.readonly()),
                        Some(ends.readonly()),
                        left_index.clone(),
                        right_index.readonly(),
                    )
                    .unwrap();
                })
            });
            group.bench_with_input(BenchmarkId::new("last", &label), &label, |b, _| {
                b.iter(|| {
                    let starts = PyArray1::from_vec(py, vec![0_i64; left_len]);
                    let ends = PyArray1::from_vec(py, vec![right_len as i64; left_len]);
                    compare_batch_indices_last(
                        py,
                        &predicates,
                        Some(starts.readonly()),
                        Some(ends.readonly()),
                        left_index.clone(),
                        right_index.readonly(),
                    )
                    .unwrap();
                })
            });
            group.bench_with_input(BenchmarkId::new("any", &label), &label, |b, _| {
                b.iter(|| {
                    let starts = PyArray1::from_vec(py, vec![0_i64; left_len]);
                    let ends = PyArray1::from_vec(py, vec![right_len as i64; left_len]);
                    compare_batch_indices_any(
                        py,
                        &predicates,
                        Some(starts.readonly()),
                        Some(ends.readonly()),
                        left_index.clone(),
                        right_index.readonly(),
                    )
                    .unwrap();
                })
            });
            group.bench_with_input(BenchmarkId::new("all", &label), &label, |b, _| {
                b.iter(|| {
                    let starts = PyArray1::from_vec(py, vec![0_i64; left_len]);
                    let ends = PyArray1::from_vec(py, vec![right_len as i64; left_len]);
                    compare_batch_indices_all(
                        py,
                        &predicates,
                        Some(starts.readonly()),
                        Some(ends.readonly()),
                        left_index.clone(),
                        right_index.readonly(),
                    )
                    .unwrap();
                })
            });
            group.bench_with_input(BenchmarkId::new("vector_all", &label), &label, |b, _| {
                b.iter(|| {
                    std::hint::black_box(baseline_all(&left_values, &right_values, &right_labels));
                })
            });
            group.bench_with_input(
                BenchmarkId::new("baseline_first", &label),
                &label,
                |b, _| {
                    b.iter(|| {
                        std::hint::black_box(baseline_indices(
                            &left_values,
                            &right_values,
                            &right_labels,
                            true,
                        ));
                    })
                },
            );
            group.bench_with_input(BenchmarkId::new("baseline_last", &label), &label, |b, _| {
                b.iter(|| {
                    std::hint::black_box(baseline_indices(
                        &left_values,
                        &right_values,
                        &right_labels,
                        false,
                    ));
                })
            });
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
