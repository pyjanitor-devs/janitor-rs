//! Experiments for replacing label-keyed match aggregation with ordinal state.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numpy::ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use std::hint::black_box;
use std::time::Duration;

mod support;
use janitor_rs::bench_support::{
    compute_max_rev_end_match_int64, compute_min_rev_end_match_int64,
    compute_prod_rev_end_match_int64, compute_size_rev_end_matches,
    compute_sum_rev_end_match_int64, max_rev_end_match_core,
};
use support::count_allocations;

type Output = Vec<(i64, i64)>;

#[derive(Clone)]
struct Fixture {
    arr: Vec<i64>,
    ends: Vec<usize>,
    index: Vec<i64>,
    matches: Vec<u8>,
    counts: Vec<i64>,
}

struct ProductionFixture {
    arr: Array1<i64>,
    ends: Array1<i64>,
    index: Array1<i64>,
    counts: Array1<i64>,
    matches: Array1<i8>,
    booleans: Array1<bool>,
}

impl ProductionFixture {
    fn from_fixture(f: &Fixture) -> Self {
        Self {
            arr: Array1::from_vec(f.arr.clone()),
            ends: Array1::from_vec(f.ends.iter().map(|&value| value as i64).collect()),
            index: Array1::from_vec(f.index.clone()),
            counts: Array1::from_vec(f.counts.clone()),
            matches: Array1::from_vec(f.matches.iter().map(|&value| value as i8).collect()),
            booleans: Array1::from_elem(f.arr.len(), false),
        }
    }
}

impl Fixture {
    fn new(rows: usize, domain: usize, width: usize, survivor_percent: usize) -> Self {
        let width = width.min(domain);
        let mut matches = Vec::with_capacity(rows.saturating_mul(width));
        let mut counts = Vec::with_capacity(rows);
        for _ in 0..rows {
            let survivors = (width * survivor_percent / 100).max(usize::from(width > 0));
            counts.push(survivors as i64);
            matches.extend((0..width).map(|item| u8::from(item < survivors)));
        }
        Fixture {
            arr: (0..rows).map(|row| (rows - row) as i64).collect(),
            ends: vec![width; rows],
            index: (0..domain).map(|item| (item as i64) * 10 + 7).collect(),
            matches,
            counts,
        }
    }
}

/// Baseline: the current implementation, with one map for the winning row
/// and a second map for the winning value.
fn old_two_maps(f: &Fixture) -> Output {
    use std::collections::HashMap;
    let mut rows: HashMap<i64, i64> = HashMap::with_capacity(f.index.len());
    let mut values: HashMap<i64, i64> = HashMap::with_capacity(f.index.len());
    let mut tape = 0;
    for (row, ((&current, &end), &count)) in f
        .arr
        .iter()
        .zip(f.ends.iter())
        .zip(f.counts.iter())
        .enumerate()
    {
        for item in 0..end {
            if f.matches[tape] == 0 {
                tape += 1;
                continue;
            }
            let label = f.index[item];
            let best_row = rows.entry(label).or_insert(-1);
            let best_value = values.entry(label).or_insert(current);
            if count != 0 && (*best_row == -1 || current > *best_value) {
                *best_value = current;
                *best_row = row as i64;
            }
            tape += 1;
        }
    }
    rows.into_iter().collect()
}

/// One sparse map, keyed by ordinal position, with row and value in one state.
fn one_sparse_map(f: &Fixture) -> Output {
    use std::collections::HashMap;
    // Reserve against the covered prefix, not the full right domain. This is
    // the important sparse case: `index.len()` may be very large while `end`
    // remains a small prefix.
    let capacity = f.ends.iter().copied().max().unwrap_or(0);
    let mut states: HashMap<usize, (i64, i64)> = HashMap::with_capacity(capacity);
    let mut tape = 0;
    for (row, ((&current, &end), &count)) in f
        .arr
        .iter()
        .zip(f.ends.iter())
        .zip(f.counts.iter())
        .enumerate()
    {
        for item in 0..end {
            if f.matches[tape] == 0 {
                tape += 1;
                continue;
            }
            let state = states.entry(item).or_insert((current, -1));
            if count != 0 && (state.1 == -1 || current > state.0) {
                *state = (current, row as i64);
            }
            tape += 1;
        }
    }
    states
        .into_iter()
        .map(|(item, (_, row))| (f.index[item], row))
        .collect()
}

/// Dense ordinal state: one `seen` bit and one state tuple per right position.
fn dense_ordinal(f: &Fixture) -> Output {
    let mut seen = vec![false; f.index.len()];
    let mut states = vec![(0_i64, -1_i64); f.index.len()];
    let mut touched = Vec::with_capacity(f.index.len());
    let mut tape = 0;
    for (row, ((&current, &end), &count)) in f
        .arr
        .iter()
        .zip(f.ends.iter())
        .zip(f.counts.iter())
        .enumerate()
    {
        for item in 0..end {
            if f.matches[tape] == 0 {
                tape += 1;
                continue;
            }
            if !seen[item] {
                seen[item] = true;
                touched.push(item);
            }
            let state = &mut states[item];
            if count != 0 && (state.1 == -1 || current > state.0) {
                *state = (current, row as i64);
            }
            tape += 1;
        }
    }
    touched
        .into_iter()
        .map(|item| (f.index[item], states[item].1))
        .collect()
}

fn normalize(mut output: Output) -> Output {
    output.sort_unstable();
    output
}

fn assert_equivalent(f: &Fixture) {
    let old = normalize(old_two_maps(f));
    assert_eq!(old, normalize(one_sparse_map(f)));
    assert_eq!(old, normalize(dense_ordinal(f)));
}

fn production_ordinal(f: &ProductionFixture) -> Output {
    let (labels, rows) = max_rev_end_match_core(
        f.arr.view(),
        f.index.view(),
        f.ends.view(),
        f.counts.view(),
        f.matches.view(),
        f.booleans.view(),
    )
    .expect("benchmark fixture must satisfy the match-tape contract");
    labels.into_iter().zip(rows).collect()
}

fn production_wrapper_end_matches(f: &Fixture, aggregation: &str) {
    Python::attach(|py| {
        let arr = PyArray1::from_vec(py, f.arr.clone());
        let index = PyArray1::from_vec(py, f.index.clone());
        let ends = PyArray1::from_vec(py, f.ends.iter().map(|&value| value as i64).collect());
        let counts = PyArray1::from_vec(py, f.counts.clone());
        let matches = PyArray1::from_vec(py, f.matches.iter().map(|&value| value as i8).collect());
        let booleans = PyArray1::from_vec(py, vec![false; f.arr.len()]);
        match aggregation {
            "max" => {
                let _ = compute_max_rev_end_match_int64(
                    py,
                    arr.readonly(),
                    index.readonly(),
                    ends.readonly(),
                    counts.readonly(),
                    matches.readonly(),
                    booleans.readonly(),
                )
                .unwrap();
            }
            "min" => {
                let _ = compute_min_rev_end_match_int64(
                    py,
                    arr.readonly(),
                    index.readonly(),
                    ends.readonly(),
                    counts.readonly(),
                    matches.readonly(),
                    booleans.readonly(),
                )
                .unwrap();
            }
            "sum" => {
                let _ = compute_sum_rev_end_match_int64(
                    py,
                    arr.readonly(),
                    index.readonly(),
                    ends.readonly(),
                    counts.readonly(),
                    matches.readonly(),
                    booleans.readonly(),
                )
                .unwrap();
            }
            "prod" => {
                let _ = compute_prod_rev_end_match_int64(
                    py,
                    arr.readonly(),
                    index.readonly(),
                    ends.readonly(),
                    counts.readonly(),
                    matches.readonly(),
                    booleans.readonly(),
                )
                .unwrap();
            }
            "size" => {
                let _ = compute_size_rev_end_matches(
                    py,
                    ends.readonly(),
                    index.readonly(),
                    matches.readonly(),
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
    });
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    eprintln!("\\nmatches ordinal allocation report (bytes / allocs / peak-live):");
    for &(rows, domain, width, survivor_percent) in &[
        (100, 1_000, 1, 100),
        (100, 1_000, 64, 100),
        (1_000, 10_000, 8, 100),
        (1_000, 10_000, 1_000, 100),
        (10_000, 100_000, 8, 100),
        (10_000, 100_000, 10_000, 100),
        (10_000, 100_000, 100_000, 100),
        (1_000, 1_000_000, 8, 100),
    ] {
        let fixture = Fixture::new(rows, domain, width, survivor_percent);
        let label = format!("rows={rows}/domain={domain}/width={width}");
        let old = count_allocations(|| old_two_maps(&fixture));
        let sparse = count_allocations(|| one_sparse_map(&fixture));
        let dense = count_allocations(|| dense_ordinal(&fixture));
        eprintln!("  {label}");
        eprintln!("    two maps:       {old:?}");
        eprintln!("    sparse ordinal: {sparse:?}");
        eprintln!("    dense ordinal:  {dense:?}");
    }

    let mut group = c.benchmark_group("matches_ordinal_state");
    group.measurement_time(Duration::from_secs(2));
    for &(rows, domain, width) in &[
        (100, 1_000, 1),
        (100, 1_000, 8),
        (100, 1_000, 64),
        (1_000, 10_000, 8),
        (1_000, 10_000, 1_000),
        (10_000, 100_000, 8),
        (10_000, 100_000, 10_000),
        (1_000, 10_000, 10_000),
        (1_000, 1_000_000, 8),
    ] {
        for &survivor_percent in &[1, 50, 100] {
            let fixture = Fixture::new(rows, domain, width, survivor_percent);
            let production_fixture = ProductionFixture::from_fixture(&fixture);
            assert_equivalent(&fixture);
            assert_eq!(
                normalize(old_two_maps(&fixture)),
                normalize(production_ordinal(&production_fixture))
            );
            let label =
                format!("rows={rows}/domain={domain}/width={width}/survivors={survivor_percent}%");
            group.bench_with_input(BenchmarkId::new("two_maps", &label), &fixture, |b, f| {
                b.iter(|| black_box(old_two_maps(black_box(f))))
            });
            group.bench_with_input(
                BenchmarkId::new("one_sparse_map", &label),
                &fixture,
                |b, f| b.iter(|| black_box(one_sparse_map(black_box(f)))),
            );
            group.bench_with_input(
                BenchmarkId::new("dense_ordinal", &label),
                &fixture,
                |b, f| b.iter(|| black_box(dense_ordinal(black_box(f)))),
            );
            group.bench_with_input(
                BenchmarkId::new("production_ordinal", &label),
                &production_fixture,
                |b, f| b.iter(|| black_box(production_ordinal(black_box(f)))),
            );
        }
    }
    group.finish();

    let fixture = Fixture::new(1_000, 10_000, 1_000, 50);
    let mut wrappers = c.benchmark_group("production_wrapper_end_matches");
    wrappers.measurement_time(Duration::from_secs(2));
    for aggregation in ["max", "min", "sum", "prod", "size"] {
        wrappers.bench_function(aggregation, |b| {
            b.iter(|| production_wrapper_end_matches(black_box(&fixture), aggregation))
        });
    }
    wrappers.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
