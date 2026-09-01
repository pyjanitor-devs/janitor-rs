use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::{
    compute_max_rev_end_match_int64, compute_min_rev_end_match_int64,
    compute_prod_rev_end_match_int64, compute_size_rev_end_matches,
    compute_sum_rev_end_match_int64,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use std::time::Duration;

struct Fixture {
    arr: Vec<i64>,
    ends: Vec<i64>,
    index: Vec<i64>,
    counts: Vec<i64>,
    matches: Vec<i8>,
}

struct Prepared<'py> {
    arr: Bound<'py, PyArray1<i64>>,
    index: Bound<'py, PyArray1<i64>>,
    ends: Bound<'py, PyArray1<i64>>,
    counts: Bound<'py, PyArray1<i64>>,
    matches: Bound<'py, PyArray1<i8>>,
    booleans: Bound<'py, PyArray1<bool>>,
}

impl Fixture {
    fn new(rows: usize, domain: usize, width: usize, survivors: usize) -> Self {
        let width = width.min(domain);
        let mut matches = Vec::with_capacity(rows * width);
        let mut counts = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut count = 0;
            for item in 0..width {
                let live = (row + item) % 100 < survivors;
                matches.push(i8::from(live));
                count += i64::from(live);
            }
            counts.push(count);
        }
        Self {
            arr: (0..rows).map(|row| row as i64).collect(),
            ends: vec![width as i64; rows],
            index: (0..domain).map(|item| (item * 10 + 7) as i64).collect(),
            counts,
            matches,
        }
    }
}

fn call(f: &Fixture, aggregation: &str) {
    Python::attach(|py| {
        let arr = PyArray1::from_vec(py, f.arr.clone());
        let index = PyArray1::from_vec(py, f.index.clone());
        let ends = PyArray1::from_vec(py, f.ends.clone());
        let counts = PyArray1::from_vec(py, f.counts.clone());
        let matches = PyArray1::from_vec(py, f.matches.clone());
        let booleans = PyArray1::from_vec(py, vec![false; f.arr.len()]);
        match aggregation {
            "max" => compute_max_rev_end_match_int64(
                py,
                arr.readonly(),
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap(),
            "min" => compute_min_rev_end_match_int64(
                py,
                arr.readonly(),
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap(),
            "sum" => compute_sum_rev_end_match_int64(
                py,
                arr.readonly(),
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap(),
            "prod" => compute_prod_rev_end_match_int64(
                py,
                arr.readonly(),
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap(),
            "size" => compute_size_rev_end_matches(
                py,
                ends.readonly(),
                index.readonly(),
                matches.readonly(),
            )
            .unwrap(),
            _ => unreachable!(),
        };
    });
}

fn call_prepared<'py>(py: Python<'py>, f: &Fixture, aggregation: &str, prepared: &Prepared<'py>) {
    let _ = f;
    match aggregation {
        "max" => compute_max_rev_end_match_int64(
            py,
            prepared.arr.readonly(),
            prepared.index.readonly(),
            prepared.ends.readonly(),
            prepared.counts.readonly(),
            prepared.matches.readonly(),
            prepared.booleans.readonly(),
        )
        .unwrap(),
        "min" => compute_min_rev_end_match_int64(
            py,
            prepared.arr.readonly(),
            prepared.index.readonly(),
            prepared.ends.readonly(),
            prepared.counts.readonly(),
            prepared.matches.readonly(),
            prepared.booleans.readonly(),
        )
        .unwrap(),
        "sum" => compute_sum_rev_end_match_int64(
            py,
            prepared.arr.readonly(),
            prepared.index.readonly(),
            prepared.ends.readonly(),
            prepared.counts.readonly(),
            prepared.matches.readonly(),
            prepared.booleans.readonly(),
        )
        .unwrap(),
        "prod" => compute_prod_rev_end_match_int64(
            py,
            prepared.arr.readonly(),
            prepared.index.readonly(),
            prepared.ends.readonly(),
            prepared.counts.readonly(),
            prepared.matches.readonly(),
            prepared.booleans.readonly(),
        )
        .unwrap(),
        "size" => compute_size_rev_end_matches(
            py,
            prepared.ends.readonly(),
            prepared.index.readonly(),
            prepared.matches.readonly(),
        )
        .unwrap(),
        _ => unreachable!(),
    };
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("production_match_wrappers");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(500));
    for &(rows, domain, width, survivors) in &[
        (100, 1_000, 1, 100),
        (100, 1_000, 64, 50),
        (1_000, 10_000, 8, 1),
        (1_000, 10_000, 1_000, 50),
        // Many rows over a narrow prefix: sparse despite the large batch.
        (10_000, 100_000, 8, 100),
        (10_000, 100_000, 10_000, 50),
        // The whole positional domain is covered: dense storage is selected.
        (1_000, 10_000, 10_000, 100),
        (1_000, 1_000_000, 8, 100),
    ] {
        let fixture = Fixture::new(rows, domain, width, survivors);
        let label = format!("rows={rows}/domain={domain}/width={width}/survivors={survivors}%");
        for aggregation in ["max", "min", "sum", "prod", "size"] {
            group.bench_with_input(
                BenchmarkId::new(aggregation, &label),
                &fixture,
                |b, fixture| b.iter(|| call(fixture, aggregation)),
            );
        }
    }
    group.finish();

    let mut prepared_group = c.benchmark_group("production_match_prebuilt_inputs");
    prepared_group.sample_size(10);
    prepared_group.warm_up_time(Duration::from_millis(200));
    prepared_group.measurement_time(Duration::from_millis(500));
    for &(rows, domain, width, survivors) in &[
        (100, 1_000, 1, 100),
        (100, 1_000, 64, 50),
        (1_000, 10_000, 8, 1),
        (1_000, 10_000, 1_000, 50),
        // Many rows over a narrow prefix: sparse despite the large batch.
        (10_000, 100_000, 8, 100),
        (10_000, 100_000, 10_000, 50),
        // The whole positional domain is covered: dense storage is selected.
        (1_000, 10_000, 10_000, 100),
        (1_000, 1_000_000, 8, 100),
    ] {
        let fixture = Fixture::new(rows, domain, width, survivors);
        let label = format!("rows={rows}/domain={domain}/width={width}/survivors={survivors}%");
        Python::attach(|py| {
            let prepared = Prepared {
                arr: PyArray1::from_vec(py, fixture.arr.clone()),
                index: PyArray1::from_vec(py, fixture.index.clone()),
                ends: PyArray1::from_vec(py, fixture.ends.clone()),
                counts: PyArray1::from_vec(py, fixture.counts.clone()),
                matches: PyArray1::from_vec(py, fixture.matches.clone()),
                booleans: PyArray1::from_vec(py, vec![false; fixture.arr.len()]),
            };
            for aggregation in ["max", "min", "sum", "prod", "size"] {
                prepared_group.bench_with_input(
                    BenchmarkId::new(aggregation, &label),
                    &fixture,
                    |b, fixture| b.iter(|| call_prepared(py, fixture, aggregation, &prepared)),
                );
            }
        });
    }
    prepared_group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
