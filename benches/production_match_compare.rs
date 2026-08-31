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

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("production_match_wrappers");
    group.measurement_time(Duration::from_millis(500));
    for &(rows, domain, width, survivors) in &[
        (100, 1_000, 1, 100),
        (100, 1_000, 64, 50),
        (1_000, 10_000, 8, 1),
        (1_000, 10_000, 1_000, 50),
        (10_000, 100_000, 8, 100),
        (10_000, 100_000, 10_000, 50),
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
}

criterion_group!(benches, bench);
criterion_main!(benches);
