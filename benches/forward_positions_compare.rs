//! Forward `_positions` direct-loop versus predecoded-ordinal experiments.
//!
//! The predecoded candidate is intentionally benchmarked end-to-end: it pays
//! to validate and convert every `positions` entry before answering ranges.
//! This shows where that extra pass is repaid by avoiding `checked_index` in
//! repeated/broad queries.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::{
    forward_max_positions_core, forward_min_positions_core, prod_positions_core, sum_positions_core,
};
use numpy::ndarray::Array1;
use std::hint::black_box;

struct Fixture {
    arr: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
    positions: Array1<i64>,
    booleans: Array1<bool>,
}

impl Fixture {
    fn new(rows: usize, broad: bool, duplicate: bool) -> Self {
        const WIDTH: usize = 8;
        let position_len = rows * WIDTH;
        let query_count = if broad { rows.min(256) } else { rows };
        let starts = if broad {
            Array1::zeros(query_count)
        } else {
            Array1::from_iter((0..query_count).map(|row| (row * WIDTH) as i64))
        };
        let ends = if broad {
            Array1::from_elem(query_count, position_len as i64)
        } else {
            Array1::from_iter((0..query_count).map(|row| (row * WIDTH + 1) as i64))
        };
        let positions = Array1::from_iter((0..position_len).map(|slot| {
            if duplicate {
                0
            } else {
                (slot % rows.max(1)) as i64
            }
        }));
        Self {
            arr: Array1::from_iter((0..rows).map(|row| (rows - row) as i64)),
            starts,
            ends,
            positions,
            booleans: Array1::from_elem(rows, false),
        }
    }
}

fn decode_positions(f: &Fixture) -> Vec<usize> {
    f.positions
        .iter()
        .map(|position| {
            usize::try_from(*position)
                .ok()
                .filter(|&position| position < f.arr.len())
                .unwrap_or(usize::MAX)
        })
        .collect()
}

fn predecoded_sum(f: &Fixture) -> Array1<i64> {
    let positions = decode_positions(f);
    let mut result = Array1::<i64>::zeros(f.starts.len());
    for (output, (start, end)) in f.starts.iter().zip(f.ends.iter()).enumerate() {
        let Ok(start) = usize::try_from(*start) else {
            continue;
        };
        let Ok(end) = usize::try_from(*end) else {
            continue;
        };
        if start >= end || end > positions.len() {
            continue;
        }
        for &position in &positions[start..end] {
            if position != usize::MAX && !f.booleans[position] {
                result[output] = result[output].wrapping_add(f.arr[position]);
            }
        }
    }
    result
}

fn predecoded_prod(f: &Fixture) -> Array1<i64> {
    let positions = decode_positions(f);
    let mut result = Array1::<i64>::from_elem(f.starts.len(), 1);
    for (output, (start, end)) in f.starts.iter().zip(f.ends.iter()).enumerate() {
        let Ok(start) = usize::try_from(*start) else {
            continue;
        };
        let Ok(end) = usize::try_from(*end) else {
            continue;
        };
        if start >= end || end > positions.len() {
            continue;
        }
        for &position in &positions[start..end] {
            if position != usize::MAX && !f.booleans[position] {
                result[output] = result[output].wrapping_mul(f.arr[position]);
            }
        }
    }
    result
}

fn predecoded_min(f: &Fixture) -> Array1<i64> {
    let positions = decode_positions(f);
    let mut result = Array1::<i64>::from_elem(f.starts.len(), -1);
    for (output, (start, end)) in f.starts.iter().zip(f.ends.iter()).enumerate() {
        let Ok(start) = usize::try_from(*start) else {
            continue;
        };
        let Ok(end) = usize::try_from(*end) else {
            continue;
        };
        if start >= end || end > positions.len() {
            continue;
        }
        let mut best = -1_i64;
        for &position in &positions[start..end] {
            if position == usize::MAX || f.booleans[position] {
                continue;
            }
            if best == -1 || f.arr[position] < f.arr[best as usize] {
                best = position as i64;
            }
        }
        result[output] = best;
    }
    result
}

fn predecoded_max(f: &Fixture) -> Array1<i64> {
    let positions = decode_positions(f);
    let mut result = Array1::<i64>::from_elem(f.starts.len(), -1);
    for (output, (start, end)) in f.starts.iter().zip(f.ends.iter()).enumerate() {
        let Ok(start) = usize::try_from(*start) else {
            continue;
        };
        let Ok(end) = usize::try_from(*end) else {
            continue;
        };
        if start >= end || end > positions.len() {
            continue;
        }
        let mut best = -1_i64;
        for &position in &positions[start..end] {
            if position == usize::MAX || f.booleans[position] {
                continue;
            }
            if best == -1 || f.arr[position] > f.arr[best as usize] {
                best = position as i64;
            }
        }
        result[output] = best;
    }
    result
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_positions_direct_vs_predecoded");
    for rows in [32, 1_000, 10_000] {
        for broad in [false, true] {
            for duplicate in [false, true] {
                let fixture = Fixture::new(rows, broad, duplicate);
                let shape = if broad { "broad" } else { "narrow" };
                let distribution = if duplicate { "duplicate" } else { "scattered" };
                let label = format!("rows={rows}/{shape}/{distribution}");

                assert_eq!(
                    sum_positions_core(
                        fixture.arr.view(),
                        fixture.starts.view(),
                        fixture.ends.view(),
                        fixture.positions.view(),
                        fixture.booleans.view(),
                    ),
                    predecoded_sum(&fixture),
                    "sum candidate must preserve direct results"
                );
                assert_eq!(
                    prod_positions_core(
                        fixture.arr.view(),
                        fixture.starts.view(),
                        fixture.ends.view(),
                        fixture.positions.view(),
                        fixture.booleans.view(),
                    ),
                    predecoded_prod(&fixture),
                    "product candidate must preserve direct results"
                );
                assert_eq!(
                    forward_min_positions_core(
                        fixture.arr.view(),
                        fixture.starts.view(),
                        fixture.ends.view(),
                        fixture.positions.view(),
                        fixture.booleans.view(),
                    ),
                    predecoded_min(&fixture),
                    "min candidate must preserve direct results"
                );
                assert_eq!(
                    forward_max_positions_core(
                        fixture.arr.view(),
                        fixture.starts.view(),
                        fixture.ends.view(),
                        fixture.positions.view(),
                        fixture.booleans.view(),
                    ),
                    predecoded_max(&fixture),
                    "max candidate must preserve direct results"
                );

                group.bench_with_input(BenchmarkId::new("sum_direct", &label), &fixture, |b, f| {
                    b.iter(|| {
                        black_box(sum_positions_core(
                            black_box(f.arr.view()),
                            black_box(f.starts.view()),
                            black_box(f.ends.view()),
                            black_box(f.positions.view()),
                            black_box(f.booleans.view()),
                        ))
                    })
                });
                group.bench_with_input(
                    BenchmarkId::new("sum_predecoded", &label),
                    &fixture,
                    |b, f| b.iter(|| black_box(predecoded_sum(black_box(f)))),
                );
                group.bench_with_input(
                    BenchmarkId::new("prod_direct", &label),
                    &fixture,
                    |b, f| {
                        b.iter(|| {
                            black_box(prod_positions_core(
                                black_box(f.arr.view()),
                                black_box(f.starts.view()),
                                black_box(f.ends.view()),
                                black_box(f.positions.view()),
                                black_box(f.booleans.view()),
                            ))
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("prod_predecoded", &label),
                    &fixture,
                    |b, f| b.iter(|| black_box(predecoded_prod(black_box(f)))),
                );
                group.bench_with_input(BenchmarkId::new("min_direct", &label), &fixture, |b, f| {
                    b.iter(|| {
                        black_box(forward_min_positions_core(
                            black_box(f.arr.view()),
                            black_box(f.starts.view()),
                            black_box(f.ends.view()),
                            black_box(f.positions.view()),
                            black_box(f.booleans.view()),
                        ))
                    })
                });
                group.bench_with_input(
                    BenchmarkId::new("min_predecoded", &label),
                    &fixture,
                    |b, f| b.iter(|| black_box(predecoded_min(black_box(f)))),
                );
                group.bench_with_input(BenchmarkId::new("max_direct", &label), &fixture, |b, f| {
                    b.iter(|| {
                        black_box(forward_max_positions_core(
                            black_box(f.arr.view()),
                            black_box(f.starts.view()),
                            black_box(f.ends.view()),
                            black_box(f.positions.view()),
                            black_box(f.booleans.view()),
                        ))
                    })
                });
                group.bench_with_input(
                    BenchmarkId::new("max_predecoded", &label),
                    &fixture,
                    |b, f| b.iter(|| black_box(predecoded_max(black_box(f)))),
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
