//! Historical label-keyed versus sparse and dense ordinal-keyed reverse `_positions` aggregation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::{
    max_positions_core_with_storage, min_positions_core_with_storage,
    prod_positions_i64_with_storage, size_positions_core_with_storage,
    sum_positions_i64_with_storage,
};
use numpy::ndarray::Array1;
use std::collections::HashMap;
use std::hint::black_box;

struct Fixture {
    arr: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
    index: Array1<i64>,
    positions: Array1<i64>,
    booleans: Array1<bool>,
}

impl Fixture {
    fn new(rows: usize, duplicate_ordinals: bool) -> Self {
        const WIDTH: usize = 8;
        let domain = rows.max(WIDTH);
        Self {
            arr: Array1::from_iter((0..rows).map(|row| (rows - row) as i64)),
            starts: Array1::from_iter((0..rows).map(|row| (row * WIDTH) as i64)),
            ends: Array1::from_iter((0..rows).map(|row| ((row + 1) * WIDTH) as i64)),
            index: Array1::from_iter((0..domain).map(|ordinal| 10_000 + ordinal as i64 * 7)),
            positions: Array1::from_iter((0..rows).flat_map(|row| {
                (0..WIDTH).map(move |offset| {
                    if duplicate_ordinals {
                        0
                    } else {
                        ((row + offset) % domain) as i64
                    }
                })
            })),
            booleans: Array1::from_elem(rows, false),
        }
    }
}

fn valid_range(start: i64, end: i64, length: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(start).ok()?;
    let end = usize::try_from(end).ok()?;
    (start < end && end <= length).then_some((start, end))
}

fn old_max(f: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let capacity = f.index.len().min(f.positions.len());
    let mut slots = HashMap::<i64, usize>::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut best_positions = Vec::with_capacity(capacity);
    let mut best_values = Vec::with_capacity(capacity);
    for (row, (((current, start), end), boolean)) in f
        .arr
        .iter()
        .zip(f.starts.iter())
        .zip(f.ends.iter())
        .zip(f.booleans.iter())
        .enumerate()
    {
        let Some((start, end)) = valid_range(*start, *end, f.positions.len()) else {
            continue;
        };
        for tape_item in start..end {
            let Some(ordinal) = usize::try_from(f.positions[tape_item])
                .ok()
                .filter(|&ordinal| ordinal < f.index.len())
            else {
                continue;
            };
            let slot = match slots.entry(f.index[ordinal]) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(f.index[ordinal]);
                    best_positions.push(-1);
                    best_values.push(*current);
                    slot
                }
            };
            if !*boolean && (best_positions[slot] == -1 || *current > best_values[slot]) {
                best_values[slot] = *current;
                best_positions[slot] = row as i64;
            }
        }
    }
    (labels, best_positions)
}

fn old_min(f: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let capacity = f.index.len().min(f.positions.len());
    let mut slots = HashMap::<i64, usize>::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut best_positions = Vec::with_capacity(capacity);
    let mut best_values = Vec::with_capacity(capacity);
    for (row, (((current, start), end), boolean)) in f
        .arr
        .iter()
        .zip(f.starts.iter())
        .zip(f.ends.iter())
        .zip(f.booleans.iter())
        .enumerate()
    {
        let Some((start, end)) = valid_range(*start, *end, f.positions.len()) else {
            continue;
        };
        for tape_item in start..end {
            let Some(ordinal) = usize::try_from(f.positions[tape_item])
                .ok()
                .filter(|&ordinal| ordinal < f.index.len())
            else {
                continue;
            };
            let slot = match slots.entry(f.index[ordinal]) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(f.index[ordinal]);
                    best_positions.push(-1);
                    best_values.push(*current);
                    slot
                }
            };
            if !*boolean && (best_positions[slot] == -1 || *current < best_values[slot]) {
                best_values[slot] = *current;
                best_positions[slot] = row as i64;
            }
        }
    }
    (labels, best_positions)
}

fn old_sum(f: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let capacity = f.index.len().min(f.positions.len());
    let mut slots = HashMap::<i64, usize>::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut totals = Vec::with_capacity(capacity);
    for (((current, start), end), boolean) in f
        .arr
        .iter()
        .zip(f.starts.iter())
        .zip(f.ends.iter())
        .zip(f.booleans.iter())
    {
        let Some((start, end)) = valid_range(*start, *end, f.positions.len()) else {
            continue;
        };
        for tape_item in start..end {
            let Some(ordinal) = usize::try_from(f.positions[tape_item])
                .ok()
                .filter(|&ordinal| ordinal < f.index.len())
            else {
                continue;
            };
            let slot = match slots.entry(f.index[ordinal]) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(f.index[ordinal]);
                    totals.push(0_i64);
                    slot
                }
            };
            if !*boolean {
                totals[slot] = totals[slot].wrapping_add(*current);
            }
        }
    }
    (labels, totals)
}

fn old_prod(f: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let capacity = f.index.len().min(f.positions.len());
    let mut slots = HashMap::<i64, usize>::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut products = Vec::with_capacity(capacity);
    for (((current, start), end), boolean) in f
        .arr
        .iter()
        .zip(f.starts.iter())
        .zip(f.ends.iter())
        .zip(f.booleans.iter())
    {
        let Some((start, end)) = valid_range(*start, *end, f.positions.len()) else {
            continue;
        };
        for tape_item in start..end {
            let Some(ordinal) = usize::try_from(f.positions[tape_item])
                .ok()
                .filter(|&ordinal| ordinal < f.index.len())
            else {
                continue;
            };
            let slot = match slots.entry(f.index[ordinal]) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(f.index[ordinal]);
                    products.push(1_i64);
                    slot
                }
            };
            if !*boolean {
                products[slot] = products[slot].wrapping_mul(*current);
            }
        }
    }
    (labels, products)
}

fn old_size(f: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut counts = HashMap::<i64, i64>::with_capacity(f.index.len());
    for (start, end) in f.starts.iter().zip(f.ends.iter()) {
        let Some((start, end)) = valid_range(*start, *end, f.positions.len()) else {
            continue;
        };
        for tape_item in start..end {
            let Some(ordinal) = usize::try_from(f.positions[tape_item])
                .ok()
                .filter(|&ordinal| ordinal < f.index.len())
            else {
                continue;
            };
            *counts.entry(f.index[ordinal]).or_insert(0) += 1;
        }
    }
    counts.into_iter().unzip()
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("positions_ordinal_old_vs_new");
    for rows in [32, 10_000, 100_000, 1_000_000] {
        for dense in [false, true] {
            for duplicate_ordinals in [true, false] {
                let fixture = Fixture::new(rows, duplicate_ordinals);
                let kind = if duplicate_ordinals {
                    "duplicate"
                } else {
                    "unique"
                };
                let storage = if dense { "dense" } else { "hashmap" };
                let label = format!("rows={rows}/{kind}/{storage}");

                group.bench_with_input(BenchmarkId::new("max_old", &label), &fixture, |b, f| {
                    b.iter(|| black_box(old_max(black_box(f))))
                });
                group.bench_with_input(
                    BenchmarkId::new("max_ordinal", &label),
                    &fixture,
                    |b, f| {
                        b.iter(|| {
                            black_box(max_positions_core_with_storage(
                                black_box(f.arr.view()),
                                black_box(f.starts.view()),
                                black_box(f.ends.view()),
                                black_box(f.index.view()),
                                black_box(f.positions.view()),
                                black_box(f.booleans.view()),
                                dense,
                            ))
                        })
                    },
                );
                group.bench_with_input(BenchmarkId::new("min_old", &label), &fixture, |b, f| {
                    b.iter(|| black_box(old_min(black_box(f))))
                });
                group.bench_with_input(
                    BenchmarkId::new("min_ordinal", &label),
                    &fixture,
                    |b, f| {
                        b.iter(|| {
                            black_box(min_positions_core_with_storage(
                                black_box(f.arr.view()),
                                black_box(f.starts.view()),
                                black_box(f.ends.view()),
                                black_box(f.index.view()),
                                black_box(f.positions.view()),
                                black_box(f.booleans.view()),
                                dense,
                            ))
                        })
                    },
                );
                group.bench_with_input(BenchmarkId::new("sum_old", &label), &fixture, |b, f| {
                    b.iter(|| black_box(old_sum(black_box(f))))
                });
                group.bench_with_input(
                    BenchmarkId::new("sum_ordinal", &label),
                    &fixture,
                    |b, f| {
                        b.iter(|| {
                            black_box(sum_positions_i64_with_storage(
                                black_box(f.arr.view()),
                                black_box(f.starts.view()),
                                black_box(f.ends.view()),
                                black_box(f.index.view()),
                                black_box(f.positions.view()),
                                black_box(f.booleans.view()),
                                dense,
                            ))
                        })
                    },
                );
                group.bench_with_input(BenchmarkId::new("prod_old", &label), &fixture, |b, f| {
                    b.iter(|| black_box(old_prod(black_box(f))))
                });
                group.bench_with_input(
                    BenchmarkId::new("prod_ordinal", &label),
                    &fixture,
                    |b, f| {
                        b.iter(|| {
                            black_box(prod_positions_i64_with_storage(
                                black_box(f.arr.view()),
                                black_box(f.starts.view()),
                                black_box(f.ends.view()),
                                black_box(f.index.view()),
                                black_box(f.positions.view()),
                                black_box(f.booleans.view()),
                                dense,
                            ))
                        })
                    },
                );
                group.bench_with_input(BenchmarkId::new("size_old", &label), &fixture, |b, f| {
                    b.iter(|| black_box(old_size(black_box(f))))
                });
                group.bench_with_input(
                    BenchmarkId::new("size_ordinal", &label),
                    &fixture,
                    |b, f| {
                        b.iter(|| {
                            black_box(size_positions_core_with_storage(
                                black_box(f.starts.view()),
                                black_box(f.ends.view()),
                                black_box(f.index.view()),
                                black_box(f.positions.view()),
                                dense,
                            ))
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
