//! Compare the old dictionary-based no-range aggregations with compact slots.
//!
//! ELI5: the old path keeps one map for output positions and another for
//! aggregate state, then walks a map again to emit results. The compact path
//! keeps each label's state in one ordinal slot and records labels in input
//! order. Both paths do the same useful work; this measures the bookkeeping
//! difference for repeated and unique labels at a large input size.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::{hash_map::Entry, HashMap};
use std::hint::black_box;

struct Fixture {
    values: Vec<i64>,
    left_index: Vec<i64>,
    right_index: Vec<i64>,
    booleans: Vec<bool>,
}

impl Fixture {
    fn new(rows: usize, labels: usize) -> Self {
        Fixture {
            values: (0..rows).map(|row| (row % 97) as i64 - 48).collect(),
            left_index: (0..rows as i64).collect(),
            right_index: (0..rows).map(|row| (row % labels) as i64).collect(),
            booleans: vec![false; rows],
        }
    }
}

fn old_max(fixture: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut positions = HashMap::<i64, i64>::with_capacity(fixture.left_index.len());
    let mut values = HashMap::<i64, i64>::with_capacity(fixture.left_index.len());
    for (&left_index, &right_index) in fixture.left_index.iter().zip(&fixture.right_index) {
        let left = left_index as usize;
        let position = positions.entry(right_index).or_insert(-1);
        let value = values.entry(right_index).or_insert(fixture.values[left]);
        if fixture.booleans[left] {
            continue;
        }
        if *position == -1 || fixture.values[left] > *value {
            *position = left_index;
            *value = fixture.values[left];
        }
    }
    positions.into_iter().unzip()
}

fn compact_max(fixture: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut slots = HashMap::<i64, usize>::new();
    let mut labels = Vec::new();
    let mut positions = Vec::new();
    let mut values = Vec::new();
    for (&left_index, &right_index) in fixture.left_index.iter().zip(&fixture.right_index) {
        let left = left_index as usize;
        let slot = match slots.entry(right_index) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot = labels.len();
                labels.push(right_index);
                positions.push(-1);
                values.push(fixture.values[left]);
                entry.insert(slot);
                slot
            }
        };
        if fixture.booleans[left] {
            continue;
        }
        if positions[slot] == -1 || fixture.values[left] > values[slot] {
            positions[slot] = left_index;
            values[slot] = fixture.values[left];
        }
    }
    (labels, positions)
}

fn old_min(fixture: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut positions = HashMap::<i64, i64>::with_capacity(fixture.left_index.len());
    let mut values = HashMap::<i64, i64>::with_capacity(fixture.left_index.len());
    for (&left_index, &right_index) in fixture.left_index.iter().zip(&fixture.right_index) {
        let left = left_index as usize;
        let position = positions.entry(right_index).or_insert(-1);
        let value = values.entry(right_index).or_insert(fixture.values[left]);
        if fixture.booleans[left] {
            continue;
        }
        if *position == -1 || fixture.values[left] < *value {
            *position = left_index;
            *value = fixture.values[left];
        }
    }
    positions.into_iter().unzip()
}

fn compact_min(fixture: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut slots = HashMap::<i64, usize>::new();
    let mut labels = Vec::new();
    let mut positions = Vec::new();
    let mut values = Vec::new();
    for (&left_index, &right_index) in fixture.left_index.iter().zip(&fixture.right_index) {
        let left = left_index as usize;
        let slot = match slots.entry(right_index) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot = labels.len();
                labels.push(right_index);
                positions.push(-1);
                values.push(fixture.values[left]);
                entry.insert(slot);
                slot
            }
        };
        if fixture.booleans[left] {
            continue;
        }
        if positions[slot] == -1 || fixture.values[left] < values[slot] {
            positions[slot] = left_index;
            values[slot] = fixture.values[left];
        }
    }
    (labels, positions)
}

fn old_prod(fixture: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut products = HashMap::<i64, i64>::with_capacity(fixture.left_index.len());
    for (&left_index, &right_index) in fixture.left_index.iter().zip(&fixture.right_index) {
        let left = left_index as usize;
        let product = products.entry(right_index).or_insert(1);
        if !fixture.booleans[left] {
            *product = product.wrapping_mul(fixture.values[left]);
        }
    }
    products.into_iter().unzip()
}

fn compact_prod(fixture: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut slots = HashMap::<i64, usize>::new();
    let mut labels = Vec::new();
    let mut products = Vec::new();
    for (&left_index, &right_index) in fixture.left_index.iter().zip(&fixture.right_index) {
        let left = left_index as usize;
        let slot = match slots.entry(right_index) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot = labels.len();
                labels.push(right_index);
                products.push(1_i64);
                entry.insert(slot);
                slot
            }
        };
        if !fixture.booleans[left] {
            products[slot] = products[slot].wrapping_mul(fixture.values[left]);
        }
    }
    (labels, products)
}

fn bench_no_range_aggs(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_no_range_aggregations");
    let rows = 100_000;
    for labels in [100, rows] {
        let fixture = Fixture::new(rows, labels);
        let input = format!("rows={rows}, labels={labels}");
        group.bench_with_input(
            BenchmarkId::new("max/hashmaps", &input),
            &fixture,
            |b, f| b.iter(|| black_box(old_max(black_box(f)))),
        );
        group.bench_with_input(BenchmarkId::new("max/compact", &input), &fixture, |b, f| {
            b.iter(|| black_box(compact_max(black_box(f))))
        });
        group.bench_with_input(
            BenchmarkId::new("min/hashmaps", &input),
            &fixture,
            |b, f| b.iter(|| black_box(old_min(black_box(f)))),
        );
        group.bench_with_input(BenchmarkId::new("min/compact", &input), &fixture, |b, f| {
            b.iter(|| black_box(compact_min(black_box(f))))
        });
        group.bench_with_input(
            BenchmarkId::new("prod/hashmap", &input),
            &fixture,
            |b, f| b.iter(|| black_box(old_prod(black_box(f)))),
        );
        group.bench_with_input(
            BenchmarkId::new("prod/compact", &input),
            &fixture,
            |b, f| b.iter(|| black_box(compact_prod(black_box(f)))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_no_range_aggs);
criterion_main!(benches);
