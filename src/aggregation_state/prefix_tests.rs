//! Independent direct-update comparisons and empirical prefix dispatch evidence.
use super::*;
use numpy::ndarray::{array, s};

fn sum_set<'a>(
    signed: ArrayView1<'a, i64>,
    unsigned: ArrayView1<'a, u64>,
    nulls: ArrayView1<'a, bool>,
    rows: usize,
) -> AggregationSet<'a> {
    AggregationSet {
        states: vec![
            State::Sum(
                View {
                    values: Values::I64(signed),
                    nulls,
                },
                vec![0; rows],
            ),
            State::SumU64(unsigned, nulls, vec![0; rows]),
            State::CountNonNullMask(nulls, vec![0; rows]),
            State::CountAll,
        ],
        output_len: rows,
        source_len: signed.len(),
        count_all: Some(vec![0; rows]),
        matched: Some(vec![false; rows]),
        successful: false,
    }
}

fn compare_with_direct(
    signed: ArrayView1<'_, i64>,
    unsigned: ArrayView1<'_, u64>,
    nulls: ArrayView1<'_, bool>,
    windows: &[(i64, i64)],
    prefix_calls: usize,
) {
    let starts = Array1::from_iter(windows.iter().map(|&(s, _)| s));
    let ends = Array1::from_iter(windows.iter().map(|&(_, e)| e));
    let mut actual = sum_set(signed, unsigned, nulls, windows.len());
    let mut reference = sum_set(signed, unsigned, nulls, windows.len());
    // The oracle visits candidates individually, without prefix tables or
    // packed-coordinate calculations. Invalid ranges contribute nothing.
    for (row, &(start, end)) in windows.iter().enumerate() {
        if start >= 0 && start < end && end <= signed.len() as i64 {
            for position in start..end {
                reference.update(position as usize, row);
            }
        }
    }
    let before = PREFIX_SUM_HITS.get();
    actual.aggregate_starts_ends(starts.view(), ends.view());
    assert_eq!(PREFIX_SUM_HITS.get() - before, prefix_calls, "{windows:?}");
    for (actual, expected) in actual.states.iter().zip(&reference.states) {
        match (actual, expected) {
            (State::Sum(_, a), State::Sum(_, e)) => assert_eq!(a, e),
            (State::SumU64(_, _, a), State::SumU64(_, _, e)) => assert_eq!(a, e),
            (State::CountNonNullMask(_, a), State::CountNonNullMask(_, e)) => assert_eq!(a, e),
            (State::CountAll, State::CountAll) => {}
            _ => panic!("unexpected test state"),
        }
    }
    assert_eq!(actual.matched, reference.matched);
    assert_eq!(actual.count_all, reference.count_all);
    assert_eq!(actual.successful, reference.successful);
}

#[test]
fn prefix_dispatch_thresholds_and_saturating_costs() {
    let signed = Array1::from_elem(16, i64::MAX);
    let unsigned = Array1::from_elem(16, u64::MAX);
    let nulls = Array1::from_elem(16, false);
    // Packed length 16, four queries: the estimated prefix cost is 24.
    for (last_width, hits) in [(3, 0), (4, 0), (5, 2)] {
        compare_with_direct(
            signed.view(),
            unsigned.view(),
            nulls.view(),
            &[(0, 16), (1, 3), (2, 4), (3, 3 + last_width)],
            hits,
        );
    }
    for (queries, hits) in [(3, 0), (4, 2)] {
        compare_with_direct(
            signed.view(),
            unsigned.view(),
            nulls.view(),
            &vec![(0, 16); queries],
            hits,
        );
    }
    for (q, width, length) in [
        (0, 0, 0),
        (4, 100, 0),
        (usize::MAX, usize::MAX, 16),
        (4, usize::MAX, usize::MAX),
    ] {
        assert!(!should_use_prefix_difference(q, width, length));
    }
}

#[test]
fn prefix_wrapping_all_windows_masks_strides_and_invalid_boundaries() {
    let signed = Array1::from_iter((0..34).map(|i| match i % 6 {
        0 => i64::MAX,
        2 => 1,
        4 => i64::MIN,
        _ => -73,
    }));
    let unsigned = Array1::from_iter((0..34).map(|i| match i % 6 {
        0 => u64::MAX,
        2 => 1,
        4 => 1_u64 << 63,
        _ => 73,
    }));
    // Include every valid interval, both subtraction-wrap directions, and
    // exact len, len+1, -1 on either bound, empty and inverted intervals.
    let windows: Vec<_> = (-1..=18)
        .flat_map(|s| (-1..=18).map(move |e| (s, e)))
        .collect();
    for all_null in [false, true] {
        let nulls = Array1::from_iter((0..34).map(|i| all_null || i % 7 == 0));
        for step in [2, -2] {
            compare_with_direct(
                signed.slice(s![..;step]),
                unsigned.slice(s![..;step]),
                nulls.slice(s![..;step]),
                &windows,
                2,
            );
        }
    }
    let empty = Array1::from_vec(vec![]);
    compare_with_direct(
        empty.view(),
        Array1::<u64>::zeros(0).view(),
        Array1::from_vec(vec![]).view(),
        &[(-1, -1), (0, 0), (0, 1), (1, 1)],
        0,
    );
    compare_with_direct(
        signed.view(),
        unsigned.view(),
        Array1::from_elem(34, false).view(),
        &[],
        0,
    );
}

#[test]
fn prefix_packed_disjoint_spans_cancel_earlier_overflow() {
    let signed = array![99, i64::MAX, 1, 4, 91, 93, 95, i64::MIN, -1, 7, 97];
    let unsigned = signed.mapv(|v| v as u64);
    let nulls = Array1::from_elem(signed.len(), false);
    let ranges = vec![Some((1, 4)), Some((7, 10)), None];
    let spans = queried_source_spans(&ranges);
    assert_eq!(spans, [(1, 4), (7, 10)]);
    let clusters = assign_range_clusters(&ranges, &spans);
    let layout = PackedSpanLayout::new(&spans, &clusters);
    assert_eq!(layout.length, 6);
    let mut a = vec![0; 3];
    update_signed_prefix_ranges(
        &View {
            values: Values::I64(signed.view()),
            nulls: nulls.view(),
        },
        &mut a,
        &ranges,
        &layout,
    );
    assert_eq!(a, [i64::MIN + 4, i64::MIN + 6, 0]);
    let mut windows = vec![(1, 4), (7, 10)];
    windows.extend(vec![(7, 10); 20]);
    compare_with_direct(signed.view(), unsigned.view(), nulls.view(), &windows, 2);
}

#[test]
fn prefix_signed_state_widens_every_integer_source_representation() {
    macro_rules! check {
        ($typ:ty, $variant:ident) => {{
            let values = Array1::from_iter(
                (0..33).map(|i| if i % 3 == 0 { <$typ>::MAX } else { 1 }),
            );
            let nulls = Array1::from_iter((0..33).map(|i| i % 5 == 0));
            let values = values.slice(s![..;-1]);
            let nulls = nulls.slice(s![..;-1]);
            let expected = (1..32)
                .filter(|&i| !nulls[i])
                .fold(0_i64, |sum, i| sum.wrapping_add(values[i] as i64));
            let mut set = AggregationSet {
                states: vec![State::Sum(
                    View { values: Values::$variant(values), nulls }, vec![0; 16],
                )],
                output_len: 16,
                source_len: 33,
                count_all: None,
                matched: None,
                successful: false,
            };
            let before = PREFIX_SUM_HITS.get();
            set.aggregate_starts_ends(
                Array1::from_elem(16, 1).view(), Array1::from_elem(16, 32).view(),
            );
            assert_eq!(PREFIX_SUM_HITS.get(), before + 1);
            let State::Sum(_, output) = &set.states[0] else { unreachable!() };
            assert_eq!(output, &vec![expected; 16], stringify!($typ));
        }};
    }
    check!(i64, I64);
    check!(i32, I32);
    check!(i16, I16);
    check!(i8, I8);
    check!(u32, U32);
    check!(u16, U16);
    check!(u8, U8);
}

#[test]
fn product_trees_keep_wrapping_zero_even_and_null_semantics() {
    let signed = array![i64::MAX, 2, 1, 0, 4, i64::MIN, -1, 2, 7];
    let unsigned = signed.mapv(|v| v as u64);
    let nulls = array![false, false, false, false, true, false, false, false, false];
    let ranges = vec![Some((0, 3)), Some((1, 6)), Some((5, 9)), Some((4, 5)), None];
    let spans = queried_source_spans(&ranges);
    let clusters = assign_range_clusters(&ranges, &spans);
    let layout = PackedSpanLayout::new(&spans, &clusters);
    let view = View {
        values: Values::I64(signed.view()),
        nulls: nulls.view(),
    };
    let mut a = vec![1; ranges.len()];
    let mut b = vec![1; ranges.len()];
    let before = PREFIX_SUM_HITS.get();
    update_signed_product_segment_ranges(&view, &mut a, &ranges, &layout);
    update_u64_product_segment_ranges(&unsigned.view(), &nulls.view(), &mut b, &ranges, &layout);
    assert_eq!(PREFIX_SUM_HITS.get(), before);
    for (row, range) in ranges.iter().enumerate() {
        let (s, e) = range.unwrap_or((0, 0));
        let positions = (s..e).filter(|&i| !nulls[i]);
        assert_eq!(
            a[row],
            positions
                .clone()
                .fold(1_i64, |a, i| a.wrapping_mul(signed[i]))
        );
        assert_eq!(
            b[row],
            positions.fold(1_u64, |a, i| a.wrapping_mul(unsigned[i]))
        );
    }
}
