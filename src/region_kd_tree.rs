//! Static k-d tree candidate generation for multi-predicate range joins.
//!
//! Region paths turn every range predicate into the same comparison:
//! `left_region <= right_region`.  A k-d tree lets all region dimensions
//! participate in pruning instead of using only the first two dimensions.
//!
//! Design inspiration: Reif and Neumann, "A Scalable and Generic Approach to
//! Range Joins" (PVLDB 15(11), 2022),
//! <https://db.in.tum.de/~reif/papers/p2535-reif.pdf>.

/// A node in the implicit k-d tree.
#[derive(Debug)]
struct Node {
    start: usize,
    end: usize,
    min: Vec<i64>,
    max: Vec<i64>,
    left: Option<usize>,
    right: Option<usize>,
}

/// A static tree over right-side region points.
pub(crate) struct RegionKdTree {
    dimensions: usize,
    points: Vec<Vec<i64>>,
    order: Vec<usize>,
    nodes: Vec<Node>,
    leaf_size: usize,
}

impl RegionKdTree {
    /// Build a deterministic tree over the right-side region vectors.
    pub(crate) fn build(right_regions: &[Vec<i64>], leaf_size: usize) -> Result<Self, String> {
        if right_regions.len() < 3 {
            return Err("k-d tree requires at least three region dimensions".to_owned());
        }
        let dimensions = right_regions.len();
        let point_count = right_regions[0].len();
        for region in right_regions.iter().skip(1) {
            if region.len() != point_count {
                return Err("k-d tree region lengths must match".to_owned());
            }
        }

        let points = (0..point_count)
            .map(|position| {
                right_regions
                    .iter()
                    .map(|region| region[position])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut tree = Self {
            dimensions,
            points,
            order: (0..point_count).collect(),
            nodes: Vec::new(),
            leaf_size: leaf_size.max(1),
        };
        if point_count != 0 {
            tree.build_node(0, point_count)?;
        }
        Ok(tree)
    }

    fn build_node(&mut self, start: usize, end: usize) -> Result<usize, String> {
        let mut min = vec![i64::MAX; self.dimensions];
        let mut max = vec![i64::MIN; self.dimensions];
        for &point_index in &self.order[start..end] {
            for dimension in 0..self.dimensions {
                min[dimension] = min[dimension].min(self.points[point_index][dimension]);
                max[dimension] = max[dimension].max(self.points[point_index][dimension]);
            }
        }

        let split_dimension = if end - start > self.leaf_size {
            Some(
                (0..self.dimensions)
                    .max_by_key(|&dimension| {
                        max[dimension].saturating_sub(min[dimension]).unsigned_abs()
                    })
                    .ok_or("k-d tree has no split dimension")?,
            )
        } else {
            None
        };
        let node_index = self.nodes.len();
        self.nodes.push(Node {
            start,
            end,
            min,
            max,
            left: None,
            right: None,
        });

        if let Some(split_dimension) = split_dimension {
            let midpoint = start + (end - start) / 2;
            self.order[start..end].sort_unstable_by_key(|&point_index| {
                (self.points[point_index][split_dimension], point_index)
            });
            let left = self.build_node(start, midpoint)?;
            let right = self.build_node(midpoint, end)?;
            self.nodes[node_index].left = Some(left);
            self.nodes[node_index].right = Some(right);
        }
        Ok(node_index)
    }

    /// Visit every right position satisfying all `left_regions <= right_regions`.
    ///
    /// The callback returns `false` to stop this left-row query early.  Zero
    /// region labels are excluded because they are the existing no-match
    /// sentinel, not ordinary rank values.
    pub(crate) fn visit<F>(&self, left_regions: &[i64], mut visit: F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        if left_regions.len() != self.dimensions || self.nodes.is_empty() {
            return true;
        }
        if left_regions.contains(&0) {
            return true;
        }
        self.visit_node(0, left_regions, &mut visit)
    }

    fn visit_node<F>(&self, node_index: usize, left_regions: &[i64], visit: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let node = &self.nodes[node_index];
        if node
            .max
            .iter()
            .zip(left_regions)
            .any(|(&maximum, &left)| maximum < left)
        {
            return true;
        }

        if node
            .min
            .iter()
            .zip(left_regions)
            .all(|(&minimum, &left)| minimum >= left)
        {
            for &point_index in &self.order[node.start..node.end] {
                if self.points[point_index].contains(&0) {
                    continue;
                }
                if !visit(point_index) {
                    return false;
                }
            }
            return true;
        }

        if let Some(left) = node.left {
            if !self.visit_node(left, left_regions, visit) {
                return false;
            }
        } else {
            for &point_index in &self.order[node.start..node.end] {
                let point = &self.points[point_index];
                if point.contains(&0)
                    || !left_regions
                        .iter()
                        .zip(point)
                        .all(|(&left, &right)| left <= right)
                {
                    continue;
                }
                if !visit(point_index) {
                    return false;
                }
            }
        }
        if let Some(right) = node.right {
            self.visit_node(right, left_regions, visit)
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RegionKdTree;

    #[test]
    fn visits_exact_dominance_matches_and_skips_zero_regions() {
        let regions = vec![vec![1, 2, 3, 0], vec![1, 2, 1, 3], vec![2, 2, 3, 3]];
        let tree = RegionKdTree::build(&regions, 1).unwrap();
        let mut actual = Vec::new();
        tree.visit(&[2, 1, 2], |position| {
            actual.push(position);
            true
        });
        actual.sort_unstable();
        assert_eq!(actual, vec![1, 2]);
    }

    #[test]
    fn can_stop_after_first_match() {
        let regions = vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]];
        let tree = RegionKdTree::build(&regions, 1).unwrap();
        let mut count = 0;
        assert!(!tree.visit(&[1, 1, 1], |_| {
            count += 1;
            false
        }));
        assert_eq!(count, 1);
    }

    #[test]
    fn randomized_queries_match_brute_force_with_ties_and_sentinels() {
        let mut seed = 0x9e37_79b9_u64;
        let mut next = |bound: i64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % bound as u64) as i64
        };

        for dimensions in 3..=8 {
            let point_count = 1 + next(80) as usize;
            let mut regions = vec![vec![0_i64; point_count]; dimensions];
            for point in 0..point_count {
                for region in &mut regions {
                    region[point] = 1 + next(12);
                }
                if point % 11 == 0 {
                    regions[point % dimensions][point] = 0;
                }
            }

            for leaf_size in [1, 2, 7, 32] {
                let tree = RegionKdTree::build(&regions, leaf_size).unwrap();
                for _ in 0..40 {
                    let left = (0..dimensions).map(|_| next(14)).collect::<Vec<_>>();
                    let mut actual = Vec::new();
                    tree.visit(&left, |position| {
                        actual.push(position);
                        true
                    });
                    actual.sort_unstable();

                    let expected = (0..point_count)
                        .filter(|&position| {
                            regions.iter().all(|region| region[position] != 0)
                                && left.iter().enumerate().all(|(dimension, &value)| {
                                    value != 0 && value <= regions[dimension][position]
                                })
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual, expected,
                        "dimensions={dimensions}, leaf={leaf_size}"
                    );
                }
            }
        }
    }
}
