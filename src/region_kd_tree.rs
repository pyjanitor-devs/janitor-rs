//! Static k-d tree candidate generation for multi-predicate range joins.
//!
//! Region paths turn every range predicate into the same comparison:
//! `left_region <= right_region`.  A k-d tree lets all region dimensions
//! participate in pruning instead of using only the first two dimensions.
//!
//! Design inspiration: Reif and Neumann, "A Scalable and Generic Approach to
//! Range Joins" (PVLDB 15(11), 2022),
//! <https://db.in.tum.de/~reif/papers/p2535-reif.pdf>.
//!
//! The rest of the region-join implementation represents one range predicate
//! as one integer *region label*.  After the labels have been aligned, a
//! candidate is valid exactly when every left label is less than or equal to
//! the corresponding right label.  In other words, a query is a point in
//! `d`-dimensional space and the matching right points are the points in its
//! upper-right orthant.
//!
//! This module owns only that geometric lookup.  It does not know about
//! Python objects, index labels, residual predicates, `keep` modes, or
//! aggregation.  Keeping those concerns outside the tree makes the same
//! traversal usable by index materialization and aggregation callers.

/// A node in the implicit k-d tree.
#[derive(Debug)]
struct Node {
    /// Half-open range into [`RegionKdTree::order`].
    start: usize,
    /// Exclusive end of the point range represented by this node.
    end: usize,
    /// Inclusive lower bound of the node in each region dimension.
    min: Vec<i64>,
    /// Inclusive upper bound of the node in each region dimension.
    max: Vec<i64>,
    /// Index of the left child in the tree's flat node arena.
    left: Option<usize>,
    /// Index of the right child in the tree's flat node arena.
    right: Option<usize>,
}

/// A static tree over right-side region points.
///
/// `points` stores right rows in their global right-position order.  `order`
/// is the mutable permutation used to lay those points out in tree nodes;
/// callbacks therefore receive the original right position, never a tree
/// slot.  Nodes keep bounding boxes so a query can reject or accept a whole
/// subtree without examining every point in it.
///
/// The tree is intentionally static: all region labels must be known before
/// construction, and queries do not mutate it.  That makes it safe to reuse
/// for every left row in one join and keeps the query path allocation-free
/// apart from the caller's left-point slice.
pub(crate) struct RegionKdTree {
    /// Number of aligned region predicates (the k in k-d tree).
    dimensions: usize,
    /// Right-side points, indexed by their global right position.
    points: Vec<Vec<i64>>,
    /// Tree order: each node owns a contiguous slice of this permutation.
    order: Vec<usize>,
    /// Flat node arena; child fields refer back into this vector.
    nodes: Vec<Node>,
    /// Maximum number of points kept in a leaf.
    leaf_size: usize,
}

impl RegionKdTree {
    /// Build a deterministic tree over columnar right-side region vectors.
    ///
    /// # Arguments
    ///
    /// * `right_regions` - One aligned region-label array per predicate. All
    ///   arrays must have the same length; position `j` describes right row
    ///   `j` in the global right layout.
    /// * `leaf_size` - Target maximum number of points scanned directly in a
    ///   leaf. Values below one are treated as one.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than three dimensions are supplied or if the
    /// region arrays have different lengths. An empty right side is valid and
    /// produces an empty tree that simply answers no queries.
    ///
    /// # Complexity
    ///
    /// Construction uses a full sort at every recursive level and is therefore
    /// `O(n log² n)` for `n` right rows in the current implementation. Point
    /// and bounding-box storage is `O(n * d)` for `d` dimensions. The explicit
    /// complexity note is intentional: replacing per-node sorting with a
    /// selection/partition build is a separate optimization opportunity.
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

    /// Recursively add the half-open `order[start..end]` range to the node
    /// arena and return its arena index.
    ///
    /// Each node first computes a bounding box. If it is larger than the leaf
    /// target, the widest dimension becomes the split axis and the point
    /// order is divided at its midpoint. The node is inserted before its
    /// children so child indices remain stable even while recursion appends
    /// more nodes.
    fn build_node(&mut self, start: usize, end: usize) -> Result<usize, String> {
        let mut min = vec![i64::MAX; self.dimensions];
        let mut max = vec![i64::MIN; self.dimensions];
        for &point_index in &self.order[start..end] {
            for dimension in 0..self.dimensions {
                min[dimension] = min[dimension].min(self.points[point_index][dimension]);
                max[dimension] = max[dimension].max(self.points[point_index][dimension]);
            }
        }

        // ELI5: put nearby points in the same box, then split the widest box
        // in half. A later query can throw away a whole box when its limits
        // prove that none of its points could possibly match.
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
            // The point index is a tie-breaker so equal labels produce the
            // same tree and callback order on every build.
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
    /// The callback returns `false` to stop this left-row query early. Zero
    /// region labels are excluded because they are the existing no-match
    /// sentinel, not ordinary rank values.
    ///
    /// # Arguments
    ///
    /// * `left_regions` - One query label per dimension, in exactly the same
    ///   predicate order used to build the tree.
    /// * `visit` - Called with each matching global right position. Returning
    ///   `false` stops this query and is reported back to the caller.
    ///
    /// # Returns
    ///
    /// `true` means traversal completed. `false` means the callback requested
    /// an early stop. Invalid dimensionality, an empty tree, or a zero query
    /// label are treated as a completed query with no matches.
    ///
    /// # Complexity
    ///
    /// The traversal is output-sensitive when bounding boxes are selective;
    /// its worst case is `O(n)` for `n` right rows. It uses no per-query heap
    /// allocation and does not change the tree.
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

    /// Visit every stored point satisfying `point <= query` coordinate-wise.
    ///
    /// This is the orientation-reversed form of [`Self::visit`]. It lets the
    /// caller build the tree over the smaller side of a join while preserving
    /// the same dominance relation. The callback receives the stored point's
    /// original position.
    pub(crate) fn visit_lower<F>(&self, query: &[i64], mut visit: F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        if query.len() != self.dimensions || self.nodes.is_empty() {
            return true;
        }
        if query.contains(&0) {
            return true;
        }
        self.visit_node_lower(0, query, &mut visit)
    }

    /// Traverse one node, applying bounding-box pruning and forwarding every
    /// matching point to `visit`.
    ///
    /// Returning `false` from the callback propagates all the way up the
    /// recursion so a caller can implement `keep="any"` without scanning the
    /// rest of the right side for that left row.
    fn visit_node<F>(&self, node_index: usize, left_regions: &[i64], visit: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let node = &self.nodes[node_index];
        // If one maximum is below the query label, every point in this box
        // fails that dimension. ELI5: the whole box is too far to the left.
        if node
            .max
            .iter()
            .zip(left_regions)
            .any(|(&maximum, &left)| maximum < left)
        {
            return true;
        }

        // If every minimum reaches the query, every point in the box passes
        // every dimension. ELI5: the whole box is safely in the answer area,
        // so we can report its points without rechecking each coordinate.
        if node
            .min
            .iter()
            .zip(left_regions)
            .all(|(&minimum, &left)| minimum >= left)
        {
            for &point_index in &self.order[node.start..node.end] {
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
            // A leaf is the only place where individual points need the full
            // coordinate-by-coordinate dominance check.
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

    /// Traverse one node for the lower-orthant query `point <= query`.
    fn visit_node_lower<F>(&self, node_index: usize, query: &[i64], visit: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let node = &self.nodes[node_index];
        // If one minimum is above the query, every point in this box fails
        // that dimension. ELI5: the whole box is too far to the right.
        if node
            .min
            .iter()
            .zip(query)
            .any(|(&minimum, &value)| minimum > value)
        {
            return true;
        }

        // If every maximum is within the query, the whole box is a match.
        if node
            .max
            .iter()
            .zip(query)
            .all(|(&maximum, &value)| maximum <= value)
        {
            for &point_index in &self.order[node.start..node.end] {
                if !visit(point_index) {
                    return false;
                }
            }
            return true;
        }

        if let Some(left) = node.left {
            if !self.visit_node_lower(left, query, visit) {
                return false;
            }
        } else {
            for &point_index in &self.order[node.start..node.end] {
                let point = &self.points[point_index];
                if point.contains(&0)
                    || !point.iter().zip(query).all(|(&left, &right)| left <= right)
                {
                    continue;
                }
                if !visit(point_index) {
                    return false;
                }
            }
        }
        if let Some(right) = node.right {
            self.visit_node_lower(right, query, visit)
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
    fn lower_queries_match_points_on_the_other_side() {
        let regions = vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1], vec![1, 2, 3, 4]];
        let tree = RegionKdTree::build(&regions, 1).unwrap();
        let mut actual = Vec::new();
        assert!(tree.visit_lower(&[3, 3, 3], |position| {
            actual.push(position);
            true
        }));
        actual.sort_unstable();
        assert_eq!(actual, vec![1, 2]);
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
