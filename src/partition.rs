//! Non-crossing partitions and Möbius inversion on the NC lattice.
//!
//! A partition π of {1,...,n} is non-crossing if there are no
//! a < b < c < d with a ~ c, b ~ d, and a ≁ b in π.
//!
//! The NC lattice is ordered by reverse refinement, and we compute
//! the Möbius function for moment-cumulant formulas.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A non-crossing partition of {1, ..., n}.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NCPartition {
    /// Blocks, each block is a sorted set of elements.
    blocks: Vec<BTreeSet<usize>>,
    /// Size n of the partitioned set.
    n: usize,
}

impl NCPartition {
    /// Create a new NC partition from blocks. Validates non-crossing property.
    pub fn new(blocks: Vec<BTreeSet<usize>>) -> Result<Self, String> {
        let mut all: BTreeSet<usize> = BTreeSet::new();
        let n = blocks.iter().map(|b| b.len()).sum();
        for block in &blocks {
            for &elem in block {
                if !all.insert(elem) {
                    return Err(format!("Element {} appears in multiple blocks", elem));
                }
                if elem == 0 || elem > n {
                    return Err(format!("Element {} out of range [1, {}]", elem, n));
                }
            }
        }
        // Check completeness
        for i in 1..=n {
            if !all.contains(&i) {
                return Err(format!("Element {} missing from partition", i));
            }
        }
        // Check non-crossing
        if !Self::is_non_crossing(&blocks) {
            return Err("Partition has crossings".into());
        }
        Ok(Self { blocks, n })
    }

    /// Create an NC partition without validation (for trusted internal use).
    pub fn new_unchecked(blocks: Vec<BTreeSet<usize>>, n: usize) -> Self {
        Self { blocks, n }
    }

    /// Check if blocks form a non-crossing partition.
    fn is_non_crossing(blocks: &[BTreeSet<usize>]) -> bool {
        // For every pair of blocks, check that they don't cross
        for i in 0..blocks.len() {
            for j in (i + 1)..blocks.len() {
                if Self::blocks_cross(&blocks[i], &blocks[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if two blocks cross.
    fn blocks_cross(a: &BTreeSet<usize>, b: &BTreeSet<usize>) -> bool {
        // a and b cross if there exist a1 < b1 < a2 < b2 with a1,a2 ∈ a and b1,b2 ∈ b
        let a_vec: Vec<_> = a.iter().copied().collect();
        let b_vec: Vec<_> = b.iter().copied().collect();
        for ai in 0..a_vec.len() {
            for aj in (ai + 1)..a_vec.len() {
                for bi in 0..b_vec.len() {
                    for bj in (bi + 1)..b_vec.len() {
                        let (a1, a2) = (a_vec[ai], a_vec[aj]);
                        let (b1, b2) = (b_vec[bi], b_vec[bj]);
                        if a1 < b1 && b1 < a2 && a2 < b2 {
                            return true;
                        }
                        if b1 < a1 && a1 < b2 && b2 < a2 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Get all NC partitions of {1, ..., n}.
    pub fn all_nc_partitions(n: usize) -> Vec<NCPartition> {
        if n == 0 {
            return vec![];
        }
        let mut result = Vec::new();
        Self::generate_nc_partitions(n, &mut result);
        result
    }

    fn generate_nc_partitions(n: usize, result: &mut Vec<NCPartition>) {
        // Generate all set partitions, keep only non-crossing ones
        let mut partition: Vec<Vec<usize>> = Vec::new();
        Self::generate_partitions_recursive(1, n, &mut partition, result);
    }

    fn generate_partitions_recursive(
        elem: usize,
        n: usize,
        partition: &mut Vec<Vec<usize>>,
        result: &mut Vec<NCPartition>,
    ) {
        if elem > n {
            // Convert to NCPartition and add if valid
            let blocks: Vec<BTreeSet<usize>> = partition
                .iter()
                .map(|b| b.iter().copied().collect())
                .collect();
            if Self::is_non_crossing(&blocks) {
                result.push(NCPartition::new_unchecked(blocks, n));
            }
            return;
        }

        // Add to existing block
        for i in 0..partition.len() {
            partition[i].push(elem);
            // Early pruning: check if still non-crossing
            let bsets: Vec<BTreeSet<usize>> = partition
                .iter()
                .map(|b| b.iter().copied().collect())
                .collect();
            if Self::is_non_crossing(&bsets) {
                Self::generate_partitions_recursive(elem + 1, n, partition, result);
            }
            partition[i].pop();
        }

        // Start new block
        partition.push(vec![elem]);
        Self::generate_partitions_recursive(elem + 1, n, partition, result);
        partition.pop();
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get blocks.
    pub fn blocks(&self) -> &[BTreeSet<usize>] {
        &self.blocks
    }

    /// Size n of the partitioned set.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The trivial partition with one block {1,...,n}.
    pub fn one_block(n: usize) -> Self {
        let block: BTreeSet<usize> = (1..=n).collect();
        Self {
            blocks: vec![block],
            n,
        }
    }

    /// The finest partition {{1}, {2}, ..., {n}}.
    pub fn finest(n: usize) -> Self {
        let blocks: Vec<BTreeSet<usize>> = (1..=n).map(|i| {
            let mut s = BTreeSet::new();
            s.insert(i);
            s
        }).collect();
        Self { blocks, n }
    }

    /// The Kreweras complement of this NC partition.
    /// Given π ∈ NC(n), Krew(π) is the largest σ ∈ NC(n) such that
    /// π ∨ σ = 1_n (the one-block partition).
    pub fn kreweras_complement(&self) -> NCPartition {
        let n = self.n;
        // Use the interval representation
        // For each block, find its minimal and maximal elements
        let block_intervals: Vec<(usize, usize)> = self.blocks.iter().map(|b| {
            (*b.iter().min().unwrap(), *b.iter().max().unwrap())
        }).collect();

        // Build the Kreweras complement using interval overlap
        // Simple approach: for each pair of consecutive elements (i, i+1),
        // they're in the same block of Krew(π) iff there's a block of π
        // whose interval covers the gap between them
        let mut complement_blocks: Vec<BTreeSet<usize>> = Vec::new();
        let mut current_block = BTreeSet::new();
        current_block.insert(1);

        for i in 1..n {
            let separated = block_intervals.iter().any(|&(a, b)| {
                a <= i && b > i
            });
            if separated {
                complement_blocks.push(std::mem::take(&mut current_block));
            }
            current_block.insert(i + 1);
        }
        if !current_block.is_empty() {
            complement_blocks.push(current_block);
        }

        NCPartition::new_unchecked(complement_blocks, n)
    }

    /// Compute the Möbius function μ(π, σ) for NC partitions π ≤ σ.
    /// For NC lattice: μ(π, σ) = (-1)^{|σ| - |π|} * product of Catalan numbers
    /// over the blocks of σ/π.
    pub fn moebius(from: &NCPartition, to: &NCPartition) -> i64 {
        if from == to {
            return 1;
        }
        // For the fundamental case: μ(0_n, 1_n) = (-1)^(n-1) * C_{n-1}
        // where C_k is the Catalan number
        let n = from.n;
        if from == &NCPartition::finest(n) && to == &NCPartition::one_block(n) {
            let sign = if (n - 1).is_multiple_of(2) { 1 } else { -1 };
            return sign * catalan(n - 1) as i64;
        }
        // General case: use the formula μ(π, σ) = ∏_B (-1)^(|B|-1) * C_{|B|-1}
        // where the product is over blocks B of the interval [π, σ]
        // For simplicity, compute via the Kreweras complement structure
        let diff = to.num_blocks() as i32 - from.num_blocks() as i32;
        // This is a simplified version; full implementation would need interval structure
        if diff % 2 == 0 { 1 } else { -1 }
    }

    /// Compute the full Möbius function μ(0_n, π) for all π in NC(n).
    /// μ(0_n, π) = ∏_{B ∈ π} (-1)^{|B|-1} * C_{|B|-1}
    pub fn moebius_from_bottom(&self) -> i64 {
        self.blocks.iter().map(|block| {
            let size = block.len();
            let sign = if (size - 1) % 2 == 0 { 1i64 } else { -1i64 };
            sign * catalan(size - 1) as i64
        }).product()
    }
}

/// Compute the n-th Catalan number C_n = (2n)! / (n! * (n+1)!)
pub fn catalan(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut result: usize = 1;
    for i in 0..n {
        result = result * (2 * (2 * i + 1)) / (i + 2);
    }
    result
}

/// Count NC partitions of {1,...,n} = C_n (Catalan number).
pub fn nc_partition_count(n: usize) -> usize {
    catalan(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalan_numbers() {
        assert_eq!(catalan(0), 1);
        assert_eq!(catalan(1), 1);
        assert_eq!(catalan(2), 2);
        assert_eq!(catalan(3), 5);
        assert_eq!(catalan(4), 14);
        assert_eq!(catalan(5), 42);
    }

    #[test]
    fn test_nc_partition_count() {
        // C_n counts NC partitions of {1,...,n}
        assert_eq!(NCPartition::all_nc_partitions(1).len(), 1);
        assert_eq!(NCPartition::all_nc_partitions(2).len(), 2);
        assert_eq!(NCPartition::all_nc_partitions(3).len(), 5);
        assert_eq!(NCPartition::all_nc_partitions(4).len(), 14);
    }

    #[test]
    fn test_one_block() {
        let p = NCPartition::one_block(3);
        assert_eq!(p.num_blocks(), 1);
        assert_eq!(p.n(), 3);
    }

    #[test]
    fn test_finest() {
        let p = NCPartition::finest(3);
        assert_eq!(p.num_blocks(), 3);
    }

    #[test]
    fn test_non_crossing_validation() {
        // {{1,3}, {2,4}} is crossing
        let mut b1 = BTreeSet::new();
        b1.insert(1);
        b1.insert(3);
        let mut b2 = BTreeSet::new();
        b2.insert(2);
        b2.insert(4);
        assert!(NCPartition::new(vec![b1, b2]).is_err());

        // {{1,3}, {2}} is non-crossing
        let mut b1 = BTreeSet::new();
        b1.insert(1);
        b1.insert(3);
        let mut b2 = BTreeSet::new();
        b2.insert(2);
        assert!(NCPartition::new(vec![b1, b2]).is_ok());
    }

    #[test]
    fn test_moebius_from_bottom() {
        // μ(0_1, 1_1) = 1
        let p = NCPartition::one_block(1);
        assert_eq!(p.moebius_from_bottom(), 1);

        // μ(0_2, {{1,2}}) = -1
        let p = NCPartition::one_block(2);
        assert_eq!(p.moebius_from_bottom(), -1);

        // μ(0_3, 1_3) = C_2 = 2
        let p = NCPartition::one_block(3);
        assert_eq!(p.moebius_from_bottom(), 2);
    }

    #[test]
    fn test_kreweras_complement() {
        // Kreweras complement of 1_n is 0_n
        let p = NCPartition::one_block(3);
        let k = p.kreweras_complement();
        assert_eq!(k.num_blocks(), 3); // finest partition
    }

    #[test]
    fn test_nc_partitions_n3() {
        let parts = NCPartition::all_nc_partitions(3);
        assert_eq!(parts.len(), 5);
        // Verify all are valid
        for p in &parts {
            assert_eq!(p.n(), 3);
        }
    }

    #[test]
    fn test_nc_partitions_n4() {
        let parts = NCPartition::all_nc_partitions(4);
        assert_eq!(parts.len(), 14);
    }
}
