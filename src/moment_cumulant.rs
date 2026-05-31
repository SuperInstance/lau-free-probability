//! Moment-cumulant formula via non-crossing partitions.
//!
//! The fundamental relation between moments and free cumulants:
//!   m_n = Σ_{π ∈ NC(n)} ∏_{B ∈ π} κ_{|B|}
//!
//! And its inverse (Möbius inversion):
//!   κ_n = Σ_{π ∈ NC(n)} μ(0_n, π) ∏_{B ∈ π} m_{|B|}

use crate::partition::NCPartition;
#[cfg(test)]
use crate::partition::catalan;

/// Convert moments to free cumulants using Möbius inversion on NC lattice.
///
/// Given moments m_1, m_2, ..., m_n, compute free cumulants κ_1, ..., κ_n.
///
/// κ_n = Σ_{π ∈ NC(n)} μ(0_n, π) ∏_{B ∈ π} m_{|B|}
pub fn moments_to_free_cumulants(moments: &[f64]) -> Vec<f64> {
    let n = moments.len();
    if n == 0 {
        return vec![];
    }

    // Use the recursive formula instead of enumerating all partitions
    // for efficiency. The moment-cumulant relation:
    // m_n = Σ_{k=1}^{n} κ_k * Σ_{π ∈ NC(n) with first block of size k} ∏ m-values
    //
    // We use the direct recurrence:
    // κ_n = m_n - Σ_{π ∈ NC(n), π ≠ 1_n} ∏_{B ∈ π} κ_{|B|}

    let mut cumulants = Vec::with_capacity(n);

    for order in 1..=n {
        // κ_n = m_n - Σ_{π ∈ NC(n), |π| ≥ 2} ∏_{B ∈ π} κ_{|B|}
        let partitions = NCPartition::all_nc_partitions(order);
        let mut cumulant = moments[order - 1];

        for pi in &partitions {
            if pi.num_blocks() == 1 {
                continue; // Skip the one-block partition (that's what we're solving for)
            }
            let product: f64 = pi.blocks().iter().map(|block| {
                let size = block.len();
                if size <= cumulants.len() {
                    cumulants[size - 1]
                } else {
                    moments[size - 1]
                }
            }).product();
            cumulant -= product;
        }
        cumulants.push(cumulant);
    }

    cumulants
}

/// Convert free cumulants to moments.
///
/// m_n = Σ_{π ∈ NC(n)} ∏_{B ∈ π} κ_{|B|}
pub fn free_cumulants_to_moments(cumulants: &[f64]) -> Vec<f64> {
    let n = cumulants.len();
    if n == 0 {
        return vec![];
    }

    let mut moments = Vec::with_capacity(n);

    for order in 1..=n {
        let partitions = NCPartition::all_nc_partitions(order);
        let mut moment = 0.0;
        for pi in &partitions {
            let product: f64 = pi.blocks().iter().map(|block| {
                cumulants[block.len() - 1]
            }).product();
            moment += product;
        }
        moments.push(moment);
    }

    moments
}

/// Compute classical cumulants from moments (for comparison).
///
/// Classical cumulants use ALL partitions (not just NC).
/// c_n = m_n - Σ_{π ∈ P(n), π ≠ 1_n} ∏ c_{|B|}
pub fn moments_to_classical_cumulants(moments: &[f64]) -> Vec<f64> {
    let n = moments.len();
    let mut cumulants = Vec::with_capacity(n);

    for order in 1..=n {
        // Use the recursive formula
        // c_n = m_n - Σ_{π ∈ P(n), π ≠ 1_n} ∏ c_{|B|}
        // For classical cumulants, use Bell polynomial approach
        let mut cumulant = moments[order - 1];

        // Sum over all set partitions (not just NC) — for small n this is fine
        // Use the recurrence: c_n = m_n - Σ_{k=1}^{n-1} C(n-1, k-1) * c_k * m_{n-k}
        for k in 1..order {
            let binom = binomial(order - 1, k - 1) as f64;
            cumulant -= binom * cumulants[k - 1] * moments[order - k - 1];
        }

        cumulants.push(cumulant);
    }

    cumulants
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Compute the n-th free cumulant from a moment sequence (1-indexed).
pub fn nth_free_cumulant(moments: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let cumulants = moments_to_free_cumulants(&moments[..n.min(moments.len())]);
    cumulants[n - 1]
}

/// Compute moments of a freely independent sum from individual moment sequences.
/// If X and Y are freely independent, then m_n(X+Y) can be computed
/// from the free cumulants: κ_n(X+Y) = κ_n(X) + κ_n(Y).
pub fn free_sum_cumulants(moments_x: &[f64], moments_y: &[f64]) -> Vec<f64> {
    let n = moments_x.len().max(moments_y.len());
    let kappa_x = moments_to_free_cumulants(moments_x);
    let kappa_y = moments_to_free_cumulants(moments_y);

    let mut sum_cumulants = Vec::with_capacity(n);
    for i in 0..n {
        let kx = if i < kappa_x.len() { kappa_x[i] } else { 0.0 };
        let ky = if i < kappa_y.len() { kappa_y[i] } else { 0.0 };
        sum_cumulants.push(kx + ky);
    }
    sum_cumulants
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_cumulants_moments() {
        // Semicircle law moments: m_1=0, m_2=1, m_3=0, m_4=2, m_5=0, m_6=5
        let moments = vec![0.0, 1.0, 0.0, 2.0, 0.0, 5.0];
        let cumulants = moments_to_free_cumulants(&moments);
        let recovered = free_cumulants_to_moments(&cumulants);
        for (i, (a, b)) in moments.iter().zip(&recovered).enumerate() {
            assert!((a - b).abs() < 1e-10, "Mismatch at order {}: {} vs {}", i + 1, a, b);
        }
    }

    #[test]
    fn test_semicircle_cumulants() {
        // Wigner semicircle: κ_1 = 0, κ_2 = 1, κ_n = 0 for n ≥ 3
        let moments = vec![0.0, 1.0, 0.0, 2.0, 0.0, 5.0];
        let cumulants = moments_to_free_cumulants(&moments);
        assert!((cumulants[0]).abs() < 1e-10, "κ_1 should be 0, got {}", cumulants[0]);
        assert!((cumulants[1] - 1.0).abs() < 1e-10, "κ_2 should be 1, got {}", cumulants[1]);
        assert!((cumulants[2]).abs() < 1e-10, "κ_3 should be 0, got {}", cumulants[2]);
        assert!((cumulants[3]).abs() < 1e-10, "κ_4 should be 0, got {}", cumulants[3]);
    }

    #[test]
    fn test_free_sum_cumulants_additivity() {
        // Free cumulants of X+Y where X,Y freely independent = κ(X) + κ(Y)
        let mx = vec![0.0, 1.0]; // semicircle-like
        let my = vec![0.0, 2.0];
        let sum_k = free_sum_cumulants(&mx, &my);
        assert!((sum_k[0]).abs() < 1e-10); // κ_1 = 0 + 0
        assert!((sum_k[1] - 3.0).abs() < 1e-10); // κ_2 = 1 + 2
    }

    #[test]
    fn test_deterministic_cumulants() {
        // For a deterministic value c: moments are c, c^2, c^3, ...
        // Free cumulants: κ_1 = c, κ_n = 0 for n ≥ 2
        let c: f64 = 3.0;
        let moments: Vec<f64> = (1..=6).map(|n| c.powi(n)).collect();
        let cumulants = moments_to_free_cumulants(&moments);
        assert!((cumulants[0] - c).abs() < 1e-8, "κ_1 = c = {}", cumulants[0]);
        for i in 1..cumulants.len() {
            assert!(cumulants[i].abs() < 1e-8, "κ_{} should be 0, got {}", i + 1, cumulants[i]);
        }
    }

    #[test]
    fn test_marchenko_pastur_cumulants() {
        // Marchenko-Pastur with λ=1: moments are Catalan numbers
        // m_n = C_n, free cumulants: κ_n = 1 for all n
        let moments: Vec<f64> = (1..=6).map(|n| catalan(n) as f64).collect();
        let cumulants = moments_to_free_cumulants(&moments);
        for (i, k) in cumulants.iter().enumerate() {
            assert!((k - 1.0).abs() < 1e-8, "κ_{} should be 1, got {}", i + 1, k);
        }
    }

    #[test]
    fn test_classical_vs_free_cumulants() {
        // For Gaussian: classical cumulants c_1=0, c_2=σ², c_n=0 for n≥3
        // For semicircle: free cumulants κ_1=0, κ_2=1, κ_n=0 for n≥3
        let moments = vec![0.0, 1.0, 0.0, 3.0]; // Gaussian moments with σ²=1
        let classical = moments_to_classical_cumulants(&moments);
        assert!((classical[0]).abs() < 1e-10);
        assert!((classical[1] - 1.0).abs() < 1e-10);
        assert!((classical[2]).abs() < 1e-10);
        assert!((classical[3]).abs() < 1e-10);
    }
}
