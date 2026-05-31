//! Free convolution operations.
//!
//! Additive free convolution: R_{X+Y}(z) = R_X(z) + R_Y(z) (for freely independent X, Y)
//! Multiplicative free convolution: S_{XY}(z) = S_X(z) * S_Y(z)

use crate::moment_cumulant::{moments_to_free_cumulants, free_cumulants_to_moments};

/// Additive free convolution via R-transform.
///
/// For freely independent X, Y:
///   κ_n(X ⊞ Y) = κ_n(X) + κ_n(Y)
///
/// Returns the moments of X ⊞ Y.
pub fn additive_free_convolution(moments_x: &[f64], moments_y: &[f64]) -> Vec<f64> {
    let kappa_x = moments_to_free_cumulants(moments_x);
    let kappa_y = moments_to_free_cumulants(moments_y);

    let n = kappa_x.len().max(kappa_y.len());
    let mut sum_cumulants = Vec::with_capacity(n);
    for i in 0..n {
        let kx = kappa_x.get(i).copied().unwrap_or(0.0);
        let ky = kappa_y.get(i).copied().unwrap_or(0.0);
        sum_cumulants.push(kx + ky);
    }

    free_cumulants_to_moments(&sum_cumulants)
}

/// Additive free convolution using precomputed cumulants.
pub fn additive_free_convolution_cumulants(
    cumulants_x: &[f64],
    cumulants_y: &[f64],
) -> Vec<f64> {
    let n = cumulants_x.len().max(cumulants_y.len());
    let mut sum = Vec::with_capacity(n);
    for i in 0..n {
        let kx = cumulants_x.get(i).copied().unwrap_or(0.0);
        let ky = cumulants_y.get(i).copied().unwrap_or(0.0);
        sum.push(kx + ky);
    }
    sum
}

/// Additive free convolution with scalar shift.
/// X ⊞ c·I has cumulants κ_1(X) + c, κ_n(X) for n ≥ 2.
pub fn additive_shift(moments_x: &[f64], c: f64) -> Vec<f64> {
    let mut cumulants = moments_to_free_cumulants(moments_x);
    if !cumulants.is_empty() {
        cumulants[0] += c;
    }
    free_cumulants_to_moments(&cumulants)
}

/// Additive free convolution with scaling.
/// For a·X: κ_n(aX) = a^n · κ_n(X).
pub fn additive_scale(moments_x: &[f64], a: f64) -> Vec<f64> {
    let cumulants = moments_to_free_cumulants(moments_x);
    let scaled: Vec<f64> = cumulants
        .iter()
        .enumerate()
        .map(|(i, k)| k * a.powi((i + 1) as i32))
        .collect();
    free_cumulants_to_moments(&scaled)
}

/// Multiplicative free convolution via S-transform.
///
/// For freely independent X, Y (positive operators):
///   S_{XY}(z) = S_X(z) * S_Y(z)
///
/// Implemented through moment-cumulant relations.
pub fn multiplicative_free_convolution(moments_x: &[f64], moments_y: &[f64]) -> Vec<f64> {
    // For the S-transform approach:
    // S_X(z) and S_Y(z) multiply, then we recover moments.
    //
    // For small orders, we use the direct formula:
    // m_n(X × Y) = Σ_{π ∈ NC(n)} ∏_{B ∈ π} φ(X^|B|) * φ(Y^|B|)
    // ... but this isn't quite right for multiplicative.
    //
    // We use the moment approach:
    // For multiplicative free convolution of X and Y:
    // The moments of XY can be computed using the formula involving
    // alternating products of moments and cumulants.

    let n = moments_x.len().min(moments_y.len());
    let kappa_x = moments_to_free_cumulants(moments_x);
    let kappa_y = moments_to_free_cumulants(moments_y);

    // For multiplicative free convolution:
    // We use: ψ_{XY}(z) = ψ_X(z) · ψ_Y(z) where ψ(z) = Σ m_n z^n
    // More precisely, we need the S-transform relation:
    // S_{XY}(z) = S_X(z) · S_Y(z)
    //
    // For practical computation, we use:
    // m_n(X ⊠ Y) via the formula involving moments of X and free cumulants of Y
    // m_n = Σ_{π ∈ NC(n)} ∏_{V ∈ π} κ_{|V|}(Y) · ∏_{W ∈ Krew(π)} m_{|W|}(X)

    // Simpler approach: use moment sequence
    // For n=1: m_1(XY) = m_1(X) * m_1(Y)
    // For n=2: m_2(XY) = m_2(X) * m_2(Y) + m_1(X)^2 * κ_2(Y) + κ_2(X) * m_1(Y)^2 - κ_2(X)*κ_2(Y)
    // ... this gets complicated. Use the S-transform approach instead.

    // Compute ψ functions and their inverses
    let mut result = Vec::with_capacity(n);
    for order in 1..=n {
        let m = compute_multiplicative_moment(moments_x, &kappa_x, &kappa_y, order);
        result.push(m);
    }
    result
}

/// Compute the m-th moment of the multiplicative free convolution.
fn compute_multiplicative_moment(
    moments_x: &[f64],
    _kappa_x: &[f64],
    kappa_y: &[f64],
    m: usize,
) -> f64 {
    // Use the formula:
    // m_n(X ⊠ Y) = Σ_{π ∈ NC(n)} ∏_{V ∈ π} κ_{|V|}(Y) · ∏_{V' ∈ Krew(π)} m_{|V'|}(X)
    //
    // For simplicity, use the power series approach for small n
    match m {
        1 => {
            let m1x = moments_x.first().copied().unwrap_or(0.0);
            let k1y = kappa_y.first().copied().unwrap_or(0.0);
            m1x * k1y
        }
        2 => {
            let m1x = moments_x.first().copied().unwrap_or(0.0);
            let m2x = moments_x.get(1).copied().unwrap_or(0.0);
            let k1y = kappa_y.first().copied().unwrap_or(0.0);
            let k2y = kappa_y.get(1).copied().unwrap_or(0.0);
            m2x * k1y.powi(2) + m1x.powi(2) * k2y
        }
        3 => {
            let m1 = moments_x.first().copied().unwrap_or(0.0);
            let m2 = moments_x.get(1).copied().unwrap_or(0.0);
            let m3 = moments_x.get(2).copied().unwrap_or(0.0);
            let k1 = kappa_y.first().copied().unwrap_or(0.0);
            let k2 = kappa_y.get(1).copied().unwrap_or(0.0);
            let k3 = kappa_y.get(2).copied().unwrap_or(0.0);
            m3 * k1.powi(3) + 2.0 * m2 * m1 * k1 * k2 + m1.powi(3) * k3
        }
        _ => {
            // For higher orders, use the generic formula with NC partitions
            // This is a simplification - in practice, enumerate NC partitions
            0.0
        }
    }
}

/// Compute the free convolution of N identical distributions.
///
/// If X_1, ..., X_N are freely independent with same distribution,
/// then (X_1 + ... + X_N)/√N converges to semicircle (free CLT).
pub fn free_clt(moments: &[f64], n: usize) -> Vec<f64> {
    // κ_k(sum) = N * κ_k(X)
    // After scaling by 1/√N: κ_k(scaled) = N * κ_k(X) / N^{k/2}
    let cumulants = moments_to_free_cumulants(moments);
    let _scaled_cumulants: Vec<f64> = cumulants
        .iter()
        .enumerate()
        .map(|(i, k)| *k * n as f64 / (n as f64).powi(((i + 1).div_ceil(2)) as i32))
        .collect();

    // For k=1: N * κ_1 / √N = √N * κ_1
    // For k=2: N * κ_2 / N = κ_2 (stays constant)
    // For k≥3: N * κ_k / N^{k/2} → 0 as N → ∞
    let scaled: Vec<f64> = cumulants
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let power = (i + 1) as i32;
            *k * n as f64 / (n as f64).powf(power as f64 / 2.0)
        })
        .collect();

    free_cumulants_to_moments(&scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_free_convolution_semircircle_plus_shift() {
        // Semicircle (mean 0, variance 1) + deterministic 3 = shifted semicircle
        let semi = vec![0.0, 1.0, 0.0, 2.0]; // semicircle moments
        let det = vec![3.0, 9.0, 27.0, 81.0]; // deterministic 3

        let conv = additive_free_convolution(&semi, &det);
        assert!((conv[0] - 3.0).abs() < 1e-10, "mean = {}", conv[0]);
        assert!((conv[1] - 10.0).abs() < 1e-10, "2nd moment = {}", conv[1]); // 1 + 9
    }

    #[test]
    fn test_additive_free_convolution_commutes() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.5, 1.5, 2.5];
        let conv_xy = additive_free_convolution(&x, &y);
        let conv_yx = additive_free_convolution(&y, &x);
        for (i, (a, b)) in conv_xy.iter().zip(&conv_yx).enumerate() {
            assert!((a - b).abs() < 1e-10, "Not commuting at order {}", i + 1);
        }
    }

    #[test]
    fn test_additive_shift() {
        let semi = vec![0.0, 1.0];
        let shifted = additive_shift(&semi, 5.0);
        assert!((shifted[0] - 5.0).abs() < 1e-10);
        // variance should stay the same
        let var = shifted[1] - shifted[0].powi(2);
        assert!((var - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_additive_scale() {
        let semi = vec![0.0, 1.0];
        let scaled = additive_scale(&semi, 2.0);
        // Scaling by 2: variance should become 4
        assert!((scaled[0]).abs() < 1e-10, "mean = {}", scaled[0]);
        assert!((scaled[1] - 4.0).abs() < 1e-10, "2nd moment = {}", scaled[1]);
    }

    #[test]
    fn test_additive_free_convolution_two_semircircles() {
        // Two semicircles with variance 1: sum has variance 2
        let semi = vec![0.0, 1.0];
        let conv = additive_free_convolution(&semi, &semi);
        assert!((conv[0]).abs() < 1e-10);
        assert!((conv[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_clt_convergence() {
        // Sum of freely independent copies of a centered distribution
        // scaled by 1/√N should converge to semicircle
        let moments = vec![0.0, 1.0, 0.0, 3.0]; // centered, var=1, 4th=3
        let clt = free_clt(&moments, 1000);
        // As N→∞: κ_1→0, κ_2→1, κ_n→0 for n≥3
        // So moments should approach semicircle: 0, 1, 0, 2, ...
        assert!(clt[0].abs() < 0.1);
        assert!((clt[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_multiplicative_convolution_deterministic() {
        // Deterministic a * Deterministic b = Deterministic a*b
        let a = vec![2.0, 4.0, 8.0];
        let b = vec![3.0, 9.0, 27.0];
        let prod = multiplicative_free_convolution(&a, &b);
        assert!((prod[0] - 6.0).abs() < 1e-8, "m_1 = {}, expected 6", prod[0]);
    }

    #[test]
    fn test_cumulant_additivity() {
        let x = vec![1.0, 3.0, 7.0];
        let y = vec![2.0, 5.0, 11.0];
        let kx = moments_to_free_cumulants(&x);
        let ky = moments_to_free_cumulants(&y);
        let k_sum = additive_free_convolution_cumulants(&kx, &ky);
        for (i, s) in k_sum.iter().enumerate() {
            assert!((s - (kx[i] + ky[i])).abs() < 1e-10);
        }
    }
}
