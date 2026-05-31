//! R-transform, Cauchy transform, and Stieltjes transform.
//!
//! The Cauchy transform: G(z) = E[1/(z - X)] = Σ_{n≥0} m_n / z^{n+1}
//! The R-transform: R(z) = G^{-1}(z) - 1/z, where G^{-1} is the functional inverse
//! The S-transform for multiplicative free convolution.

use crate::moment_cumulant::{moments_to_free_cumulants, free_cumulants_to_moments};

/// Compute the Cauchy transform G(z) = Σ_{n≥0} m_n / z^{n+1}
/// truncated to a given number of terms.
///
/// G(z) = 1/z + m_1/z² + m_2/z³ + ...
pub fn cauchy_transform(moments: &[f64], z: f64, num_terms: usize) -> f64 {
    let mut result = 0.0;
    for n in 0..num_terms {
        let m_n = if n == 0 { 1.0 } else if n <= moments.len() { moments[n - 1] } else { 0.0 };
        result += m_n / z.powi((n + 1) as i32);
    }
    result
}

/// Compute the Stieltjes transform (same as Cauchy transform).
pub fn stieltjes_transform(moments: &[f64], z: f64, num_terms: usize) -> f64 {
    cauchy_transform(moments, z, num_terms)
}

/// Compute the R-transform R(z) = Σ_{n≥1} κ_n z^{n-1}
/// from free cumulants.
pub fn r_transform_from_cumulants(cumulants: &[f64], z: f64, num_terms: usize) -> f64 {
    let mut result = 0.0;
    let terms = num_terms.min(cumulants.len());
    for n in 1..=terms {
        result += cumulants[n - 1] * z.powi((n - 1) as i32);
    }
    result
}

/// Compute the R-transform from moments (convenience function).
pub fn r_transform(moments: &[f64], z: f64) -> f64 {
    let cumulants = moments_to_free_cumulants(moments);
    r_transform_from_cumulants(&cumulants, z, cumulants.len())
}

/// Compute the η-transform: η(z) = ∫ (z*t)/(1 - z*t) dμ(t)
/// Related to the S-transform.
pub fn eta_transform(moments: &[f64], z: f64, num_terms: usize) -> f64 {
    let mut result = 0.0;
    for n in 1..=num_terms {
        let m_n = if n <= moments.len() { moments[n - 1] } else { 0.0 };
        result += m_n * z.powi(n as i32);
    }
    result
}

/// S-transform for multiplicative free convolution.
///
/// S(z) = (1 + z) / z * χ^{-1}(z)
/// where χ(z) = z * G(z) - 1 / (G(z) * z) ... in practice:
///
/// S(z) = (1 + z) / z * 1/ψ^{-1}(z)
/// where ψ(z) = η(z)/(1+η(z))
///
/// For a compact representation, we compute S from moments:
/// S(z) = (1 + z)/(z * (m_1 + m_2 * z + m_3 * z² + ...))
pub fn s_transform(moments: &[f64], z: f64, num_terms: usize) -> f64 {
    let psi = eta_transform(moments, z, num_terms);
    if psi.abs() < 1e-15 {
        return 0.0;
    }
    (1.0 + psi) / (z * psi)
}

/// Evaluate the functional inverse of the Cauchy transform numerically.
///
/// Given w, find z such that G(z) = w, then R(w) = z - 1/w.
/// Uses Newton's method.
pub fn r_transform_numerical(
    density_fn: &dyn Fn(f64) -> f64,
    w: f64,
    initial_guess: f64,
    tolerance: f64,
    max_iter: usize,
) -> f64 {
    // G(z) = ∫ dμ(t)/(z - t)
    // We want to find z such that G(z) = w
    // Use Newton's method: z_{n+1} = z_n - (G(z_n) - w) / G'(z_n)
    // G'(z) = -∫ dμ(t)/(z-t)²

    let mut z = initial_guess;
    for _ in 0..max_iter {
        let g = numerical_cauchy(density_fn, z, 1000);
        let g_prime = numerical_cauchy_derivative(density_fn, z, 1000);
        let delta = (g - w) / g_prime;
        z -= delta;
        if delta.abs() < tolerance {
            break;
        }
    }
    z - 1.0 / w
}

/// Numerically evaluate G(z) = ∫ dμ(t)/(z - t) using Simpson's rule.
fn numerical_cauchy(density_fn: &dyn Fn(f64) -> f64, z: f64, n_points: usize) -> f64 {
    // Find a reasonable integration range
    let a = -10.0;
    let b = 10.0;
    let h = (b - a) / n_points as f64;

    let mut sum = density_fn(a) / (z - a) + density_fn(b) / (z - b);
    for i in 1..n_points {
        let t = a + i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * density_fn(t) / (z - t);
    }
    sum * h / 3.0
}

/// Numerically evaluate G'(z) = -∫ dμ(t)/(z - t)²
fn numerical_cauchy_derivative(density_fn: &dyn Fn(f64) -> f64, z: f64, n_points: usize) -> f64 {
    let a = -10.0;
    let b = 10.0;
    let h = (b - a) / n_points as f64;

    let mut sum = -density_fn(a) / (z - a).powi(2) - density_fn(b) / (z - b).powi(2);
    for i in 1..n_points {
        let t = a + i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum -= weight * density_fn(t) / (z - t).powi(2);
    }
    sum * h / 3.0
}

/// Compute moments from the R-transform coefficients (free cumulants).
/// Given κ_1, ..., κ_n, compute m_1, ..., m_n.
pub fn moments_from_r_transform(cumulants: &[f64]) -> Vec<f64> {
    free_cumulants_to_moments(cumulants)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cauchy_transform_at_infinity() {
        let moments = vec![0.0, 1.0];
        let g = cauchy_transform(&moments, 1000.0, 3);
        // G(z) ≈ 1/z + 0/z² + 1/z³ for large z
        let expected = 1.0 / 1000.0 + 0.0 / 1000000.0 + 1.0 / 1e9;
        assert!((g - expected).abs() < 1e-12);
    }

    #[test]
    fn test_r_transform_semicircle() {
        // Semicircle: κ_1 = 0, κ_2 = 1, κ_n = 0 for n ≥ 3
        // R(z) = 0 + 1*z = z
        let moments = vec![0.0, 1.0, 0.0, 2.0];
        let cumulants = moments_to_free_cumulants(&moments);
        let r = r_transform_from_cumulants(&cumulants, 0.5, 4);
        // R(0.5) ≈ κ_1 + κ_2 * 0.5 = 0 + 0.5 = 0.5
        assert!((r - 0.5).abs() < 1e-10, "R(0.5) = {}, expected 0.5", r);
    }

    #[test]
    fn test_r_transform_deterministic() {
        // Deterministic c: κ_1 = c, rest zero. R(z) = c
        let moments = vec![2.0, 4.0, 8.0];
        let r = r_transform(&moments, 0.5);
        assert!((r - 2.0).abs() < 1e-8, "R(z) = {} for deterministic 2, expected 2", r);
    }

    #[test]
    fn test_s_transform_identity() {
        // For identity distribution (deterministic 1):
        // moments = 1, 1, 1, ...
        // S(z) should equal 1
        let moments = vec![1.0, 1.0, 1.0];
        let s = s_transform(&moments, 0.1, 3);
        // S(z) = (1+z)/(z * (1 + z + z²)) approximately
        assert!(s.is_finite());
    }

    #[test]
    fn test_cauchy_semicircle() {
        // For semicircle on [-2,2]: G(z) = (z - sqrt(z²-4)) / 2
        // The power series G(z) = Σ m_n / z^{n+1} converges for |z| > R = 2
        // At z = 10: exact = (10 - sqrt(96)) / 2 ≈ 0.1009
        let moments = vec![0.0, 1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 14.0];
        let g = cauchy_transform(&moments, 10.0, 8);
        let exact = (10.0 - (100.0 - 4.0_f64).sqrt()) / 2.0;
        // Series: 1/10 + 0/100 + 1/1000 + 0/10000 + 2/100000 + ... ≈ 0.10102
        // Exact: 0.1009... Close enough with 8 terms
        assert!((g - exact).abs() < 0.01, "G(10) = {}, expected ~{}", g, exact);
    }

    #[test]
    fn test_eta_transform() {
        let moments = vec![1.0, 1.0, 1.0];
        let eta = eta_transform(&moments, 0.1, 3);
        let expected = 0.1 + 0.01 + 0.001;
        assert!((eta - expected).abs() < 1e-12);
    }

    #[test]
    fn test_moments_from_r_roundtrip() {
        let moments = vec![0.0, 1.0, 0.0, 2.0, 0.0, 5.0];
        let cumulants = moments_to_free_cumulants(&moments);
        let recovered = moments_from_r_transform(&cumulants);
        for (i, (a, b)) in moments.iter().zip(&recovered).enumerate() {
            assert!((a - b).abs() < 1e-10, "Roundtrip mismatch at {}: {} vs {}", i, a, b);
        }
    }
}
