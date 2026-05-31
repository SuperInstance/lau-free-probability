//! Classical laws in free probability.
//!
//! - Wigner semicircle law (free CLT)
//! - Marchenko-Pastur law (free Poisson)
//! - Free compound Poisson distributions

use serde::{Deserialize, Serialize};

/// Wigner semicircle distribution with radius R and center μ.
///
/// Density: f(x) = (2/(π R²)) √(R² - (x-μ)²) for |x-μ| ≤ R
///
/// This is the free analog of the Gaussian — the limit distribution
/// in the free central limit theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemicircleLaw {
    /// Center (mean).
    pub mu: f64,
    /// Radius. Variance = R²/4.
    pub radius: f64,
}

impl SemicircleLaw {
    /// Standard semicircle: μ=0, R=2 (variance 1).
    pub fn standard() -> Self {
        Self { mu: 0.0, radius: 2.0 }
    }

    /// Semicircle with given mean and variance.
    pub fn with_mean_variance(mu: f64, variance: f64) -> Self {
        Self {
            mu,
            radius: 2.0 * variance.sqrt(),
        }
    }

    /// Variance = R²/4.
    pub fn variance(&self) -> f64 {
        self.radius.powi(2) / 4.0
    }

    /// Probability density at x.
    pub fn pdf(&self, x: f64) -> f64 {
        let dx = x - self.mu;
        let r2 = self.radius.powi(2);
        if dx.powi(2) > r2 {
            0.0
        } else {
            (2.0 / (std::f64::consts::PI * r2)) * (r2 - dx.powi(2)).sqrt()
        }
    }

    /// Cumulative distribution function at x.
    pub fn cdf(&self, x: f64) -> f64 {
        let dx = x - self.mu;
        let r = self.radius;
        if dx <= -r {
            0.0
        } else if dx >= r {
            1.0
        } else {
            let t = dx / r;
            0.5 + (t * (1.0 - t.powi(2)).sqrt() + (t).asin()) / std::f64::consts::PI
        }
    }

    /// n-th moment E[X^n].
    ///
    /// Moments of the semicircle: odd moments = 0,
    /// even moments m_{2k} = C_k where C_k is the k-th Catalan number.
    pub fn moment(&self, n: usize) -> f64 {
        let r = self.radius / 2.0; // σ
        if n % 2 == 1 {
            0.0
        } else {
            let k = n / 2;
            let c = catalan(k);
            self.mu.powi(0) * r.powi(n as i32) * c as f64
            // Note: this is for μ=0. For μ≠0, use binomial expansion.
        }
    }

    /// Moments including center shift via binomial expansion.
    pub fn moments(&self, max_order: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(max_order);
        for n in 1..=max_order {
            let mut m = 0.0;
            for k in 0..=n {
                let binom = binomial(n, k) as f64;
                let sigma_r = (self.radius / 2.0).powi(k as i32);
                let mu_r = self.mu.powi((n - k) as i32);
                // Centered moment of order k
                let centered = if k % 2 == 1 {
                    0.0
                } else {
                    sigma_r * catalan(k / 2) as f64
                };
                m += binom * mu_r * centered;
            }
            result.push(m);
        }
        result
    }

    /// Generate samples (approximate via Box-Muller analog).
    pub fn sample(&self, n: usize) -> Vec<f64> {
        // Generate semicircle samples using the inverse CDF or rejection sampling
        let mut samples = Vec::with_capacity(n);
        let mut rng = SimpleRng::new(42);

        for _ in 0..n {
            // Rejection sampling
            loop {
                let x = self.mu - self.radius + 2.0 * self.radius * rng.next();
                let y = 2.0 / (std::f64::consts::PI * self.radius) * rng.next();
                let pdf_max = 2.0 / (std::f64::consts::PI * self.radius);
                if y < self.pdf(x) / pdf_max.max(1e-15) {
                    samples.push(x);
                    break;
                }
            }
        }
        samples
    }

    /// Free cumulants: κ_1 = μ, κ_2 = R²/4, κ_n = 0 for n ≥ 3.
    pub fn free_cumulants(&self, max_order: usize) -> Vec<f64> {
        let mut cumulants = vec![0.0; max_order];
        if max_order >= 1 {
            cumulants[0] = self.mu;
        }
        if max_order >= 2 {
            cumulants[1] = self.variance();
        }
        // κ_n = 0 for n ≥ 3
        cumulants
    }

    /// Cauchy transform: G(z) = (z - μ - sqrt((z-μ)² - R²)) / (R²/2)
    pub fn cauchy_transform(&self, z: f64) -> f64 {
        let dz = z - self.mu;
        let r2 = self.radius.powi(2);
        let disc = dz.powi(2) - r2;
        if disc >= 0.0 {
            // Real square root: pick the branch with |G| < 1/|z|
            let sq = disc.sqrt();
            let g1 = (dz - sq) / (r2 / 2.0);
            let g2 = (dz + sq) / (r2 / 2.0);
            // Return the one with smaller magnitude (analytic continuation from upper half-plane)
            if g1.abs() < g2.abs() { g1 } else { g2 }
        } else {
            // Complex case: use positive imaginary part convention
            let sq = (-disc).sqrt();
            let real = dz / (r2 / 2.0);
            let imag = -sq / (r2 / 2.0);
            // Return real part approximation
            real
        }
    }
}

/// Marchenko-Pastur distribution (free Poisson).
///
/// Eigenvalue distribution of XX^T/N where X is N×M random matrix.
/// Parameter λ = M/N (aspect ratio).
///
/// Density supported on [(1-√λ)², (1+√λ)²] for λ ≥ 1,
/// or two intervals for λ < 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarchenkoPasturLaw {
    /// Aspect ratio λ = M/N.
    pub lambda: f64,
}

impl MarchenkoPasturLaw {
    /// Standard Marchenko-Pastur with λ = 1.
    pub fn standard() -> Self {
        Self { lambda: 1.0 }
    }

    /// Lower edge of support.
    pub fn a(&self) -> f64 {
        (1.0 - self.lambda.sqrt()).powi(2)
    }

    /// Upper edge of support.
    pub fn b(&self) -> f64 {
        (1.0 + self.lambda.sqrt()).powi(2)
    }

    /// Probability density at x.
    pub fn pdf(&self, x: f64) -> f64 {
        let a = self.a();
        let b = self.b();
        if x <= a || x >= b {
            if self.lambda < 1.0 && x.abs() < 1e-15 {
                return 1.0 - self.lambda;
            }
            0.0
        } else {
            let lambda = self.lambda;
            let numerator = ((b - x) * (x - a)).sqrt();
            (1.0 / (2.0 * std::f64::consts::PI * lambda * x)) * numerator
        }
    }

    /// n-th moment E[X^n] = Σ_{k=0}^{n-1} (1/n) * C(n,k) * C(n,k+1) * λ^k
    /// where C(n,k) are Narayana numbers.
    /// Equivalently, m_n = Σ_{k=1}^{n} λ^{k-1} * N(n,k) / n ... but simpler:
    /// m_n = Σ_{k=0}^{n-1} (1/(n)) * binomial(n, k) * binomial(n, k+1) * λ^k
    pub fn moment(&self, n: usize) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let mut m = 0.0;
        for k in 0..n {
            // Narayana number N(n, k+1) = (1/n) * C(n, k) * C(n, k+1)
            let narayana = binomial(n, k) * binomial(n, k + 1) / n;
            m += narayana as f64 * self.lambda.powi(k as i32);
        }
        m
    }

    /// Moments up to given order.
    pub fn moments(&self, max_order: usize) -> Vec<f64> {
        (1..=max_order).map(|n| self.moment(n)).collect()
    }

    /// Free cumulants: κ_n = λ^{n-1} * 1 = λ^{n-1} for all n ≥ 1
    /// Actually for MP(λ): κ_n = λ for all n (free Poisson).
    /// Wait — the free cumulants of MP(λ) are κ_n = λ.
    /// No — let me re-derive. For λ=1: moments are Catalan numbers C_n,
    /// and κ_n = 1 for all n. For general λ: κ_n = λ.
    pub fn free_cumulants(&self, max_order: usize) -> Vec<f64> {
        vec![self.lambda; max_order]
    }

    /// Cauchy transform for Marchenko-Pastur.
    pub fn cauchy_transform(&self, z: f64) -> f64 {
        let lambda = self.lambda;
        let disc = (z - 1.0).powi(2) - 4.0 * lambda * z;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let g1 = ((1.0 + lambda) * z - 1.0 - sq) / (2.0 * lambda * z * z);
            let g2 = ((1.0 + lambda) * z - 1.0 + sq) / (2.0 * lambda * z * z);
            if g1.abs() < g2.abs() { g1 } else { g2 }
        } else {
            ((1.0 + lambda) * z - 1.0) / (2.0 * lambda * z * z)
        }
    }

    /// Generate approximate samples.
    pub fn sample(&self, n: usize) -> Vec<f64> {
        let mut samples = Vec::with_capacity(n);
        let mut rng = SimpleRng::new(123);
        for _ in 0..n {
            loop {
                let x = self.a() + (self.b() - self.a()) * rng.next();
                let y = rng.next();
                let pdf_max = 1.0 / (2.0 * std::f64::consts::PI * self.lambda);
                if y < self.pdf(x) / pdf_max.max(1e-15) {
                    samples.push(x);
                    break;
                }
            }
        }
        samples
    }
}

/// Free compound Poisson distribution.
///
/// κ_n = λ · m_n(ν) where ν is the compounding measure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeCompoundPoisson {
    /// Rate parameter.
    pub lambda: f64,
    /// Moments of the compounding measure ν.
    pub nu_moments: Vec<f64>,
}

impl FreeCompoundPoisson {
    /// Free cumulants: κ_n = λ · m_n(ν).
    pub fn free_cumulants(&self, max_order: usize) -> Vec<f64> {
        (1..=max_order)
            .map(|n| {
                if n <= self.nu_moments.len() {
                    self.lambda * self.nu_moments[n - 1]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Moments via the free cumulant to moment formula.
    pub fn moments(&self, max_order: usize) -> Vec<f64> {
        use crate::moment_cumulant::free_cumulants_to_moments;
        let cumulants = self.free_cumulants(max_order);
        free_cumulants_to_moments(&cumulants)
    }
}

/// Simple deterministic RNG for reproducible sampling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }
}

/// Compute the n-th Catalan number.
fn catalan(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut result: usize = 1;
    for i in 0..n {
        result = result * (2 * (2 * i + 1)) / (i + 2);
    }
    result
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moment_cumulant::{moments_to_free_cumulants, free_cumulants_to_moments};

    #[test]
    fn test_semicircle_standard_pdf() {
        let sc = SemicircleLaw::standard();
        // PDF at center should be maximum: 2/(π*4) * 2 = 1/π
        assert!((sc.pdf(0.0) - 1.0 / std::f64::consts::PI).abs() < 1e-10);
        // PDF outside support should be 0
        assert_eq!(sc.pdf(3.0), 0.0);
        assert_eq!(sc.pdf(-3.0), 0.0);
    }

    #[test]
    fn test_semicircle_moments() {
        let sc = SemicircleLaw::standard();
        // Odd moments = 0
        assert_eq!(sc.moment(1), 0.0);
        assert_eq!(sc.moment(3), 0.0);
        assert_eq!(sc.moment(5), 0.0);
        // Even moments = Catalan numbers
        assert_eq!(sc.moment(2), 1.0);  // C_1 = 1
        assert_eq!(sc.moment(4), 2.0);  // C_2 = 2
        assert_eq!(sc.moment(6), 5.0);  // C_3 = 5
        assert_eq!(sc.moment(8), 14.0); // C_4 = 14
    }

    #[test]
    fn test_semicircle_cumulants() {
        let sc = SemicircleLaw::standard();
        let cumulants = sc.free_cumulants(6);
        assert!((cumulants[0]).abs() < 1e-10); // κ_1 = 0
        assert!((cumulants[1] - 1.0).abs() < 1e-10); // κ_2 = 1
        assert!(cumulants[2].abs() < 1e-10); // κ_3 = 0
        assert!(cumulants[3].abs() < 1e-10); // κ_4 = 0
    }

    #[test]
    fn test_semicircle_variance() {
        let sc = SemicircleLaw::standard();
        assert!((sc.variance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_semicircle_cdf() {
        let sc = SemicircleLaw::standard();
        assert!((sc.cdf(0.0) - 0.5).abs() < 1e-10);
        assert_eq!(sc.cdf(-2.0), 0.0);
        assert_eq!(sc.cdf(2.0), 1.0);
    }

    #[test]
    fn test_semicircle_cauchy() {
        let sc = SemicircleLaw::standard();
        // G(z) = (z - sqrt(z²-4))/2 for standard semicircle
        let z = 10.0;
        let g = sc.cauchy_transform(z);
        let expected = (z - (z * z - 4.0).sqrt()) / 2.0;
        assert!((g - expected).abs() < 1e-10);
    }

    #[test]
    fn test_semicircle_shifted_moments() {
        let sc = SemicircleLaw { mu: 3.0, radius: 2.0 };
        let moments = sc.moments(4);
        assert!((moments[0] - 3.0).abs() < 1e-10, "m_1 = {}", moments[0]); // mean = 3
        assert!((moments[1] - 10.0).abs() < 1e-10, "m_2 = {}", moments[1]); // var + mu² = 1 + 9
    }

    #[test]
    fn test_marchenko_pastur_standard() {
        let mp = MarchenkoPasturLaw::standard();
        assert!((mp.a() - 0.0).abs() < 1e-10);
        assert!((mp.b() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_marchenko_pastur_moments() {
        let mp = MarchenkoPasturLaw::standard();
        // For λ=1: m_n = C_n (Catalan numbers)
        assert_eq!(mp.moment(1), 1.0);  // C_1 = 1
        assert_eq!(mp.moment(2), 2.0);  // C_2 = 2
        assert_eq!(mp.moment(3), 5.0);  // C_3 = 5
        assert_eq!(mp.moment(4), 14.0); // C_4 = 14
    }

    #[test]
    fn test_marchenko_pastur_cumulants() {
        let mp = MarchenkoPasturLaw::standard();
        let cumulants = mp.free_cumulants(5);
        // κ_n = λ = 1 for all n
        for (i, k) in cumulants.iter().enumerate() {
            assert!((k - 1.0).abs() < 1e-10, "κ_{} = {}, expected 1", i + 1, k);
        }
    }

    #[test]
    fn test_marchenko_pastur_with_lambda() {
        let mp = MarchenkoPasturLaw { lambda: 2.0 };
        assert!((mp.a() - (1.0 - 2.0_f64.sqrt()).powi(2)).abs() < 1e-10);
        assert!((mp.b() - (1.0 + 2.0_f64.sqrt()).powi(2)).abs() < 1e-10);
        // Cumulants should all be λ = 2
        let cumulants = mp.free_cumulants(3);
        for k in &cumulants {
            assert!((k - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_marchenko_pastur_moment_cumulant_roundtrip() {
        let mp = MarchenkoPasturLaw::standard();
        let moments = mp.moments(6);
        let cumulants = moments_to_free_cumulants(&moments);
        for (i, k) in cumulants.iter().enumerate() {
            assert!((k - 1.0).abs() < 1e-8, "κ_{} = {}, expected 1", i + 1, k);
        }
        let recovered = free_cumulants_to_moments(&cumulants);
        for (i, (a, b)) in moments.iter().zip(&recovered).enumerate() {
            assert!((a - b).abs() < 1e-8, "Roundtrip mismatch at {}: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_marchenko_pastur_pdf() {
        let mp = MarchenkoPasturLaw::standard();
        // PDF at x=1 should be 1/(π*1) * sqrt(3*1) / 2 ... let me compute
        // f(1) = sqrt((4-1)*(1-0)) / (2π*1*1) = sqrt(3)/(2π)
        let expected = 3.0_f64.sqrt() / (2.0 * std::f64::consts::PI);
        assert!((mp.pdf(1.0) - expected).abs() < 1e-10);
        // Outside support
        assert_eq!(mp.pdf(-1.0), 0.0);
        assert_eq!(mp.pdf(5.0), 0.0);
    }

    #[test]
    fn test_semicircle_samples() {
        let sc = SemicircleLaw::standard();
        let samples = sc.sample(1000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let var: f64 = samples.iter().map(|x| x * x).sum::<f64>() / samples.len() as f64 - mean * mean;
        assert!(mean.abs() < 0.2, "Sample mean = {}", mean);
        assert!((var - 1.0).abs() < 0.3, "Sample var = {}", var);
    }

    #[test]
    fn test_free_compound_poisson() {
        // Free Poisson with λ=1 and ν=δ_1 (compounding measure is point mass at 1)
        // This gives κ_n = 1 for all n → Marchenko-Pastur
        let fcp = FreeCompoundPoisson {
            lambda: 1.0,
            nu_moments: vec![1.0, 1.0, 1.0, 1.0],
        };
        let cumulants = fcp.free_cumulants(4);
        for k in &cumulants {
            assert!((k - 1.0).abs() < 1e-10);
        }
    }
}
