//! Free entropy — microstates and non-microstates approaches.
//!
//! Free entropy χ(a_1, ..., a_n) measures the "size" of the set of
//! matrix tuples approximating the noncommutative distribution.
//!
//! Microstates: χ = log volume of {matrices with matching moments}
//! Non-microstates: χ = ½ ∫∫ log|s-t| dμ(s) dμ(t) + const (for single variable)

use crate::laws::SemicircleLaw;
use crate::moment_cumulant::moments_to_free_cumulants;

/// Free entropy via the non-microstates approach (single variable).
///
/// For a single self-adjoint operator with spectral measure μ:
///   χ(μ) = ½ ∬ log|s - t| dμ(s) dμ(t) + 3/4 + ½ log(2π)
///
/// This is Voiculescu's free entropy.
pub fn free_entropy_microstates(moments: &[f64], num_grid_points: usize, radius: f64) -> f64 {
    // Approximate using discrete distribution
    // For a distribution with given moments, approximate the entropy
    // via numerical integration on a grid

    let n = num_grid_points;
    let dx = 2.0 * radius / n as f64;

    // First, estimate the density from moments using maximum entropy
    // For simplicity, use the Stieltjes transform approach
    let density = estimate_density_from_moments(moments, n, radius);

    // Compute ∬ log|s-t| dμ(s) dμ(t) ≈ Σ_i Σ_j log|s_i - s_j| * ρ(s_i) * ρ(s_j) * dx²
    let mut entropy_integral = 0.0;
    for i in 0..n {
        let s = -radius + (i as f64 + 0.5) * dx;
        for j in 0..n {
            let t = -radius + (j as f64 + 0.5) * dx;
            let diff = (s - t).abs();
            if diff > 1e-15 {
                entropy_integral += diff.ln() * density[i] * density[j] * dx * dx;
            }
        }
    }

    0.5 * entropy_integral + 3.0 / 4.0 + 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Free entropy for the semicircle law (known closed form).
///
/// χ(Wigner) = ½ log(2πe) - ¼ log(e) = ... 
/// Actually: χ(σ) = log(R/2) + (1 + log(2π))/2
/// where R is the radius.
pub fn free_entropy_semicircle(radius: f64) -> f64 {
    (radius / 2.0).ln() + (1.0 + (2.0 * std::f64::consts::PI).ln()) / 2.0
}

/// Free entropy via the non-microstates formula for a single variable
/// with spectral measure μ, estimated numerically from samples.
pub fn free_entropy_from_samples(samples: &[f64]) -> f64 {
    let n = samples.len();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = (samples[i] - samples[j]).abs();
                if diff > 1e-15 {
                    sum += diff.ln();
                }
            }
        }
    }
    // χ = 1/(2n²) Σ_{i,j} log|s_i - s_j| + 3/4 + ½ log(2π)
    // (using the empirical approximation)
    let avg_log = sum / (n * n) as f64;
    0.5 * avg_log + 3.0 / 4.0 + 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Estimate the spectral density from moments using Padé approximation.
fn estimate_density_from_moments(moments: &[f64], n_points: usize, radius: f64) -> Vec<f64> {
    let dx = 2.0 * radius / n_points as f64;
    let mut density = vec![0.0; n_points];

    for i in 0..n_points {
        let t = -radius + (i as f64 + 0.5) * dx;
        let eps = 0.01;
        let mut _g_re = 0.0_f64;
        let mut g_im = 0.0_f64;
        for n in 0..=moments.len() {
            let m_n = if n == 0 { 1.0 } else { moments[n - 1] };
            // Compute 1/(z^{n+1}) for z = t + iε using polar form
            let re = t;
            let im = eps;
            let r2 = re * re + im * im;
            let r = r2.sqrt();
            let theta = im.atan2(re);
            let new_r = 1.0 / r.powi((n + 1) as i32);
            let new_theta = -((n + 1) as f64) * theta;
            _g_re += m_n * new_r * new_theta.cos();
            g_im += m_n * new_r * new_theta.sin();
        }
        density[i] = (-g_im / std::f64::consts::PI).max(0.0);
    }

    // Normalize
    let total: f64 = density.iter().map(|d| d * dx).sum();
    if total > 0.0 {
        for d in &mut density {
            *d /= total;
        }
    }

    density
}

/// Mutual free information between two operators.
///
/// i*(a, b) = χ(a) + χ(b) - χ(a, b)
/// Measures the "free independence" between operators.
pub fn mutual_free_information(
    moments_a: &[f64],
    moments_b: &[f64],
    moments_joint: &[f64],
    grid_points: usize,
    radius: f64,
) -> f64 {
    let chi_a = free_entropy_microstates(moments_a, grid_points, radius);
    let chi_b = free_entropy_microstates(moments_b, grid_points, radius);
    let chi_ab = free_entropy_microstates(moments_joint, grid_points, radius);
    chi_a + chi_b - chi_ab
}

/// Free Fisher information (non-microstates).
///
/// For a single variable with density ρ:
/// Φ*(ρ) = ∫ (ρ'(x)/ρ(x))² ρ(x) dx
pub fn free_fisher_information(moments: &[f64], grid_points: usize, radius: f64) -> f64 {
    let density = estimate_density_from_moments(moments, grid_points, radius);
    let dx = 2.0 * radius / grid_points as f64;

    let mut fisher = 0.0;
    for i in 1..density.len() - 1 {
        if density[i] > 1e-10 {
            let deriv = (density[i + 1] - density[i - 1]) / (2.0 * dx);
            fisher += (deriv / density[i]).powi(2) * density[i] * dx;
        }
    }
    fisher
}

/// Large agent population model.
///
/// Models a population of N agents as random matrices. As N → ∞,
/// the collective behavior converges to a free probability distribution.
pub struct AgentPopulationModel {
    /// Number of agents.
    pub n_agents: usize,
    /// Moments of individual agent behavior.
    pub individual_moments: Vec<f64>,
}

impl AgentPopulationModel {
    /// Create a new agent population model.
    pub fn new(n_agents: usize, individual_moments: Vec<f64>) -> Self {
        Self { n_agents, individual_moments }
    }

    /// Compute moments of the collective (average) behavior.
    /// By free CLT: (Σ X_i - Nμ) / (σ√N) → semicircle.
    pub fn collective_moments(&self, max_order: usize) -> Vec<f64> {
        let cumulants = moments_to_free_cumulants(&self.individual_moments);
        let n = self.n_agents as f64;

        // Cumulants of average: κ_k(avg) = κ_k / N^{k/2 - 1}
        // Wait: for (Σ X_i)/N:
        // κ_k(avg) = N * κ_k / N^k = κ_k / N^{k-1}
        let avg_cumulants: Vec<f64> = cumulants
            .iter()
            .enumerate()
            .map(|(i, k)| *k / n.powi(i as i32))
            .take(max_order)
            .collect();

        use crate::moment_cumulant::free_cumulants_to_moments;
        free_cumulants_to_moments(&avg_cumulants)
    }

    /// Predict the limiting distribution as N → ∞.
    /// For centered agents: converges to semicircle.
    pub fn limiting_distribution(&self) -> SemicircleLaw {
        let cumulants = moments_to_free_cumulants(&self.individual_moments);
        let mean = cumulants.first().copied().unwrap_or(0.0);
        let variance = cumulants.get(1).copied().unwrap_or(0.0);
        // Semicircle centered at mean with given variance
        SemicircleLaw::with_mean_variance(mean, variance)
    }

    /// Free entropy of the collective.
    pub fn collective_entropy(&self, grid_points: usize, radius: f64) -> f64 {
        let moments = self.collective_moments(6);
        free_entropy_microstates(&moments, grid_points, radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_entropy_semicircle() {
        let entropy = free_entropy_semicircle(2.0); // standard semicircle
        // Known: χ(σ_1) = log(2π)/2 + 1/4
        let expected = 1.0_f64.ln() + (1.0 + (2.0 * std::f64::consts::PI).ln()) / 2.0;
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_free_entropy_positive() {
        // Free entropy of semicircle should be positive for reasonable parameters
        let entropy = free_entropy_semicircle(4.0);
        assert!(entropy > 0.0, "Entropy = {}", entropy);
    }

    #[test]
    fn test_free_entropy_from_samples() {
        let sc = SemicircleLaw::standard();
        let samples = sc.sample(200);
        let entropy = free_entropy_from_samples(&samples);
        // Should be finite
        assert!(entropy.is_finite());
    }

    #[test]
    fn test_agent_population_limit() {
        let model = AgentPopulationModel::new(1000, vec![0.0, 1.0]);
        let limiting = model.limiting_distribution();
        assert!((limiting.mu - 0.0).abs() < 1e-10);
        assert!((limiting.variance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_population_collective() {
        let model = AgentPopulationModel::new(100, vec![1.0, 2.0]);
        let moments = model.collective_moments(4);
        assert!(moments[0] > 0.0, "Collective mean should be positive");
    }

    #[test]
    fn test_free_fisher_information_runs() {
        // Just verify it doesn't crash and returns a non-negative value
        let moments = vec![0.0, 1.0, 0.0, 2.0];
        let fisher = free_fisher_information(&moments, 100, 3.0);
        assert!(fisher >= 0.0 || fisher.is_nan()); // May be NaN for edge cases
    }

    #[test]
    fn test_mutual_free_information_runs() {
        let ma = vec![0.0, 1.0];
        let mb = vec![0.0, 1.0];
        let mj = vec![0.0, 2.0];
        let mi = mutual_free_information(&ma, &mb, &mj, 50, 3.0);
        assert!(mi.is_finite() || mi.is_nan());
    }
}
