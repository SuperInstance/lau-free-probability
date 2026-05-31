//! Noncommutative probability spaces.
//!
//! A noncommutative probability space consists of a unital algebra A
//! equipped with a linear functional φ: A → ℂ satisfying φ(1) = 1.

use serde::{Deserialize, Serialize};

/// A noncommutative probability space.
///
/// Contains a unital algebra with a positive, normalized linear functional
/// (the "expectation" or "state").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NCProbabilitySpace {
    /// Moments E[X^n] for n = 1, 2, ...
    /// moments[n-1] = E[X^n]
    moments: Vec<f64>,
}

impl NCProbabilitySpace {
    /// Create a space from a known moment sequence.
    pub fn from_moments(moments: Vec<f64>) -> Self {
        assert!(!moments.is_empty(), "Need at least one moment");
        Self { moments }
    }

    /// Create a space from an empirical distribution of samples.
    pub fn from_samples(samples: &[f64], max_order: usize) -> Self {
        let mut moments = Vec::with_capacity(max_order);
        for n in 1..=max_order {
            let m: f64 = samples.iter().map(|x| x.powi(n as i32)).sum::<f64>() / samples.len() as f64;
            moments.push(m);
        }
        Self { moments }
    }

    /// Get E[X^n] for order n (1-indexed).
    pub fn moment(&self, n: usize) -> f64 {
        if n == 0 {
            1.0
        } else {
            self.moments.get(n - 1).copied().unwrap_or(0.0)
        }
    }

    /// Get all moments up to given order.
    pub fn moments(&self, max_order: usize) -> Vec<f64> {
        (0..=max_order).map(|n| self.moment(n)).collect()
    }

    /// Compute the variance E[X^2] - E[X]^2.
    pub fn variance(&self) -> f64 {
        self.moment(2) - self.moment(1).powi(2)
    }

    /// Number of moments stored.
    pub fn order(&self) -> usize {
        self.moments.len()
    }
}

/// An element of a noncommutative algebra, represented by its moments
/// in a given state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NCElement {
    /// The probability space this element lives in.
    pub space: NCProbabilitySpace,
    /// Optional name/label.
    pub label: Option<String>,
}

impl NCElement {
    /// Create a named element from moments.
    pub fn new(moments: Vec<f64>, label: Option<String>) -> Self {
        Self {
            space: NCProbabilitySpace::from_moments(moments),
            label,
        }
    }

    /// Get E[X^n].
    pub fn moment(&self, n: usize) -> f64 {
        self.space.moment(n)
    }

    /// Mean E[X].
    pub fn mean(&self) -> f64 {
        self.moment(1)
    }

    /// Variance.
    pub fn variance(&self) -> f64 {
        self.space.variance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_space() {
        let space = NCProbabilitySpace::from_moments(vec![0.0, 1.0, 0.0, 3.0]);
        assert_eq!(space.moment(0), 1.0);
        assert_eq!(space.moment(1), 0.0);
        assert_eq!(space.moment(2), 1.0);
        assert_eq!(space.moment(3), 0.0);
        assert_eq!(space.moment(4), 3.0);
    }

    #[test]
    fn test_from_samples() {
        let samples: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let space = NCProbabilitySpace::from_samples(&samples, 2);
        assert!((space.moment(1) - 3.0).abs() < 1e-10);
        assert!((space.moment(2) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_nc_element() {
        let elem = NCElement::new(vec![2.0, 5.0, 14.0], Some("test".into()));
        assert_eq!(elem.mean(), 2.0);
        assert!((elem.variance() - 1.0).abs() < 1e-10);
        assert_eq!(elem.label.as_deref(), Some("test"));
    }

    #[test]
    fn test_zeroth_moment() {
        let space = NCProbabilitySpace::from_moments(vec![1.0, 2.0]);
        assert_eq!(space.moment(0), 1.0); // φ(1) = 1 always
    }
}
