# lau-free-probability

**Free probability theory for large-scale agent systems — non-crossing partitions, R-transforms, free convolution, Wigner semicircle, Marchenko-Pastur, and Voiculescu's free entropy.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

56 tests · 1,987 lines of Rust · 7 modules

---

## What This Does

When random matrices get large enough, their eigenvalues stop being random in the usual sense — they converge to deterministic distributions governed by **free probability**. The free CLT says sums of freely independent operators converge to the semicircle law, just as the classical CLT gives the Gaussian. This crate implements the full mathematical toolkit.

You get:
- **Non-crossing partitions** — the combinatorial backbone, enumerated by Catalan numbers
- **Moment-cumulant formulas** — converting between moments and free cumulants via Möbius inversion on the NC lattice
- **R-transform and Cauchy transform** — the analytic tools: R_{X+Y} = R_X + R_Y for freely independent operators
- **S-transform** — multiplicative free convolution: S_{XY} = S_X · S_Y
- **Wigner semicircle law** — the free Gaussian, with PDF, CDF, sampling, and moments (Catalan numbers)
- **Marchenko-Pastur law** — the free Poisson, eigenvalue limit of sample covariance matrices
- **Free compound Poisson** — κ_n = λ · m_n(ν) for arbitrary compounding measures
- **Free entropy** — Voiculescu's microstates and non-microstates approaches, free Fisher information
- **Noncommutative probability spaces** — the algebraic framework
- **Agent population models** — N agents behaving like random matrices, with free CLT convergence

---

## Key Idea

Free probability replaces classical independence with **free independence** — a noncommutative condition where alternating products of centered variables vanish in expectation. The parallel with classical probability is exact:

| Classical Probability | Free Probability |
|---|---|
| Independence | Free independence |
| Gaussian (CLT limit) | Semicircle (free CLT limit) |
| Poisson | Marchenko-Pastur (free Poisson) |
| Moments via all partitions | Moments via non-crossing partitions |
| Classical cumulants (Bell polynomials) | Free cumulants (NC partitions + Möbius) |
| Characteristic function | R-transform: R_{X+Y} = R_X + R_Y |
| Cumulants add (independence) | Free cumulants add (free independence) |
| Entropy | Free entropy (Voiculescu) |
| Fisher information | Free Fisher information |

The combinatorial heart: the Catalan numbers C_n = (2n)!/(n!(n+1)!) count non-crossing partitions of {1,…,n}, replacing the Bell numbers of classical probability.

---

## Install

```toml
[dependencies]
lau-free-probability = "0.1"
```

Requires Rust 2021 edition. No external dependencies.

---

## Quick Start

```rust
use lau_free_probability::*;

// 1. Wigner semicircle: the free Gaussian
let sc = SemicircleLaw::standard(); // μ=0, R=2, variance=1
assert_eq!(sc.moment(1), 0.0);     // odd moments = 0
assert_eq!(sc.moment(2), 1.0);     // C_1 = 1
assert_eq!(sc.moment(4), 2.0);     // C_2 = 2
assert_eq!(sc.moment(6), 5.0);     // C_3 = 5
assert_eq!(sc.moment(8), 14.0);    // C_4 = 14

// Free cumulants of semicircle: κ_1=0, κ_2=1, κ_n=0 for n≥3
let cumulants = sc.free_cumulants(6);

// 2. Marchenko-Pastur: the free Poisson (eigenvalue limit of XX^T/N)
let mp = MarchenkoPasturLaw::standard(); // λ=1
assert_eq!(mp.moment(1), 1.0);  // C_1
assert_eq!(mp.moment(2), 2.0);  // C_2
assert_eq!(mp.moment(3), 5.0);  // C_3
// Free cumulants: κ_n = 1 for all n

// 3. Non-crossing partitions of {1,2,3,4}
let ncp = NCPartition::all_nc_partitions(4);
assert_eq!(ncp.len(), 14); // Catalan(4) = 14

// 4. Moment-cumulant conversion
let moments = vec![0.0, 1.0, 0.0, 2.0, 0.0, 5.0]; // semicircle moments
let kappa = moments_to_free_cumulants(&moments);
// κ_1=0, κ_2=1, κ_3=0, κ_4=0, κ_5=0, κ_6=0

// Roundtrip
let recovered = free_cumulants_to_moments(&kappa);
assert_eq!(recovered, moments);

// 5. Additive free convolution: semicircle + semicircle = wider semicircle
let sc1 = vec![0.0, 1.0];
let sc2 = vec![0.0, 1.0];
let sum = additive_free_convolution(&sc1, &sc2);
assert!((sum[1] - 2.0).abs() < 1e-10); // variance doubles
```

---

## API Reference

### `NCPartition`

Non-crossing partitions of {1,…,n}: partitions where no two blocks "cross" (no a < b < c < d with a~c, b~d, a≁b).

```rust
// Enumerate all NC partitions
let parts = NCPartition::all_nc_partitions(4);
assert_eq!(parts.len(), 14); // Catalan(4)

// Construct specific partitions
let one = NCPartition::one_block(4);   // {{1,2,3,4}}
let fine = NCPartition::finest(4);     // {{1},{2},{3},{4}}

// Validate non-crossing property
let valid = NCPartition::new(vec![
    vec![1, 3].into_iter().collect(),
    vec![2].into_iter().collect(),
]); // OK: {1,3} and {2} don't cross

let invalid = NCPartition::new(vec![
    vec![1, 3].into_iter().collect(),
    vec![2, 4].into_iter().collect(),
]); // Error: 1<2<3<4 with 1~3 and 2~4 — crossing!

// Kreweras complement
let krew = one.kreweras_complement(); // gives finest partition

// Möbius function on NC lattice
let mu = one.moebius_from_bottom(); // μ(0_n, 1_n)
```

### Moment-Cumulant Conversion

The fundamental relations:

**m_n = Σ_{π ∈ NC(n)} ∏_{B ∈ π} κ_{|B|}** (moments from free cumulants)

**κ_n = Σ_{π ∈ NC(n)} μ(0_n, π) ∏_{B ∈ π} m_{|B|}** (free cumulants from moments, via Möbius inversion)

```rust
// Free cumulants from moments
let kappa = moments_to_free_cumulants(&[0.0, 1.0, 0.0, 2.0]);

// Moments from free cumulants
let moments = free_cumulants_to_moments(&kappa);

// Classical cumulants (for comparison — uses all partitions)
let classical = moments_to_classical_cumulants(&[0.0, 1.0, 0.0, 3.0]);

// Free cumulants of a sum of freely independent operators
let sum_k = free_sum_cumulants(&moments_x, &moments_y);
```

### Transforms

```rust
// Cauchy transform G(z) = Σ m_n / z^{n+1}
let g = cauchy_transform(&moments, 10.0, 8);

// R-transform R(z) = Σ κ_n z^{n-1} (free cumulant generating function)
let r = r_transform(&moments, 0.5);

// R-transform from precomputed cumulants
let r = r_transform_from_cumulants(&kappa, 0.5, 6);

// η-transform: η(z) = Σ m_n z^n
let eta = eta_transform(&moments, 0.1, 6);

// S-transform: S_{XY} = S_X · S_Y for multiplicative free convolution
let s = s_transform(&moments, 0.1, 6);

// Numerical R-transform from a density (Newton's method on Cauchy inverse)
let r_num = r_transform_numerical(&density_fn, 0.5, 1.0, 1e-10, 100);
```

### `SemicircleLaw`

The free Gaussian. Density: f(x) = (2/(πR²))√(R²−(x−μ)²) for |x−μ| ≤ R.

```rust
let sc = SemicircleLaw::standard();            // μ=0, R=2, variance=1
let sc2 = SemicircleLaw::with_mean_variance(3.0, 2.0); // μ=3, σ²=2

sc.pdf(0.0);              // 1/π (maximum)
sc.cdf(0.0);              // 0.5
sc.variance();            // 1.0
sc.moment(6);             // 5.0 (Catalan number)
sc.moments(8);            // all moments including center shift
sc.free_cumulants(6);     // [0, 1, 0, 0, 0, 0]
sc.cauchy_transform(10.0); // G(10)
let samples = sc.sample(1000); // rejection sampling
```

### `MarchenkoPasturLaw`

The free Poisson — eigenvalue distribution of XX^T/N where X is N×M. Parameter λ = M/N.

```rust
let mp = MarchenkoPasturLaw::standard(); // λ=1, support [0, 4]
let mp2 = MarchenkoPasturLaw { lambda: 2.0 };

mp.a();  // lower edge: (1-√λ)²
mp.b();  // upper edge: (1+√λ)²
mp.pdf(1.0);     // √3/(2π)
mp.moment(4);    // 14.0 (Catalan number for λ=1)
mp.free_cumulants(5); // [1, 1, 1, 1, 1] — κ_n = λ for all n
mp.cauchy_transform(z);
let samples = mp.sample(100);
```

### Free Convolution

**Additive:** R_{X⊞Y}(z) = R_X(z) + R_Y(z) — free cumulants add.

```rust
// Full additive convolution
let sum = additive_free_convolution(&moments_x, &moments_y);

// Shift: X ⊞ cI just adds c to the mean
let shifted = additive_shift(&moments, 5.0);

// Scale: κ_n(aX) = a^n · κ_n(X)
let scaled = additive_scale(&moments, 2.0);

// Free CLT: sum of N freely independent copies, scaled by 1/√N
let limiting = free_clt(&moments, 1000); // converges to semicircle
```

**Multiplicative:** S_{X⊠Y}(z) = S_X(z) · S_Y(z).

```rust
let prod = multiplicative_free_convolution(&moments_x, &moments_y);
```

### Free Entropy

Voiculescu's free entropy χ(a₁,…,aₙ) measures the "size" of the set of approximating matrix tuples.

```rust
// From moments (microstates approach)
let chi = free_entropy_microstates(&moments, 100, 3.0);

// Closed form for semicircle
let chi_sc = free_entropy_semicircle(2.0);

// From samples (non-microstates)
let chi = free_entropy_from_samples(&samples);

// Free Fisher information: Φ*(ρ) = ∫ (ρ'/ρ)² ρ dx
let fisher = free_fisher_information(&moments, 100, 3.0);

// Mutual free information: i*(a,b) = χ(a) + χ(b) - χ(a,b)
let mi = mutual_free_information(&moments_a, &moments_b, &moments_ab, 50, 3.0);
```

### `AgentPopulationModel`

Models N agents as random matrices — as N → ∞, collective behavior converges to a free probability distribution.

```rust
let model = AgentPopulationModel::new(1000, vec![0.0, 1.0]);
let limiting = model.limiting_distribution(); // SemicircleLaw
let moments = model.collective_moments(6);
let entropy = model.collective_entropy(100, 3.0);
```

---

## How It Works

The crate builds from combinatorial foundations to analytic tools:

```
Layer 1: Combinatorics      NCPartition (Catalan numbers, Möbius function)
              │
              ▼
Layer 2: Algebraic          moment_cumulant (NC lattice Möbius inversion)
              │
              ▼
Layer 3: Analytic           transform (Cauchy, R-transform, S-transform)
              │
              ▼
Layer 4: Convolution        convolution (additive via R, multiplicative via S)
              │
              ▼
Layer 5: Classical Laws     laws (Semicircle, Marchenko-Pastur, compound Poisson)
              │
              ▼
Layer 6: Entropy            entropy (microstates, non-microstates, Fisher info)
              │
              ▼
Layer 7: Probability Space  space (NC probability spaces and elements)
              │
              ▼
Layer 8: Application        entropy::AgentPopulationModel (free CLT for agents)
```

**Layer 1** provides the combinatorial backbone: non-crossing partitions ordered by reverse refinement form the NC lattice, with Möbius function μ(0_n, π) = ∏_{B ∈ π} (-1)^{|B|-1} C_{|B|-1}.

**Layer 2** implements the moment-cumulant duality. Free cumulants κ_n are the "linearizers" of free independence: κ_n(X+Y) = κ_n(X) + κ_n(Y) for freely independent X, Y.

**Layer 3** converts between moment and cumulant representations via the Cauchy transform G(z) = E[1/(z−X)] and its functional inverse (the R-transform).

**Layer 4** implements free convolution. Additive uses R-transforms (they add); multiplicative uses S-transforms (they multiply).

**Layer 5** provides the canonical distributions with closed-form PDFs, CDFs, moments, cumulants, and sampling.

**Layer 6** computes free entropy — the free analog of Shannon entropy, with both microstates and non-microstates approaches.

---

## The Math

### Free Independence

Operators a₁, …, aₙ are **freely independent** if for any alternating centered products:

φ(a_{i₁} · a_{i₂} · … · a_{iₖ}) = 0

whenever i₁ ≠ i₂ ≠ … ≠ iₖ (adjacent indices differ) and φ(a_{i_j}) = 0.

This is the noncommutative replacement for classical independence.

### Non-Crossing Partitions and Catalan Numbers

A partition π of {1,…,n} is **non-crossing** if there are no a < b < c < d with a ~π c, b ~π d, and a ≁π b. The number of NC partitions of {1,…,n} is the Catalan number C_n = (2n)!/(n!(n+1)!).

NC(1)=1, NC(2)=2, NC(3)=5, NC(4)=14, NC(5)=42, NC(6)=132…

The NC lattice (ordered by reverse refinement) supports Möbius inversion, which gives the moment-cumulant formulas.

### R-Transform and Free Convolution

The **Cauchy transform** G(z) = E[1/(z−X)] encodes the distribution as an analytic function. Its functional inverse defines the **R-transform**: R(z) = G⁻¹(z) − 1/z.

The key property: **R_{X⊞Y}(z) = R_X(z) + R_Y(z)** for freely independent X, Y. Free cumulants are the coefficients: R(z) = Σ κ_n z^{n−1}. This is why "free cumulants add" — exactly parallel to how classical cumulants add for independent random variables.

For the **semicircle**: R(z) = σ²z (only κ₂ = σ² is nonzero).
For **Marchenko-Pastur**: R(z) = λ/(1−z) (geometric series, κ_n = λ).

### S-Transform and Multiplicative Convolution

For positive operators X, Y that are freely independent, the **S-transform** satisfies S_{XY}(z) = S_X(z) · S_Y(z). This gives multiplicative free convolution.

### Free CLT

If X₁, X₂, … are freely independent, identically distributed with mean 0 and variance σ², then (X₁ + … + Xₙ)/√n converges to the **semicircle law** with variance σ². This is the exact parallel of the classical CLT → Gaussian.

### Marchenko-Pastur Law

For an N×M random matrix X with i.i.d. entries of variance 1/N, the eigenvalues of XX^T converge (as N,M → ∞ with λ = M/N fixed) to the Marchenko-Pastur distribution supported on [(1−√λ)², (1+√λ)²]. For λ=1: moments are Catalan numbers, free cumulants are all 1.

### Free Entropy

Voiculescu's free entropy χ(μ) = ½ ∬ log|s−t| dμ(s) dμ(t) + const measures the "volume" of matrices approximating a given noncommutative distribution. It satisfies:

- χ(μ ⊞ ν) ≥ χ(μ) + χ(ν) (subadditivity under free convolution)
- Maximum entropy for fixed variance → semicircle

---

## Module Overview

| Module | Tests | Key Types | Purpose |
|--------|-------|-----------|---------|
| `partition` | 8 | `NCPartition` | Non-crossing partitions, Catalan numbers, Möbius function |
| `moment_cumulant` | 7 | — | Free/classical moment-cumulant conversion |
| `transform` | 7 | — | Cauchy, R-transform, S-transform, η-transform |
| `convolution` | 8 | — | Additive and multiplicative free convolution, free CLT |
| `laws` | 14 | `SemicircleLaw`, `MarchenkoPasturLaw`, `FreeCompoundPoisson` | Canonical distributions |
| `entropy` | 7 | `AgentPopulationModel` | Free entropy, Fisher info, agent populations |
| `space` | 4 | `NCProbabilitySpace`, `NCElement` | Noncommutative probability spaces |

---

## References

- **Free Probability:** Voiculescu, Dykema & Nica, *Free Random Variables* (1992)
- **Combinatorics:** Nica & Speicher, *Lectures on the Combinatorics of Free Probability* (2006)
- **Random Matrices:** Anderson, Guionnet & Zeitouni, *An Introduction to Random Matrices* (2010)
- **Free Entropy:** Voiculescu, *The analogues of entropy and of Fisher's information measure in free probability theory* (1998)
- **Marchenko-Pastur:** Marchenko & Pastur, *Distribution of eigenvalues for some sets of random matrices* (1967)

---

## License

MIT
