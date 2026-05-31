# lau-free-probability

When random matrices get large enough, their eigenvalues stop being random. Free probability theory describes this limit — the "free CLT" says sums of freely independent operators converge to the semicircle law, just as the classical CLT gives the Gaussian.

This is the mathematics of large agent populations behaving like random matrices.

## The math in 60 seconds

In **free probability**, independence is replaced by **free independence** — a noncommutative condition where alternating products of centered variables vanish in expectation. The key transforms:

- **R-transform:** R(z) = G⁻¹(z) - 1/z, where G is the Cauchy transform. Free independence ↔ R-transforms add.
- **S-transform:** encodes multiplicative free convolution.
- **Wigner semicircle:** the free analog of the Gaussian. Sum of freely independent = semicircle.
- **Marchenko-Pastur:** the free Poisson distribution (eigenvalue limit of sample covariance matrices).
- **Non-crossing partitions:** Catalan numbers enumerate the moments — the combinatorial backbone.

References: Voiculescu, Dykema & Nica, *Free Random Variables* (1992); Nica & Speicher, *Lectures on the Combinatorics of Free Probability* (2006)

## Quick start

```rust
use lau_free_probability::{WignerSemicircle, RTransform, FreeConvolution, NCPartition};

// Wigner semicircle law with radius σ
let wigner = WignerSemicircle::new(1.0);
let moments = wigner.moments(6); // Catalan numbers: [1, 0, 1, 0, 2, 0]

// R-transform: R(z) = σ²z for the semicircle
let r = RTransform::from_cauchy(&wigner.cauchy_transform());
assert!((r.evaluate(0.5) - 0.25).abs() < 1e-10);

// Free convolution: R_{X+Y} = R_X + R_Y
let x = WignerSemicircle::new(1.0);
let y = WignerSemicircle::new(2.0);
let sum = FreeConvolution::additive(&x, &y); // semicircle with σ²=3

// Non-crossing partitions of {1,2,3,4}
let ncp = NCPartition::enumerate(4);
assert_eq!(ncp.len(), 14); // Catalan(4) = 14

// Marchenko-Pastur (free Poisson)
let mp = MarchenkoPastur::new(1.0, 100);
```

## Key types

| Type | What it is |
|------|-----------|
| `WignerSemicircle` | The free CLT limit — semicircle distribution |
| `MarchenkoPastur` | The free Poisson — eigenvalue limit of covariance matrices |
| `RTransform` | Additive free convolution via R_X+Y = R_X + R_Y |
| `STransform` | Multiplicative free convolution |
| `NCPartition` | Non-crossing partitions — the Catalan combinatorics |
| `FreeEntropy` | Voiculescu's free entropy (microstate and non-microstate) |

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-free-probability/issues) or PR.
