#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]
//! # lau-free-probability
//!
//! Free probability theory for large-scale agent systems.
//!
//! Implements noncommutative probability, free independence, free convolution,
//! Wigner semicircle law, Marchenko-Pastur law, and moment-cumulant formulas
//! via non-crossing partitions.

pub mod partition;
pub mod moment_cumulant;
pub mod transform;
pub mod convolution;
pub mod laws;
pub mod entropy;
pub mod space;

pub use partition::*;
pub use moment_cumulant::*;
pub use transform::*;
pub use convolution::*;
pub use laws::*;
pub use entropy::*;
pub use space::*;
