//! A tiny deterministic PRNG.
//!
//! Everything random in this crate - source synthesis noise, the mixing matrix,
//! and the FastICA weight initialization - draws from one seeded generator so
//! results are reproducible across runs and machines. We deliberately avoid the
//! `rand` crate to keep the dependency surface minimal and the determinism
//! obvious. The core is `xorshift64*`, which has good enough statistical quality
//! for simulation and mixing while being trivial to seed.

/// Seedable `xorshift64*` PRNG.
///
/// Construct with [`Rng::new`] (any seed, including 0, is accepted - it is
/// remapped to a nonzero state internally).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    /// Cached second Gaussian sample from Box-Muller (drawn in pairs).
    spare: Option<f64>,
}

impl Rng {
    /// Create a generator from `seed`. A zero seed is remapped so the state is
    /// never the all-zeros fixed point of xorshift.
    pub fn new(seed: u64) -> Self {
        // SplitMix64-style avalanche of the seed so nearby seeds give
        // well-separated streams.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self {
            state: z | 1,
            spare: None,
        }
    }

    /// Next raw 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f64` in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        // Use the top 53 bits for a full-precision double in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform `f64` in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform `f64` in `[-1, 1)`.
    #[inline]
    pub fn bipolar(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    /// Standard normal `N(0, 1)` sample via Box-Muller (drawn in pairs and
    /// cached so successive calls are cheap and still deterministic).
    pub fn gaussian(&mut self) -> f64 {
        if let Some(g) = self.spare.take() {
            return g;
        }
        // Guard u1 away from 0 so ln is finite.
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn unit_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn gaussian_roughly_standard() {
        let mut r = Rng::new(123);
        let n = 200_000;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for _ in 0..n {
            let g = r.gaussian();
            mean += g;
            m2 += g * g;
        }
        mean /= n as f64;
        let var = m2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
