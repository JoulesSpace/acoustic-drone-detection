//! Cue fusion: turn the three per-frame f0 candidates into one robust estimate.
//!
//! Each cue (HPS, cepstrum, autocorrelation) returns an `(f0, confidence)`
//! candidate. Individually they have characteristic failure modes — HPS likes
//! octave errors, autocorrelation can lock onto a sub-harmonic, the cepstrum
//! can be noisy at low SNR. Fusing them removes most of these:
//!
//! 1. Build the candidate set (any cue that produced a value).
//! 2. For each candidate, score it by the total harmonic energy it explains in
//!    the magnitude spectrum, plus agreement bonuses when another cue lands on
//!    the same f0 (within tolerance) or on a clean octave of it.
//! 3. Pick the highest-scoring candidate. Octave disambiguation falls out of the
//!    harmonic-energy term: the true f0 explains its harmonics, f0/2 wastes half
//!    its "harmonics" on empty bins and 2·f0 misses the odd partials.

use drone_dsp::{hz_to_bin, Spectrum, NUM_BINS};

/// A single cue's f0 candidate.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// Estimated fundamental in Hz.
    pub f0: f32,
    /// Per-cue confidence in `[0, 1]`.
    pub conf: f32,
}

/// Result of fusing the per-frame cues.
#[derive(Clone, Copy, Debug)]
pub struct FrameEstimate {
    /// Fused fundamental in Hz.
    pub f0: f32,
    /// Fused confidence in `[0, 1]`.
    pub conf: f32,
}

/// Two frequencies agree if within `tol` (relative).
fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * a.max(b)
}

/// Harmonic energy a candidate f0 explains in the spectrum: sum of magnitudes
/// at the first few harmonic bins. The true fundamental maximizes this.
fn harmonic_energy(spectrum: &Spectrum, f0: f32, sr: u32) -> f32 {
    let mut acc = 0.0_f32;
    for h in 1..=8 {
        let bin = hz_to_bin(f0 * h as f32, sr);
        if bin >= NUM_BINS {
            break;
        }
        // Take the max in a tiny window to tolerate sub-bin offsets.
        let lo = bin.saturating_sub(1);
        let hi = (bin + 1).min(NUM_BINS - 1);
        let local = spectrum[lo..=hi].iter().copied().fold(0.0_f32, f32::max);
        acc += local;
    }
    acc
}

/// Fuse cue candidates into a single frame estimate using the magnitude
/// spectrum to arbitrate. `candidates` may contain `None` for cues that failed.
///
/// Returns `None` if no cue produced a candidate.
pub fn fuse(
    candidates: &[Option<Candidate>],
    spectrum: &Spectrum,
    sr: u32,
) -> Option<FrameEstimate> {
    let cands: Vec<Candidate> = candidates.iter().filter_map(|c| *c).collect();
    if cands.is_empty() {
        return None;
    }

    const TOL: f32 = 0.06; // 6% — about a bin's worth at these frequencies.

    let mut best: Option<(f32, Candidate, f32)> = None; // (score, cand, agreement)
    for &c in &cands {
        // Base term: how much harmonic energy this f0 explains, scaled by the
        // cue's own confidence so a confident cue counts for more.
        let he = harmonic_energy(spectrum, c.f0, sr);
        let mut score = he * (0.5 + 0.5 * c.conf);

        // Agreement bonus: other cues that land on the same f0 reinforce it; an
        // octave match (×2 or ×0.5) is weaker evidence but still meaningful and
        // is how we damp octave errors — the candidate that the others agree
        // with *directly* outscores its own octave.
        let mut agree = 0.0_f32;
        for &other in &cands {
            if std::ptr::eq(&other, &c) {
                continue;
            }
            if close(c.f0, other.f0, TOL) {
                agree += 1.0;
            } else if close(c.f0, other.f0 * 2.0, TOL) || close(c.f0, other.f0 * 0.5, TOL) {
                agree += 0.25;
            }
        }
        score *= 1.0 + 0.5 * agree;

        match &best {
            Some((bs, _, _)) if *bs >= score => {}
            _ => best = Some((score, c, agree)),
        }
    }

    let (_score, c, agree) = best?;
    // Confidence: blend the cue's own confidence with cross-cue agreement.
    let conf = (0.5 * c.conf + 0.5 * (agree / (cands.len() as f32 - 1.0).max(1.0))).clamp(0.0, 1.0);
    Some(FrameEstimate { f0: c.f0, conf })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_spectrum_with_harmonics(f0: f32, sr: u32) -> Spectrum {
        let mut s = [0.0_f32; NUM_BINS];
        for h in 1..=8 {
            let bin = hz_to_bin(f0 * h as f32, sr);
            if bin < NUM_BINS {
                s[bin] = 1.0;
            }
        }
        s
    }

    #[test]
    fn agreeing_cues_win() {
        let sr = 16_000;
        let spec = flat_spectrum_with_harmonics(120.0, sr);
        let cands = [
            Some(Candidate {
                f0: 120.0,
                conf: 0.8,
            }),
            Some(Candidate {
                f0: 121.0,
                conf: 0.7,
            }),
            Some(Candidate {
                f0: 240.0,
                conf: 0.6,
            }), // octave error
        ];
        let est = fuse(&cands, &spec, sr).unwrap();
        assert!((est.f0 - 120.0).abs() < 5.0, "f0 was {}", est.f0);
    }

    #[test]
    fn rejects_subharmonic_via_harmonic_energy() {
        // Spectrum has harmonics of 150 Hz only. A 75 Hz subharmonic candidate
        // explains the same bins but should lose to the direct 150 Hz one.
        let sr = 16_000;
        let spec = flat_spectrum_with_harmonics(150.0, sr);
        let cands = [
            Some(Candidate {
                f0: 150.0,
                conf: 0.7,
            }),
            Some(Candidate {
                f0: 75.0,
                conf: 0.7,
            }),
            None,
        ];
        let est = fuse(&cands, &spec, sr).unwrap();
        assert!((est.f0 - 150.0).abs() < 5.0, "f0 was {}", est.f0);
    }

    #[test]
    fn empty_is_none() {
        let sr = 16_000;
        let spec = [0.0_f32; NUM_BINS];
        assert!(fuse(&[None, None, None], &spec, sr).is_none());
    }
}
