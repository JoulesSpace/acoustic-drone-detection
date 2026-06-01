//! Time-domain autocorrelation pitch cue.
//!
//! The autocorrelation of a quasi-periodic signal peaks at lags that are
//! integer multiples of its fundamental period. We search the lag range that
//! corresponds to the f0 band of interest and return the frequency of the
//! strongest peak, together with a normalized peak height in `[0, 1]` we use as
//! a per-cue confidence.

/// Estimate f0 from a frame using normalized autocorrelation.
///
/// Searches lags corresponding to `[f_lo, f_hi]` Hz. Returns
/// `(f0_hz, confidence)` where confidence is the autocorrelation peak value
/// normalized by the zero-lag energy (≈ 1.0 for a clean periodic signal).
/// Returns `None` if no usable peak is found.
pub fn autocorr_f0(samples: &[f32], sr: u32, f_lo: f32, f_hi: f32) -> Option<(f32, f32)> {
    let sr_f = sr as f32;
    // Lag bounds (in samples) for the frequency band: higher freq -> smaller lag.
    let min_lag = (sr_f / f_hi).floor() as usize;
    let max_lag = (sr_f / f_lo).ceil() as usize;
    if min_lag < 1 || max_lag <= min_lag || max_lag >= samples.len() {
        return None;
    }

    // Remove DC so a constant offset doesn't dominate the correlation.
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let x: Vec<f32> = samples.iter().map(|s| s - mean).collect();

    let energy: f32 = x.iter().map(|v| v * v).sum();
    if energy <= 1e-12 {
        return None;
    }

    // Compute normalized autocorrelation over the lag window and find the peak.
    let n = x.len();
    let mut best_lag = min_lag;
    let mut best_val = f32::MIN;
    let mut r = vec![0.0_f32; max_lag + 1];
    for (lag, slot) in r.iter_mut().enumerate().take(max_lag + 1).skip(min_lag) {
        let mut acc = 0.0_f32;
        for i in 0..(n - lag) {
            acc += x[i] * x[i + lag];
        }
        *slot = acc / energy;
        if *slot > best_val {
            best_val = *slot;
            best_lag = lag;
        }
    }

    if best_val <= 0.0 {
        return None;
    }

    // Parabolic interpolation around the integer peak for sub-sample lag.
    let lag = parabolic_peak(&r, best_lag);
    let f0 = sr_f / lag;
    if !(f_lo..=f_hi).contains(&f0) {
        return None;
    }
    Some((f0, best_val.clamp(0.0, 1.0)))
}

/// Refine an integer peak index in `data` to a sub-sample position via
/// parabolic interpolation of the three points around the peak.
fn parabolic_peak(data: &[f32], idx: usize) -> f32 {
    if idx == 0 || idx + 1 >= data.len() {
        return idx as f32;
    }
    let a = data[idx - 1];
    let b = data[idx];
    let c = data[idx + 1];
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        return idx as f32;
    }
    let delta = 0.5 * (a - c) / denom;
    idx as f32 + delta.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn tone(f0: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * PI * f0 * t).sin() + 0.5 * (2.0 * PI * 2.0 * f0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn recovers_fundamental_of_a_tone() {
        let sr = 16_000;
        let x = tone(120.0, sr, 4096);
        let (f0, conf) = autocorr_f0(&x, sr, 50.0, 400.0).unwrap();
        assert!((f0 - 120.0).abs() < 3.0, "f0 was {f0}");
        assert!(conf > 0.5);
    }

    #[test]
    fn silent_returns_none() {
        let sr = 16_000;
        let x = vec![0.0_f32; 4096];
        assert!(autocorr_f0(&x, sr, 50.0, 400.0).is_none());
    }
}
