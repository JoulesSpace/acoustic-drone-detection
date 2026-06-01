//! Multi-vendor dataset wiring: build a labelled set of brand/model classes
//! from real audio on disk or from a deterministic synthetic generator, plus a
//! K-class-stratified train/test split.
//!
//! Reuses [`drone_bench::dataset::Sample`] as the clip type and its WAV decoder.
//! `Sample.label` is a `u8`, which doubles as the brand class id `0..K`.
//!
//! ## Real data layout
//!
//! Two on-disk shapes are accepted, auto-detected by [`VendorDataset::load_dir`]:
//!
//! 1. **Class folders** - each immediate subdirectory is a brand, its `*.wav`
//!    files are the clips (the Al-Emadi / Multiclass_Drone_Audio shape).
//! 2. **Flat brand-named files** - a single directory of `*.wav` whose
//!    filenames encode the brand, e.g. `DJI_Mavic2pro_81.wav` (the College of
//!    Charleston 32-brand visualization set). The trailing `_<index>` is
//!    stripped to recover the brand name.
//!
//! Because the flat set ships a single recording per brand, each clip is
//! windowed into fixed-length, non-overlapping **segments**, each treated as an
//! independent labelled example. This yields a real, honest stratified split.
//! The caveat - train and test segments come from the same recording, so this
//! measures within-recording brand separability, not cross-recording
//! generalization - is surfaced in the run report.

use drone_bench::dataset::{read_mono_wav, Sample};
use std::error::Error;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;

/// Default segment length (seconds) when windowing a flat single-clip-per-brand
/// dataset into multiple examples.
pub const DEFAULT_SEGMENT_SECS: f32 = 0.75;

/// Synthetic brand signature:
/// `(name, f0 Hz, harmonic roll-off exponent, n harmonics, rotor count,
///   AM depth, broadband motor-whine level, whine center Hz)`.
type BrandSig = (&'static str, f32, f32, usize, f32, f32, f32, f32);

/// A labelled multi-vendor dataset plus the id <-> brand-name mapping that gives
/// the integer class ids their human-readable meaning.
pub struct VendorDataset {
    pub samples: Vec<Sample>,
    /// `class_names[id]` is the brand/model name for class `id`.
    pub class_names: Vec<String>,
}

impl VendorDataset {
    /// Number of brand classes.
    pub fn n_classes(&self) -> usize {
        self.class_names.len()
    }

    /// Load a real dataset, auto-detecting the on-disk shape.
    ///
    /// If `root` contains subdirectories, the class-folder layout is used and
    /// `max_per_class` caps clips per class (`0` = no cap). If `root` contains a
    /// flat set of brand-named `*.wav` files, each is windowed into
    /// `segment_secs`-long segments (see module docs). A `min_per_class` of 2 is
    /// enforced so every brand can appear in both the train and test split.
    pub fn load_dir(
        root: &Path,
        max_per_class: usize,
        segment_secs: f32,
    ) -> Result<Self, Box<dyn Error>> {
        let subdirs: Vec<_> = fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();

        if subdirs.is_empty() {
            Self::load_flat(root, segment_secs)
        } else {
            Self::load_class_folders(root, max_per_class)
        }
    }

    /// Class-folder layout loader (one subdirectory per brand).
    fn load_class_folders(root: &Path, max_per_class: usize) -> Result<Self, Box<dyn Error>> {
        let mut dirs: Vec<_> = fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();

        let mut class_names = Vec::new();
        let mut samples = Vec::new();
        for (id, dir) in dirs.iter().enumerate() {
            let name = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            let mut wavs: Vec<_> = fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("wav"))
                .collect();
            wavs.sort();
            if max_per_class > 0 && wavs.len() > max_per_class {
                wavs.truncate(max_per_class);
            }
            for wav in wavs {
                let (audio, sr) = read_mono_wav(&wav)?;
                let rel = wav
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_string();
                samples.push(Sample {
                    id: format!("{name}/{rel}"),
                    samples: audio,
                    sample_rate: sr,
                    label: id as u8,
                });
            }
            class_names.push(name);
        }
        if samples.is_empty() {
            return Err("found class folders but no WAV files".into());
        }
        Ok(Self {
            samples,
            class_names,
        })
    }

    /// Flat brand-named-file loader: one recording per brand, segment-windowed.
    fn load_flat(root: &Path, segment_secs: f32) -> Result<Self, Box<dyn Error>> {
        let mut wavs: Vec<_> = fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("wav"))
            .collect();
        wavs.sort();
        if wavs.is_empty() {
            return Err(format!("no WAV files under {}", root.display()).into());
        }

        // Brands appear in sorted filename order; assign ids deterministically.
        let mut class_names: Vec<String> = Vec::new();
        let mut samples: Vec<Sample> = Vec::new();
        for wav in &wavs {
            let stem = wav
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            let brand = brand_from_stem(&stem);
            let id = match class_names.iter().position(|b| b == &brand) {
                Some(i) => i,
                None => {
                    class_names.push(brand.clone());
                    class_names.len() - 1
                }
            };

            let (audio, sr) = read_mono_wav(wav)?;
            let seg_len = ((segment_secs * sr as f32) as usize).max(1);
            // Non-overlapping segments; drop a trailing partial shorter than a
            // quarter segment so every example is comparable in length.
            let mut start = 0;
            let mut seg_idx = 0;
            while start + seg_len <= audio.len() {
                samples.push(Sample {
                    id: format!("{stem}#{seg_idx}"),
                    samples: audio[start..start + seg_len].to_vec(),
                    sample_rate: sr,
                    label: id as u8,
                });
                start += seg_len;
                seg_idx += 1;
            }
            // If the clip was shorter than one segment, keep it whole so the
            // brand is not silently dropped.
            if seg_idx == 0 && !audio.is_empty() {
                samples.push(Sample {
                    id: format!("{stem}#0"),
                    samples: audio,
                    sample_rate: sr,
                    label: id as u8,
                });
            }
        }

        // Enforce >= 2 segments per brand so the stratified split can populate
        // both train and test for every class.
        for (id, name) in class_names.iter().enumerate() {
            let n = samples.iter().filter(|s| s.label as usize == id).count();
            if n < 2 {
                return Err(format!(
                    "brand '{name}' produced only {n} segment(s) at {segment_secs}s; \
                     use a shorter --segment-secs"
                )
                .into());
            }
        }

        Ok(Self {
            samples,
            class_names,
        })
    }

    /// Deterministic synthetic multi-brand set: `K = 12` "brands" with distinct
    /// acoustic signatures - different blade-pass fundamentals, harmonic
    /// roll-off shapes, rotor counts (which set the AM beat rate), and broadband
    /// motor-whine levels. No downloads, fully verifiable, and separable enough
    /// that a working pipeline should reach a high macro-F1. This proves the
    /// capability and keeps the architecture real-data-ready.
    pub fn synth(n_per_class: usize, sample_rate: u32, seed: u32) -> Self {
        // Rotor count drives the AM rate (blade-pass beat ~ rotors * a base
        // rate), and the whine level/center shape the broadband floor - the
        // mix of cues a real brand classifier keys on. See [`BrandSig`] for the
        // tuple field order.
        let brands: [BrandSig; 12] = [
            ("aero_quad_s", 95.0, 1.2, 9, 4.0, 0.30, 0.04, 2200.0),
            ("aero_hex_m", 140.0, 1.7, 7, 6.0, 0.35, 0.05, 2600.0),
            ("aero_octo_l", 210.0, 0.8, 11, 8.0, 0.22, 0.07, 3000.0),
            ("volans_quad", 120.0, 2.0, 6, 4.0, 0.40, 0.03, 1800.0),
            ("volans_fpv", 260.0, 1.0, 8, 4.0, 0.18, 0.09, 4200.0),
            ("nimbus_tri", 165.0, 1.5, 7, 3.0, 0.33, 0.05, 2400.0),
            ("nimbus_heavy", 80.0, 2.4, 5, 8.0, 0.45, 0.06, 1500.0),
            ("kestrel_micro", 320.0, 0.7, 10, 4.0, 0.15, 0.10, 5200.0),
            ("kestrel_mini", 240.0, 1.3, 8, 4.0, 0.25, 0.06, 3600.0),
            ("orca_marine", 110.0, 1.9, 6, 6.0, 0.38, 0.04, 2000.0),
            ("falcon_fixed", 300.0, 2.6, 4, 2.0, 0.10, 0.08, 3300.0),
            ("sparrow_toy", 360.0, 0.9, 9, 4.0, 0.50, 0.12, 4800.0),
        ];
        let mut rng = Rng(seed.max(1));
        let dur_s = 1.0_f32;
        let n = (dur_s * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(n_per_class * brands.len());

        for (id, &(name, f0_base, rolloff, n_harm, rotors, am_depth, whine, whine_hz)) in
            brands.iter().enumerate()
        {
            // Blade-pass beat: a base per-rotor rate times the rotor count.
            let am_base = 1.6 * rotors;
            for k in 0..n_per_class {
                // Small per-clip jitter so each brand is a cluster, not a point.
                let f0 = f0_base * (1.0 + 0.04 * (rng.unit01() - 0.5));
                let am_hz = am_base * (1.0 + 0.08 * (rng.unit01() - 0.5));
                let whine_c = whine_hz * (1.0 + 0.05 * (rng.unit01() - 0.5));
                let mut clip = vec![0.0_f32; n];
                for (i, s) in clip.iter_mut().enumerate() {
                    let t = i as f32 / sample_rate as f32;
                    let am = 1.0 + am_depth * (2.0 * PI * am_hz * t).sin();
                    // Harmonic stack with brand-specific roll-off.
                    let mut v = 0.0;
                    for h in 1..=n_harm {
                        let amp = 0.6 / (h as f32).powf(rolloff);
                        v += amp * (2.0 * PI * f0 * h as f32 * t).sin();
                    }
                    // Narrowband motor whine plus a touch of broadband noise.
                    let whine_sig = whine * (2.0 * PI * whine_c * t).sin();
                    *s = (0.55 * am * v + whine_sig + 0.03 * rng.bipolar()).clamp(-1.0, 1.0);
                }
                samples.push(Sample {
                    id: format!("synth/{name}_{k:04}"),
                    samples: clip,
                    sample_rate,
                    label: id as u8,
                });
            }
        }
        Self {
            samples,
            class_names: brands.iter().map(|b| b.0.to_string()).collect(),
        }
    }

    /// K-class-stratified, deterministic train/test split. Within each class the
    /// samples are Fisher-Yates shuffled with a seeded RNG and the first
    /// `train_frac` go to train (at least one example each side when possible).
    pub fn split(&self, train_frac: f32, seed: u32) -> (Vec<Sample>, Vec<Sample>) {
        let mut rng = Rng(seed.max(1));
        let mut train = Vec::new();
        let mut test = Vec::new();
        for class in 0..self.n_classes() as u8 {
            let mut idx: Vec<usize> = self
                .samples
                .iter()
                .enumerate()
                .filter(|(_, s)| s.label == class)
                .map(|(i, _)| i)
                .collect();
            for i in (1..idx.len()).rev() {
                let j = (rng.next() as usize) % (i + 1);
                idx.swap(i, j);
            }
            let mut cut = (idx.len() as f32 * train_frac) as usize;
            // Guarantee at least one train and one test example per class when
            // the class has >= 2 samples.
            if idx.len() >= 2 {
                cut = cut.clamp(1, idx.len() - 1);
            }
            for (rank, &i) in idx.iter().enumerate() {
                if rank < cut {
                    train.push(self.samples[i].clone());
                } else {
                    test.push(self.samples[i].clone());
                }
            }
        }
        (train, test)
    }
}

/// Recover a brand name from a flat-file stem by stripping a trailing numeric
/// index, e.g. `DJI_Mavic2pro_81` -> `DJI_Mavic2pro`, `PhenoBee_80` ->
/// `PhenoBee`. If the last `_`-segment is not all digits the stem is kept whole.
fn brand_from_stem(stem: &str) -> String {
    match stem.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => {
            head.to_string()
        }
        _ => stem.to_string(),
    }
}

/// Tiny deterministic xorshift32 PRNG (matches `drone-bench`'s, no rng crate).
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn unit01(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32
    }
    fn bipolar(&mut self) -> f32 {
        self.unit01() * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_has_twelve_balanced_brands() {
        let ds = VendorDataset::synth(10, 16_000, 7);
        assert_eq!(ds.n_classes(), 12);
        assert_eq!(ds.samples.len(), 120);
        for c in 0..12u8 {
            let n = ds.samples.iter().filter(|s| s.label == c).count();
            assert_eq!(n, 10);
        }
    }

    #[test]
    fn split_is_stratified_and_deterministic() {
        let ds = VendorDataset::synth(10, 16_000, 7);
        let (tr1, te1) = ds.split(0.7, 1);
        let (tr2, _te2) = ds.split(0.7, 1);
        let ids1: Vec<_> = tr1.iter().map(|s| s.id.clone()).collect();
        let ids2: Vec<_> = tr2.iter().map(|s| s.id.clone()).collect();
        assert_eq!(ids1, ids2);
        for c in 0..12u8 {
            assert_eq!(tr1.iter().filter(|s| s.label == c).count(), 7);
            assert_eq!(te1.iter().filter(|s| s.label == c).count(), 3);
        }
    }

    #[test]
    fn brand_from_stem_strips_numeric_index() {
        assert_eq!(brand_from_stem("DJI_Mavic2pro_81"), "DJI_Mavic2pro");
        assert_eq!(brand_from_stem("PhenoBee_80"), "PhenoBee");
        assert_eq!(brand_from_stem("Syma_X5SW_68"), "Syma_X5SW");
        // No trailing numeric segment: kept whole.
        assert_eq!(brand_from_stem("PhenoBee"), "PhenoBee");
    }

    #[test]
    fn split_gives_each_class_train_and_test() {
        // Two segments per class -> one train, one test each.
        let ds = VendorDataset::synth(2, 16_000, 3);
        let (tr, te) = ds.split(0.7, 1);
        for c in 0..12u8 {
            assert!(tr.iter().any(|s| s.label == c));
            assert!(te.iter().any(|s| s.label == c));
        }
    }
}
