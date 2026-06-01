//! Multiclass dataset wiring: build a labelled set from class folders (real
//! Al-Emadi data) or from a deterministic synthetic generator, plus a
//! K-class-stratified train/test split.
//!
//! Reuses [`drone_bench::dataset::Sample`] as the clip type and its CSV loader /
//! WAV decoder. `Sample.label` is a `u8`, which doubles as the class id `0..K`.

use drone_bench::dataset::{read_mono_wav, Sample};
use std::error::Error;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;

/// A labelled multiclass dataset plus the id <-> name mapping that gives the
/// integer class ids their human-readable meaning.
pub struct MultiDataset {
    pub samples: Vec<Sample>,
    /// `class_names[id]` is the folder/type name for class `id`.
    pub class_names: Vec<String>,
}

impl MultiDataset {
    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.class_names.len()
    }

    /// Load the Al-Emadi `Multiclass_Drone_Audio/` layout: each immediate
    /// subdirectory is a class, its `*.wav` files are the clips. Subfolders are
    /// sorted by name so class ids are deterministic. `max_per_class` caps the
    /// number of clips drawn from each class (keeps the heavily-skewed
    /// `unknown` folder from dwarfing the others and keeps runtime sane); `0`
    /// means "no cap".
    pub fn load_dir(root: &Path, max_per_class: usize) -> Result<Self, Box<dyn Error>> {
        let mut dirs: Vec<_> = fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        if dirs.is_empty() {
            return Err(format!("no class subdirectories under {}", root.display()).into());
        }

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

    /// Deterministic synthetic multiclass set: `K = 4` drone "types" with
    /// distinct blade-pass fundamentals, harmonic roll-off profiles and AM
    /// rates. No downloads, fully verifiable, and easy enough that a working
    /// pipeline should score a high macro-F1.
    pub fn synth(n_per_class: usize, sample_rate: u32, seed: u32) -> Self {
        // (name, f0 Hz, harmonic roll-off exponent, n harmonics, AM Hz, AM depth)
        let types: [(&str, f32, f32, usize, f32, f32); 4] = [
            ("type_a_quad", 110.0, 1.0, 8, 6.0, 0.25),
            ("type_b_hex", 165.0, 1.6, 6, 11.0, 0.35),
            ("type_c_octo", 240.0, 0.7, 10, 18.0, 0.20),
            ("type_d_fixed", 320.0, 2.2, 4, 3.5, 0.45),
        ];
        let mut rng = Rng(seed.max(1));
        let dur_s = 1.0_f32;
        let n = (dur_s * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(n_per_class * types.len());

        for (id, &(name, f0_base, rolloff, n_harm, am_base, am_depth)) in types.iter().enumerate() {
            for k in 0..n_per_class {
                // Small per-clip jitter so classes are clusters, not single points.
                let f0 = f0_base * (1.0 + 0.04 * (rng.unit01() - 0.5));
                let am_hz = am_base * (1.0 + 0.10 * (rng.unit01() - 0.5));
                let mut clip = vec![0.0_f32; n];
                for (i, s) in clip.iter_mut().enumerate() {
                    let t = i as f32 / sample_rate as f32;
                    let am = 1.0 + am_depth * (2.0 * PI * am_hz * t).sin();
                    let mut v = 0.0;
                    for h in 1..=n_harm {
                        let amp = 0.6 / (h as f32).powf(rolloff);
                        v += amp * (2.0 * PI * f0 * h as f32 * t).sin();
                    }
                    *s = (0.6 * am * v + 0.05 * rng.bipolar()).clamp(-1.0, 1.0);
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
            class_names: types.iter().map(|t| t.0.to_string()).collect(),
        }
    }

    /// K-class-stratified, deterministic train/test split. Within each class the
    /// samples are Fisher-Yates shuffled with a seeded RNG and the first
    /// `train_frac` go to train.
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
            let cut = (idx.len() as f32 * train_frac) as usize;
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
    fn synth_is_balanced_and_labelled() {
        let ds = MultiDataset::synth(20, 16_000, 7);
        assert_eq!(ds.n_classes(), 4);
        assert_eq!(ds.samples.len(), 80);
        for c in 0..4u8 {
            let n = ds.samples.iter().filter(|s| s.label == c).count();
            assert_eq!(n, 20);
        }
    }

    #[test]
    fn split_is_stratified_and_deterministic() {
        let ds = MultiDataset::synth(20, 16_000, 7);
        let (tr1, te1) = ds.split(0.7, 1);
        let (tr2, _te2) = ds.split(0.7, 1);
        // Deterministic.
        let ids1: Vec<_> = tr1.iter().map(|s| s.id.clone()).collect();
        let ids2: Vec<_> = tr2.iter().map(|s| s.id.clone()).collect();
        assert_eq!(ids1, ids2);
        // Stratified: every class appears in both splits at the right count.
        for c in 0..4u8 {
            assert_eq!(tr1.iter().filter(|s| s.label == c).count(), 14);
            assert_eq!(te1.iter().filter(|s| s.label == c).count(), 6);
        }
    }
}
