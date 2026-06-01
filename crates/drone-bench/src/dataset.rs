//! Labelled dataset: loading from disk, synthetic generation, and splitting.

use std::error::Error;
use std::f32::consts::PI;
use std::path::Path;

/// One labelled clip of mono audio.
#[derive(Clone)]
pub struct Sample {
    /// Identifier (filename or synthetic id), used for leakage-aware grouping.
    pub id: String,
    /// Mono audio in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// 1 = drone present, 0 = no drone.
    pub label: u8,
}

/// A collection of labelled samples.
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// Load from a CSV manifest with a header `path,label` (label 0/1). Paths
    /// are resolved relative to `root`. WAV files are decoded to mono `f32`.
    pub fn load_csv(root: &Path, manifest: &Path) -> Result<Self, Box<dyn Error>> {
        let text = std::fs::read_to_string(manifest)?;
        let mut samples = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Skip a header row if present.
            if i == 0 && line.to_ascii_lowercase().starts_with("path") {
                continue;
            }
            let (path_str, label_str) = line
                .rsplit_once(',')
                .ok_or_else(|| format!("bad manifest line {}: {line}", i + 1))?;
            let label: u8 = label_str.trim().parse()?;
            let wav_path = root.join(path_str.trim());
            let (audio, sr) = read_mono_wav(&wav_path)?;
            samples.push(Sample {
                id: path_str.trim().to_string(),
                samples: audio,
                sample_rate: sr,
                label,
            });
        }
        Ok(Self { samples })
    }

    /// Generate a deterministic synthetic dataset: `n_per_class` drone clips and
    /// `n_per_class` non-drone clips. Lets the whole pipeline run with no
    /// downloads. NOT a substitute for real data — it validates plumbing and
    /// gives a sanity baseline only.
    pub fn synth(n_per_class: usize, sample_rate: u32, seed: u32) -> Self {
        let mut rng = Rng(seed.max(1));
        let dur_s = 1.0_f32;
        let n = (dur_s * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(n_per_class * 2);

        for k in 0..n_per_class {
            // --- drone-like positive: harmonic stack + AM + noise ---
            let f0 = 90.0 + 120.0 * rng.unit01(); // 90..210 Hz blade-pass
            let am_hz = 5.0 + 10.0 * rng.unit01();
            let mut clip = vec![0.0_f32; n];
            for (i, s) in clip.iter_mut().enumerate() {
                let t = i as f32 / sample_rate as f32;
                let am = 1.0 + 0.25 * (2.0 * PI * am_hz * t).sin();
                let mut v = 0.0;
                for h in 1..=8 {
                    v += (0.5 / h as f32) * (2.0 * PI * f0 * h as f32 * t).sin();
                }
                *s = (0.6 * am * v + 0.06 * rng.bipolar()).clamp(-1.0, 1.0);
            }
            samples.push(Sample {
                id: format!("synth/drone_{k:04}"),
                samples: clip,
                sample_rate,
                label: 1,
            });

            // --- negative: a mix of confounders chosen by k % 3 ---
            let mut clip = vec![0.0_f32; n];
            match k % 3 {
                0 => {
                    // broadband white noise
                    for s in clip.iter_mut() {
                        *s = 0.5 * rng.bipolar();
                    }
                }
                1 => {
                    // out-of-band low hum + noise (e.g. 50/60 Hz mains-ish)
                    let f = 45.0 + 30.0 * rng.unit01();
                    for (i, s) in clip.iter_mut().enumerate() {
                        let t = i as f32 / sample_rate as f32;
                        *s = (0.6 * (2.0 * PI * f * t).sin() + 0.05 * rng.bipolar())
                            .clamp(-1.0, 1.0);
                    }
                }
                _ => {
                    // single bright tone (whistle/insect-ish), out of harmonic context
                    let f = 2000.0 + 4000.0 * rng.unit01();
                    for (i, s) in clip.iter_mut().enumerate() {
                        let t = i as f32 / sample_rate as f32;
                        *s =
                            (0.5 * (2.0 * PI * f * t).sin() + 0.1 * rng.bipolar()).clamp(-1.0, 1.0);
                    }
                }
            }
            samples.push(Sample {
                id: format!("synth/neg_{k:04}"),
                samples: clip,
                sample_rate,
                label: 0,
            });
        }
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn n_pos(&self) -> usize {
        self.samples.iter().filter(|s| s.label == 1).count()
    }

    /// Stratified, deterministic train/test split. Within each class, samples
    /// are shuffled by a seeded RNG and the first `train_frac` go to train.
    pub fn split(&self, train_frac: f32, seed: u32) -> (Vec<Sample>, Vec<Sample>) {
        let mut rng = Rng(seed.max(1));
        let mut train = Vec::new();
        let mut test = Vec::new();
        for class in [0u8, 1u8] {
            let mut idx: Vec<usize> = self
                .samples
                .iter()
                .enumerate()
                .filter(|(_, s)| s.label == class)
                .map(|(i, _)| i)
                .collect();
            // Fisher-Yates with the seeded RNG.
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

/// Decode a WAV file to mono `f32` in `[-1, 1]`, downmixing channels.
pub fn read_mono_wav(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

/// Tiny deterministic xorshift32 PRNG (no rng crate dependency).
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
