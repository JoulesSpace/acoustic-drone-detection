//! Corpus loading for the head-to-head: DADS (train + in-distribution test) and
//! the leakage-proof unseen-drone test set (32-brand positives + ESC-50
//! negatives).
//!
//! The unseen-drone loading mirrors `drone_bench`'s `heldout32` binary verbatim
//! (same anti-aliased 44.1k->16k resampler, same ~1 s windowing, same brand
//! parsing, same ESC-50 negative classes) so the CNN is evaluated on EXACTLY the
//! corpus our heuristic detectors were on. We re-implement the loader here
//! rather than editing `drone-bench`, per the task's "do not edit existing
//! crates" constraint; the algorithm is identical.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use drone_bench::dataset::{read_mono_wav, Sample};

/// Target sample rate the 16 kHz-tuned pipeline expects.
pub const TARGET_SR: u32 = 16_000;

/// Window length (seconds) each unseen-drone clip is cut into (~1 s, matching
/// DADS / Al-Emadi clip granularity).
const WINDOW_SECS: f32 = 1.0;

/// ESC-50 negative classes (same split heldout32 uses). The confusable group is
/// acoustically rotor/engine-like; the control group is a few easy negatives.
/// These ARE inside DADS, so the ROC/F1 they produce is indicative, not clean.
const NEG_CONFUSABLE_CLASSES: &[&str] = &["airplane", "helicopter", "engine", "chainsaw", "wind"];
const NEG_CONTROL_CLASSES: &[&str] = &["rain", "sea_waves", "clapping"];

/// One unseen-drone positive clip plus the brand it came from.
pub struct DroneClip {
    pub sample: Sample,
    pub brand: String,
}

// ---------------------------------------------------------------------------
// Resampling - identical scheme to heldout32/xeval. 44.1k -> 16k: box low-pass
// of width floor(src/dst) then linear interpolation onto the target grid.
// Deterministic and dependency-free.
// ---------------------------------------------------------------------------

/// Resample `x` from `src_sr` to `dst_sr` with anti-aliased linear interpolation.
pub fn resample(x: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || x.is_empty() {
        return x.to_vec();
    }
    let filtered: Vec<f32> = if src_sr > dst_sr {
        let width = (src_sr / dst_sr).max(1) as usize;
        if width <= 1 {
            x.to_vec()
        } else {
            box_lowpass(x, width)
        }
    } else {
        x.to_vec()
    };
    let ratio = src_sr as f64 / dst_sr as f64;
    let out_len = ((x.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let a = filtered[i0.min(filtered.len() - 1)];
        let b = filtered[(i0 + 1).min(filtered.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Centered moving-average low-pass of the given width (clamped at the ends).
fn box_lowpass(x: &[f32], width: usize) -> Vec<f32> {
    let w = width.max(1);
    let half = w / 2;
    let mut out = vec![0.0f32; x.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(half);
        let hi = (i + half).min(x.len() - 1);
        let mut acc = 0.0f32;
        for &v in &x[lo..=hi] {
            acc += v;
        }
        *o = acc / (hi - lo + 1) as f32;
    }
    out
}

// ---------------------------------------------------------------------------
// Unseen-drone positives (32-brand drone-visualization set).
// ---------------------------------------------------------------------------

/// Parse the brand/model from a 32-set filename, dropping the trailing take
/// index, e.g. `DJI_Mavic_Air2_63.wav` -> `DJI_Mavic_Air2`.
fn brand_of(file_stem: &str) -> String {
    match file_stem.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => {
            head.to_string()
        }
        _ => file_stem.to_string(),
    }
}

/// Load the 32-brand drone clips from `<root>/public/droneAudio`: decode to
/// mono, resample to 16 kHz, cut into non-overlapping ~`WINDOW_SECS` windows
/// (label 1). Deterministic by filename sort.
pub fn load_heldout_drones(root: &Path) -> Result<Vec<DroneClip>, Box<dyn Error>> {
    let dir = root.join("public").join("droneAudio");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("reading {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wav").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no WAVs under {}", dir.display()).into());
    }

    let win = (WINDOW_SECS * TARGET_SR as f32) as usize;
    let mut out = Vec::new();
    for p in &files {
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        let brand = brand_of(&stem);
        let (audio, sr) = read_mono_wav(p)?;
        let audio = resample(&audio, sr, TARGET_SR);
        let n_full = audio.len() / win;
        let mut made = 0usize;
        for w in 0..n_full {
            let start = w * win;
            out.push(DroneClip {
                sample: Sample {
                    id: format!("dv/{stem}#{w}"),
                    samples: audio[start..start + win].to_vec(),
                    sample_rate: TARGET_SR,
                    label: 1,
                },
                brand: brand.clone(),
            });
            made += 1;
        }
        let rem = audio.len() - n_full * win;
        if made == 0 && rem >= win / 2 && !audio.is_empty() {
            out.push(DroneClip {
                sample: Sample {
                    id: format!("dv/{stem}#0"),
                    samples: audio.clone(),
                    sample_rate: TARGET_SR,
                    label: 1,
                },
                brand: brand.clone(),
            });
        }
    }
    Ok(out)
}

/// Load ESC-50 negatives for the chosen confusable + control classes, resampled
/// to 16 kHz. `per_class` caps clips per class. Deterministic by filename sort.
pub fn load_esc50_negatives(root: &Path, per_class: usize) -> Result<Vec<Sample>, Box<dyn Error>> {
    let meta = root.join("meta").join("esc50.csv");
    let text =
        std::fs::read_to_string(&meta).map_err(|e| format!("reading {}: {e}", meta.display()))?;
    let wanted: Vec<&str> = NEG_CONFUSABLE_CLASSES
        .iter()
        .chain(NEG_CONTROL_CLASSES)
        .copied()
        .collect();

    let mut by_class: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 4 {
            continue;
        }
        let filename = cols[0].trim();
        let category = cols[3].trim();
        if wanted.contains(&category) {
            by_class
                .entry(category.to_string())
                .or_default()
                .push(filename.to_string());
        }
    }

    let audio_dir = root.join("audio");
    let mut out = Vec::new();
    for (category, mut files) in by_class {
        files.sort();
        for f in files
            .into_iter()
            .take(if per_class > 0 { per_class } else { usize::MAX })
        {
            let p = audio_dir.join(&f);
            let (audio, sr) = read_mono_wav(&p)?;
            let audio = resample(&audio, sr, TARGET_SR);
            out.push(Sample {
                id: format!("esc50/{category}/{f}"),
                samples: audio,
                sample_rate: TARGET_SR,
                label: 0,
            });
        }
    }
    Ok(out)
}

/// The ESC-50 class lists, exposed for the JSON report.
pub fn neg_classes() -> (Vec<String>, Vec<String>) {
    (
        NEG_CONFUSABLE_CLASSES
            .iter()
            .map(|s| s.to_string())
            .collect(),
        NEG_CONTROL_CLASSES.iter().map(|s| s.to_string()).collect(),
    )
}
