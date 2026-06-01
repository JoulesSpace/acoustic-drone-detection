//! Spectral template matching (the owner's hypothesis).
//!
//! `fit` averages L2-normalized log-magnitude spectra of the drone training
//! clips into a single template. `score` is the cosine similarity of a clip's
//! mean log-spectrum to that template - a bounded, amplitude-invariant
//! confidence. Expected to be a decent baseline but to trail harmonic-structure
//! methods under noise / model variation (see approaches survey).

use crate::dataset::Sample;
use crate::{util, Approach};
use drone_dsp::NUM_BINS;

#[derive(Default)]
pub struct Template {
    template: Vec<f32>,
}

impl Template {
    pub fn new() -> Self {
        Self::default()
    }
}

fn log_magnitude(spec: &mut [f32]) {
    for x in spec.iter_mut() {
        *x = (1.0 + *x).ln();
    }
}

impl Approach for Template {
    fn name(&self) -> &str {
        "template"
    }
    fn description(&self) -> &str {
        "Cosine similarity to averaged drone log-magnitude spectrum"
    }

    fn fit(&mut self, train: &[Sample]) {
        let mut acc = vec![0.0_f32; NUM_BINS];
        let mut count = 0usize;
        for s in train.iter().filter(|s| s.label == 1) {
            let mut mean = util::mean_spectrum(&s.samples);
            log_magnitude(&mut mean);
            let mut v = mean.to_vec();
            util::l2_normalize(&mut v);
            for (a, x) in acc.iter_mut().zip(v.iter()) {
                *a += *x;
            }
            count += 1;
        }
        if count > 0 {
            for a in acc.iter_mut() {
                *a /= count as f32;
            }
        }
        util::l2_normalize(&mut acc);
        self.template = acc;
    }

    fn score(&self, samples: &[f32], _sample_rate: u32) -> f32 {
        if self.template.is_empty() {
            return 0.0;
        }
        let mut mean = util::mean_spectrum(samples);
        log_magnitude(&mut mean);
        let mut v = mean.to_vec();
        util::l2_normalize(&mut v);
        util::cosine(&v, &self.template).clamp(0.0, 1.0)
    }
}
