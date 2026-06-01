//! The mel-spectrogram CNN: a compact 2-conv-block convolutional classifier,
//! the upstream-literature representative.
//!
//! Architecture (deliberately small, matching the Al-Emadi / MDPI family):
//!   input  : 1 x N_MELS x N_FRAMES   (log-mel image)
//!   conv1  : 1->8,  3x3, pad 1  -> ReLU -> maxpool 2x2
//!   conv2  : 8->16, 3x3, pad 1  -> ReLU -> maxpool 2x2
//!   flatten
//!   dense1 : -> 32 -> ReLU
//!   dense2 : -> 1  -> sigmoid (binary drone / no-drone)
//!
//! Built on `candle-nn` layers, but every weight is initialized from our OWN
//! seeded xorshift RNG (Kaiming-uniform for conv/dense, zero bias). candle's CPU
//! RNG is explicitly not seedable (`set_seed` bails on the CPU backend), so to
//! keep training reproducible we never touch it: we synthesize the initial
//! weights deterministically and feed candle's `Conv2d::new` / `Linear::new`
//! the resulting tensors. The collected `Var`s drive the AdamW optimizer.

use candle_core::{Device, Result, Tensor, Var};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module};

use crate::mel::{N_FRAMES, N_MELS};

/// Channels after conv1 / conv2.
const C1: usize = 8;
const C2: usize = 16;
/// Hidden width of the first dense layer.
const DENSE: usize = 32;

/// Frequency/time after two 2x2 max-pools (padding-1, stride-1 3x3 convs keep
/// the spatial size, so each pool halves it).
const POOLED_MELS: usize = N_MELS / 4;
const POOLED_FRAMES: usize = N_FRAMES / 4;
/// Flattened length feeding the first dense layer.
const FLAT: usize = C2 * POOLED_MELS * POOLED_FRAMES;

/// Deterministic xorshift128+ -ish PRNG (same family as the harness's loader,
/// kept local so model init has no external RNG dependency).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the zero state; mix the seed.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform f32 in `[-1, 1)`.
    fn bipolar(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
}

/// Kaiming-uniform weight vector of length `n`, fan-in `fan_in`, drawn from the
/// seeded RNG. Bound = sqrt(6 / fan_in) (ReLU gain), the standard He-uniform.
fn kaiming_uniform(rng: &mut Rng, n: usize, fan_in: usize) -> Vec<f32> {
    let bound = (6.0f32 / fan_in as f32).sqrt();
    (0..n).map(|_| rng.bipolar() * bound).collect()
}

/// The CNN. Holds the layers plus the flat list of trainable `Var`s.
pub struct DroneCnn {
    conv1: Conv2d,
    conv2: Conv2d,
    dense1: Linear,
    dense2: Linear,
    vars: Vec<Var>,
    device: Device,
}

impl DroneCnn {
    /// Build the network with deterministic, seed-driven weight init on `device`.
    pub fn new(seed: u64, device: &Device) -> Result<Self> {
        let mut rng = Rng::new(seed);
        let mut vars: Vec<Var> = Vec::new();

        // Helper: make a Var from explicit data and register it for the optimizer.
        let mut mk = |data: Vec<f32>, shape: &[usize]| -> Result<Tensor> {
            let v = Var::from_vec(data, shape, device)?;
            let t = v.as_tensor().clone();
            vars.push(v);
            Ok(t)
        };

        // conv1: (C1, 1, 3, 3), fan_in = 1*3*3 = 9
        let w1 = mk(kaiming_uniform(&mut rng, C1 * 9, 9), &[C1, 1, 3, 3])?;
        let b1 = mk(vec![0.0; C1], &[C1])?;
        // conv2: (C2, C1, 3, 3), fan_in = C1*3*3
        let w2 = mk(
            kaiming_uniform(&mut rng, C2 * C1 * 3 * 3, C1 * 9),
            &[C2, C1, 3, 3],
        )?;
        let b2 = mk(vec![0.0; C2], &[C2])?;
        // dense1: (DENSE, FLAT)
        let wd1 = mk(
            kaiming_uniform(&mut rng, DENSE * FLAT, FLAT),
            &[DENSE, FLAT],
        )?;
        let bd1 = mk(vec![0.0; DENSE], &[DENSE])?;
        // dense2: (1, DENSE)
        let wd2 = mk(kaiming_uniform(&mut rng, DENSE, DENSE), &[1, DENSE])?;
        let bd2 = mk(vec![0.0; 1], &[1])?;

        let cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        Ok(Self {
            conv1: Conv2d::new(w1, Some(b1), cfg),
            conv2: Conv2d::new(w2, Some(b2), cfg),
            dense1: Linear::new(wd1, Some(bd1)),
            dense2: Linear::new(wd2, Some(bd2)),
            vars,
            device: device.clone(),
        })
    }

    /// Trainable variables (for the optimizer).
    pub fn vars(&self) -> Vec<Var> {
        self.vars.clone()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Forward pass returning the pre-sigmoid logit, shape `(batch, 1)`.
    /// `x` is `(batch, 1, N_MELS, N_FRAMES)`.
    pub fn logits(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?;
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?;
        let x = x.flatten_from(1)?; // (batch, FLAT)
        let x = self.dense1.forward(&x)?.relu()?;
        self.dense2.forward(&x) // (batch, 1)
    }

    /// Forward pass returning the sigmoid probability, shape `(batch, 1)`.
    pub fn probs(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::sigmoid(&self.logits(x)?)
    }

    /// Build the `(n, 1, N_MELS, N_FRAMES)` input tensor from flat mel images.
    pub fn batch_tensor(&self, images: &[Vec<f32>]) -> Result<Tensor> {
        let n = images.len();
        let mut flat = Vec::with_capacity(n * N_MELS * N_FRAMES);
        for img in images {
            flat.extend_from_slice(img);
        }
        Tensor::from_vec(flat, (n, 1, N_MELS, N_FRAMES), &self.device)
    }
}

/// Binary cross-entropy from logits, mean over the batch. Numerically stable
/// form: `max(z,0) - z*y + log(1 + exp(-|z|))`.
pub fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let zeros = logits.zeros_like()?;
    let max_part = logits.maximum(&zeros)?;
    let zy = (logits * targets)?;
    let abs = logits.abs()?;
    let log_part = ((abs.neg()?.exp()? + 1.0)?).log()?;
    let loss = ((max_part - zy)? + log_part)?;
    loss.mean_all()
}
