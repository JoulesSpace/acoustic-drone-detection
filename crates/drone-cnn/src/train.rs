//! Deterministic training loop for the mel-spectrogram CNN.
//!
//! Full-batch-shuffled mini-batch AdamW with BCE-from-logits loss and early
//! stopping on a validation slice (best-val-loss checkpoint, kept in memory as
//! the raw weight vectors). Batch order is a seeded Fisher-Yates permutation,
//! so a fixed seed reproduces the run bit-for-bit on the CPU backend.

use candle_core::{Result, Tensor};
use candle_nn::Optimizer;

use crate::model::{bce_with_logits, DroneCnn};

/// Hyperparameters. Small + early-stopping, matching the modest upstream nets.
pub struct TrainCfg {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Stop if val loss fails to improve for this many epochs.
    pub patience: usize,
    pub seed: u64,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            epochs: 60,
            batch_size: 32,
            lr: 1e-3,
            patience: 8,
            seed: 1,
        }
    }
}

/// Local seeded xorshift for batch shuffling (independent of model init RNG).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0xD1B5_4A32_D192_ED03).max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Fisher-Yates shuffle of `idx` with the given RNG.
fn shuffle(idx: &mut [usize], rng: &mut Rng) {
    for i in (1..idx.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
}

/// Snapshot of all model weights (flat data), for the best-val checkpoint.
struct Checkpoint {
    data: Vec<Vec<f32>>,
}

impl Checkpoint {
    fn capture(model: &DroneCnn) -> Result<Self> {
        let mut data = Vec::new();
        for v in model.vars() {
            data.push(v.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(Self { data })
    }
    fn restore(&self, model: &DroneCnn) -> Result<()> {
        for (v, d) in model.vars().iter().zip(self.data.iter()) {
            let shape = v.as_tensor().shape().clone();
            let t = Tensor::from_vec(d.clone(), shape, model.device())?;
            v.set(&t)?;
        }
        Ok(())
    }
}

/// Outcome of a training run (for reporting).
pub struct TrainReport {
    pub epochs_run: usize,
    pub best_epoch: usize,
    pub best_val_loss: f32,
    pub final_train_loss: f32,
}

/// Train `model` on `(train_imgs, train_labels)`, early-stopping on
/// `(val_imgs, val_labels)`. Restores the best-val-loss weights at the end.
#[allow(clippy::too_many_arguments)]
pub fn train(
    model: &DroneCnn,
    train_imgs: &[Vec<f32>],
    train_labels: &[f32],
    val_imgs: &[Vec<f32>],
    val_labels: &[f32],
    cfg: &TrainCfg,
    verbose: bool,
) -> Result<TrainReport> {
    let mut opt = candle_nn::AdamW::new_lr(model.vars(), cfg.lr)?;
    let n = train_imgs.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(cfg.seed);

    let val_x = model.batch_tensor(val_imgs)?;
    let val_y = Tensor::from_vec(val_labels.to_vec(), (val_labels.len(), 1), model.device())?;

    let mut best = Checkpoint::capture(model)?;
    let mut best_val = f32::INFINITY;
    let mut best_epoch = 0;
    let mut since_improve = 0usize;
    let mut final_train_loss = f32::NAN;
    let mut epochs_run = 0;

    for epoch in 0..cfg.epochs {
        epochs_run = epoch + 1;
        shuffle(&mut order, &mut rng);
        let mut epoch_loss = 0.0f32;
        let mut nb = 0usize;
        let mut start = 0;
        while start < n {
            let end = (start + cfg.batch_size).min(n);
            let batch_idx = &order[start..end];
            let imgs: Vec<Vec<f32>> = batch_idx.iter().map(|&i| train_imgs[i].clone()).collect();
            let labels: Vec<f32> = batch_idx.iter().map(|&i| train_labels[i]).collect();
            let x = model.batch_tensor(&imgs)?;
            let y = Tensor::from_vec(labels, (batch_idx.len(), 1), model.device())?;
            let logits = model.logits(&x)?;
            let loss = bce_with_logits(&logits, &y)?;
            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()?;
            nb += 1;
            start = end;
        }
        final_train_loss = epoch_loss / nb.max(1) as f32;

        // Validation loss for early stopping (no grad needed; reuse logits).
        let val_logits = model.logits(&val_x)?;
        let val_loss = bce_with_logits(&val_logits, &val_y)?.to_scalar::<f32>()?;

        if verbose {
            println!(
                "  epoch {:>3}: train_loss {:.4}  val_loss {:.4}",
                epoch + 1,
                final_train_loss,
                val_loss
            );
        }

        if val_loss + 1e-5 < best_val {
            best_val = val_loss;
            best_epoch = epoch + 1;
            best = Checkpoint::capture(model)?;
            since_improve = 0;
        } else {
            since_improve += 1;
            if since_improve >= cfg.patience {
                if verbose {
                    println!("  early stop at epoch {} (no val improvement)", epoch + 1);
                }
                break;
            }
        }
    }

    best.restore(model)?;
    Ok(TrainReport {
        epochs_run,
        best_epoch,
        best_val_loss: best_val,
        final_train_loss,
    })
}

/// Score a batch of mel images, returning per-clip drone probabilities in [0,1].
/// Chunked to keep peak memory bounded for large test sets.
pub fn predict(model: &DroneCnn, imgs: &[Vec<f32>]) -> Result<Vec<f32>> {
    const CHUNK: usize = 256;
    let mut out = Vec::with_capacity(imgs.len());
    let mut start = 0;
    while start < imgs.len() {
        let end = (start + CHUNK).min(imgs.len());
        let x = model.batch_tensor(&imgs[start..end])?;
        let p = model.probs(&x)?.flatten_all()?.to_vec1::<f32>()?;
        out.extend_from_slice(&p);
        start = end;
    }
    Ok(out)
}
