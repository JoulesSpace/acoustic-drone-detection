//! The live listener: open a mic, resample to 16 kHz, frame, score, alert.
//!
//! Data flow:
//!
//! ```text
//!  cpal capture thread                main thread
//!  ┌──────────────────┐   mpsc   ┌──────────────────────────────┐
//!  │ device callback  │ ──────▶  │ accumulate ≥ WINDOW samples   │
//!  │ to-f32 + downmix │  mono    │ score via Approach            │
//!  │ + resample 16kHz │  f32     │ EMA confidence + hold logic   │
//!  └──────────────────┘          │ live meter + ⚠ DRONE DETECTED │
//!                                 └──────────────────────────────┘
//! ```
//!
//! The audio callback must never block, so it does only cheap, allocation-light
//! work (sample conversion, downmix, linear resample) and ships mono 16 kHz
//! samples to the main thread over a channel. The main thread owns all detection
//! and printing. We score on a sliding **window** of several frames (so the
//! [`Approach`]'s internal robust per-frame aggregation has frames to work with),
//! advancing by [`drone_dsp::FRAME_SIZE`] each step (~64 ms at 16 kHz).

use std::error::Error;
use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use drone_bench::Approach;
use drone_dsp::FRAME_SIZE;

use crate::resample::{Resampler, TARGET_RATE};

/// Detection window: how many 16 kHz samples each `score()` call sees.
///
/// Eight frames (~0.5 s of audio with 50% overlap) gives the harmonic-comb
/// aggregation enough frames for its robust high-quantile while keeping the
/// alert latency around half a second.
const WINDOW: usize = FRAME_SIZE * 8;

/// How far the window advances between scores: one frame (~64 ms at 16 kHz).
const STEP: usize = FRAME_SIZE;

/// EMA smoothing factor for the rolling confidence. Smaller = smoother / slower.
const EMA_ALPHA: f32 = 0.4;

/// Width of the printed confidence bar, in characters.
const BAR_WIDTH: usize = 30;

/// Parameters for one live-listen session.
pub struct ListenConfig<'a> {
    /// Substring to match a device name against; `None` = default input device.
    pub device: Option<&'a str>,
    /// EMA confidence above which a window counts toward the hold.
    pub threshold: f32,
    /// Consecutive over-threshold windows required to declare a detection.
    pub hold: usize,
    /// Pre-built, ready-to-score detector (see [`crate::detector::build`]).
    pub approach: Box<dyn Approach>,
}

/// Open the chosen input device and run the live detection loop forever
/// (until the process is interrupted). Blocks the calling thread.
pub fn run(cfg: ListenConfig) -> Result<(), Box<dyn Error>> {
    let host = cpal::default_host();
    let device = pick_device(&host, cfg.device)?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());

    let supported = device.default_input_config()?;
    let in_rate = supported.sample_rate().0;
    let channels = supported.channels();
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.clone().into();

    let path = if Resampler::new(in_rate, channels).is_passthrough() {
        "passthrough"
    } else {
        "downmix + resample"
    };
    println!(
        "listening on '{name}': {channels} ch, {in_rate} Hz, {sample_format:?} \
         → mono {TARGET_RATE} Hz ({path})"
    );
    println!(
        "approach '{}', threshold {:.2}, hold {} windows. press Ctrl-C to stop.\n",
        cfg.approach.name(),
        cfg.threshold,
        cfg.hold
    );

    // Channel from the audio callback to the detection loop.
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    // Each format gets its own typed callback; all funnel mono f32 into `tx`.
    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(&device, &stream_config, channels, in_rate, tx),
        SampleFormat::I16 => build_stream::<i16>(&device, &stream_config, channels, in_rate, tx),
        SampleFormat::U16 => build_stream::<u16>(&device, &stream_config, channels, in_rate, tx),
        SampleFormat::I32 => build_stream::<i32>(&device, &stream_config, channels, in_rate, tx),
        SampleFormat::I8 => build_stream::<i8>(&device, &stream_config, channels, in_rate, tx),
        SampleFormat::U8 => build_stream::<u8>(&device, &stream_config, channels, in_rate, tx),
        SampleFormat::F64 => build_stream::<f64>(&device, &stream_config, channels, in_rate, tx),
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    }?;
    stream.play()?;

    detection_loop(rx, &cfg);
    Ok(())
}

/// Pick the input device: the first whose name contains `want`, or the default.
fn pick_device(host: &cpal::Host, want: Option<&str>) -> Result<cpal::Device, Box<dyn Error>> {
    match want {
        Some(substr) => {
            let needle = substr.to_lowercase();
            for d in host.input_devices()? {
                if let Ok(n) = d.name() {
                    if n.to_lowercase().contains(&needle) {
                        return Ok(d);
                    }
                }
            }
            Err(format!("no input device matching '{substr}'").into())
        }
        None => host
            .default_input_device()
            .ok_or_else(|| "no default input device".into()),
    }
}

/// Build a typed input stream that converts samples to f32, downmixes, resamples
/// to [`TARGET_RATE`], and sends mono buffers down `tx`. One [`Resampler`] is
/// owned by the closure so its cross-buffer state persists for the whole stream.
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    in_rate: u32,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream, Box<dyn Error>>
where
    T: cpal::SizedSample + ToF32 + Send + 'static,
{
    let mut resampler = Resampler::new(in_rate, channels);
    let mut scratch: Vec<f32> = Vec::new();
    let mut mono_out: Vec<f32> = Vec::new();
    let err_fn = |e| eprintln!("audio stream error: {e}");
    let stream = device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _| {
            // Convert this callback's interleaved samples to f32 in `scratch`.
            scratch.clear();
            scratch.extend(data.iter().map(|s| s.to_f32()));
            mono_out.clear();
            resampler.push(&scratch, &mut mono_out);
            if !mono_out.is_empty() {
                // If the receiver is gone the loop has exited; drop silently.
                let _ = tx.send(mono_out.clone());
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

/// Main-thread loop: accumulate mono samples, score sliding windows, smooth with
/// an EMA, and alert when the EMA holds above threshold for `hold` windows.
fn detection_loop(rx: mpsc::Receiver<Vec<f32>>, cfg: &ListenConfig) {
    let mut buf: Vec<f32> = Vec::with_capacity(WINDOW * 2);
    let mut ema = 0.0_f32;
    let mut ema_primed = false;
    let mut over_count = 0usize;
    let mut alerting = false;

    // Block on the channel; the audio thread feeds us. When the sender is
    // dropped (never, in normal operation) `recv` errors and we return.
    while let Ok(chunk) = rx.recv() {
        buf.extend_from_slice(&chunk);

        // Score every time we have advanced one STEP past a full WINDOW.
        while buf.len() >= WINDOW {
            let conf = cfg
                .approach
                .score(&buf[..WINDOW], TARGET_RATE)
                .clamp(0.0, 1.0);

            ema = if ema_primed {
                EMA_ALPHA * conf + (1.0 - EMA_ALPHA) * ema
            } else {
                ema_primed = true;
                conf
            };

            update_alert(ema, cfg, &mut over_count, &mut alerting);
            print_meter(conf, ema, cfg.threshold, alerting);

            // Slide the window forward by one frame; keep the overlap.
            buf.drain(..STEP);
        }
    }
}

/// Advance the hold counter and flip the alerting state. Returns nothing; the
/// transition into the alert is what prints the warning banner.
fn update_alert(ema: f32, cfg: &ListenConfig, over_count: &mut usize, alerting: &mut bool) {
    if ema >= cfg.threshold {
        *over_count += 1;
        if *over_count >= cfg.hold && !*alerting {
            *alerting = true;
            // Newline so the banner isn't overwritten by the carriage-return meter.
            println!("\n⚠ DRONE DETECTED (confidence {ema:.2})");
        }
    } else {
        if *alerting {
            println!("\n· clear (confidence {ema:.2})");
        }
        *over_count = 0;
        *alerting = false;
    }
}

/// Print a one-line, carriage-returned live meter: a bar for the instantaneous
/// confidence plus the EMA value and current state.
fn print_meter(conf: f32, ema: f32, threshold: f32, alerting: bool) {
    use std::io::Write;
    let filled = ((ema * BAR_WIDTH as f32).round() as usize).min(BAR_WIDTH);
    let bar: String = "█".repeat(filled) + &"·".repeat(BAR_WIDTH - filled);
    let state = if alerting { "DRONE" } else { "  ·  " };
    // `\r` keeps the meter on one updating line; flush so it shows immediately.
    print!("\r[{bar}] ema {ema:.2} (now {conf:.2}, thr {threshold:.2}) {state}",);
    let _ = std::io::stdout().flush();
}

/// Convert a cpal sample type into a normalized `f32` in roughly `[-1, 1]`.
///
/// cpal's own `FromSample`/`Sample` conversions would also work, but a tiny
/// local trait keeps the conversion explicit and the dependency surface small.
trait ToF32 {
    fn to_f32(self) -> f32;
}

impl ToF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}
impl ToF32 for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl ToF32 for i8 {
    fn to_f32(self) -> f32 {
        self as f32 / i8::MAX as f32
    }
}
impl ToF32 for i16 {
    fn to_f32(self) -> f32 {
        self as f32 / i16::MAX as f32
    }
}
impl ToF32 for i32 {
    fn to_f32(self) -> f32 {
        self as f32 / i32::MAX as f32
    }
}
impl ToF32 for u8 {
    fn to_f32(self) -> f32 {
        (self as f32 / u8::MAX as f32) * 2.0 - 1.0
    }
}
impl ToF32 for u16 {
    fn to_f32(self) -> f32 {
        (self as f32 / u16::MAX as f32) * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_f32_normalizes_extremes() {
        assert!((i16::MAX.to_f32() - 1.0).abs() < 1e-3);
        assert!((0i16.to_f32()).abs() < 1e-6);
        assert!((u8::MAX.to_f32() - 1.0).abs() < 1e-2);
        assert!(((u8::MAX / 2).to_f32()).abs() < 0.01);
    }

    #[test]
    fn alert_requires_hold_then_clears() {
        let cfg = ListenConfig {
            device: None,
            threshold: 0.5,
            hold: 3,
            approach: crate::detector::build("band_ratio", None).unwrap().0,
        };
        let mut over = 0;
        let mut alerting = false;

        // Two over-threshold windows: not yet alerting (hold = 3).
        update_alert(0.9, &cfg, &mut over, &mut alerting);
        update_alert(0.9, &cfg, &mut over, &mut alerting);
        assert!(!alerting);
        // Third tips it over.
        update_alert(0.9, &cfg, &mut over, &mut alerting);
        assert!(alerting);
        // A below-threshold window clears it and resets the counter.
        update_alert(0.1, &cfg, &mut over, &mut alerting);
        assert!(!alerting);
        assert_eq!(over, 0);
    }
}
