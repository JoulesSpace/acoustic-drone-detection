//! `drone` - host-side CLI for the acoustic drone detector.
//!
//! Two subcommands:
//! - `synth` - generate a drone-like (or plain-tone) test WAV so the pipeline
//!   is exercisable without real recordings.
//! - `analyze` - run a sliding-window STFT over a WAV, compute spectral
//!   features per frame, run the detector, and summarise.
//!
//! The host crate owns all the std-only concerns (file IO, the over-the-whole-
//! signal loop). The actual DSP and detection live in the `no_std`-friendly
//! `drone-dsp` / `drone-detect` crates so they can be reused on the edge.

use std::f32::consts::PI;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use drone_detect::Detector;
use drone_dsp::{hann_in_place, magnitude_spectrum, spectral_centroid, Frame, FRAME_SIZE};

#[derive(Parser)]
#[command(
    name = "drone",
    version,
    about = "Acoustic drone detection toolkit (v0.1.0)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a test WAV (mono, 16-bit PCM).
    Synth {
        /// Output path.
        #[arg(short, long, default_value = "drone.wav")]
        out: PathBuf,
        /// Duration in seconds.
        #[arg(short, long, default_value_t = 3.0)]
        seconds: f32,
        /// Sample rate in Hz.
        #[arg(short = 'r', long, default_value_t = 16_000)]
        sample_rate: u32,
        /// Fundamental (blade-pass) frequency in Hz.
        #[arg(short, long, default_value_t = 120.0)]
        fundamental: f32,
        /// Generate a plain sine instead of a harmonic "drone" stack + noise.
        #[arg(long)]
        plain: bool,
    },
    /// Analyze a WAV file frame by frame and report detections.
    Analyze {
        /// Input WAV path.
        #[arg(short, long)]
        input: PathBuf,
        /// Emit per-frame features as CSV to stdout instead of a summary.
        #[arg(long)]
        csv: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Synth {
            out,
            seconds,
            sample_rate,
            fundamental,
            plain,
        } => synth(&out, seconds, sample_rate, fundamental, plain),
        Command::Analyze { input, csv } => analyze(&input, csv),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Cheap deterministic PRNG (xorshift32) so synth needs no rng crate and is
/// reproducible across runs.
struct XorShift(u32);
impl XorShift {
    fn next_unit(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        // Map to [-1, 1).
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn synth(
    out: &PathBuf,
    seconds: f32,
    sample_rate: u32,
    fundamental: f32,
    plain: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let total = (seconds * sample_rate as f32) as usize;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out, spec)?;
    let mut rng = XorShift(0x1234_5678);

    for n in 0..total {
        let t = n as f32 / sample_rate as f32;
        let sample = if plain {
            0.6 * (2.0 * PI * fundamental * t).sin()
        } else {
            // Stack of harmonics with decaying amplitude - a crude but
            // recognisable multirotor signature - plus broadband noise and a
            // slow amplitude modulation (rotor wobble).
            let am = 1.0 + 0.2 * (2.0 * PI * 8.0 * t).sin();
            let mut s = 0.0_f32;
            for k in 1..=6 {
                let amp = 0.5 / k as f32;
                s += amp * (2.0 * PI * fundamental * k as f32 * t).sin();
            }
            0.7 * am * s + 0.05 * rng.next_unit()
        };
        let clamped = sample.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    println!(
        "wrote {} ({:.1}s, {} Hz, {})",
        out.display(),
        seconds,
        sample_rate,
        if plain {
            "plain tone"
        } else {
            "synthetic drone"
        }
    );
    Ok(())
}

/// Read a WAV into mono `f32` samples in `[-1, 1]`, downmixing if needed.
fn read_mono(input: &PathBuf) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(input)?;
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
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

fn analyze(input: &PathBuf, csv: bool) -> Result<(), Box<dyn std::error::Error>> {
    let (samples, sample_rate) = read_mono(input)?;
    if samples.len() < FRAME_SIZE {
        return Err(format!(
            "signal too short: {} samples, need at least {FRAME_SIZE}",
            samples.len()
        )
        .into());
    }

    let detector = Detector::new(sample_rate);
    let hop = FRAME_SIZE / 2; // 50% overlap

    let mut frames = 0usize;
    let mut drone_frames = 0usize;
    let mut sum_ratio = 0.0_f32;
    let mut sum_dom_hz = 0.0_f32;
    let mut sum_centroid = 0.0_f32;

    if csv {
        println!("frame,time_s,dominant_hz,centroid_hz,band_ratio,confidence,is_drone");
    }

    let mut start = 0usize;
    while start + FRAME_SIZE <= samples.len() {
        let mut frame: Frame = [0.0; FRAME_SIZE];
        frame.copy_from_slice(&samples[start..start + FRAME_SIZE]);
        hann_in_place(&mut frame);

        // magnitude_spectrum needs a separate buffer copy for the centroid,
        // since the FFT consumes the frame in place.
        let spectrum = magnitude_spectrum(&mut frame);
        let centroid = spectral_centroid(&spectrum, sample_rate);
        let det = detector.analyze(&spectrum);

        if csv {
            let time_s = start as f32 / sample_rate as f32;
            println!(
                "{frames},{time_s:.3},{:.1},{:.1},{:.4},{:.4},{}",
                det.dominant_hz, centroid, det.band_ratio, det.confidence, det.is_drone as u8
            );
        }

        frames += 1;
        if det.is_drone {
            drone_frames += 1;
        }
        sum_ratio += det.band_ratio;
        sum_dom_hz += det.dominant_hz;
        sum_centroid += centroid;

        start += hop;
    }

    if !csv {
        let f = frames as f32;
        let pct = 100.0 * drone_frames as f32 / f;
        println!("file:            {}", input.display());
        println!("sample_rate:     {sample_rate} Hz");
        println!("frames:          {frames} ({FRAME_SIZE}-sample, 50% overlap)");
        println!("drone frames:    {drone_frames} ({pct:.1}%)");
        println!("mean band ratio: {:.3}", sum_ratio / f);
        println!("mean dominant:   {:.1} Hz", sum_dom_hz / f);
        println!("mean centroid:   {:.1} Hz", sum_centroid / f);
        println!(
            "verdict:         {}",
            if pct >= 50.0 {
                "DRONE PRESENT"
            } else {
                "no drone"
            }
        );
    }

    Ok(())
}
