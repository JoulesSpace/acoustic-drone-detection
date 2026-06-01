//! `drone-live` — real-time acoustic drone detection from a live microphone,
//! plus a hardware/mic probe.
//!
//! Two subcommands:
//!
//! * **`devices`** — the hardware probe. Enumerate every audio input device and
//!   its supported configs (channels, sample-rate range, sample format), marking
//!   the default. Answers "what mics do we have and what can they do?".
//!
//! * **`listen`** — open an input device, downmix + resample to 16 kHz, frame the
//!   stream, score each window with a [`drone_bench`] detection approach, smooth
//!   the confidence with an EMA, and alert when it holds above a threshold.
//!
//! The detection logic is reused verbatim from the benchmark harness
//! (`drone_bench::approaches`) so the live detector and the offline benchmark
//! share one implementation. See [`detector`] for which approaches are usable
//! without labelled training data at runtime.

mod detector;
mod devices;
mod listen;
mod resample;

use std::error::Error;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use detector::DEFAULT_APPROACH;
use listen::ListenConfig;

#[derive(Parser)]
#[command(
    name = "drone-live",
    version,
    about = "Real-time drone detection from a live mic, plus a mic/hardware probe"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Probe: list audio input devices and their supported configs.
    Devices,
    /// Listen on a microphone and detect drones in real time.
    Listen {
        /// Input device name substring; default = the system default input.
        #[arg(long)]
        device: Option<String>,
        /// EMA confidence above which a window counts toward a detection.
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        /// Consecutive over-threshold windows required to declare a detection.
        #[arg(long, default_value_t = 3)]
        hold: usize,
        /// Detection approach. Default is unsupervised (no training needed).
        #[arg(long, default_value = DEFAULT_APPROACH)]
        approach: String,
        /// Optional labelled dir (with labels.csv, header `path,label`) to fit a
        /// supervised approach on before listening.
        #[arg(long)]
        train: Option<PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Devices => devices::run(),
        Command::Listen {
            device,
            threshold,
            hold,
            approach,
            train,
        } => {
            let (detector, note) = detector::build(&approach, train.as_deref())?;
            println!("{note}");
            listen::run(ListenConfig {
                device: device.as_deref(),
                threshold,
                hold: hold.max(1),
                approach: detector,
            })
        }
    }
}
