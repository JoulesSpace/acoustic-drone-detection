//! `drone-live` - real-time acoustic drone detection from a live microphone,
//! plus a hardware/mic probe.
//!
//! Two subcommands:
//!
//! * **`devices`** - the hardware probe. Enumerate every audio input device and
//!   its supported configs (channels, sample-rate range, sample format), marking
//!   the default. Answers "what mics do we have and what can they do?".
//!
//! * **`listen`** - open an input device, downmix + resample to 16 kHz, frame the
//!   stream, score each window with a [`drone_bench`] detection approach, smooth
//!   the confidence with an EMA, and alert when it holds above a threshold.
//!
//! * **`record`** - open an input device, downmix + resample to 16 kHz, segment
//!   the stream into fixed-length clips, and write them as labelled mono 16 kHz
//!   WAVs (plus a `labels.csv` manifest) for building a truly held-out field
//!   dataset. The recorded clips share NO recordings with DADS, so evaluating on
//!   them (via `drone-bench`'s `fieldeval` bin) is leakage-free. See [`record`].
//!
//! The detection logic is reused verbatim from the benchmark harness
//! (`drone_bench::approaches`) so the live detector and the offline benchmark
//! share one implementation. See [`detector`] for which approaches are usable
//! without labelled training data at runtime.

mod detector;
mod devices;
mod listen;
mod record;
mod resample;

use std::error::Error;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use detector::DEFAULT_APPROACH;
use listen::ListenConfig;
use record::RecordConfig;

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
    /// Record labelled mic clips into a dataset dir for training / held-out eval.
    Record {
        /// Binary label: `drone` (written as 1) or `nondrone` (written as 0).
        /// Mutually exclusive with `--class`; one of the two is required.
        #[arg(long)]
        label: Option<Label>,
        /// Multiclass label: a name + integer class id, e.g. `--class quadcopter:2`.
        /// The name is the clip subdirectory; the id is written to labels.csv.
        /// Mutually exclusive with `--label`.
        #[arg(long, value_parser = parse_class)]
        class: Option<(String, u32)>,
        /// Total seconds to record across all clips.
        #[arg(long, default_value_t = 60.0)]
        seconds: f32,
        /// Length of each clip, in seconds.
        #[arg(long, default_value_t = 1.0)]
        clip_len: f32,
        /// Output dataset root: clips go to `<out>/<label>/NNNNN.wav`.
        #[arg(long, default_value = "data/field")]
        out: PathBuf,
        /// Input device name substring; default = the system default input.
        #[arg(long)]
        device: Option<String>,
    },
}

/// The two binary labels, with their canonical subdirectory name and CSV id.
#[derive(Clone, Copy, clap::ValueEnum)]
enum Label {
    Drone,
    Nondrone,
}

impl Label {
    /// Clip subdirectory name (also the convention DADS/field manifests expect).
    fn dir(self) -> &'static str {
        match self {
            Label::Drone => "drone",
            Label::Nondrone => "nondrone",
        }
    }
    /// Integer label written to `labels.csv`: 1 = drone, 0 = nondrone.
    fn id(self) -> u32 {
        match self {
            Label::Drone => 1,
            Label::Nondrone => 0,
        }
    }
}

/// Parse a `--class name:id` argument into `(name, id)`. Accepts a bare `name`
/// too, defaulting the id to 1 (treated as a positive multiclass member).
fn parse_class(s: &str) -> Result<(String, u32), String> {
    match s.rsplit_once(':') {
        Some((name, id)) => {
            let id: u32 = id
                .trim()
                .parse()
                .map_err(|_| format!("class id must be an integer in '{s}'"))?;
            let name = name.trim();
            if name.is_empty() {
                return Err(format!("class name must be non-empty in '{s}'"));
            }
            Ok((name.to_string(), id))
        }
        None => {
            let name = s.trim();
            if name.is_empty() {
                return Err("class name must be non-empty".to_string());
            }
            Ok((name.to_string(), 1))
        }
    }
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
        Command::Record {
            label,
            class,
            seconds,
            clip_len,
            out,
            device,
        } => {
            // Resolve the class dir + integer id from exactly one of --label / --class.
            let (class_dir, id) = match (label, class) {
                (Some(_), Some(_)) => {
                    return Err("pass only one of --label or --class".into());
                }
                (Some(l), None) => (l.dir().to_string(), l.id()),
                (None, Some((name, id))) => (name, id),
                (None, None) => {
                    return Err("pass --label <drone|nondrone> or --class <name[:id]>".into());
                }
            };
            record::run(RecordConfig {
                device: device.as_deref(),
                label: id,
                class_dir,
                seconds,
                clip_len_s: clip_len,
                out,
            })
        }
    }
}
