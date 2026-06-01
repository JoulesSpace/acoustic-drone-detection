//! The `record` subcommand: capture labelled clips from a live mic for building
//! a TRULY held-out field dataset (training and, more importantly, evaluation).
//!
//! Why this exists: our in-distribution DADS numbers are inflated by
//! recording-level leakage (clips from one recording land in both train and
//! test). The only honest way to claim "beats upstream" is a held-out set
//! recorded with the owner's own drone and own mics - sharing NO recordings with
//! DADS. This subcommand makes collecting that set one command:
//!
//! ```text
//! drone-live record --label drone    --seconds 120
//! drone-live record --label nondrone --seconds 120
//! cargo run --manifest-path crates/drone-bench/Cargo.toml --bin fieldeval -- --field data/field
//! ```
//!
//! Data flow mirrors [`crate::listen`]: the cpal callback converts + downmixes +
//! resamples to mono 16 kHz and ships samples over a channel; the main thread
//! accumulates them into fixed-length clips and writes each clip as a mono 16 kHz
//! WAV, appending a row to `labels.csv`. The capture/resample path is reused
//! verbatim so recorded clips match exactly what the detector sees at runtime.
//!
//! The audio plumbing (open device, typed callback) is genuinely interactive -
//! it needs a real mic - so it can't run headless in CI. The piece that *can* be
//! tested deterministically is the clip segmentation + WAV writing, which is
//! factored into [`ClipWriter`] and covered by unit tests with synthetic buffers.

use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

use crate::resample::{Resampler, TARGET_RATE};

/// Default clip length in seconds when `--clip-len` is not given.
const DEFAULT_CLIP_LEN_S: f32 = 1.0;

/// Parameters for one recording session.
pub struct RecordConfig<'a> {
    /// Substring to match a device name against; `None` = default input device.
    pub device: Option<&'a str>,
    /// Integer class label written to `labels.csv` (1 = drone, 0 = nondrone, or
    /// any integer for multiclass) and used as the clip subdirectory name basis.
    pub label: u32,
    /// Subdirectory name under `out` the clips are written into (e.g. `drone`).
    pub class_dir: String,
    /// Total seconds to record (across all clips).
    pub seconds: f32,
    /// Length of each individual clip, in seconds.
    pub clip_len_s: f32,
    /// Output dataset root; clips go to `<out>/<class_dir>/NNNNN.wav` and rows are
    /// appended to `<out>/labels.csv`.
    pub out: PathBuf,
}

/// Run an interactive recording session: open the chosen input device and write
/// labelled clips until `seconds` of audio have been captured. Blocks the calling
/// thread. The structure of the output (filenames, CSV rows) is deterministic.
pub fn run(cfg: RecordConfig) -> Result<(), Box<dyn Error>> {
    let host = cpal::default_host();
    let device = pick_device(&host, cfg.device)?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());

    let supported = device.default_input_config()?;
    let in_rate = supported.sample_rate().0;
    let channels = supported.channels();
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.clone().into();

    let clip_len_s = if cfg.clip_len_s > 0.0 {
        cfg.clip_len_s
    } else {
        DEFAULT_CLIP_LEN_S
    };
    let clip_samples = (clip_len_s * TARGET_RATE as f32).round() as usize;
    let target_clips = ((cfg.seconds / clip_len_s).floor() as usize).max(1);

    let mut writer = ClipWriter::new(&cfg.out, &cfg.class_dir, cfg.label, clip_samples)?;

    println!(
        "recording on '{name}': {channels} ch, {in_rate} Hz, {sample_format:?} \
         -> mono {TARGET_RATE} Hz"
    );
    println!(
        "label {} (dir '{}'), clip-len {:.2}s ({} samples), target {} clips (~{:.0}s) -> {}",
        cfg.label,
        cfg.class_dir,
        clip_len_s,
        clip_samples,
        target_clips,
        target_clips as f32 * clip_len_s,
        cfg.out.display(),
    );
    println!(
        "press Ctrl-C to stop early. starting from clip index {}.\n",
        writer.next_index()
    );

    // Channel from the audio callback to the writing loop.
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

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

    let written = capture_loop(rx, &mut writer, target_clips)?;
    println!(
        "\ndone: wrote {written} clip(s) for label {} to {} ({} total rows in labels.csv)",
        cfg.label,
        cfg.out.join(&cfg.class_dir).display(),
        writer.next_index(),
    );
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
/// to [`TARGET_RATE`], and sends mono buffers down `tx`. Identical capture path
/// to [`crate::listen`] so recorded clips match runtime audio exactly.
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
            scratch.clear();
            scratch.extend(data.iter().map(|s| s.to_f32()));
            mono_out.clear();
            resampler.push(&scratch, &mut mono_out);
            if !mono_out.is_empty() {
                let _ = tx.send(mono_out.clone());
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

/// Main-thread loop: accumulate mono samples and flush a clip every time a full
/// clip's worth has arrived, until `target_clips` clips are written. Returns the
/// number of clips written.
fn capture_loop(
    rx: mpsc::Receiver<Vec<f32>>,
    writer: &mut ClipWriter,
    target_clips: usize,
) -> Result<usize, Box<dyn Error>> {
    let mut written = 0usize;
    while written < target_clips {
        match rx.recv() {
            Ok(chunk) => {
                let n = writer.push(&chunk)?;
                for _ in 0..n {
                    written += 1;
                    print_progress(written, target_clips);
                    if written >= target_clips {
                        break;
                    }
                }
            }
            // Sender dropped (audio thread gone): stop.
            Err(_) => break,
        }
    }
    Ok(written)
}

/// Print a one-line, carriage-returned progress indicator of clips written.
fn print_progress(written: usize, target: usize) {
    use std::io::Write as _;
    print!("\rrecorded {written}/{target} clips");
    let _ = std::io::stdout().flush();
}

/// Accumulates mono 16 kHz samples and writes fixed-length labelled WAV clips,
/// appending one `path,label` row to `labels.csv` per clip.
///
/// This is the deterministic, headless-testable core of `record`: it owns
/// segmentation (slicing the stream into `clip_samples`-length clips), filename
/// numbering (zero-padded, sequential, resuming past any clips already on disk),
/// and the WAV + CSV writing. The interactive cpal capture lives outside it.
pub struct ClipWriter {
    /// `<out>/<class_dir>` - where clip WAVs are written.
    clip_dir: PathBuf,
    /// Open handle to `<out>/labels.csv` in append mode.
    csv: File,
    /// Integer label written to `labels.csv`.
    label: u32,
    /// Samples per clip (mono 16 kHz).
    clip_samples: usize,
    /// Next sequential clip index (also the count of clips written so far in this
    /// directory, including any pre-existing ones we resumed past).
    next_index: usize,
    /// Carry-over partial clip across `push` calls.
    buf: Vec<f32>,
}

impl ClipWriter {
    /// Create the output layout under `out`: ensure `<out>/<class_dir>` exists,
    /// open (creating with a header if absent) `<out>/labels.csv` for appending,
    /// and resume the clip index past any `NNNNN.wav` already in the class dir so
    /// repeated sessions don't clobber earlier recordings.
    pub fn new(
        out: &Path,
        class_dir: &str,
        label: u32,
        clip_samples: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let clip_dir = out.join(class_dir);
        fs::create_dir_all(&clip_dir)?;

        let csv_path = out.join("labels.csv");
        let need_header = !csv_path.exists();
        let mut csv = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)?;
        if need_header {
            writeln!(csv, "path,label")?;
        }

        let next_index = next_clip_index(&clip_dir)?;

        Ok(Self {
            clip_dir,
            csv,
            label,
            clip_samples: clip_samples.max(1),
            next_index,
            buf: Vec::with_capacity(clip_samples.max(1) * 2),
        })
    }

    /// The next clip index that will be used (zero-padded in the filename).
    pub fn next_index(&self) -> usize {
        self.next_index
    }

    /// Append mono samples; flush as many whole clips as are now buffered. Returns
    /// the number of clips written by this call.
    pub fn push(&mut self, samples: &[f32]) -> Result<usize, Box<dyn Error>> {
        self.buf.extend_from_slice(samples);
        let mut written = 0;
        while self.buf.len() >= self.clip_samples {
            // Take exactly one clip's worth off the front, keep the remainder.
            let clip: Vec<f32> = self.buf.drain(..self.clip_samples).collect();
            self.write_clip(&clip)?;
            written += 1;
        }
        Ok(written)
    }

    /// Write one clip as a mono 16 kHz WAV and append its `labels.csv` row. The
    /// stored CSV path is relative to the dataset root (`<class_dir>/NNNNN.wav`),
    /// using forward slashes so the manifest is portable across platforms.
    fn write_clip(&mut self, clip: &[f32]) -> Result<(), Box<dyn Error>> {
        let filename = format!("{:05}.wav", self.next_index);
        let wav_path = self.clip_dir.join(&filename);
        write_wav_16k_mono(&wav_path, clip)?;

        let class_name = self
            .clip_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        writeln!(self.csv, "{class_name}/{filename},{}", self.label)?;
        self.csv.flush()?;

        self.next_index += 1;
        Ok(())
    }
}

/// Highest existing `NNNNN.wav` index in `dir`, plus one (0 if the dir is empty).
/// Lets repeat sessions append rather than overwrite.
fn next_clip_index(dir: &Path) -> Result<usize, Box<dyn Error>> {
    let mut max_seen: Option<usize> = None;
    if dir.exists() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|x| x == "wav").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(n) = stem.parse::<usize>() {
                        max_seen = Some(max_seen.map_or(n, |m| m.max(n)));
                    }
                }
            }
        }
    }
    Ok(max_seen.map_or(0, |m| m + 1))
}

/// Write mono `f32` samples (in `[-1, 1]`) as a 16-bit PCM, 16 kHz, mono WAV.
///
/// 16-bit PCM matches how DADS/field clips are stored and decoded by
/// `drone_bench::dataset::read_mono_wav`, so the round-trip is lossless enough
/// for the harmonic detectors and keeps files small. Samples are clamped before
/// scaling to avoid wrap-around on overshoot.
fn write_wav_16k_mono(path: &Path, samples: &[f32]) -> Result<(), Box<dyn Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Convert a cpal sample type into a normalized `f32` in roughly `[-1, 1]`.
/// Mirrors the conversion in [`crate::listen`] so capture is identical.
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
    use drone_bench::dataset::read_mono_wav;

    /// A unique temp dir under the OS temp location, created fresh per test.
    fn temp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("drone-live-record-{tag}-{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// A ramp signal that is easy to verify after a round-trip through 16-bit WAV.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect()
    }

    #[test]
    fn segments_into_fixed_length_clips() {
        let dir = temp_dir("segment");
        let clip_samples = 100;
        let mut w = ClipWriter::new(&dir, "drone", 1, clip_samples).unwrap();

        // Push 250 samples in two uneven chunks: expect floor(250/100) = 2 clips,
        // with 50 samples carried over (not yet written).
        assert_eq!(w.push(&ramp(130)).unwrap(), 1);
        assert_eq!(w.push(&ramp(120)).unwrap(), 1);
        assert_eq!(w.next_index(), 2);

        // Files are zero-padded sequential under the class dir.
        assert!(dir.join("drone/00000.wav").exists());
        assert!(dir.join("drone/00001.wav").exists());
        assert!(!dir.join("drone/00002.wav").exists());

        // Each written clip has exactly clip_samples mono 16 kHz samples.
        let (audio, sr) = read_mono_wav(&dir.join("drone/00000.wav")).unwrap();
        assert_eq!(sr, TARGET_RATE);
        assert_eq!(audio.len(), clip_samples);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn does_not_write_partial_trailing_clip() {
        let dir = temp_dir("partial");
        let mut w = ClipWriter::new(&dir, "nondrone", 0, 100).unwrap();
        // 99 samples: not enough for a clip, nothing written.
        assert_eq!(w.push(&ramp(99)).unwrap(), 0);
        assert_eq!(w.next_index(), 0);
        assert!(!dir.join("nondrone/00000.wav").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn labels_csv_has_header_and_rows() {
        let dir = temp_dir("csv");
        let mut w = ClipWriter::new(&dir, "drone", 1, 50).unwrap();
        w.push(&ramp(150)).unwrap(); // 3 clips
        drop(w);

        let csv = fs::read_to_string(dir.join("labels.csv")).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "path,label");
        assert_eq!(lines[1], "drone/00000.wav,1");
        assert_eq!(lines[2], "drone/00001.wav,1");
        assert_eq!(lines[3], "drone/00002.wav,1");
        assert_eq!(lines.len(), 4);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resumes_index_and_appends_csv_across_sessions() {
        let dir = temp_dir("resume");

        // Session 1: two drone clips.
        {
            let mut w = ClipWriter::new(&dir, "drone", 1, 50).unwrap();
            w.push(&ramp(100)).unwrap();
        }
        // Session 2: a different class; index for THIS dir starts fresh at 0, but
        // the shared labels.csv keeps its header and gains new rows.
        {
            let mut w = ClipWriter::new(&dir, "nondrone", 0, 50).unwrap();
            w.push(&ramp(50)).unwrap();
        }
        // Session 3: back to drone; must resume at index 2 (past 00000/00001).
        {
            let mut w = ClipWriter::new(&dir, "drone", 1, 50).unwrap();
            assert_eq!(w.next_index(), 2);
            w.push(&ramp(50)).unwrap();
            assert!(dir.join("drone/00002.wav").exists());
        }

        let csv = fs::read_to_string(dir.join("labels.csv")).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        // One header + 2 (session1) + 1 (session2) + 1 (session3) = 5 lines.
        assert_eq!(lines[0], "path,label");
        assert_eq!(lines.len(), 5);
        assert!(lines.contains(&"nondrone/00000.wav,0"));
        assert!(lines.contains(&"drone/00002.wav,1"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wav_roundtrip_preserves_signal() {
        let dir = temp_dir("roundtrip");
        let mut w = ClipWriter::new(&dir, "drone", 1, 64).unwrap();
        let signal = ramp(64);
        w.push(&signal).unwrap();
        drop(w);

        let (audio, _) = read_mono_wav(&dir.join("drone/00000.wav")).unwrap();
        assert_eq!(audio.len(), signal.len());
        // 16-bit quantization error is at most ~1/32767.
        for (a, b) in audio.iter().zip(signal.iter()) {
            assert!((a - b).abs() < 1e-3, "sample drift {a} vs {b}");
        }
        fs::remove_dir_all(&dir).ok();
    }
}
