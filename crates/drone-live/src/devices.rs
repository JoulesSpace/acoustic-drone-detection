//! The hardware/mic probe: enumerate audio **input** devices and capabilities.
//!
//! This answers "what microphones / capture hardware does this machine have, and
//! what can each one do?" — sample-rate ranges, channel counts, and sample
//! formats (i16 / f32 / ...). It runs headless and prints real devices from the
//! platform's default `cpal` host (WASAPI on Windows, CoreAudio on macOS,
//! ALSA/JACK on Linux, AAudio on Android).

use std::error::Error;

use cpal::traits::{DeviceTrait, HostTrait};

/// Enumerate input devices on the default host and print a capability table.
///
/// The default input device (if any) is marked with `*`. For each device we list
/// every *supported input config range* the backend reports: channel count,
/// sample-rate range (min..max Hz), and sample format. Devices that error on
/// enumeration are still listed, with the error noted, rather than aborting the
/// whole probe — one flaky device should not hide the others.
pub fn run() -> Result<(), Box<dyn Error>> {
    let host = cpal::default_host();
    println!("audio host: {}", host.id().name());

    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();
    if default_name.is_empty() {
        println!("default input device: (none)");
    } else {
        println!("default input device: {default_name}");
    }

    let devices: Vec<_> = match host.input_devices() {
        Ok(it) => it.collect(),
        Err(e) => {
            return Err(format!("could not enumerate input devices: {e}").into());
        }
    };

    if devices.is_empty() {
        println!("\nno audio input devices found.");
        return Ok(());
    }

    println!("\n{} input device(s):", devices.len());
    for (i, device) in devices.iter().enumerate() {
        let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        let marker = if !default_name.is_empty() && name == default_name {
            " *"
        } else {
            ""
        };
        println!("\n[{i}] {name}{marker}");

        // Default config first (the rate/format the backend prefers).
        match device.default_input_config() {
            Ok(cfg) => println!(
                "    default: {} ch, {} Hz, {:?}",
                cfg.channels(),
                cfg.sample_rate().0,
                cfg.sample_format()
            ),
            Err(e) => println!("    default: (unavailable: {e})"),
        }

        // All supported input config ranges.
        match device.supported_input_configs() {
            Ok(ranges) => {
                let ranges: Vec<_> = ranges.collect();
                if ranges.is_empty() {
                    println!("    supported: (none reported)");
                } else {
                    println!(
                        "    {:<8} {:>20} {:>10}",
                        "channels", "sample-rate (Hz)", "format"
                    );
                    for r in ranges {
                        let lo = r.min_sample_rate().0;
                        let hi = r.max_sample_rate().0;
                        let rate = if lo == hi {
                            format!("{lo}")
                        } else {
                            format!("{lo}..{hi}")
                        };
                        println!(
                            "    {:<8} {:>20} {:>10}",
                            r.channels(),
                            rate,
                            format!("{:?}", r.sample_format())
                        );
                    }
                }
            }
            Err(e) => println!("    supported: (unavailable: {e})"),
        }
    }

    println!("\n(* marks the default input device)");
    Ok(())
}
