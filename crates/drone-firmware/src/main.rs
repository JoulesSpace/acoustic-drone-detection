//! esp32-C6 edge drone-detector firmware (`#![no_std] #![no_main]`).
//!
//! This is the project's *deployment proof*: a real bare-metal RISC-V firmware
//! for the esp32-C6 (target `riscv32imac-unknown-none-elf`) built on the esp-rs
//! [`esp_hal`] 1.0 stack. It runs the **real** edge detector
//! ([`drone_edge::EdgeDetector`], a no_std/alloc-free spectral-rule scorer) over
//! [`drone_dsp::FRAME_SIZE`]-sample audio frames and, when a drone is detected,
//! latches an alert: it lights an LED GPIO and prints a log line over the
//! USB-Serial-JTAG console.
//!
//! ## Signal flow
//!
//! ```text
//!   mic frame (FRAME_SIZE f32)  ->  EdgeDetector::push_frame  ->  bool alert
//!         (TODO: real I2S)            (window+FFT+features+rule, EMA+hold)
//!                                                |
//!                                                v
//!                                   LED GPIO on/off + esp-println log
//! ```
//!
//! ## Wiring (esp32-C6-DevKitC-1 / -M-1 defaults)
//!
//! * Alert LED -> `GPIO8` (drive an external LED + resistor, or the on-board
//!   user LED on boards that wire one there). Active-high in this firmware.
//! * Serial   -> the on-board USB-Serial-JTAG; `esp-println` logs land there.
//! * I2S MEMS mic (e.g. INMP441 / ICS-43434) -> **not yet wired**; see the
//!   `read_mic_frame` TODO below. That driver is the single remaining stub; the
//!   detector path it feeds is fully real.

#![no_std]
#![no_main]

use drone_dsp::{Frame, FRAME_SIZE};
use drone_edge::EdgeDetector;
use esp_backtrace as _;
use esp_hal::{
    delay::Delay,
    gpio::{Level, Output, OutputConfig},
    main,
};

// Emit the esp-idf-style application descriptor the 2nd-stage bootloader needs
// to recognise and launch this image.
esp_bootloader_esp_idf::esp_app_desc!();

/// Sample rate the (eventual) I2S front-end clocks the mic at, in Hz. The
/// detector uses this only to map FFT bins to the 100-4000 Hz drone band, so it
/// must match whatever `read_mic_frame` actually produces.
const SAMPLE_RATE: u32 = 16_000;

/// Two pi, for the synthetic-frame generator (no `core::f32::consts` import
/// churn at the call site).
const TWO_PI: f32 = 2.0 * core::f32::consts::PI;

#[main]
fn main() -> ! {
    // Bring up the serial logger first so we see everything, including a panic
    // backtrace, from the very first line.
    esp_println::logger::init_logger_from_env();
    let peripherals = esp_hal::init(esp_hal::Config::default());

    // Alert LED on GPIO8, starting off (active-high).
    let mut led = Output::new(peripherals.GPIO8, Level::Low, OutputConfig::default());
    let delay = Delay::new();

    // The REAL detector: EMA-smoothed spectral-rule scorer with a hold counter.
    // Firmware defaults: alpha=0.4, threshold=0.5, hold=3 frames.
    let mut detector = EdgeDetector::with_defaults(SAMPLE_RATE);

    esp_println::println!(
        "drone-firmware up: esp32-C6, frame={} samples @ {} Hz, detector=drone-edge",
        FRAME_SIZE,
        SAMPLE_RATE
    );

    // Reusable frame buffer - exactly how firmware would hand a DMA block to the
    // detector. No heap: one fixed [f32; FRAME_SIZE] on the stack.
    let mut frame: Frame = [0.0; FRAME_SIZE];

    // Demo schedule: alternate a few seconds of "drone-like" harmonic frames with
    // a few seconds of broadband noise so the alert visibly latches and clears on
    // real hardware. Replace `read_mic_frame` with a live I2S read to make this a
    // continuous real-world listener.
    let mut tick: u32 = 0;
    let mut was_alerting = false;

    loop {
        // ~one second of audio is ~16 frames at 16 kHz / 1024 samples; we don't
        // sleep a full frame-time here because there is no real-time mic clock yet.
        let drone_phase = (tick / 16).is_multiple_of(2);
        read_mic_frame(&mut frame, tick, drone_phase);

        // === REAL detector call ===
        let alerting = detector.push_frame(&frame);
        let confidence = detector.confidence();

        // Drive the LED off the latched alert state every frame.
        led.set_level(if alerting { Level::High } else { Level::Low });

        // Log on edges (and periodically) rather than every frame, to keep the
        // serial console readable.
        if alerting && !was_alerting {
            esp_println::println!("ALERT: drone detected (confidence {:.2})", confidence);
        } else if !alerting && was_alerting {
            esp_println::println!("clear: alert reset (confidence {:.2})", confidence);
        } else if tick.is_multiple_of(16) {
            esp_println::println!(
                "frame {}: confidence {:.2}, alerting={}",
                tick,
                confidence,
                alerting
            );
        }
        was_alerting = alerting;

        tick = tick.wrapping_add(1);
        // Small pacing delay so the serial log is watchable; not a real audio clock.
        delay.delay_millis(60);
    }
}

/// Fill `frame` with one block of audio samples.
///
/// TODO(mic): This is the **only** stub in the firmware. Replace the synthetic
/// generator below with a real I2S read from a MEMS microphone (e.g. INMP441 /
/// ICS-43434 on `esp_hal::i2s::master`): configure I2S in std/Philips mode at
/// [`SAMPLE_RATE`], DMA `FRAME_SIZE` samples into a buffer, and convert the
/// 24-bit left-justified words to `f32` in roughly `[-1.0, 1.0]`. Until then we
/// synthesize frames so the detector + alert path can be exercised on real
/// silicon: a harmonic multirotor-like signature when `drone_like`, broadband
/// pseudo-noise otherwise. The detector that consumes this frame is fully real.
fn read_mic_frame(frame: &mut Frame, seed: u32, drone_like: bool) {
    if drone_like {
        // Fundamental + a few harmonics inside the 100-4000 Hz drone band, like a
        // multirotor's blade-pass / motor whine. Scores high on the spectral rule.
        let fundamental_hz = 150.0;
        for (i, s) in frame.iter_mut().enumerate() {
            let t = i as f32 / SAMPLE_RATE as f32;
            let mut v = 0.0_f32;
            for h in 1..=5u32 {
                let amp = 1.0 / h as f32;
                v += amp * libm::sinf(TWO_PI * fundamental_hz * h as f32 * t);
            }
            *s = 0.3 * v;
        }
    } else {
        // Deterministic LCG broadband noise - flat spectrum, scores low.
        let mut state = seed | 1;
        for s in frame.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
            *s = 2.0 * u - 1.0;
        }
    }
}
