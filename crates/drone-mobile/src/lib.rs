//! Mobile FFI for the drone detector (Android / iOS).
//!
//! A thin, stable C ABI over the `no_std`-friendly [`drone_edge`] detector so it
//! can be loaded from a phone app (JNI on Android, a bridging header on iOS) and
//! ship to millions of commodity devices. The DSP and detection are the same
//! code that runs on the desktop and on the esp32 firmware; this crate only adds
//! the FFI boundary.
//!
//! ## Calling convention
//! Audio must be mono `f32` in roughly `[-1, 1]`. Call
//! [`drone_mobile_frame_size`] to learn the block size, then feed blocks of that
//! many samples. Stateless scoring uses [`drone_mobile_score`]; a smoothed,
//! latching detector uses [`drone_mobile_new`] / [`drone_mobile_push`] /
//! [`drone_mobile_confidence`] / [`drone_mobile_free`].
//!
//! `unsafe` is confined to the pointer-dereferencing FFI shims below; the safe
//! core (`drone-edge`/`drone-dsp`) is `#![forbid(unsafe_code)]`.

use drone_dsp::FRAME_SIZE;
use drone_edge::EdgeDetector;

/// Number of mono samples per analysis block. Feed blocks of this length.
#[no_mangle]
pub extern "C" fn drone_mobile_frame_size() -> usize {
    FRAME_SIZE
}

/// Copy a caller buffer into a fixed `FRAME_SIZE` frame (zero-padded / truncated).
///
/// # Safety
/// `ptr` must be valid for reads of `len` `f32`s, or null (treated as empty).
unsafe fn frame_from_raw(ptr: *const f32, len: usize) -> [f32; FRAME_SIZE] {
    let mut frame = [0.0_f32; FRAME_SIZE];
    if ptr.is_null() || len == 0 {
        return frame;
    }
    let n = len.min(FRAME_SIZE);
    // SAFETY: caller guarantees `ptr` is valid for `len` reads; we read `n <= len`.
    let src = core::slice::from_raw_parts(ptr, n);
    frame[..n].copy_from_slice(src);
    frame
}

/// Stateless confidence in `[0, 1]` that a drone is present in one block.
/// Returns a negative value on a null pointer.
///
/// # Safety
/// `ptr` must be valid for reads of `len` `f32`s, or null.
#[no_mangle]
pub unsafe extern "C" fn drone_mobile_score(ptr: *const f32, len: usize, sample_rate: u32) -> f32 {
    if ptr.is_null() {
        return -1.0;
    }
    let frame = frame_from_raw(ptr, len);
    drone_edge::drone_confidence(&frame, sample_rate)
}

/// Allocate a smoothing/latching detector. Free it with [`drone_mobile_free`].
#[no_mangle]
pub extern "C" fn drone_mobile_new(sample_rate: u32) -> *mut EdgeDetector {
    Box::into_raw(Box::new(EdgeDetector::with_defaults(sample_rate)))
}

/// Feed one block to a detector. Returns `1` if it is alerting, `0` if not,
/// `-1` on a null detector/buffer.
///
/// # Safety
/// `det` must be a pointer from [`drone_mobile_new`] (not yet freed); `ptr` must
/// be valid for reads of `len` `f32`s, or null.
#[no_mangle]
pub unsafe extern "C" fn drone_mobile_push(
    det: *mut EdgeDetector,
    ptr: *const f32,
    len: usize,
) -> i32 {
    if det.is_null() || ptr.is_null() {
        return -1;
    }
    let frame = frame_from_raw(ptr, len);
    // SAFETY: caller guarantees `det` came from `drone_mobile_new` and is live.
    let detector = &mut *det;
    detector.push_frame(&frame) as i32
}

/// Current smoothed confidence in `[0, 1]` (negative on a null detector).
///
/// # Safety
/// `det` must be a live pointer from [`drone_mobile_new`].
#[no_mangle]
pub unsafe extern "C" fn drone_mobile_confidence(det: *const EdgeDetector) -> f32 {
    if det.is_null() {
        return -1.0;
    }
    // SAFETY: caller guarantees `det` is a live `drone_mobile_new` pointer.
    (*det).confidence()
}

/// Free a detector allocated by [`drone_mobile_new`]. Null is a no-op.
///
/// # Safety
/// `det` must be a pointer from [`drone_mobile_new`] that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn drone_mobile_free(det: *mut EdgeDetector) {
    if !det.is_null() {
        // SAFETY: reclaim the Box we leaked in `drone_mobile_new`.
        drop(Box::from_raw(det));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_size_is_exposed() {
        assert_eq!(drone_mobile_frame_size(), FRAME_SIZE);
    }

    #[test]
    fn score_handles_null_and_short_buffers() {
        // Null -> negative sentinel.
        let s = unsafe { drone_mobile_score(core::ptr::null(), 0, 16_000) };
        assert!(s < 0.0);
        // Short buffer -> padded, finite, in range.
        let buf = [0.1_f32; 100];
        let s = unsafe { drone_mobile_score(buf.as_ptr(), buf.len(), 16_000) };
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
    }

    #[test]
    fn detector_lifecycle() {
        let det = drone_mobile_new(16_000);
        assert!(!det.is_null());
        let frame = [0.0_f32; FRAME_SIZE];
        let alert = unsafe { drone_mobile_push(det, frame.as_ptr(), frame.len()) };
        assert!(alert == 0 || alert == 1);
        let c = unsafe { drone_mobile_confidence(det) };
        assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        unsafe { drone_mobile_free(det) };
    }
}
