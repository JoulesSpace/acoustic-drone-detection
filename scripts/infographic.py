#!/usr/bin/env python3
"""Generate the acoustic-drone-detection signal-chain infographic.

A single data-dense, scientifically-correct figure tracing the physics from a
spinning rotor to an edge alert: acoustic radiation → pressure wave → sampling
(Nyquist) → ADC quantization → STFT spectrum (blade-pass comb) → detection
pipeline → hardware tiers. Numbers are this project's real measurements.

Pure matplotlib + numpy (both in the `plot` container). Deterministic. Output:
`assets/signal_chain.png`. Regenerate with:
    docker compose run --rm --entrypoint python plot scripts/infographic.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ---- shared constants (this project's real parameters) --------------------
FS = 16_000          # Hz, working sample rate
FRAME = 1024         # samples / analysis frame
F0 = 120.0           # Hz, illustrative blade-pass fundamental
C_SOUND = 343.0      # m/s @ 20 °C
INK = "#12263a"
ACCENT = "#1f77b4"
WARN = "#c1121f"
GOOD = "#2a9d8f"
GRID = "#d8dee9"
plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def drone_signal(t, f0=F0, harmonics=6, am_hz=8.0):
    """Quasi-periodic multirotor pressure signal: harmonic stack + rotor AM."""
    am = 1.0 + 0.22 * np.sin(2 * np.pi * am_hz * t)
    s = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        s += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    return am * s / 1.7


# ===========================================================================
def panel_rotor(ax):
    ax.set_title("1 · acoustic source", loc="left")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # wavefronts (spherical radiation)
    for r in (1.2, 2.2, 3.2, 4.2):
        ax.add_patch(Circle((5, 5.4), r, fill=False, ec=ACCENT, lw=0.7, alpha=0.45))
    # quadrotor: body + 4 arms + rotors
    ax.plot([3.2, 6.8], [3.6, 7.2], color=INK, lw=2, zorder=3)
    ax.plot([3.2, 6.8], [7.2, 3.6], color=INK, lw=2, zorder=3)
    for (x, y) in [(3.2, 3.6), (6.8, 3.6), (3.2, 7.2), (6.8, 7.2)]:
        ax.add_patch(Circle((x, y), 0.95, fill=False, ec=INK, lw=1.6, zorder=4))
        ax.add_patch(Circle((x, y), 0.08, color=INK, zorder=5))
        # blade
        ax.plot([x - 0.8, x + 0.8], [y, y], color=WARN, lw=2.2, zorder=4)
    ax.add_patch(FancyBboxPatch((4.4, 4.6), 1.2, 1.2, boxstyle="round,pad=0.05",
                                fc="#e9eef5", ec=INK, lw=1.4, zorder=4))
    ax.text(5, 1.7,
            "rotor RPM × blades → blade-pass $f_0$\n"
            "$f_0 \\approx$ 100–250 Hz   (DADS drones ≈ 230 Hz)\n"
            "radiates at $c$ = 343 m/s,  $\\lambda=c/f$",
            ha="center", va="center", fontsize=8)


def panel_wave(ax):
    ax.set_title("2 · pressure wave  $p(t)$", loc="left")
    t = np.linspace(0, 0.04, 2000)
    ax.plot(t * 1000, drone_signal(t), color=ACCENT, lw=1.3)
    T = 1000.0 / F0
    ax.annotate("", xy=(T, 1.18), xytext=(0, 1.18),
                arrowprops=dict(arrowstyle="<->", color=WARN, lw=1.2))
    ax.text(T / 2, 1.34, f"$T=1/f_0$ = {T:.1f} ms", color=WARN, ha="center", fontsize=8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("pressure")
    ax.set_ylim(-1.5, 1.6)
    ax.set_yticks([])
    ax.grid(True, color=GRID, lw=0.5)
    ax.text(0.5, -1.32, "quasi-periodic: harmonic stack + slow rotor AM",
            fontsize=7.5, color=INK)


def panel_sampling(ax):
    ax.set_title("3 · sampling  ($f_s$ = 16 kHz, Nyquist 8 kHz)", loc="left")
    t = np.linspace(0, 0.01, 1500)
    ax.plot(t * 1000, drone_signal(t), color=ACCENT, lw=1.0, alpha=0.55)
    # sample dots at fs (decimated visually for clarity ~ every 1/4 ms)
    ts = np.arange(0, 0.01, 1.0 / 4000)
    ax.plot(ts * 1000, drone_signal(ts), "o", color=INK, ms=3.2)
    ax.vlines(ts * 1000, 0, drone_signal(ts), color=INK, lw=0.5, alpha=0.4)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude")
    ax.set_yticks([])
    ax.grid(True, color=GRID, lw=0.5)
    ax.text(0.0, -1.5,
            "$\\Delta t = 62.5\\,\\mu s$ · capture > $2f_{max}$ or it aliases",
            fontsize=7.5)
    ax.set_ylim(-1.7, 1.3)


def panel_adc(ax):
    ax.set_title("4 · ADC quantization", loc="left")
    t = np.linspace(0, 1, 500)
    sig = 0.9 * np.sin(2 * np.pi * 1.5 * t)
    ax.plot(t, sig, color=ACCENT, lw=1.0, alpha=0.6, label="analog")
    for bits, col, a in [(3, INK, 1.0)]:
        levels = 2 ** bits
        q = np.round((sig + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
        ax.step(t, q, where="mid", color=col, lw=1.3, alpha=a, label=f"{bits}-bit")
    # level grid
    for lv in np.linspace(-1, 1, 2 ** 3):
        ax.axhline(lv, color=GRID, lw=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("code")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(0.0, -1.62,
            "step $q=2^{-(N-1)}$ · SQNR ≈ 6.02$N$+1.76 dB\n"
            "ours: ROC-AUC holds to 4-bit (strong detectors)",
            fontsize=7.5)
    ax.set_ylim(-1.75, 1.2)


def panel_spectrum(ax):
    ax.set_title("5 · STFT magnitude — blade-pass comb", loc="left")
    n = FRAME
    t = np.arange(n) / FS
    x = drone_signal(t) * np.hanning(n)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    ax.plot(freqs, spec / spec.max(), color=ACCENT, lw=1.0)
    for h in range(1, 8):
        fx = F0 * h
        if fx < freqs[-1]:
            ax.axvline(fx, color=WARN, lw=0.7, ls=":", alpha=0.7)
    ax.axvspan(100, 4000, color=GOOD, alpha=0.10)
    ax.set_xlim(0, 4200)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|X| (norm)")
    ax.set_yticks([])
    ax.grid(True, color=GRID, lw=0.5)
    ax.text(0.0, 1.04,
            "frame 1024 @16 kHz = 64 ms, 15.6 Hz/bin · Hann · band 100–4000 Hz",
            transform=ax.transAxes, fontsize=7.3)


def _box(ax, x, y, w, h, text, fc, ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)


def panel_hardware(ax):
    ax.set_title("6 · hardware tiers (real numbers)", loc="left")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    _box(ax, 0.3, 6.6, 9.4, 2.6,
         "TINY-EDGE · esp32-C3/C6 (riscv32imc)\n"
         "drone-edge no_std · ~17–27 KB flash · ~0 static RAM\n"
         "~26 µs/frame · band-ratio / HPS / spectral-gate", "#eaf4ef", GOOD)
    _box(ax, 0.3, 3.7, 9.4, 2.6,
         "BALANCED · phone / Raspberry Pi\n"
         "MFCC+logistic, GTCC, cepstrum, envelope\n"
         "90–2400× real-time", "#eaf0f7", ACCENT)
    _box(ax, 0.3, 0.8, 9.4, 2.6,
         "MAX-ACCURACY · server\n"
         "feature-fusion / ensemble\n"
         "best in-dist; honest cross-dataset ROC-AUC ≤ 0.87", "#f6ecec", WARN)


def pipeline_strip(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.5, 9.0, "detection pipeline", fontsize=9.5, fontweight="bold")
    stages = [
        ("mic\n16 kHz", "#e9eef5"),
        ("frame 1024\n+ Hann", "#e9eef5"),
        ("real FFT\n(microfft)", "#e9eef5"),
        ("features\nMFCC·HPS·spectral", "#eaf0f7"),
        ("classifier\nlogistic / ensemble", "#eaf0f7"),
        ("EMA + hold\nthreshold", "#eaf4ef"),
        ("⚠ ALERT", "#f6ecec"),
    ]
    x = 1.0
    w = 12.6
    gap = 1.6
    for i, (txt, fc) in enumerate(stages):
        _box(ax, x, 2.2, w, 4.2, txt, fc)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 4.3), (x + w + gap, 4.3),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=INK, lw=1.3))
        x += w + gap
    ax.text(0.5, 0.4,
            "frame-synchronous, no_std-portable · ~0.5 s window + hold latency · "
            "deterministic, GPU-free",
            fontsize=7.6)


def main() -> int:
    os.makedirs("assets", exist_ok=True)
    fig = plt.figure(figsize=(13.5, 11.0), dpi=130)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.62], hspace=0.5, wspace=0.22,
                          left=0.045, right=0.985, top=0.88, bottom=0.055)
    fig.suptitle("Acoustic Drone Detection — Signal Chain & Physics",
                 fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.045, 0.93,
             "rotor → pressure wave → sampling → ADC → STFT → detection → edge alert."
             "  Parameters are this project's real measurements.",
             fontsize=9)

    panel_rotor(fig.add_subplot(gs[0, 0]))
    panel_wave(fig.add_subplot(gs[0, 1]))
    panel_sampling(fig.add_subplot(gs[0, 2]))
    panel_adc(fig.add_subplot(gs[1, 0]))
    panel_spectrum(fig.add_subplot(gs[1, 1]))
    panel_hardware(fig.add_subplot(gs[1, 2]))
    pipeline_strip(fig.add_subplot(gs[2, :]))

    out = "assets/signal_chain.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
