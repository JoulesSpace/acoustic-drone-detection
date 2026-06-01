#!/usr/bin/env python3
"""Plot detector accuracy vs sample rate and vs bit depth.

Reads `benchmarks/results/ratesweep.json` (produced by the `ratesweep` binary)
and draws two panels:

  * left  — ROC-AUC vs sample rate (one line per detector); the 16 kHz native
             rate is marked, since rates above it can only test pipeline
             behaviour, not extra signal.
  * right — ROC-AUC vs bit depth (one line per detector).

Outputs `benchmarks/plots/ratesweep.png`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def _by_detector(rows, x_key, y_key):
    """rows -> {detector: ([x...], [y...])} sorted by x."""
    series: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        series.setdefault(r["detector"], []).append((float(r[x_key]), float(r[y_key])))
    for d in series:
        series[d].sort(key=lambda t: t[0])
    return {d: ([x for x, _ in v], [y for _, y in v]) for d, v in series.items()}


def main() -> int:
    path = RESULTS / "ratesweep.json"
    if not path.exists():
        print(f"missing {path}; run the ratesweep binary first", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    native = data.get("native_sample_rate", 16000)

    rate_s = _by_detector(data["rate_sweep"], "rate_hz", "roc_auc")
    bit_s = _by_detector(data["bit_sweep"], "bits", "roc_auc")

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5.5))

    for det in sorted(rate_s):
        xs, ys = rate_s[det]
        axl.plot(xs, ys, marker="o", ms=4, lw=1.4, label=det)
    axl.axvline(native, color="k", ls="--", lw=1, alpha=0.5,
                label=f"native {native} Hz")
    axl.set_xlabel("sample rate (Hz) — above native = no new signal")
    axl.set_ylabel("ROC-AUC")
    axl.set_title("Accuracy vs sample rate")
    axl.grid(alpha=0.3)
    axl.legend(fontsize=7, loc="lower right")

    for det in sorted(bit_s):
        xs, ys = bit_s[det]
        axr.plot(xs, ys, marker="o", ms=4, lw=1.4, label=det)
    axr.set_xlabel("bit depth (bits)")
    axr.set_ylabel("ROC-AUC")
    axr.set_title("Accuracy vs bit depth")
    axr.grid(alpha=0.3)
    axr.invert_xaxis()  # fewer bits (cheaper ADC) to the right
    axr.legend(fontsize=7, loc="lower left")

    fig.suptitle("ratesweep: detection accuracy vs sample rate and bit depth (DADS)")
    fig.tight_layout()
    out = PLOTS / "ratesweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
