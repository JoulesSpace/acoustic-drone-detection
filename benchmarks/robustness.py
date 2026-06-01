#!/usr/bin/env python3
"""Plot detector robustness vs additive-noise SNR.

Reads `benchmarks/results/snr_<level>/<approach>.json` directories (produced by
running `drone-bench --snr <level> --out-dir ...` at several SNRs, plus a
`snr_clean` run with no added noise) and plots ROC-AUC and F1 as a function of
SNR for every approach — the degradation curves that show which detectors hold
up under stress.

Outputs `benchmarks/plots/robustness_roc.png` and `robustness_f1.png`.
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

# "clean" (no added noise) is drawn at this pseudo-SNR on the x-axis.
CLEAN_X = 40.0


def load() -> dict[str, list[tuple[float, dict]]]:
    """approach -> list of (snr, metrics) sorted by snr."""
    series: dict[str, list[tuple[float, dict]]] = {}
    for d in sorted(RESULTS.glob("snr_*")):
        if not d.is_dir():
            continue
        tag = d.name[len("snr_") :]
        snr = CLEAN_X if tag == "clean" else float(tag)
        for jf in d.glob("*.json"):
            try:
                data = json.loads(jf.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if "approach" not in data:
                continue
            series.setdefault(data["approach"], []).append((snr, data))
    for a in series:
        series[a].sort(key=lambda t: t[0])
    return series


def _num(x, default=float("nan")):
    return default if x is None else float(x)


def plot_metric(series, key, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Order legend by clean-condition performance (best first).
    def clean_val(items):
        for snr, m in items:
            if snr == CLEAN_X:
                return _num(m.get(key), 0.0)
        return _num(items[-1][1].get(key), 0.0)

    for approach in sorted(series, key=lambda a: clean_val(series[a]), reverse=True):
        items = series[approach]
        xs = [snr for snr, _ in items]
        ys = [_num(m.get(key), float("nan")) for _, m in items]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=approach)
    ax.set_xlabel("SNR (dB) — rightmost = clean")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    series = load()
    if not series:
        print("no snr_* result dirs found; run the SNR sweep first", file=sys.stderr)
        return 1
    PLOTS.mkdir(parents=True, exist_ok=True)
    n_levels = len(next(iter(series.values())))
    print(f"loaded {len(series)} approaches across {n_levels} SNR levels")
    plot_metric(series, "roc_auc", "ROC-AUC", "Robustness: ROC-AUC vs noise SNR",
                PLOTS / "robustness_roc.png")
    plot_metric(series, "f1_best", "F1 (calibrated)", "Robustness: F1 vs noise SNR",
                PLOTS / "robustness_f1.png")
    print(f"wrote robustness plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
