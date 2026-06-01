#!/usr/bin/env python3
"""Plot the speed/accuracy Pareto trade-off from ``benchmarks/results/pareto.json``.

The Rust ``pareto`` binary (``crates/drone-bench/src/bin/pareto.rs``) emits one
row per approach with its measured latency, ROC-AUC, hardware tier, and a
``pareto_frontier`` flag. This script renders the classic trade-off view:

  * x — inference latency (microseconds per 1024-sample frame), **log scale**
        (lower / left is faster);
  * y — ROC-AUC (higher / up is more accurate);
  * points coloured and legended by hardware tier, each annotated with its name;
  * the **Pareto frontier** drawn as a connecting line through the non-dominated
    points (no other approach is both faster AND more accurate).

Saves ``benchmarks/plots/pareto.png``. Pure matplotlib + stdlib (no numpy). Run
via the ``plot`` compose service so matplotlib is containerized (docker-first),
or directly if matplotlib is installed on the host.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PARETO_JSON = RESULTS / "pareto.json"

# Stable colour per tier (cheapest -> most capable).
TIER_COLOR = {
    "tiny-edge": "#2ca02c",     # green: runs anywhere
    "balanced": "#1f77b4",      # blue: phone / Pi class
    "max-accuracy": "#d62728",  # red: server / workstation
}
TIER_ORDER = ["tiny-edge", "balanced", "max-accuracy"]


def _num(x, default=float("nan")):
    return default if x is None else float(x)


def load_rows() -> list[dict]:
    if not PARETO_JSON.exists():
        print(
            f"no {PARETO_JSON} — run the `pareto` binary first "
            "(cargo run --release --bin pareto)",
            file=sys.stderr,
        )
        return []
    data = json.loads(PARETO_JSON.read_text())
    return [r for r in data if isinstance(r, dict) and "approach" in r]


def plot_pareto(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Clamp latency to a small positive value so the log axis is well-defined.
    def lat(r):
        return max(_num(r.get("latency_us_per_frame"), 0.0), 1e-3)

    # Scatter, grouped by tier so the legend lists each tier once.
    for tier in TIER_ORDER:
        pts = [r for r in rows if r.get("tier") == tier]
        if not pts:
            continue
        ax.scatter(
            [lat(r) for r in pts],
            [_num(r.get("roc_auc"), 0.0) for r in pts],
            s=90,
            color=TIER_COLOR.get(tier, "#7f7f7f"),
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label=tier,
        )

    # Annotate every point with its approach name.
    for r in rows:
        ax.annotate(
            r["approach"],
            (lat(r), _num(r.get("roc_auc"), 0.0)),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
        )

    # Pareto frontier: the non-dominated points, drawn left-to-right (ascending
    # latency). With latency increasing, a true frontier has non-decreasing
    # ROC-AUC, so the connecting line steps up as you pay more.
    frontier = sorted(
        (r for r in rows if r.get("pareto_frontier")),
        key=lat,
    )
    if len(frontier) >= 2:
        ax.plot(
            [lat(r) for r in frontier],
            [_num(r.get("roc_auc"), 0.0) for r in frontier],
            color="black",
            ls="--",
            lw=1.4,
            zorder=2,
            label="Pareto frontier",
        )
    elif len(frontier) == 1:
        # Single non-dominated point: mark it so the frontier is still visible.
        r = frontier[0]
        ax.scatter(
            [lat(r)],
            [_num(r.get("roc_auc"), 0.0)],
            s=220,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            zorder=2,
            label="Pareto frontier",
        )

    ax.set_xscale("log")
    ax.set_xlabel("inference latency — microseconds per 1024-sample frame (log scale)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Speed vs accuracy by hardware tier (upper-left is better)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "pareto.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    rows = load_rows()
    if not rows:
        return 1
    print(
        f"loaded {len(rows)} approach row(s): "
        f"{', '.join(r['approach'] for r in rows)}"
    )
    plot_pareto(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
