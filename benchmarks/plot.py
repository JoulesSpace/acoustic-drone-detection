#!/usr/bin/env python3
"""Generate comparison plots from drone-bench result JSON.

Reads every ``benchmarks/results/*.json`` (the per-approach output of the Rust
``drone-bench`` harness) and writes PNGs into ``benchmarks/plots/``:

  * metrics_bar.png   - grouped accuracy/precision/recall/F1 per approach
  * roc.png           - ROC curves overlaid, labelled with ROC-AUC
  * pr.png            - Precision-Recall curves overlaid, labelled with PR-AUC
  * cost_quality.png  - inference cost (ms/clip, log x) vs F1 (the
                        insightface-style "what do you pay for quality" view)

Pure matplotlib + stdlib; no numpy required. Run via the ``plot`` compose
service so matplotlib is containerized (docker-first).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def load_results() -> list[dict]:
    out = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and "approach" in data:
            out.append(data)
    return out


def _num(x, default=float("nan")):
    return default if x is None else float(x)


def plot_metrics_bar(results: list[dict]) -> None:
    results = sorted(results, key=lambda r: _num(r.get("f1"), -1), reverse=True)
    names = [r["approach"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    n_groups = len(names)
    n_bars = len(metrics)
    width = 0.8 / n_bars
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_groups), 5))
    for i, m in enumerate(metrics):
        xs = [g + (i - (n_bars - 1) / 2) * width for g in range(n_groups)]
        ys = [_num(r.get(m), 0.0) for r in results]
        ax.bar(xs, ys, width=width, label=m)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Detection metrics by approach (threshold = 0.5)")
    ax.legend(loc="lower right", ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "metrics_bar.png", dpi=130)
    plt.close(fig)


def _curve(ax, results, key, auc_key, xlabel, ylabel, title, diagonal):
    for r in sorted(results, key=lambda r: _num(r.get(auc_key), -1), reverse=True):
        pts = r.get(key) or []
        pts = sorted(((p["x"], p["y"]) for p in pts), key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        auc = _num(r.get(auc_key))
        label = r["approach"] + (f" ({auc:.3f})" if not math.isnan(auc) else "")
        ax.plot(xs, ys, marker=".", ms=3, lw=1.3, label=label)
    if diagonal:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)


def plot_roc(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    _curve(ax, results, "roc", "roc_auc", "False positive rate",
           "True positive rate", "ROC curves (AUC in legend)", diagonal=True)
    fig.tight_layout()
    fig.savefig(PLOTS / "roc.png", dpi=130)
    plt.close(fig)


def plot_pr(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    _curve(ax, results, "pr", "pr_auc", "Recall", "Precision",
           "Precision-Recall curves (AUC in legend)", diagonal=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "pr.png", dpi=130)
    plt.close(fig)


def plot_cost_quality(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in results:
        ms = max(_num(r.get("mean_infer_ms"), 0.0), 1e-3)  # clamp for log axis
        f1 = _num(r.get("f1"), 0.0)
        ax.scatter(ms, f1, s=60)
        ax.annotate(r["approach"], (ms, f1), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("inference cost - ms per clip (log scale)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cost vs quality (upper-left is better)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PLOTS / "cost_quality.png", dpi=130)
    plt.close(fig)


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    results = load_results()
    if not results:
        print(f"no result JSON found in {RESULTS} - run drone-bench first", file=sys.stderr)
        return 1
    print(f"loaded {len(results)} approach result(s): {', '.join(r['approach'] for r in results)}")
    plot_metrics_bar(results)
    plot_roc(results)
    plot_pr(results)
    plot_cost_quality(results)
    print(f"wrote plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
