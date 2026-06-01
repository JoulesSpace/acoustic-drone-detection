#!/usr/bin/env python3
"""Download a balanced SUBSET of the DADS drone-audio dataset into ./data.

DADS (geronimobasso/drone-audio-detection-samples) is 6.81 GB and very class-
imbalanced (≈164k drone / ≈17k no-drone). We stream it (no full download) and
write a capped, balanced subset of 16 kHz mono WAVs plus a `labels.csv` manifest
that the Rust `drone-bench` harness reads directly.

Run via the `data` compose service (docker-first):
    docker compose run --rm data --per-class 300

Output layout:
    data/dads/drone/00000.wav ...
    data/dads/nondrone/00000.wav ...
    data/dads/labels.csv         (header: path,label   label 1=drone 0=no-drone)
"""
from __future__ import annotations

import argparse
import csv
import io
import pathlib
import sys

import soundfile as sf
from datasets import Audio, load_dataset

DATASET = "geronimobasso/drone-audio-detection-samples"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/dads", help="output directory")
    ap.add_argument("--per-class", type=int, default=300, help="clips per class")
    ap.add_argument(
        "--max-scan",
        type=int,
        default=200_000,
        help="stop scanning the stream after this many examples (safety bound)",
    )
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    for sub in ("drone", "nondrone"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print(f"streaming {DATASET} (this does not download the full 6.81 GB)…", flush=True)
    # decode=False: keep the raw WAV bytes and decode them ourselves with
    # soundfile, so we don't need librosa (datasets' default audio decoder).
    ds = load_dataset(DATASET, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    counts = {0: 0, 1: 0}
    rows: list[tuple[str, int]] = []
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > args.max_scan:
            print(f"hit --max-scan={args.max_scan}; stopping", file=sys.stderr)
            break
        label = int(ex["label"])
        if counts[label] >= args.per_class:
            if counts[0] >= args.per_class and counts[1] >= args.per_class:
                break
            continue
        audio = ex["audio"]
        raw = audio.get("bytes")
        if raw is None:
            continue  # path-only entries can't be fetched offline; skip
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if arr.ndim > 1:  # downmix to mono
            arr = arr.mean(axis=1)
        sub = "drone" if label == 1 else "nondrone"
        rel = f"{sub}/{counts[label]:05d}.wav"
        sf.write(str(out / rel), arr, sr, subtype="PCM_16")
        rows.append((rel, label))
        counts[label] += 1
        if scanned % 500 == 0:
            print(f"  scanned {scanned}, kept drone={counts[1]} nondrone={counts[0]}", flush=True)

    manifest = out / "labels.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

    print(f"wrote {len(rows)} clips to {out} (drone={counts[1]}, nondrone={counts[0]})")
    print(f"manifest: {manifest}")
    if counts[0] < args.per_class or counts[1] < args.per_class:
        print("WARNING: did not reach --per-class for both classes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
