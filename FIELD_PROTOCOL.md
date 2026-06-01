# Field capture protocol - a truly held-out drone-detection test set

Every benchmark number in this repo so far is at risk of recording-level
leakage. DADS clips cut from the same recording can land in both the train and
test split, so a detector can "recognize the recording" instead of "recognizing
a drone". `xeval` mitigates this by testing cross-dataset (Al-Emadi + ESC-50),
but those are still public corpora someone could have tuned against.

This protocol closes the last gap. You record your OWN drone with your OWN mics,
in your OWN environment, and evaluate on it. Because the resulting field set
**shares no recordings with DADS**, its ROC-AUC and calibrated-F1 are the honest,
leakage-free answer to the only question that matters: *does this actually detect
my drone, in my world?* That is the sole basis on which "beats upstream" can be
claimed truthfully.

> The leakage point, stated plainly: the field set is held out **because it
> shares NO recordings, microphones, drones, or locations with DADS**. Train on
> DADS, test on field. Never the reverse, and never mix them.

---

## What you need

- The one real drone you want to detect (your unit), with charged batteries.
- Two recorders for mic diversity: a **phone** and a **laptop**. Using two
  different mics keeps the model from learning one mic's coloration.
- A way to measure rough distance (paces are fine: ~1 m per stride).
- Somewhere safe and legal to fly. Follow local drone rules.
- A build of `drone-live` (`cargo build --manifest-path crates/drone-live/Cargo.toml`).

Identify your input device first so you can pass it explicitly:

```
drone-live devices
```

---

## The capture matrix

Record the **positive** class (drone) across a spread of conditions so the test
set is representative, not a single easy clip. Vary three axes:

### Distances (the most important axis)

Fly/hover at roughly these ranges and record at each:

| Distance | Why |
| -------- | --- |
| 5 m   | loud, easy positive - the floor for "obviously detectable" |
| 15 m  | typical close approach |
| 30 m  | the realistic detection range you care about |
| 60 m  | getting hard; harmonics thinning into the noise |
| 100 m | the honest upper limit - expect misses, that is the point |

### Azimuths

At a couple of the distances (say 15 m and 30 m), record from a few directions
relative to the mic: in front, off to one side, and behind. Rotor directivity
and obstacles change the spectrum.

### Backgrounds

Repeat the distance sweep under different ambient conditions if you can:

- **quiet** (calm, low ambient) - the clean baseline.
- **wind** - wind noise is a classic false-positive driver.
- **traffic** / urban hum - broadband + low-frequency confusers.

### Hard negatives (record these too - they are what make the test fair)

A test set that is "drone vs silence" is meaningless. Record the **non-drone**
class from sounds that are acoustically *confusable* with a drone (harmonic,
mechanical, broadband) so the false-positive rate is honest:

- **car** idling / passing (engine harmonics).
- **lawnmower / power tool** (drill, blower, chainsaw) - rotor-like and the
  single most important hard negative.
- **aircraft / helicopter** if any pass overhead (real rotor/engine harmonics).
- **wind** gusting into the mic.
- **music** with bass and sustained tones.
- general ambient: voices, footsteps, birds, traffic.

---

## How many clips

Clips are 1 second by default. Aim for a set that is big enough for the
ROC-AUC / F1 to mean something and roughly class-balanced:

- **Per positive condition** (each distance x background, a few azimuths):
  about **60-120 s** of recording -> 60-120 clips. Hit at least the 5/15/30 m
  rows; 60/100 m if you can.
- **Hard negatives:** about **60-120 s per negative type**. Cover at least
  car, a power tool, and wind.
- **Totals to aim for:** on the order of **300+ drone** and **300+ nondrone**
  clips overall. More is better; balance matters more than raw size.
- Record some positives **and** negatives on **both** the phone and the laptop
  so neither mic is purely one class.

Keep each continuous recording to one condition (don't let a car drive through
your "quiet drone at 30 m" take) - the label is per clip, and 1 s clips inherit
the condition you were recording.

---

## Recording the clips

`drone-live record` opens the mic, downmixes + resamples to mono 16 kHz (the
exact path the live detector uses), slices the stream into fixed-length clips,
writes them to `<out>/<label>/NNNNN.wav`, and appends rows to `<out>/labels.csv`
(header `path,label`; `1` = drone, `0` = nondrone). Filenames are zero-padded
and sequential, and **repeat sessions resume the numbering** instead of
overwriting - so you can record many short takes into the same dataset.

Drone present (hover/fly during the whole take):

```
drone-live record --label drone --seconds 120
```

Non-drone / hard negative (drone OFF and away; make the confuser sound happen):

```
drone-live record --label nondrone --seconds 120
```

Useful flags:

- `--device "<name substring>"` - pick a specific mic (match against
  `drone-live devices`). Use this to record on the phone vs. laptop deliberately.
- `--seconds <total>` - how long to record this take.
- `--clip-len <s>` - clip length (default `1.0`).
- `--out <dir>` - dataset root (default `data/field`).
- `--class <name[:id]>` - for multiclass instead of `--label`, e.g.
  `--class quadcopter:1` / `--class fixedwing:2`. The name is the subdirectory;
  the integer is written to `labels.csv`.

Recommended workflow per condition: announce the condition to yourself, start
the drone (or the negative sound), run the command, let it capture the take,
stop. Move to the next distance/azimuth/background and repeat. Everything lands
in `data/field/` with a single growing `labels.csv`.

Example session:

```
drone-live record --label drone    --seconds 90  --device "Headset"   # 5 m, quiet
drone-live record --label drone    --seconds 90  --device "Headset"   # 30 m, quiet
drone-live record --label drone    --seconds 60  --device "Built-in"  # 30 m, phone-ish mic
drone-live record --label nondrone --seconds 90  --device "Headset"   # car idling
drone-live record --label nondrone --seconds 90  --device "Headset"   # leaf blower
drone-live record --label nondrone --seconds 60  --device "Headset"   # wind
```

---

## Evaluating

Once `data/field` has both classes, run the held-out evaluation. It FITs every
approach on **all of DADS** and TESTs on your field set, reporting per-approach
ROC-AUC, PR-AUC, fixed-threshold F1, and **calibrated (best-threshold) F1**:

```
cargo run --manifest-path crates/drone-bench/Cargo.toml --bin fieldeval -- --field data/field
```

It writes `benchmarks/results/fieldeval.json` and prints a table. If `data/field`
is empty it tells you to record some first. These are the numbers to quote: they
cannot be inflated by leakage, because no field recording was ever in training.

Flags: `--field <dir>` (default `data/field`), `--dads <dir>` (the train
source), `--threshold <t>` (fixed operating point, default `0.5`),
`--out <path>` (JSON output).

---

## Honesty checklist

- [ ] Field recordings were made with mics/drone/locations **not** in DADS.
- [ ] Both classes present, roughly balanced.
- [ ] Hard negatives included (car, power tool, wind at minimum).
- [ ] A spread of distances, including ones you expect to be hard (60/100 m).
- [ ] At least two different mics used across the set.
- [ ] You trained on DADS and tested on field - never the reverse, never mixed.
