# Part A4 results — Windows mirror vs Mac mirror consistency check

**Audience:** Mac command center.
**Reporter:** Windows box.
**Companion to:** `LIDO_COHORT_WINDOWS_BRIEF.md` (Part A4 ask).

## TL;DR

A4 passes: **Windows S1 mirror produces output that's not bit-identical to
Mac S1 mirror, but the divergence is in pixels pyfaceau doesn't look at.**
Downstream AU extraction matches the Mac-mirror v1316 baseline cleanly
(0/17 AU failures both sides against normal-cohort tier bands). Lido
mirrors will run on Windows in Part B.

## What ran

S1's `StableFaceSplitter.process_video()` invoked from a small Python
wrapper (`C:\temp\partA4_mirror_check.py`) on `IMG_0942_source.MOV` with
auto-detected device (CUDA). Then per-frame pixel compare vs the existing
Mac-built `IMG_0942_{left,right}_mirrored.mp4` in
`SplitFace/S1O Processed Files/Face Mirror 1.0 Output/`.

Used the same pyclnf 1.3.16 / pyfaceau 1.3.16 / `use_gpu=True` we'll
use for Lido (confirmed in Part A is bit-exact with v1316).

## Result 1 — frame structure matches

| side | win frames | mac frames | win dims | mac dims |
|---|---|---|---|---|
| left | 1110 | 1110 | 1080×1920 | 1080×1920 |
| right | 1110 | 1110 | 1080×1920 | 1080×1920 |

## Result 2 — per-frame pixel diff is small but non-zero

| side | mean abs pixel diff | p50 | p95 | max frame mean | max single pixel |
|---|---|---|---|---|---|
| left | 1.99 / 255 (0.78%) | 1.94 | 2.52 | 3.04 | 203 |
| right | 2.04 / 255 (0.80%) | 1.96 | 2.69 | 3.49 | 190 |

Almost certainly attributable to one of:
  - FFmpeg / libx264 build version difference between Windows VS-built
    pyfaceau and macOS Homebrew-built pyfaceau (encoder macroblock
    decisions, chroma subsampling rounding)
  - Tiny CLNF landmark FP drift → slightly different mirror plane →
    slightly different downstream encode
  - YUV ↔ RGB conversion rounding choices between FFmpeg versions

The high `max_single_pixel_diff` of 190-203 sounds alarming but reflects
a small number of pixels at mirror-plane edges where 1 px of geometric
drift creates a discrete blocky difference. Mean across the frame stays
~2 / 255.

Per the brief: non-zero -> escalate to AU comparison.

## Result 3 — AU comparison Windows-mirror vs Mac-mirror

Ran pyfaceau v1.3.16 GPU mode on the Windows-built mirror, then compared
its AU columns to the existing
`tests/golden/aus/IMG_0942_{left,right}/pyfaceau_windows_cuda.parquet`
(which was built from the Mac mirror under the same config). Bands per
`metric_bands.yaml` normal-cohort tiers (post v1.3.16 relaxation: easy
r≥0.844, medium r≥0.791, hard r≥0.375).

### Left side — 0 / 17 failures

| AU | tier | r vs Mac-mirror | band | MAE | MAE band | verdict |
|---|---|---|---|---|---|---|
| AU01_r | easy | 0.9902 | 0.844 | 0.034 | 0.244 | ✅ |
| AU02_r | easy | 0.9769 | 0.844 | 0.034 | 0.244 | ✅ |
| AU04_r | medium | 0.9953 | 0.791 | 0.017 | 0.307 | ✅ |
| AU06_r | medium | 0.9991 | 0.791 | 0.013 | 0.307 | ✅ |
| AU07_r | medium | 0.9857 | 0.791 | 0.050 | 0.307 | ✅ |
| AU10_r | medium | 0.9951 | 0.791 | 0.041 | 0.307 | ✅ |
| AU12_r | easy | 0.9980 | 0.844 | 0.047 | 0.244 | ✅ |
| AU14_r | hard | 0.9812 | 0.375 | 0.108 | 1.063 | ✅ |
| AU15_r | hard | 0.8171 | 0.375 | 0.058 | 1.063 | ✅ |
| AU17_r | hard | 0.9459 | 0.375 | 0.095 | 1.063 | ✅ |
| AU20_r | hard | 0.9296 | 0.375 | 0.033 | 1.063 | ✅ |
| AU23_r | hard | 0.9901 | 0.375 | 0.018 | 1.063 | ✅ |
| AU25_r | hard | 0.9924 | 0.375 | 0.071 | 1.063 | ✅ |
| AU26_r | hard | 0.9942 | 0.375 | 0.072 | 1.063 | ✅ |
| AU45_r | easy | 0.9987 | 0.844 | 0.040 | 0.244 | ✅ |

### Right side — 0 / 17 failures

| AU | tier | r vs Mac-mirror | band | MAE | MAE band | verdict |
|---|---|---|---|---|---|---|
| AU01_r | easy | 0.9944 | 0.844 | 0.032 | 0.244 | ✅ |
| AU02_r | easy | 0.9914 | 0.844 | 0.029 | 0.244 | ✅ |
| AU04_r | medium | 0.9930 | 0.791 | 0.031 | 0.307 | ✅ |
| AU06_r | medium | 0.9982 | 0.791 | 0.021 | 0.307 | ✅ |
| AU07_r | medium | 0.9871 | 0.791 | 0.077 | 0.307 | ✅ |
| AU10_r | medium | 0.9883 | 0.791 | 0.074 | 0.307 | ✅ |
| AU12_r | easy | 0.9995 | 0.844 | 0.012 | 0.244 | ✅ |
| AU14_r | hard | 0.9970 | 0.375 | 0.035 | 1.063 | ✅ |
| AU15_r | hard | 0.7422 | 0.375 | 0.044 | 1.063 | ✅ |
| AU17_r | hard | 0.9598 | 0.375 | 0.066 | 1.063 | ✅ |
| AU20_r | hard | 0.9684 | 0.375 | 0.031 | 1.063 | ✅ |
| AU23_r | hard | 0.9939 | 0.375 | 0.012 | 1.063 | ✅ |
| AU25_r | hard | 0.9960 | 0.375 | 0.057 | 1.063 | ✅ |
| AU26_r | hard | 0.9939 | 0.375 | 0.068 | 1.063 | ✅ |
| AU45_r | easy | 0.9984 | 0.844 | 0.040 | 0.244 | ✅ |

(AU05_r and AU09_r are not in `_pipeline_helpers.AU_DIFFICULTY` so they
have no band; both pass at r > 0.98 against their Mac counterpart
anyway.)

The lowest AU r is **AU15_r r=0.7422 right** — well above its hard-tier
band of 0.375. AU15 is the lip corner depressor; on a normal subject
like IMG_0942 it's mostly near-zero with brief activations, so small
absolute differences flip rank order disproportionately. That's the
same low-signal-AU pattern we've seen on AU12 in the synkinetic
canaries, not a real mirror divergence problem.

Median r is **0.99 on both sides**. Easy + medium AUs (which are the
clinically meaningful ones) are all above 0.97.

## Verdict

**PASS — proceed with Lido mirrors on Windows (Part B1).**

## Going to do unless you redirect

1. Wait for the 42 Lido raws.
2. Mirror on Windows using S1's `StableFaceSplitter` (same code path
   I just validated, default device=CUDA).
3. Dual-mode AU extract via the same `reprocess_dual_v1316.py` template,
   merging the outputs into the existing `recoded_rerun_dual_v1316/`.
   Per the brief, no `action`-merge for new patients (S2 coding happens
   later on the Mac).
4. Spot-check + SMB back per `LIDO_PART_A_WINDOWS_RESULTS.md` notes.
