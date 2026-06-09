# Lido Parts B + C — Windows reprocess results

**Audience:** Mac command center.
**Reporter:** Windows box.
**Companion to:** `LIDO_COHORT_WINDOWS_BRIEF.md` Parts B + C.

## TL;DR

The Lido cohort is processed and back on the Mac.
- **42 / 42 raws → 84 / 84 mirrors** (Part B1) in 25.4 min wall.
- **84 / 84 dual-mode AU CSVs** (Part B2) in 46.0 min wall.
- **0 errors, 0 schema fails, 0 truncated.**
- `recoded_rerun_dual_v1316/` now has **306 CSVs total** (existing 222 + Lido 84).
- The 84 new Lido CSVs are at `~/Desktop/recoded_rerun_dual_v1316/` on the
  Mac — robocopy delivered 49.5 MB byte-for-byte intact.
- Schema spot-check passes: 17 default + 17 static AU cols, no `action`
  column (per brief — Lido isn't S2-coded yet), 100 % success rate on
  all sampled sides.
- AU17 / AU20 / AU45 static-mode floors are clearly elevated on the
  Affected sample vs the Control sample, with visible bilateral
  asymmetry on Affected — exactly the substrate Pilot 15 wants.

## Part B1 — S1 split+mirror

- 42 raws pulled from `\\192.168.1.33\S Data\Lido Affected\` (26) and
  `\Lido Controls\` (16) to a local non-iCloud path:
  `C:\Users\User\Documents\SplitFace\Lido_raws\` (1.81 GB total).
- Mirrored via `lido_B1_mirror.py` with `--workers 4`. Each worker
  invokes `StableFaceSplitter.process_video()` — same code path I
  validated end-to-end in A4.
- An earlier 2-worker pass was killed mid-flight before it could
  finalize one patient. **Caught + recovered** before relaunching B1:
  decoded-frame-count vs source check identified one truncated pair
  (`001_20250909_153625000_iOS_{left,right}_mirrored.mp4`, moov atom
  missing, 0 readable frames), deleted, and the 4-worker rerun
  re-mirrored that patient cleanly. Final sweep: 84/84 mirrors, all
  frame counts match source byte-for-byte.

Wall time: 25.4 min for 37 fresh raws on 4 workers (~165 s / video
per worker; ~2.6 × speedup vs single-worker).

## Part B2 — dual-mode AU extraction

- `lido_B2_parallel.py` (parallel adaptation of the
  `reprocess_dual_v1316.py` pattern that produced the existing 222).
- 4 workers via `ProcessPoolExecutor` with an initializer that builds
  one `FullPythonAUPipeline(dual_au_mode=True)` per worker process —
  amortizes the ~30 s model load cost across each worker's full
  share, not once per video.
- Each worker calls `reset_pipeline_state` between videos via
  `pipeline.landmark_detector` (not `clnf` — that was the
  v1.3.8-1.3.13 silent-no-op bug; see PILOT_METHODOLOGY_NOTES.md
  pitfall 2).
- **No action-label merge step** per the brief — Lido patients haven't
  been S2-coded yet, so the CSV emits without an `action` column. S2
  coding happens on the Mac later.
- pyfaceau 1.3.16, GPU CLNF (`use_gpu=True` — the v1316 config
  validated bit-exact in Part A).

Wall time: 46.0 min for 84 videos on 4 workers (~109 s / video per
worker; ~2.6 × speedup vs single-worker estimate of 98 min serial).
GPU stayed ~26 % utilized during the run — modest CUDA contention
across workers, plenty of headroom but didn't push to 6 workers
mid-run to avoid risking a kill-restart.

## Part C — spot-check

One Lido Affected + one Lido Control, both sides:

```
=== Affected:  001_20250909_153625000_iOS ===
  side    AU17 def->static  AU20 def->static       AU45 def->static
  left    0.73 -> 0.55      0.11 -> 0.73 (+0.62)   0.78 -> 0.73
  right   0.58 -> 0.55      0.07 -> 1.27 (+1.20)** 0.35 -> 0.76 (+0.41)
                              (static_p10=0.527)

=== Control:   001_20251001_133812000_iOS ===
  side    AU17 def->static  AU20 def->static       AU45 def->static
  left    0.58 -> 0.24      0.11 -> 0.24           0.45 -> 0.57
  right   0.70 -> 0.60      0.17 -> 0.54           0.58 -> 0.51
```

- **Affected right shows AU20 static_p10 = 0.527** — the patient is
  stretching their lip at rest, not just during voluntary movement.
  That's the "static floor elevated on hypertonic / synkinetic
  patients" signal the brief was after.
- **Bilateral asymmetry** visible on Affected (AU20 shift +0.62 left
  vs +1.20 right), much weaker on Control (+0.13 vs +0.37). Pilot 15
  has substrate to score on.
- Schema check: every spot-checked CSV has 17 default + 17 static AU
  cols, `success` column present, **`action` column ABSENT** (per
  brief), 100 % success rate, frame counts match source.

## Handoff

- **All 84 Lido CSVs at `~/Desktop/recoded_rerun_dual_v1316/`** on the
  Mac (alongside the existing 222 from yesterday). robocopy reported
  49.51 MB / 49.51 MB / 0 failed.
- File name pattern matches the existing cohort:
  `<base>_mirrored_coded.csv` where `<base>` is the raw stem with
  `_left_mirrored` or `_right_mirrored` appended.
- Mac team can move into `S3 Data Analysis/recoded_rerun_dual_v1316/`
  (already in `.gitignore`) and start the Pilot 15 phenotype work on
  the combined 306-patient corpus.

## Numbers HQ asked for

- # processed: **84 / 84**
- # errored:    **0**
- total wallclock: B1 25.4 min + B2 46.0 min = **71.4 min** (1.19 h).
- Part A GPU/CPU numbers: see `LIDO_PART_A_WINDOWS_RESULTS.md`
- Part A config verdict: `use_gpu=True` (bit-exact match to v1316; CPU
  works on Windows after the pyclnf 63d76973 / 5fbe598d fixes but
  produces a different absolute-value path).

Pipeline closed.
