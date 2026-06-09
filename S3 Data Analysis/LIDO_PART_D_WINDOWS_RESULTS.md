# Lido Part D — mirrored mp4s shipped (response to LIDO_WINDOWS_BRIEF.md)

**Audience:** Mac command center.
**Reporter:** Windows box.
**Companion to:** `LIDO_WINDOWS_BRIEF.md` (the "ship remaining assets" ask).

## What landed on the Mac

- **84 / 84 Lido mirrored mp4s** at
  `~/Desktop/Lido_mirrored_videos/` on the Mac.
- **873 MB** delivered via SMB / robocopy, 0 failed,
  byte-for-byte intact (42 s actual copy time at 1.2 GB/min).
- File-name pattern matches the existing cohort:
  `<patient>_<side>_mirrored.mp4` where `<patient>` is the raw stem
  (e.g. `001_20250909_153625000_iOS_left_mirrored.mp4`).
- All 84 had their frame counts verified against the source `.MOV`
  during B1 — no truncation.

## What was NOT shipped (heads-up)

The brief expected B1 to have also written per-frame single-mode AU
CSVs to `Combined Data/<patient>_<side>_mirrored.csv` as a byproduct.
**It didn't, and they don't exist on this box.**

Why: `lido_B1_mirror.py` invokes `StableFaceSplitter.process_video()`
on its own, which produces only the mirrored mp4s. The single-mode
AU CSVs are written by a separate `openface_processor.process_video()`
call that `S1_FaceMirror/main.py` chains after the splitter (see
`main.py:538` — `openface_processor.process_video(...)` inside the
"finalize mirrors then run OpenFace" block). My B1 wrapper skipped
that second step because B2 was going to re-extract AUs in dual mode
anyway.

Per the brief, this is the optional half of the transfer:
> *Optional* because the Mac can re-derive them from the dual CSVs
> via `ensure_mirrored_csvs`, but shipping them is simpler and avoids
> any derivation mismatch.

**Recommended Mac-side action:** run `ensure_mirrored_csvs` on the
existing dual CSVs we already shipped in `recoded_rerun_dual_v1316/`
to derive the single-mode `<patient>_<side>_mirrored.csv`. The dual
output's default-mode `*_r` columns are produced by the same
`FullPythonAUPipeline` pass as single-mode would have been — the
extra `*_r_static` columns are computed alongside, not in place of,
the defaults — so derivation should be byte-identical to what B1's
openface step would have written.

If "byte-identical" matters and we can't get that from derivation,
I can re-run pyfaceau on the 84 mirrored mp4s in single mode
(`dual_au_mode=False`) and ship the resulting CSVs. ~45 min on
4 workers. Ping if needed.

## Inventory state on Mac after this

```
~/Desktop/Lido_mirrored_videos/                       (NEW, this push)
  84 × <patient>_<side>_mirrored.mp4   (873 MB)

~/Desktop/recoded_rerun_dual_v1316/                   (from earlier)
  306 × <patient>_<side>_mirrored_coded.csv  (139.4 MB total)
    - 222 existing (have action column from S2)
    - 84 Lido     (no action column -- S2 hasn't coded these yet)

\\192.168.1.33\S Data\Lido {Affected,Controls}\       (Mac-original)
  42 × <patient>_iOS.MOV   raw sources (untouched by this transfer)
```

The Mac team has everything they need to run S2 coding on the 42
Lido patients. Once S2 codes them, the action labels can be merged
into the corresponding dual CSVs the same way the existing 222 had
their actions merged.
