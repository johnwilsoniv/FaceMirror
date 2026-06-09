# Lido — ship the remaining assets (mirrored videos + Combined Data)

**Audience:** Windows box.  **Command center:** Mac.

## Status
Lido is processed: the **84 dual-mode AU CSVs are on the Mac, 0 errors** (verified). But
the last handback shipped only those analysis CSVs. **S2 action-coding on the Mac — which
must run before S2.5 and the phenotype work — needs the videos + the single-mode AU CSVs,
and those weren't included.** S2 plays the mirrored mp4 to code actions; a CSV can't
substitute for the video. So the pipeline is paused on this one transfer.

## Ask — no reprocessing, just transfer files you already produced in Part B1
1. **84 mirrored mp4s** (essential) — `Face Mirror 1.0 Output/<patient>_<side>_mirrored.mp4`
   for the 42 Lido patients (the `001_*_iOS_*` set), ~1.3 GB. These drive S2 coding and
   S2.5 frame rendering.
2. **Combined Data AU CSVs** (optional but easiest) — the per-frame single-mode S1 output
   B1 wrote to `Combined Data/` for these patients. Small. S2 loads them alongside the
   videos. *Optional* because the Mac can re-derive them from the dual CSVs via
   `ensure_mirrored_csvs`, but shipping them is simpler and avoids any derivation mismatch.

Nothing to re-run — pyfaceau 1.3.16 / `use_gpu=True` is settled (Part A), and these files
already exist from B1. This is purely a file transfer.

## Transfer
Same method as the dual-CSV delivery; command center coordinates the share path. Don't
stage on iCloud Drive on Windows (files-on-demand hangs the reader).

## After this lands (Mac side — FYI, no action for you)
S2 codes the 42 → merge action codes into the dual CSVs → move into
`recoded_rerun_dual_v1316/` → S2.5 curation → re-run the analysis matrix. The Lido clinical
labels get added to `FPRS_FP_Key_v2.csv` for the validation join.
