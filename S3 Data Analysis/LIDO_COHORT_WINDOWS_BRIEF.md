# Brief — add the Lido cohort to v1316 + resolve the GPU-CLNF question (Windows-CUDA)

**Audience:** the Windows-CUDA box (the machine that produced `recoded_rerun_dual_v1316/`).
**Command center:** the Mac (writes this brief, does the split+mirror, ships you the
mirrored clips, ingests your results). You are the worker.

## Why this runs on Windows
The existing 111-patient AU dataset (`recoded_rerun_dual_v1316/`) was AU-extracted on
THIS Windows-CUDA box with **pyfaceau 1.3.16, `dual_au_mode=True`** (see
`WINDOWS_DUAL_MODE_REPROCESS_BRIEF.md`). Cross-platform AU values are NOT bit-exact
(~1e-6 BLAS/cuDNN differences), so to keep the new patients comparable to the existing
cohort, their **AU extraction must happen on this same box, same version, same config.**

**Full pipeline (split+mirror AND AU) runs on Windows for the Lido batch** — the CUDA GPU
makes mirroring much faster, and keeping both steps on one box avoids a Mac→Windows hop.
Caveat to verify (Part A4): the *existing* cohort's mirrored clips were produced on the
**Mac**. The split+mirror geometry should be platform-identical, but it uses face
detection + landmarks (which have a GPU/CPU path), so we confirm with a one-canary
re-mirror diff before trusting the Windows-made Lido mirrors. If that diff is non-trivial,
fall back to mirroring on the Mac (command center) and shipping the mirrored clips.

## Two jobs
- **A. Resolve the GPU-CLNF question.** The Mac tier0 gate flags
  `pyfaceau.config.CLNF_CONFIG['use_gpu'] == True` in installed pyfaceau 1.3.16, while a
  code comment claims CPU is "~38% better landmark accuracy on paralyzed faces." We need
  to (a) learn which config the v1316 goldens were actually made with, and (b) test
  whether that 38% gap still exists in 1.3.16 or is a stale comment.
- **B. AU-extract 42 new "Lido" patients** (26 synkinesis-affected + 16 controls),
  dual-mode, consistent with v1316.

---

## Part 0 — Preflight (do NOT skip; ~15 min)

```powershell
cd <repo-root>
git fetch origin && git checkout s25-autocurator-handoff && git pull   # gets this brief + latest tests
cd "S1_FaceMirror"; .\.venv\Scripts\Activate.ps1
python -c "import pyfaceau; print(pyfaceau.__version__)"   # MUST print 1.3.16
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # CUDAExecutionProvider present
```

**Canary parity (proves this box still reproduces v1316):**
```powershell
cd "..\S3 Data Analysis"
$env:SPLITFACE_BASE = "$env:USERPROFILE/Documents/SplitFace"
python -m pytest tests/test_tier1_windows_cuda_parity.py -v
```
All canaries × sides must pass within `tests/golden/metric_bands.yaml`. If anything fails,
**stop and report to command center** — the env has drifted and Lido would be inconsistent.

---

## Part A — GPU-on/off test (resolve the stale-comment question)

The 38% claim is specifically about **paralyzed faces vs the C++ ground truth.** Test it
directly on a Complete-severity canary (**IMG_0861** and/or **IMG_2259**) plus the normal
baseline canary (IMG_0942).

1. **Existing divergence harness** (CPU vs GPU on IMG_0942):
   ```powershell
   python tests/update_goldens.py --stage gpu_divergence --reason "pyfaceau 1.3.16 GPU vs CPU reverify"
   python -m pytest tests/test_pyfaceau_gpu_divergence_within_band -v
   ```
   (If the `gpu_divergence` stage no longer exists, do the manual comparison below instead.)

2. **The accuracy question (the part that matters).** For each of IMG_0861, IMG_2259
   (paralyzed) and IMG_0942 (normal), each side, run pyfaceau on the mirrored clip
   (`$SPLITFACE_BASE/S1O Processed Files/Face Mirror 1.0 Output/<id>_<side>_mirrored.mp4`)
   TWICE — once with `CLNF_CONFIG['use_gpu']=False`, once `=True` — and compare each run's
   AU columns to the **C++ ground truth** `tests/golden/aus/<id>_<side>/cpp.parquet`
   (Pearson r + MAE per AU). Report, per canary:
   - `r(CPU vs cpp)` and `MAE(CPU vs cpp)`
   - `r(GPU vs cpp)` and `MAE(GPU vs cpp)`
   - the **GPU−CPU accuracy delta** on the paralyzed canaries specifically.
   To flip `use_gpu` at runtime: `from pyfaceau.config import CLNF_CONFIG; CLNF_CONFIG['use_gpu']=False`
   BEFORE constructing the pipeline (it reads the dict at init).

3. **Also determine v1316's config.** Compare each run to the existing Windows golden
   `tests/golden/aus/<id>_<side>/pyfaceau_windows_cuda.parquet` — whichever config
   (GPU or CPU) matches it near-exactly is what v1316/the goldens were built with.

**Decisions from Part A:**
- **Lido must use the SAME config that the windows_cuda goldens were made with** (step 3),
  whatever the 38% verdict — that's what keeps Lido comparable to v1316.
- If GPU now ≈ CPU on paralyzed faces (comment is stale): note it, and update the comment
  in `pyfaceau` + the `test_clnf_config_use_gpu_disabled` invariant accordingly (separate
  PR). If GPU is still materially worse: the comment stands; keep CPU.
- Report all numbers back to command center before starting Part B.

---

## Part A4 — Mirror-consistency check (confirm Windows mirror == Mac mirror)
Before mirroring the Lido batch on Windows, prove a Windows re-mirror reproduces the
Mac-made mirror for one existing canary (this also confirms S1 split+mirror runs on this box):
1. Command center ships one canary's RAW source video (e.g. `IMG_0942.MOV`) plus its
   Mac-made `IMG_0942_left_mirrored.mp4` / `_right_mirrored.mp4`.
2. On Windows, run S1 split+mirror on the raw video (`batch_process.py` or the FaceMirror
   GUI) → Windows mirrors.
3. Compare Windows vs Mac mirror: mean-abs pixel diff per frame (expect ≈0 if identical).
   If non-zero, run pyfaceau on both and compare AUs within `metric_bands.yaml` tolerance.
4. **PASS** (≈identical / within tolerance) → mirror Lido on Windows (Part B).
   **FAIL** → stop; command center will mirror Lido on the Mac and ship the clips instead.
Report the pixel-diff / AU numbers to command center.

## Part B — Split+mirror + AU-extract the Lido cohort

Input: **42 raw Lido `.MOV`** shipped from command center (26 affected + 16 controls).

**Step B1 — split+mirror** each raw video via S1 (`batch_process.py` or the FaceMirror GUI)
→ 84 `<patient>_<side>_mirrored.mp4`, same naming as the existing 221. Use the CLNF config
resolved in Part A. **Keep `batch_process.py`'s normal S1 outputs** (the mirrored mp4s in
`Face Mirror 1.0 Output/` and the single-mode AU CSVs in `Combined Data/`) — those are the
artifacts the Mac's **S2 action coder** reads (it plays the mirrored mp4 and loads the AU
CSV from the sibling `Combined Data/`). The dual-mode pass below re-extracts AU for the
analysis format, but does NOT replace what S2 needs.

**Step B2 — dual-mode AU** on those 84 mirrors. Adapt the reprocess script from
`WINDOWS_DUAL_MODE_REPROCESS_BRIEF.md` (Step 3) with these **deltas**:
1. `MIRRORED_DIR` → the dir where B1 wrote the 84 Lido mirrors.
2. `OUTPUT_DIR` → `recoded_rerun_dual_v1316/` (append — they join the cohort).
3. **Skip the action-merge block.** New patients have no `recoded_per_frame/<base>_coded.csv`
   yet (S2 coding happens later on the Mac). Write the CSV with NO `action` column.
4. Use the CLNF `use_gpu` value resolved in Part A (set it before building the pipeline).

Everything else identical: `FullPythonAUPipeline(..., dual_au_mode=True)`,
`reset_pipeline_state(pipeline)` between videos (use `pipeline.landmark_detector`, never
`clnf`), idempotent skip-if-exists.

## Part C — Spot-check + handback

Spot-check (per `WINDOWS_DUAL_MODE_REPROCESS_BRIEF.md` Step 4): each CSV has 17 `_r` + 17
`_r_static` columns, frame counts match the source clip, default-mode p10 ≈ 0, static-mode
floor elevated for the affected patients. **No `action` column expected** for these.

Handback to command center — **S2 coding on the Mac needs the video + AU artifacts, not
just the analysis CSVs.** Ship all three:
1. **Dual CSVs** (84) → for `recoded_rerun_dual_v1316/` — the analysis format (no `action`).
2. **Mirrored mp4s** (84, `<patient>_<side>_mirrored.mp4`) → for `Face Mirror 1.0 Output/` —
   S2 plays these to code actions; S2.5 renders frames from them.
3. **Combined Data AU CSVs** (B1's single-mode S1 output) → for `Combined Data/` — S2 loads
   these alongside the videos.
4. One-line status: # processed, # errored, the Part A GPU/CPU numbers, the Part A config verdict.

---

## Data transfer (Mac ⇄ Windows)
- **Mac → Windows:** (1) the **42 raw Lido `.MOV`** (26 from `S Data/Lido Affected/`, 16
  from `S Data/Lido Controls/`); (2) for the A4 check, **one canary raw** (`IMG_0942.MOV`)
  + its two Mac mirrors. SMB share off the Mac, or zip + cloud/USB. Do NOT stage on iCloud
  Drive on Windows (files-on-demand hangs ffmpeg — copy to a local non-cloud path first;
  `fetch_canaries.ps1` at repo root is the SMB template).
- **Windows → Mac:** (1) 84 dual CSVs (small); (2) 84 mirrored mp4s (~the bulk — needed
  for S2 coding + S2.5 frame rendering); (3) 84 Combined Data AU CSVs (small). All three
  per the Part C handback.
- Command center will provide the share path / method.

## Guardrails
- pyfaceau **must be 1.3.16**; do not upgrade for this batch.
- `reset_pipeline_state` between videos is mandatory (`landmark_detector`, not `clnf`).
- Lido CSVs join `recoded_rerun_dual_v1316/` — keep the exact `<patient>_<side>_mirrored_coded.csv`
  filename pattern (Mac analysis scripts glob on it).
