# S25 Auto-Curator — method & validation (as deployed)

Reverse-engineered from the curating clinician's hand selections in the S2.5 Frame
Curator. Re-fit on the **30-patient** ground-truth set: **277 status=done
patient×action instances** (after excluding `not_performed`; see below). All scores
are **leave-one-patient-out cross-validation F0.5** (precision-leaning, per the
"purer phenotype" objective) — honest held-out, not fit scores.

## Clinician's principles (the model we are encoding)
1. Keep frames where the REQUESTED action is at/near maximal expression.
2. Skip delayed onset (subject doesn't start immediately).
3. Stop at early relaxation / re-do (don't keep the dip between two attempts).
4. Reject blinks (AU45) — EXCEPT eye-closure tasks (ES/ET/BK) where closure is the target.
5. A baseline is REST: reject frames where the patient is smiling (AU06+AU12) — a
   resting baseline contaminated by expression is not a resting baseline.
6. For eye-closure tasks BOTH eyes must be closed; read closure as the minimum
   across hemifaces so a frame with one eye still open does not count as held.
NOT encoded: rejecting "contaminant" AUs (e.g. AU17 during BS) — spurious
correlation, not the clinician's reasoning.

## Flag handling (ground-truth hygiene)
- **`not_performed` → EXCLUDED** from fit and eval. The action wasn't done, so its
  kept set is leftover auto/whole-range or empty — not a representative-frame
  selection. Non-performance is a separate compliance gate, not a frame-selection
  target. (Excluded 8 instances; worst was ET, 6 of 24.)
- **`abnormal` → KEPT** (20 instances). The action WAS performed, just abnormally;
  the kept frames are valid, and training on them teaches the curator to handle
  abnormal performances.

## Rule archetypes
- REST (BL): keep quiet eyes-open frames (low total AU) + no blink + NON-SMILING
  (AU06+AU12 < gate). If the entire clip is smiling, keep the least-smiling frames
  and flag the baseline `smiling`; recovered baselines prefer a neutral window and
  fall back to a flagged smiling window only when none exists.
- EXPRESSION, frac-of-peak + position: keep task-signal >= frac*peak AND
  position >= pos (held portion), minus blinks. Used by RE, ES, ET, SE, SO, WN,
  FR, BK, **BC**.
- EXPRESSION, plateau: smooth the task signal, keep the LONGEST contiguous run
  held >= rel*peak (encodes onset-skip + early-stop), minus blinks. Used by BS, SS.
- POSITION-only: no panel AU and no CV-confirmed proxy -> keep position >= pos.
  Used by PL, LT.

## Task signals (which AUs define "the action is happening")
Default = the FACS in-set sum. Overridden by a CV-confirmed focused subset
(`signal_aus`) where one tracks the expression more cleanly:
- **BS = AU10+AU25**, **SS = AU12+AU23** — focused smile dyads beat the full
  7-AU sum (dilution).
- **SO = AU17+AU25+AU26** — orbicularis oris (AU18) is OFF-PANEL; this triad is
  the open-"O" articulation proxy (jaw-drop + lips-part + chin). CV 0.59→0.84.
- **BC = AU12+AU17** — cheek puff (AU33/34) is OFF-PANEL; CV-confirmed proxy dyad.
  BC therefore moved OUT of the position-only fallback into frac-of-peak.
- **LT** stays position-only: DLI/AU16 is off-panel and AU15/DAO (which we DO have)
  did not beat position under CV.

## Stronger-side reading
Read the higher-key-AU-peak hemiface (≈40% of patients are right-dominant) for
**BK, BS, SS** — proven under CV; NOT applied to RE (already near-perfect on the
default side). Side selection uses the full in-set; only the keep-signal is focused.

## Eye-closure: both eyes must close (`closure='min_both'`)
For **ES** and **ET** the closure signal is the **element-wise minimum across the
two hemifaces**, so a frame counts as a held closure only when BOTH eyes are
closed. This rejects the one-eye-open frames a single-side reading kept (a weak or
synkinetic lid lags on one side). ES reads AU45 (`signal_aus=['AU45']`); ET reads
the **min over its in-set** sum — tight closure recruits more than AU45 alone, and
AU45-only collapsed ET's signal. Implemented once in
`DataManager.task_signal(..., closure='min_both')` and shared by fit and deploy.

## Baseline WINDOW selection (`DataManager.choose_baseline`)
Separate from frame-KEEP: this picks WHICH frames are the BL action for a patient
(the auto-curator then keeps the least-smiling eyes-open frames within). The
baseline is the patient at REST; `tone` = total AU minus AU45 (eye-closure can't
look quiet). Tunable via config `BL_*`.
- **OPENING (default).** The quietest eyes-open window in the opening rest — frames
  before the FIRST task's *real onset*. Onset = where that task's defining AU signal
  ramps (`BL_ONSET_FRAC=0.3` of its peak), so leading frames the S2 coder
  mis-labelled as the first task (patient still at rest) are **reclaimed** into the
  baseline (e.g. IMG_2814: RE coded 1–50 but the brow-raise starts at 24, so 0–23 is
  reclaimed rest; that task node re-curates on its corrected extent).
- **LATER (exception).** Only when the opening is *unusable* — smiling
  (`BL_CONTAM_SMILE=2.5`) or active (`BL_CONTAM_TONE=9`) — AND a materially quieter
  later window exists (tone lower by `BL_SWITCH_MARGIN=3`, not more smiling) does it
  switch to that window (the heavy-smiler-at-the-start case, e.g. IMG_4036: opening
  smile 4.8/tone 11 → later 293–308 at tone 7.5). Later candidates are eyes-open,
  brow-quiet, uncoded-or-BL, and **not following an off-panel-target task**
  (`BL_OFFPANEL_ACTIONS` = BC/SO/PL/LT): residual cheek-puff/pucker/platysma/lip
  activity is invisible to tone, so a post-off-panel window would look quiet while
  the patient is still puffed/pursed (e.g. 20250225: a post-BC window read tone 4.8
  with cheeks still puffed → guarded → moved to 80–87, a tone-trustworthy post-RE
  rest). The guard clears at the next on-panel task.
- **Scoring.** Each window is scored on its quietest `BL_SEED_WIN=8`-frame seed and
  widened only across frames within `BL_EXTEND_TONE` of it, so a wider coded window
  can never inflate the score (a greedy widen had pulled windows into higher-tone
  neighbours). `apply_baseline` writes it hash-verified, reclaiming only the named
  first-task frames (never steals another action), with cross-hemiface consistency.

## Deployed held-out CV F0.5 (30-patient re-fit)
| action | n | CV F0.5 | rule |
|--------|---|---------|------|
| BL | 30 | 0.96 | rest (eyes-open + non-smiling, no movement gate) |
| RE | 30 | 0.91 | frac 0.60 |
| ES | 24 | 0.86 | frac 0.55 + pos + both-eyes (AU45 min) |
| ET | 18 | 0.90 | frac 0.75 + both-eyes (in-set min) |
| BS | 30 | 0.86 | plateau rel .65 + AU10+AU25 + stronger-side |
| SS | 29 | 0.92 | plateau rel .80 + AU12+AU23 + stronger-side |
| SE | 12 | 0.92 | frac 0.70 + pos |
| SO | 12 | 0.84 | frac 0.60 + pos + AU17+AU25+AU26 |
| WN | 17 | 0.93 | frac 0.50 + AU09 |
| FR |  7 | 0.67 | frac 0.65 + AU04  — see limitations |
| BK | 15 | 0.84 | frac 0.70 + AU45 + stronger-side |
| PL | 18 | 0.93 | position |
| BC | 18 | 0.83 | frac 0.60 + pos + AU12+AU17 |
| LT | 17 | 0.82 | position |

Macro held-out CV ≈ **0.87** (vs the prior 10-patient deployment's in-sample 0.835).
In-sample on the same 277 pairs: macro 0.890; 11/14 actions improved, 0 regressed.

## Fit == Deploy parity (production invariant)
The fit-time rule (`s25_auto_curator.predict_keep`) and the deploy-time rule
(`data_manager.auto_keep_frames`) are separate code. They are verified to produce
**byte-identical picks across every patient×action instance in the corpus
(currently 805 checks, incl. 95 BL; 0 mismatches)** with the deployed params — so deployed
behavior == validated behavior. The cross-hemiface closure aggregation
(`min_both`) and the baseline smile gate both live in shared `DataManager` helpers
(`task_signal`, `auto_keep_frames`) that the harness imports, so the two paths
cannot drift. Config constants (EYE_TASKS, NO_PANEL, STRONGER_SIDE, FACS_TASK_AUS)
are mirrored between the harness and `config.py` and checked equal.

## Honest limitations
- **FR is fit on n=7** and overfits: CV 0.67 vs in-sample 0.90 (gap 0.23). It is the
  only action below 0.80 held-out. `frac=0.65` still beats the prior `frac=0.95`
  (~0.46 on these patients), so it ships as the better default — but flagged
  low-confidence. RE-FIT once >7 patients have FR curated. Most patients have no FR.
- BS/BK/SO (0.84–0.86) remain the harder/smaller-n actions; the lever is more data.
- SO/BC proxies are off-panel surrogates (AU18/AU33-34 absent), validated by CV, not
  direct measurements of the target muscle.

## Files
- s25_auto_curator.py     — fit + leave-one-patient-out CV harness (ground-truth
  loader excludes `not_performed`)
- s25_au_combo_search.py  — dyad/triad proxy search for off-panel actions (diagnostic)
- s25_auto_review.py      — human vs new-auto vs old-auto comparison sheets
- s25_auto_params.json    — DEPLOYED per-action rule params (curator reads at startup)
- data_manager.auto_keep_frames() — applies the rules; _legacy_auto_keep() fallback

## Re-fit after curating more patients
Run:  `python3 s25_auto_curator.py`  (writes /tmp/s25_auto_params.json; prints the
held-out CV table). Then copy to `S3 Data Analysis/s25_auto_params.json`. Existing
hand-curation is never modified — new params only change the auto baseline for
uncurated actions and re-score resets. After deploying, re-run the parity check
(predict_keep == auto_keep_frames) before treating it as production.
