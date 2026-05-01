# Methods Divergence from Published Manuscript

This document tracks every material divergence between the methods described
in the published manuscript and the production pipeline as of 2026-05-01.
It is the input to any future Methods-section update or addendum.

**Pipeline status as of this snapshot:** validated against manuscript-era
classification accuracy on all three face zones — see
`S3 Data Analysis/models/{upper,mid,lower}_face_eval_metrics.json` for the
held-out test-set numbers (re-evaluated 2026-05-01).

| Zone | Manuscript acc / F1w | Production acc / F1w | Δ acc |
|---|---:|---:|---:|
| Upper | 0.83 / 0.83 | **0.830 / 0.839** | 0.0 |
| Mid | 0.93 / 0.92 | **0.907 / 0.910** | −2.3pp |
| Lower | 0.84 / 0.82 | **0.821 / 0.831** | −1.9pp |

Production exceeds manuscript on `f1_Partial` across all three zones
(+17/+8/+18 pp), the clinically-relevant ambiguous middle class.

---

## Architectural Divergences

### 1. AU extractor: OpenFace 2.2 (C++) → pyfaceau (pure Python)
The single largest infrastructural change. Manuscript ran the OpenFace 2.2
C++ binary against mirrored hemiface MP4s. Production now runs **pyfaceau**
(`pyfaceau` 1.3.11), a pure-Python reimplementation of OpenFace 2.2 backed
by `pyclnf` 0.3.3 (CLNF landmark refinement), `pyfhog` 0.1.4 (HOG features),
and `pymtcnn` 1.1.5 (face detection).

- Orchestrated through `S1_FaceMirror/openface_integration.py:28` and
  `S1_FaceMirror/pyfaceau_detector.py`.
- AU prediction stage applies a **3-frame moving average** in
  `pyfaceau/pipeline.py:1080,1179` and `prediction/au_predictor.py:275,368`.
- Validated against C++ on a 10-canary regression suite. Clean-shaven mean
  Pearson r vs C++: 0.972 (healthy), 0.950 (paralysis). Facial-hair
  subjects show larger divergence (mean r=0.920) for both detectors and are
  flagged as a known limitation rather than a regression.
- AU17/AU25/AU26 absolute intensities have drifted vs. the manuscript-era
  pyfaceau snapshot (`paper_combined_results.csv`). See
  `RETRAINING_REPRODUCIBILITY.md` for full details. Saved-model inference
  is robust to this drift; **fresh retraining on regenerated data drops
  Lower Face accuracy 0.84→0.64** unless `paper_combined_results.csv` is
  used as the input.

### 2. Landmark detector: dlib 68-point → MTCNN + CLNF refinement
Manuscript described "dlib 68-point predictor". Production uses MTCNN for
face detection followed by 68-point CLNF refinement (`pyfaceau_detector.py:26-84`),
GPU-accelerated where available. SPIGA was evaluated as an alternative in
Feb 2026 (`S1_FaceMirror/debug_clnf_spiga_v*.py`) but never wired into
`face_splitter.py` — production remains MTCNN+CLNF.

### 3. Classifier head: flat XGBoost multiclass → ordinal binary decomposition
Manuscript described plain XGBoost multiclass with isotonic calibration.
Production added an **OrdinalBinaryClassifier**
(`paralysis_training_helpers.py:47`, commit `547e3969` 2025-12-31) that
decomposes ordinal severity into cumulative binary decisions:
P(Y > None), P(Y > Partial). Class predictions reconstructed by threshold
optimization (`optimize_ordinal_thresholds`, line 188).

- Default for retraining is now ordinal-on
  (`paralysis_config.py:137-152`).
- The currently committed `models/*.pkl` files **predate this change** and
  use plain XGBoost — they reproduce manuscript accuracy.
- A future retrain will silently use the ordinal classifier unless
  `use_known_optimal=True` is set, in which case the saved hyperparameters
  reproduce manuscript-era behavior.

### 4. Hyperparameter search: 5-fold CV → Optuna TPE + HyperbandPruner
Manuscript described "5-fold CV tuning, key settings like learning rate".
Production runs **Optuna `TPESampler` + `HyperbandPruner`, 200 trials per
zone** (`paralysis_config.py:236-256`), with the chosen hyperparameters
frozen as `known_optimal_params` in the same file (e.g., Mid:
`learning_rate=0.0785, max_depth=6, n_estimators=356`).

When `use_known_optimal=True`, no tuning runs and the frozen params are
used — this is the canonical path for manuscript-quality reproduction.

### 5. Feature selection: per-zone hand-curated AU sets → automated 3-stage
Manuscript described per-zone AU sets without an automated FS step.
Production runs **VarianceThreshold → SelectKBest(f_classif) → RandomForest
importance** (`paralysis_utils.py:140-195`), capped per zone:

- Upper: `top_n_features=25`
- Mid: `top_n_features=40`
- Lower: `top_n_features=60`

The selected feature lists are persisted to `models/{zone}_face_features.list`
and re-used at inference.

### 6. Imbalance handling: SMOTE → multi-strategy + per-zone weights
Manuscript: SMOTE + isotonic calibration. Production:

- borderline + regular SMOTE with adaptive ratios
  (`paralysis_config.py:94-129`)
- optional SMOTEENN cleanup
- per-zone class weights (Mid override `{None:1, Partial:10, Complete:7}`
  for the 75/15/10 imbalance, `paralysis_config.py:212-215`)
- threshold optimization on the validation fold

### 7. Calibrator: uniform isotonic → per-zone (sigmoid for Upper)
Manuscript described isotonic calibration applied uniformly. Production
uses **sigmoid calibration for Upper, isotonic for Mid and Lower**
(`paralysis_config.py:372`). This was empirically chosen.

### 8. AU set adjustments
- **AU16 dropped** from Lower (commits `2f3925f3`, `a47237a7` 2025-12-30):
  not available in OpenFace 2.2 / pyfaceau output. The manuscript-era
  configuration referenced it; current `paralysis_config.py:160` excludes it.
- **AU06 restored** to Mid (commit `932681c6` 2025-12-30) after a brief
  removal during an experiment.

---

## Paralysis-Specific Engineering Beyond Manuscript

Hemiface mirroring (the manuscript's headline novelty) is unchanged. The
following enhancements were added on top:

### Asymmetry feature primitives (`paralysis_utils.py:_extract_base_au_features`, lines 197-249)
Generated for every (action, AU) pair:
- `_val_side` — raw AU intensity on the affected side
- `_val_opp` — raw AU intensity on the contralateral side
- `_Asym_Diff` — side − opposite
- `_Asym_Ratio` — side / opposite
- `_Asym_PercDiff` — percentage difference
- `_Is_Weaker_Side` — binary flag for the lower-intensity side

The manuscript described "ratios and percentage differences" only. The raw
side/opposite values and the binary weaker-side flag are production-only
additions.

### Zone-specific paralysis interactions
- **Mid** (`mid_face_features.py:31-87`): ETES (Eyes Tightly / Eyes Softly)
  ratios both same-side and contralateral, AU45/AU07 ES↔ET differences,
  per-action max/min/range
- **Lower** (`lower_face_features.py:36-58`): cross-action AU12 averages,
  `BS_Asym_Ratio_Product_12_25` for the smile zone
- **Upper** (`upper_face_features.py:30-48`): AU01+AU02 asymmetry products,
  AU01×AU02 product/sum

### Synkinesis-aware feature helpers
`_extract_coupling_features` and `_extract_paralysis_conditioned_features`
(`paralysis_utils.py:264-414`) build trigger×coupled discriminators
(`Bilateral_Min/Ratio/GeoMean`). Latent — not used by the published
paralysis models but consumed by adjacent work.

### Frame quality scoring
- `pyfaceau_detector.py:425` — quantitative head-yaw quality score
- `pyfaceau/data/quality_filter.py` — jitter detector

Manuscript described "we flagged frames with excessive rotation" without
quantification; production produces a numeric quality score per frame.

---

## Reproducibility Infrastructure (post-manuscript)

- **Deterministic patient sort** before `train_test_split`
  (`paralysis_utils.py:507`). Without this, row order from
  `main.py --batch` varied across runs and `random_state=42` did not
  produce the same split. This single line accounts for ~20pp Lower Face
  variance observed across "identical" retrains in early Apr 2026.
- **`use_known_optimal=True` flag** in `paralysis_config.py` to bypass
  hyperparameter tuning and use the frozen manuscript-era params.
- **`paper_combined_results.csv`** committed snapshot of the manuscript-era
  PyFaceAU output. Required for byte-identical re-trains.
- **`RETRAINING_REPRODUCIBILITY.md`** documents the AU17/25/26 drift and
  the recipe to reproduce manuscript numbers.
- **`tests/` regression framework** (S3 Data Analysis/tests/) — 7-stage
  per-canary comparison vs. C++, golden file system with SHA256 checksums,
  `make tier0`/`tier1`/`tier2` test tiers, pre-commit hook.
- **`*_face_eval_metrics.json`** companion files next to each saved model
  (added 2026-05-01) record the held-out test metrics on both
  paper and today's CSV inputs.

---

## Items That Need Disclosure in Any Methods Update

If this pipeline contributes to a future publication, the methods section
should explicitly note:

1. **AU extractor swap**: pyfaceau replaces OpenFace 2.2 C++. Validated
   parity on clean-shaven subjects (mean r=0.97); larger divergence on
   facial-hair subjects (mean r=0.92).
2. **Ordinal classifier head**: replaces flat multiclass for new training
   runs. Saved manuscript-era models are flat XGBoost.
3. **Per-zone calibrator choice**: not the uniform isotonic the manuscript
   reports.
4. **AU16 absent from Lower zone** (not in OpenFace 2.2 output).
5. **Optuna 200-trial tuning** with frozen `known_optimal_params`
   replaces the manuscript's "5-fold CV tuning."
6. **Quantitative frame quality scoring** replaces the manuscript's
   qualitative rotation-flagging.

---

*Generated 2026-05-01 alongside the production-2026-05-01 git tag.*
*Inputs: agent audit of pipeline structure, `RETRAINING_REPRODUCIBILITY.md`,*
*`{upper,mid,lower}_face_eval_metrics.json`, git log on*
*`paralysis_training_pipeline.py` and `*_face_features.py` since 2025-09-01.*
