# SplitFace pipeline regression-test framework

A pytest harness that benchmarks every stage of the paralysis pipeline
against C++ OpenFace 2.0 reference outputs. Catches regressions at the exact
stage they're introduced.

## Why

Multiple bugs in the past spent weeks of investigation each before being
located: a per-video state-carryover bug in pyfaceau, a BatchedCEN GPU
divergence, a deterministic-sort regression in `prepare_data_generalized`, a
dead-code `use_known_optimal` flag that silently did nothing. Every one of
those had a 1-line invariant we could have asserted but didn't. This
framework is the durable fix.

## Quick start

```bash
# Tier 0 (≤30s): determinism invariants — run on every commit
pytest -m tier0

# Tier 1 (≤10min): full quality vs C++ on 10 canary patients — run before merging
pytest -m tier1

# Tier 2 (≤15min): retrain reproducibility — run before tagging a release
pytest -m tier2  # NOTE: not yet implemented (sub-PR 3)

# Run everything that's currently implemented
pytest

# Run just one canary
pytest -k IMG_0861

# Verbose with full diffs
pytest -v --tb=long
```

## What's tested

Six pipeline stages, each with separate tests. The framework currently
covers stages 3–6a; stages 1–2 (bbox, landmarks) are marked `xfail` until
sub-PR 2 instruments pyfaceau-side capture.

| Stage | What's compared | Threshold split |
|---|---|---|
| 1. Face detection | bbox per frame | normal vs paralyzed (xfail) |
| 2. Landmarks | 68 (x,y) per frame | normal vs paralyzed (xfail) |
| 3. AU intensities | 17 AUs per frame, frame-paired | severity × difficulty bucket |
| 4. Peak frame detection | `{action}_Max Frame` per (patient, side) | shared |
| 5. Engineered features | mid_face_features output diff | shared |
| 6a. Inference parity | saved Jan 1 model on py vs cpp features | shared |
| 6b. Retrain reproducibility | fresh retrain with `use_known_optimal=True` | per-zone (sub-PR 3) |

## The 10 canary patients

Defined in `canary_patients.yaml`. Three Normal, three Mid Partial, two Mid
Complete, two edge cases. Every test that's parametrized over canaries runs
once per patient (and usually once per (patient, side) hemiface). Threshold
selection is controlled by the patient's `severity` field — paralyzed cases
get more permissive thresholds because pyfaceau drift is larger on
paralyzed faces (well-documented in `RETRAINING_REPRODUCIBILITY.md`).

Tier 0 is a smaller subset (`tier0: true` in the YAML) — currently IMG_0942
(normal) + IMG_2380 (paralyzed). Both run during the fast determinism gate.

## Initial threshold policy

Per the project decision (Apr 2026), bands are calibrated to **today's
observed values + headroom**, not to manuscript-era values. The framework's
job is to catch FUTURE drift below today's bar. Tightening thresholds
toward the manuscript-era gold standard is a deliberate separate task —
adjust `metric_bands.yaml` and re-run the full suite to confirm.

Calibration headroom (defined in `update_goldens.stage_metric_bands`):
- Pearson r threshold = `worst_observed - 0.05`
- MAE threshold = `worst_observed + 0.10`
- Feature drift hard cap = `worst_observed + 0.10`
- Inference agreement = `worst_observed - 0.05` (floor: 0.50)

## Updating goldens

When you intentionally change something — pyfaceau ships a real improvement,
expert labels are updated, a new canary is added — regenerate goldens:

```bash
# Regenerate all stages
python tests/update_goldens.py --stage all --reason "pyfaceau v0.3.2 dynamic-AU fix landed"

# Regenerate just one stage
python tests/update_goldens.py --stage aus --reason "re-snapshotted after CLNF rigid-fix"
```

Every run appends to `golden/golden_history.md` with timestamp, git SHA, and
the SHA256 of the active venv's pip-freeze (so library drift is auditable).

`update_goldens.py` is **idempotent** — running twice in a row produces
byte-identical output and an unchanged checksums.json. If a re-run produces
diffs, that's itself a regression worth investigating before committing.

## Files

```
tests/
  README.md                           ← this file
  pytest.ini                          ← markers, filters, addopts
  conftest.py                         ← shared fixtures (canary loader, paths, jan1_model)
  canary_patients.yaml                ← the 10 canary registry
  _pipeline_helpers.py                ← stage-by-stage utilities (loaders + metrics)
  update_goldens.py                   ← regenerates golden/ contents
  test_tier0_determinism.py           ← byte-equality tests
  test_tier1_quality_vs_cpp.py        ← per-stage quality tests
  test_framework_self_test.py         ← deliberate-regression tests (catches framework rot)
  golden/
    metric_bands.yaml                 ← per-stage thresholds (auto-calibrated)
    checksums.json                    ← SHA256 of every golden file
    golden_history.md                 ← append-only update log
    test_split_seed42.json            ← locked test patient IDs
    peak_frames.json                  ← locked peak frames per canary
    features_pyfaceau.parquet         ← locked engineered features (pyfaceau)
    features_cpp.parquet              ← locked engineered features (C++)
    predictions_pyfaceau.json         ← locked Jan 1 model predictions (py features)
    predictions_cpp.json              ← locked Jan 1 model predictions (cpp features)
    aus/<patient>_<side>/
      pyfaceau.parquet                ← snapshot of S2O Coded Files CSV
      cpp.parquet                     ← snapshot of S2O Coded Files OF CSV
    landmarks/<patient>_<side>/
      cpp.parquet                     ← snapshot of C++ landmarks
                                      (pyfaceau side comes in sub-PR 2)
```

## Inputs the framework needs (not in repo)

These live on each engineer's local filesystem — paths are baked into
`conftest.py`. The tests `pytest.skip` if any are absent rather than fail:

- `/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/`
  — mirrored hemiface MP4s (`{patient}_{side}_mirrored.mp4`)
- `/Users/johnwilsoniv/Documents/SplitFace/S2O Coded Files/`
  — pyfaceau per-frame AU CSVs
- `/Users/johnwilsoniv/Documents/SplitFace/S2O Coded Files OF/`
  — C++ per-frame AU + landmarks CSVs
- `/Users/johnwilsoniv/Documents/SplitFace/S3O Results/combined_results.csv`
- `/Users/johnwilsoniv/Documents/SplitFace/S3O Results/combined_results_OF_v2.csv`
- `S3 Data Analysis/dist/Paralysis Analyzer.app/.../models/mid_face_*.pkl` — saved Jan 1 model

## Coming next

- **Sub-PR 2**: Stage 1 (bbox) and Stage 2 (landmarks) tests. Requires
  capturing pyfaceau-side bbox and landmarks per frame (currently the
  pyfaceau output CSV only persists AUs). Will likely add a `--instrument`
  flag to a wrapper around pyfaceau's pipeline that saves all per-frame
  intermediates to parquet.
- **Sub-PR 3**: Stage 6b (retrain reproducibility). Runs
  `paralysis_training_pipeline.py` for each zone with `use_known_optimal=True`
  on `paper_combined_results.csv`, asserts test acc lands in the locked band.
  Will be the only test that takes more than a minute.
