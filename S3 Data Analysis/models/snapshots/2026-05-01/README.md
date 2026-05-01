# Production-2026-05-01 Model Snapshot

Frozen archival copies of the production paralysis-classification models
as of the `production-2026-05-01` git tag.

## What's here

For each face zone (upper / mid / lower):
- `{zone}_face_model.pkl`        — trained XGBoost VotingClassifier
- `{zone}_face_scaler.pkl`       — StandardScaler fit on selected features
- `{zone}_face_features.list`    — selected feature names (one per line)
- `{zone}_face_eval_metrics.json` — held-out test-set performance
- `{zone}_face_feature_importance.csv` — Random Forest importance scores

## Held-out test-set performance

| Zone  | Manuscript | This snapshot (TODAY's CSV) | Δ acc  |
|-------|-----------:|----------------------------:|-------:|
| Upper | 0.83       | 0.830                       | 0.0    |
| Mid   | 0.93       | 0.907                       | -2.3pp |
| Lower | 0.84       | 0.821                       | -1.9pp |

Partial-class F1 improved across all three zones (+17/+8/+18 pp). See
each `*_eval_metrics.json` for full per-class metrics, confusion matrices,
and the manuscript reference values.

## How to use

These are byte-identical copies of `S3 Data Analysis/models/{zone}_face_*.pkl`
at the time of the production-2026-05-01 milestone. Production code loads
from the parent directory (`models/`), not from here. This snapshot exists
purely for historical comparison — if a future retrain replaces the active
models, this directory preserves the manuscript-validated state.

To compare a future model against this snapshot, load both and re-evaluate
on the same `random_state=42` test split:

```python
import joblib, json
old = joblib.load('models/snapshots/2026-05-01/mid_face_model.pkl')
new = joblib.load('models/mid_face_model.pkl')
ref = json.load(open('models/snapshots/2026-05-01/mid_face_eval_metrics.json'))
print(f"Snapshot baseline: acc={ref['evaluation']['on_today_combined_results']['accuracy']:.3f}")
```

## Reproducibility envelope

- Python 3.10.6 on macOS arm64
- pyfaceau 1.3.11 (production-2026-05-01) — note: 1.3.12 release adds
  Cython build-deps but produces identical AU output
- pyclnf 0.3.3 (production-2026-05-01) — note: 0.3.4 release adds CMake
  fix + LFS migration but produces identical landmark output
- pyfhog 0.1.4
- pymtcnn 1.1.5
- See `requirements.lock` at FaceMirror repo root for the full pinned env

## Snapshot tag

`production-2026-05-01` (FaceMirror @ 1a61a4f5)
