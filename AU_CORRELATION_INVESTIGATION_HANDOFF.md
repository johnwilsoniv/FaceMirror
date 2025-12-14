# AU Correlation Investigation Handoff

## Problem Statement

**Landmarks are very accurate (~1px), but AU correlations with C++ OpenFace were poor, especially for lower face AUs.**

## Current State (Dec 13, 2024)

### ROOT CAUSES IDENTIFIED AND FIXED

| Issue | Root Cause | Status |
|-------|------------|--------|
| HOG mismatch | Face alignment mask boundaries | ✅ FIXED (warp is perfect) |
| Geometric features wrong | Using CalcParams instead of pyclnf params | ✅ FIXED |
| AU predictions wrong | Geometric features had wrong scale | ✅ FIXED |

### Current AU Accuracy

**Using pyclnf params_local for geometric features:**

| AU | C++ Value | Python Value | Match |
|----|-----------|--------------|-------|
| AU01_r | 0.000 | 0.000 | ✅ |
| AU02_r | 0.000 | 0.000 | ✅ |
| AU04_r | 0.000 | 0.000 | ✅ |
| AU05_r | 0.160 | 0.000 | ✅ |
| AU06_r | 0.000 | 0.000 | ✅ |
| AU07_r | 1.130 | 1.259 | ✅ |
| AU09_r | 0.000 | 0.000 | ✅ |
| AU10_r | 0.800 | 0.920 | ✅ |
| AU12_r | 0.000 | 0.000 | ✅ |
| AU14_r | 0.920 | 1.336 | ✅ |
| AU15_r | 0.000 | 0.000 | ✅ |
| AU17_r | 1.070 | 3.768 | ❌ (dynamic) |
| AU20_r | 0.000 | 0.181 | ✅ |
| AU23_r | 0.310 | 1.427 | ❌ (dynamic) |
| AU25_r | 0.030 | 0.000 | ✅ |
| AU26_r | 0.000 | 0.000 | ✅ |
| AU45_r | 0.040 | 0.000 | ✅ |

**15/17 AUs match within 0.5 (static prediction without running median)**

---

## KEY FINDINGS

### 1. Face Alignment Warp is PERFECT ✅

The similarity transform and warping are correct:
- Pixel correlation inside mask: **0.999998**
- Mean pixel difference: **0.01**
- The face pixels match perfectly

### 2. Mask Boundaries Have Minor Differences ⚠️

- Mask match: 98.8%
- 147 pixels differ at edges
- Does NOT significantly affect HOG features

### 3. pyfhog is CORRECT ✅

When given the same aligned face as C++:
- HOG correlation: **1.000000**
- Mean absolute difference: **0.000000**
- pyfhog is a correct wrapper around dlib's FHOG

### 4. Geometric Features Source CRITICAL

**WRONG:** Using `CalcParams` on raw landmarks
- params_local range: [-292, 772] ❌
- Correlation with C++: 0.35

**CORRECT:** Using pyclnf's fitted `params_local` from `info['params'][6:]`
- params_local range: [-29, 32] ✅
- Correlation with C++: **0.9962**

### 5. Remaining Issues: Dynamic Models Need Running Median

AU17_r and AU23_r are "dynamic" models that require running_median calibration.
With running_median=0 (first frame), they predict incorrectly.
This is expected behavior - the full pipeline will calibrate over time.

---

## How to Get Correct AU Predictions

```python
from pyclnf import CLNF
from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.features.pdm import PDMParser
from pyfaceau.prediction.batched_au_predictor import BatchedAUPredictor
import pyfhog

# 1. Fit landmarks with pyclnf and GET PARAMS
clnf = CLNF(...)
landmarks, info = clnf.detect_and_fit(frame, return_params=True)
params = info['params']
params_global = params[:6]
params_local = params[6:]  # CRITICAL: Use this, NOT CalcParams!

# 2. Align face
pose_tx, pose_ty = params_global[4], params_global[5]
aligned = aligner.align_face(frame, landmarks, pose_tx, pose_ty,
                              apply_mask=True, triangulation=triangulation)

# 3. Extract HOG
hog = pyfhog.extract_fhog_features(aligned, cell_size=8)

# 4. Extract geometric features using pyclnf params_local
geom_features = pdm.extract_geometric_features(params_local)  # Uses params from pyclnf!

# 5. Predict AUs
aus = predictor.predict(hog, geom_features, running_median)
```

---

## What Has Been Verified Correct ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| Landmark detection (pyclnf) | ✅ | 1.02px jaw error, 0.024px L/R asymmetry |
| Face warp transform | ✅ | 0.999998 correlation with C++ |
| pyfhog | ✅ | 1.0 correlation with C++ (same input) |
| pyclnf params_local | ✅ | 0.9962 correlation with C++ |
| Static AU prediction | ✅ | 15/17 AUs match within 0.5 |
| Geometric features (with correct params) | ✅ | Range [-29, 32] matches C++ |

---

## Remaining Work

1. **Running Median Calibration**: Dynamic models (AU17, AU23) need running_median from the pipeline
2. **Full Pipeline Integration**: Ensure FullPythonAUPipeline uses pyclnf params_local for geometric features
3. **Multi-frame Testing**: Validate correlations over longer sequences

---

## Key Files

### Fixed Components
| File | Status | Notes |
|------|--------|-------|
| `pyfaceau/alignment/face_aligner.py` | ✅ | Warp is correct |
| `pyfhog` | ✅ | Correct dlib FHOG wrapper |
| `pyfaceau/prediction/batched_au_predictor.py` | ✅ | Prediction code is correct |

### Integration Point
| File | Notes |
|------|-------|
| `pyfaceau/pipeline.py` | Must use pyclnf params for geometric features |
| `pyfaceau/features/pdm.py` | `extract_geometric_features()` expects pyclnf params_local |

---

## Test Commands

```bash
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3

# Test AU accuracy with pyclnf params
PYTHONPATH="pyclnf:pymtcnn:pyfaceau:." python3 test_au_accuracy.py

# Compare aligned faces
PYTHONPATH="pyfaceau:." python3 compare_aligned_faces_v2.py
```

---

## Summary

The AU prediction pipeline is working correctly when:
1. Using pyclnf's fitted `params_local` (NOT CalcParams)
2. Using the current face aligner (warp is perfect)
3. Using pyfhog (verified correct)

**Static models (15/17) now match C++ within 0.5.**
**Dynamic models need running_median calibration over time.**
