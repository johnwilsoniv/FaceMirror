# Critical Bug: Geometric Features Source for AU Prediction

## Bug Summary

**AU predictions were incorrect because geometric features were computed from the wrong source.**

## The Bug

The pyfaceau pipeline was using `CalcParams` to compute `params_local` from raw landmark coordinates for geometric feature extraction. This produced incorrect values that broke AU predictions.

### Wrong Approach (BUG)
```python
# BUG: CalcParams on raw landmarks produces WRONG params_local
from pyfaceau.alignment.calc_params import CalcParams
calc_params = CalcParams(pdm)
params_global, params_local = calc_params.calc_params(landmarks.flatten())
geom_features = pdm.extract_geometric_features(params_local)
# Result: params_local range [-292, 772] - WRONG SCALE!
# Correlation with C++: only 0.35
```

### Correct Approach (FIX)
```python
# CORRECT: Use params_local from pyclnf optimization
landmarks, info = clnf.detect_and_fit(frame, return_params=True)
params_local = info['params'][6:]  # params[0:6] = global, params[6:] = local
geom_features = pdm.extract_geometric_features(params_local)
# Result: params_local range [-29, 32] - CORRECT!
# Correlation with C++: 0.9962
```

## Root Cause

OpenFace C++ computes `params_local` through its CLNF optimization process, which iteratively fits the PDM shape to the detected landmarks. The resulting `params_local` values represent the optimal PCA coefficients that describe the face shape deviation from the mean.

In Python, we were incorrectly calling `CalcParams` as a standalone function on raw landmarks, which:
1. Did not have the same initialization as CLNF
2. Produced parameters ~25x larger than expected
3. Broke the geometric feature distribution expected by AU SVR models

The correct approach is to use the `params_local` that pyclnf already computes during its optimization - accessible via `info['params'][6:]` when calling `detect_and_fit(frame, return_params=True)`.

## Impact

| Metric | Bug (CalcParams) | Fixed (pyclnf params) |
|--------|------------------|----------------------|
| params_local range | [-292, 772] | [-29, 32] |
| Correlation with C++ | 0.35 | 0.9962 |
| AUs matching C++ (within 0.5) | 6/17 | 15/17 |

## Affected Components

- `pyfaceau/pipeline.py` - Must use pyclnf params, not CalcParams
- Any code that calls `pdm.extract_geometric_features()` - Must pass pyclnf params_local

## Verification

The fix was verified by comparing Python and C++ outputs:
- C++ exports geometric features to `/tmp/cpp_geom_features_frame1.txt`
- Python params_local now correlates 0.9962 with C++ values
- 15/17 AUs now match C++ within 0.5 (only dynamic models AU17, AU23 differ due to running median)

## Date Discovered
December 13, 2024

## Related Files
- `AU_CORRELATION_INVESTIGATION_HANDOFF.md` - Full investigation details
- `compare_aligned_faces_v2.py` - Test script for alignment comparison
- `test_au_accuracy.py` - Test script for AU accuracy
