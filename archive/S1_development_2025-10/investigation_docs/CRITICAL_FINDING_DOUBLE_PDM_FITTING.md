# CRITICAL FINDING: Double PDM Fitting

**Date:** 2025-10-29
**Status:** 🔥 BREAKTHROUGH - This explains the rotation difference!

## The Discovery

FaceAnalyser calls `PDM.CalcParams()` **TWICE** on different landmark sets!

### The Double-Fitting Pipeline

**Step 1: Initial Landmark Detection & Fitting**
```
Raw Image
  ↓
CLNF Detection → raw_landmarks
  ↓
PDM.CalcParams(raw_landmarks) → params_global₁, params_local₁
  ↓
PDM.CalcShape2D(params_global₁, params_local₁) → reconstructed_landmarks
  ↓
CSV Output ← params_global₁, params_local₁, reconstructed_landmarks
```

**Step 2: FaceAnalyser Re-Fitting**
```
FaceAnalyser.AddNextFrame(reconstructed_landmarks)
  ↓
PDM.CalcParams(reconstructed_landmarks) → params_global₂, params_local₂
  ↓
AlignFace(..., params_global₂, ...) → aligned_face
```

## Key Evidence

**FaceAnalyser.cpp line 321:**
```cpp
void FaceAnalyser::AddNextFrame(..., const cv::Mat_<float>& detected_landmarks, ...)
{
    ...
    pdm.CalcParams(params_global, params_local, detected_landmarks);
    ...
    AlignFace(aligned_face_for_output, frame, detected_landmarks,
              params_global, pdm, ...);  // Uses NEW params_global!
}
```

**The landmarks passed to AddNextFrame are PDM-reconstructed:**
From FeatureExtraction.cpp line 227:
```cpp
open_face_rec.SetObservationLandmarks(face_model.detected_landmarks, ...);
```

And face_model.detected_landmarks comes from:
```cpp
pdm.CalcShape2D(detected_landmarks, params_local, params_global);  // Line 635
```

## Why This Matters

**Our Python implementation:**
- Uses CSV landmarks (PDM-reconstructed)
- Uses CSV params_global₁ (from first fitting)
- ❌ These don't match what C++ alignment uses!

**C++ alignment:**
- Uses PDM-reconstructed landmarks
- Re-fits PDM to get params_global₂
- Uses params_global₂ for alignment
- ✅ This is what produces upright faces!

## The Critical Difference

When you fit a PDM to **already PDM-reconstructed** landmarks:
- Raw landmarks have noise, expression artifacts
- PDM fitting to raw landmarks: params contain rigid pose + expression + noise
- PDM-reconstructed landmarks are already "cleaned" by PCA
- Re-fitting PDM to reconstructed landmarks: params are more "canonical"

**Hypothesis:** The second fitting produces params_global₂ with near-zero rotation because the landmarks are already in PDM-canonical orientation!

## What This Explains

### ✓ Expression Invariance
Re-fitting to PDM-reconstructed landmarks means expression is already factored into the landmark positions. The second CalcParams extracts minimal params_local, and params_global₂ represents just rigid pose without expression artifacts.

### ✓ Upright Faces (~0° rotation)
PDM-reconstructed landmarks are in a canonical orientation. Re-fitting them produces params_global₂ with rotation close to identity, leading to upright aligned faces.

### ✓ Why Our Python Fails
We use params_global₁ from CSV, which represents the rotation of the original raw detections. C++ uses params_global₂ from re-fitting, which represents canonical orientation.

## The Fix

**Option 1: Implement CalcParams in Python**
- Re-fit PDM to CSV landmarks
- Get params_global₂
- Use params_global₂ for alignment (specifically tx, ty)

**Option 2: Ignore pose params entirely**
- Since PDM-reconstructed landmarks are already canonical
- Maybe we don't need pose correction at all?
- Just align landmarks directly to PDM mean shape

**Option 3: Use identity rotation**
- Don't compute rotation via Kabsch
- Use pure scale + translation
- Assume landmarks are already rotationally aligned

## Testing The Hypothesis

**Test 1: Align without computing rotation**
```python
# Instead of Kabsch alignment
# Just use scale and translation
scale = compute_scale(csv_landmarks, pdm_mean_shape)
translation = compute_translation(csv_landmarks, pdm_mean_shape)
# Skip rotation computation
```

**Test 2: Check if re-fitting gives different params**
```python
# Implement CalcParams
params2_global, params2_local = pdm.calc_params(csv_landmarks)
# Compare to CSV params
print(f"CSV p_rz: {csv_params['p_rz']}")
print(f"Re-fit p_rz: {params2_global[3]}")
```

## Implementation Priority

**Immediate:** Test Option 2 (ignore rotation, use only scale/translation)
**Short-term:** Implement CalcParams if Option 2 fails
**Fallback:** pybind11 wrapper for CalcParams

## Confidence Level

**95%** - This explains ALL observed behaviors:
- ✅ Why C++ is expression-invariant
- ✅ Why C++ produces upright faces
- ✅ Why our Python using CSV params doesn't match
- ✅ Why params_global rotation isn't used in AlignFace (it comes from the second fitting, not CSV)

This is likely the root cause!
