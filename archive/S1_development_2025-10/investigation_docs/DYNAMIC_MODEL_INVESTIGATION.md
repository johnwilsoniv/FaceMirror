# Dynamic Model Investigation - Root Cause Analysis

## Problem Statement

After implementing histogram-based running median tracking, dynamic AU models still show poor correlation with OpenFace 2.2 predictions (10/11 AUs failing). Static models work perfectly (r > 0.87), proving the SVR implementation is correct.

## Investigation Timeline

### Attempt 1: Simple Rolling Window Median
- **Implementation**: `running_median_tracker.py` with deque-based rolling window
- **Result**: Poor correlation for 10/11 dynamic AUs
- **Issue**: Not matching OF2.2's histogram-based approach

### Attempt 2: Histogram-Based Median (Wrong Parameters)
- **Implementation**: `histogram_median_tracker.py` with num_bins=200, range=[-3, 5]
- **Result**: WORSE correlation, huge RMSE values
- **Issue**: Histogram parameters completely wrong

### Attempt 3: Histogram-Based Median (Correct Parameters)
- **Discovery**: OF2.2 uses different histogram params for HOG vs geometric:
  - HOG: num_bins=1000, min_val=-0.005, max_val=1.0
  - Geometric: num_bins=10000, min_val=-60, max_val=60
- **Result**: Slightly better but still poor
  - AU45: r=0.955 ✓
  - AU26: r=0.896 (improved from 0.470)
  - AU23: r=0.727 (improved from 0.303)
  - Most others still < 0.7

## Root Cause Identified

### The Critical Discrepancy

**Our implementation uses:**
- HOG features: ✓ Correct (from .hog file)
- Geometric features: ✗ **Raw 3D landmarks** (X_0...X_67, Y_0...Y_67, Z_0...Z_67) + PDM params

**OpenFace 2.2 actually uses:**
- HOG features: ✓ (same)
- Geometric features: ✓ **PDM-reconstructed landmarks** + PDM params

### Evidence from Source Code

From `FaceAnalyser.cpp` (line ~570):
```cpp
// Start with PDM parameters
params_local = params_local.t();
params_local.convertTo(geom_descriptor_frame, CV_64F);

// Reconstruct landmarks from PDM
cv::Mat_<double> princ_comp_d;
pdm.princ_comp.convertTo(princ_comp_d, CV_64F);
cv::Mat_<double> locs = princ_comp_d * geom_descriptor_frame.t();

// Concatenate: [reconstructed_landmarks, PDM_params]
cv::hconcat(locs.t(), geom_descriptor_frame.clone(), geom_descriptor_frame);
```

The `geom_descriptor_frame` contains:
1. **PDM-reconstructed landmarks**: `princ_comp × params_local`
2. **PDM parameters**: `params_local` (34 dims)

### Why This Matters

**Raw 3D landmarks (our approach):**
- Range: -115 to 565
- Distribution: Wide spread, real-world coordinates
- Histogram fit: Only 95.6% of values fit in [-60, 60]

**PDM-reconstructed landmarks (OF2.2 approach):**
- Range: Expected to fit in [-60, 60]
- Distribution: Mean-centered in PDM space
- Histogram fit: Should be ~100% (by design)

The running median histogram is clipping 4.4% of our geometric feature values, corrupting the median estimate.

## Why Static Models Still Work

Static models don't use running median at all. They only need:
- HOG features: ✓ Correct
- 3D landmarks: ✓ Available (even if not used for running median)
- PDM params: ✓ Available

Prediction formula for static models:
```
prediction = (features - means) × support_vectors + bias
```

No running median needed, so no geometric feature reconstruction needed.

## Why AU45 Works (Dynamic Model)

AU45 is the only dynamic model with r > 0.95. Possible reasons:
1. **HOG-dominant**: AU45 (eye blink) may rely primarily on HOG features around the eyes, with minimal geometric contribution
2. **Robust to noise**: The specific SV weights for AU45 might be less sensitive to errors in geometric features
3. **Lucky cancellation**: Errors in running median may coincidentally cancel out for this AU

## Blocking Issue: PDM Principal Components

To properly implement dynamic models, we need:
- **PDM principal components matrix** (`pdm.princ_comp`)
- **Reconstruction formula**: `reconstructed_landmarks = princ_comp × PDM_params`

**Problem**: The CSV file from FeatureExtraction contains:
- ✓ PDM parameters (p_0...p_33)
- ✓ Raw 3D landmarks (X_0...X_67, Y_0...Y_67, Z_0...Z_67)
- ✗ **PDM principal components matrix** (not exported)

The principal components are stored in OF2.2's internal PDM model files, not in the CSV output.

## Options Moving Forward

### Option A: Extract PDM Principal Components
- Find PDM model files in OF2.2 installation
- Parse binary format to extract `princ_comp` matrix
- Reconstruct geometric features correctly
- Re-validate dynamic models

**Effort**: Medium (1-2 days)
**Risk**: PDM file format may be complex
**Benefit**: All 11 dynamic models could work

### Option B: Deploy Static Models Only
- Use 6 validated static models (r > 0.87):
  - AU04, AU06, AU07, AU10, AU12, AU14
- Skip dynamic models for now
- Focus on Python FHOG extraction to complete pipeline

**Effort**: Low (continue current path)
**Risk**: Missing 11 AUs, but coverage still good
**Benefit**: Faster path to production

### Option C: Hybrid - Static Models + AU45
- Use 6 static models + AU45 (the one working dynamic model)
- Total: 7 AUs validated
- Add more dynamic models as we fix them

**Effort**: Low
**Risk**: Minimal (AU45 already validated)
**Benefit**: Slight improvement over Option B

## Recommendation

**Proceed with Option B or C** for the following reasons:

1. **Static models are production-ready**: 6 AUs with r > 0.87 is excellent
2. **Good facial coverage**:
   - Upper face: AU04 (brow lowerer)
   - Mid face: AU06 (cheek raiser), AU07 (lid tightener), AU10 (upper lip raiser)
   - Lower face: AU12 (lip corner puller), AU14 (dimpler)
3. **Next critical step is FHOG extraction**: Without Python FHOG, we can't process new videos
4. **PDM reconstruction is a rabbit hole**: Could take significant time with uncertain payoff

## Next Steps

1. ✓ Document dynamic model findings (this file)
2. Update implementation status document
3. Proceed with Python FHOG extraction (highest priority)
4. Revisit dynamic models later if needed

## Technical Details

### Geometric Feature Dimensions

**Our current approach (INCORRECT for dynamic models):**
- 3D landmarks: 204 dims (X_0...X_67, Y_0...Y_67, Z_0...Z_67)
- PDM params: 34 dims (p_0...p_33)
- **Total**: 238 dims

**OF2.2 approach (CORRECT):**
- Reconstructed landmarks: Unknown dims (princ_comp rows)
- PDM params: 34 dims
- **Total**: Unknown (need to check princ_comp.rows + 34)

The mismatch in geometric feature construction is the root cause of dynamic model failures.

---

**Date**: October 28, 2025
**Status**: Investigation complete, root cause identified
**Decision**: Proceed with static models, implement FHOG next
