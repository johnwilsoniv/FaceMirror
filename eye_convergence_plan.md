# Eye Refinement Convergence Analysis

## Executive Summary

**Current Status:** ✅ **FIX IMPLEMENTED AND WORKING**
**Previous Issue:** Python eye refinement DEGRADED results (2.11px → 3.13px for eyes)
**After Fix:** Python eye refinement IMPROVES results (2.11px → 1.59px for eyes)
**Improvement:** 0.52px reduction in eye landmark error
**Target:** <0.5px error to match C++ quality (still in progress)
**Root Cause:** Missing `sim_ref_to_img` transformation for patch extraction and mean-shift scaling

---

## ✅ Fix Results (November 22, 2024)

### Before Fix
```
Without eye refinement: Eyes = 2.11px
With eye refinement:    Eyes = 3.13px
Difference:             +1.02px (WORSE)
```

### After Fix
```
Without eye refinement: Eyes = 2.11px
With eye refinement:    Eyes = 1.59px
Difference:             -0.52px (BETTER!)
```

### Individual Landmarks Improved
| Landmark | No-Refine | With-Refine | Change |
|----------|-----------|-------------|--------|
| 36 R outer | 2.67px | 1.55px | -1.12px |
| 37 R up-out | 3.33px | 2.71px | -0.62px |
| 38 R up-in | 2.82px | 1.66px | -1.15px |
| 40 R lo-in | 1.69px | 0.94px | -0.75px |
| 45 L outer | 2.09px | 0.84px | -1.25px |

### What Was Fixed
1. **Added `align_shapes_with_scale` function** - Kabsch algorithm for computing similarity transform
2. **Compute `sim_ref_to_img` before response maps** - Aligns current shape to reference
3. **Use a1/b1 in patch extraction** - Now extracts rotation-normalized patches (~3.3x scale)
4. **Transform mean-shifts to image space** - Multiplies by `sim_ref_to_img.T`

### Key Values Now
```
sim_ref_to_img: a1=3.257363, b1=-0.395850
```

Previously was using `a1=1.0, b1=0.0` (identity).

---

## Critical Finding: Response Map Values

### C++ Eye_8 Response Map (5x5 window)
```
      0.0351   0.0679   0.1699   0.2043   0.0887
      0.0104   0.0996   0.2220   0.1692   0.1204
      0.0342   0.1085   0.2506   0.2099   0.1755
      0.0666   0.1188   0.1958   0.1116   0.0417
      0.1024   0.1055   0.0663   0.0283   0.0044
```
- **Range:** 0.004 to 0.250
- **Clear peak** at row 2, col 2 (value 0.2506)
- **Mean:** 0.112

### Python Eye_8 Response Map (3x3 window)
```
  0.0034 0.0026 0.0017
  0.0045 0.0045 0.0033
  0.0029 0.0050 0.0040
```
- **Range:** 0.0017 to 0.0050
- **Nearly uniform** - no clear peak
- **Mean:** ~0.004

**The Python response maps have ~50x smaller magnitude and essentially NO structure!**

---

## Implementation Differences Map

### 1. Patch Extraction (warpAffine transformation)

| Aspect | C++ | Python | Impact |
|--------|-----|--------|--------|
| a1, b1 values | `sim_ref_to_img(0,0)`, `-sim_ref_to_img(0,1)` | Hardcoded `1.0, 0.0` | **HIGH** - C++ uses pose-normalized extraction |
| Transformation | Includes rotation + scale | Identity (no rotation) | Rotation affects patch content |
| WARP_INVERSE_MAP | Yes | Yes | OK |

**Issue:** C++ extracts rotation-normalized patches based on `sim_ref_to_img = AlignShapesWithScale(image_shape, reference_shape)`. Python uses identity transformation.

### 2. Mean-Shift Transformation

| Aspect | C++ | Python | Impact |
|--------|-----|--------|--------|
| Reference→Image transform | `mean_shifts_2D * sim_ref_to_img.t()` | None | **HIGH** - C++ transforms mean-shifts to image space |
| Scale factor | ~11.3x (from sim_ref_to_img) | 1.0 | Mean-shifts need scaling |

**Issue:** C++ transforms mean-shifts from reference coordinate space to image coordinate space. Python skips this transformation entirely.

### 3. CCNF Response Computation

| Aspect | C++ | Python | Status |
|--------|-----|--------|--------|
| Neural network weights | Loaded from model | Loaded from model | Need to verify |
| Sigmoid normalization | Yes | Need to check | **SUSPECT** |
| Sigma matrix computation | ComputeSigmas | May be missing | **SUSPECT** |

### 4. Algorithm Parameters

| Parameter | C++ | Python | Match? |
|-----------|-----|--------|--------|
| window_sizes | [3, 5, 9] | [3, 5, 9] | ✓ |
| reg_factor | 0.5 | 0.5 | ✓ (fixed) |
| sigma | 1.0 | 1.0 | ✓ |
| max_iterations | 5 | 5 | ✓ |
| damping | 0.5 | 0.5 | ✓ |

### 5. Data Layout

| Aspect | C++ | Python | Status |
|--------|-----|--------|--------|
| Mean-shift storage | [all_X; all_Y] (n rows X, n rows Y) | Interleaved [x0,y0,x1,y1,...] | OK - internally consistent |
| Jacobian layout | [X rows; Y rows] | Interleaved | OK - matches mean-shift |

---

## Initialization Comparison

### Input Landmarks (from C++ main model output)
```
C++ Eye_8: (391.1343, 827.5673)
C++ Eye_10: (409.2716, 807.8469)
C++ Eye_12: (436.2549, 806.1482)
C++ Eye_14: (461.1137, 823.6552)
C++ Eye_16: (437.5308, 830.7004)
C++ Eye_18: (410.2859, 833.9506)
```

### Fitted Global Parameters

| Parameter | C++ | Python | Difference |
|-----------|-----|--------|------------|
| scale | 3.371203 | 3.262359 | -0.109 |
| rx | -0.118319 | -0.071219 | +0.047 |
| ry | 0.176101 | 0.148709 | -0.027 |
| rz | -0.099366 | -0.089906 | +0.009 |
| tx | 425.031647 | 433.543761 | +8.51 |
| ty | 820.112061 | 819.481939 | -0.63 |

**Note:** Python initialization uses different input landmarks (from Python's own main model output).

### Initial Eye Landmark 8

| | C++ | Python | Difference |
|-|-----|--------|------------|
| x | 391.3150 | 400.0273 | +8.71 |
| y | 827.5953 | 824.9067 | -2.69 |

---

## Mean-Shift Comparison (First Iteration)

### Eye_8 Mean-Shift

| | C++ (ws=3) | Python (ws=3) | Status |
|-|------------|---------------|--------|
| ms_x | -0.176587 | -0.050792 | **3.5x smaller** |
| ms_y | +0.311911 | +0.113024 | **2.8x smaller** |
| Direction | LEFT, DOWN | LEFT, DOWN | ✓ |

**Issue:** Python mean-shifts are 3-4x smaller in magnitude. This is consistent with the ~50x smaller response map values being partially compensated by the weighted mean computation.

---

## Root Cause Analysis

### ✅ CONFIRMED: Model Parameters Are Correct

After detailed debug comparison:
- C++ landmark 21: alpha=15.060042, bias=-21.386976, norm_w=84.325770
- Python landmark 21: alpha=15.060042, bias=-21.386976, norm_w=84.325770 ✓
- Computation formula matches when using same inputs ✓

**The exported CCNF model weights and biases are correct.**

### ✅ CONFIRMED: Computation Formula Is Correct

When using C++ normalized input values with Python weights:
- C++ sigmoid_input: 8.167937
- Python (with C++ input): 8.167938 ✓

**The neuron response computation is correct.**

### 🔴 ROOT CAUSE IDENTIFIED: Missing sim_ref_to_img Transformation

**The critical difference is in patch extraction and mean-shift transformation.**

#### C++ Implementation (Patch_experts.cpp lines 156-164)
```cpp
// Compute similarity transform from current shape to reference shape
sim_img_to_ref = AlignShapesWithScale(image_shape_2D, reference_shape_2D);
sim_ref_to_img = sim_img_to_ref.inv(cv::DECOMP_LU);

float a1 = sim_ref_to_img(0, 0);  // ~3.3 for eye model
float b1 = -sim_ref_to_img(0, 1); // Small rotation component

// Used in warpAffine for patch extraction
cv::Mat sim = (cv::Mat_<float>(2, 3) <<
    a1, -b1, landmark_x - a1*half_aoi + b1*half_aoi,
    b1,  a1, landmark_y - a1*half_aoi - b1*half_aoi);
```

#### Python Implementation (eye_patch_expert.py lines 720-730)
```python
# WRONG: Uses identity transform
a1 = 1.0  # Should be ~3.3 (scale factor)
b1 = 0.0  # Should account for rotation
```

#### Impact

1. **Patch extraction is wrong**: Python extracts 1:1 scale patches, but C++ extracts patches at reference scale (~3.3x larger area mapped to same patch size)

2. **Mean-shifts are not transformed**: C++ transforms mean-shifts back to image space using `sim_ref_to_img.t()`, Python doesn't

This causes:
- Completely different pixel values in the extracted patches
- CCNF sees unrecognized patterns and produces garbage responses
- Refinement moves landmarks in wrong directions

### Input Landmark Comparison

C++ and Python start with different main model landmarks (~5px offset), but this is secondary - the eye refinement should still improve relative to its own starting point. The primary issue is the patch extraction/transformation.

**C++ Input (to eye model):**
```
Eye_8: (396.22, 827.04)
Eye_10: (414.41, 808.01)
```

**Python Input (to eye model):**
```
Eye_8: (391.12, 827.56)
Eye_10: (409.27, 807.78)
```

### Raw Patch Value Comparison

**C++ area_of_interest (first 5x5):**
```
  41.8 57.6 43.7 34.8 30.1
  98.2 63.7 32.9 92.9 45.9
  122.0 62.0 64.6 65.3 70.6
```

**Python area_of_interest (first 5x5):**
```
  49.0 49.0 54.0 59.0 62.0
  53.0 54.0 58.0 61.0 63.0
  56.0 56.0 60.0 63.0 66.0
```

**Completely different pixel values** - confirming wrong patch extraction.

---

## Fix Implementation Plan

### Phase 1: Implement sim_ref_to_img Transformation (CRITICAL)

**File:** `pyclnf/core/eye_patch_expert.py`

1. **Compute AlignShapesWithScale**
   - Get current 28-point eye landmarks
   - Compute reference shape at `patch_scaling[scale]` (scale=1.0 for eye model)
   - Compute similarity transform to align current→reference

2. **Extract rotation-normalized patches**
   - Use computed a1, b1 in warpAffine matrix
   - This will extract larger image area mapped to patch size

3. **Transform mean-shifts to image space**
   - After computing mean-shifts in reference space
   - Multiply by sim_ref_to_img.T

### Implementation Details

```python
def _compute_sim_transforms(self, eye_landmarks, params, pdm, side):
    """Compute similarity transform from current shape to reference shape."""
    # Current shape in image coordinates
    image_shape = eye_landmarks.flatten()  # (56,) = 28*2

    # Reference shape at patch_scaling (1.0 for eye model)
    ref_params = params.copy()
    ref_params[0] = 1.0  # scale = patch_scaling
    ref_params[1:4] = 0  # no rotation
    ref_params[4:6] = 0  # no translation
    reference_shape = pdm.params_to_landmarks_2d(ref_params).flatten()

    # Compute AlignShapesWithScale (Procrustes alignment)
    # Returns 2x2 matrix [a, -b; b, a] where a=scale*cos, b=scale*sin
    sim_img_to_ref = align_shapes_with_scale(image_shape, reference_shape)
    sim_ref_to_img = np.linalg.inv(sim_img_to_ref)

    return sim_ref_to_img, sim_img_to_ref
```

### Phase 2: Update Patch Extraction

**Location:** `_compute_eye_response_maps()` around line 720

```python
# Get transforms
a1 = sim_ref_to_img[0, 0]
b1 = -sim_ref_to_img[0, 1]

# Create transformation matrix with scale and rotation
tx = x - a1 * half_aoi + b1 * half_aoi
ty = y - a1 * half_aoi - b1 * half_aoi
sim = np.array([[a1, -b1, tx],
               [b1, a1, ty]], dtype=np.float32)
```

### Phase 3: Update Mean-Shift Transformation

**Location:** After computing mean-shifts in `_mean_shift_update()`

```python
# Transform mean-shifts from reference to image space
mean_shifts_2D = mean_shifts.reshape(-1, 2)  # (n, 2)
mean_shifts_2D = mean_shifts_2D @ sim_ref_to_img.T
mean_shifts = mean_shifts_2D.flatten()
```

### Phase 4: Verification

1. **Compare a1, b1 values with C++**
   - Should be ~3.3 for scale component
   - Small rotation component

2. **Compare extracted patch pixel values**
   - Should now match C++ area_of_interest

3. **Verify response maps have structure**
   - Values in 0.01-0.5 range
   - Clear peak

4. **Confirm refinement improves results**

---

## Test Procedure

```bash
# Run eye refinement test
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
env PYTHONPATH="pyclnf:pymtcnn:." python3 test_eye_refinement.py
```

### Success Criteria
- [ ] Response map values in same range as C++ (0.001 to 0.5)
- [ ] Clear peak structure in response maps
- [ ] Mean-shift magnitudes similar to C++ (within 2x)
- [ ] Eye refinement IMPROVES results (reduces error)
- [ ] Final eye error < 1.0px

---

## Files to Investigate

1. **pyclnf/core/eye_patch_expert.py**
   - `EyeCCNFPatchExpert.compute_response()` - lines 87-191
   - `_compute_ccnf_response_map()` - lines 726-776
   - `_compute_eye_response_maps()` - lines 641-704

2. **C++ Reference**
   - `OpenFace/lib/local/LandmarkDetector/src/CCNF_patch_expert.cpp`
   - `OpenFace/lib/local/LandmarkDetector/src/Patch_experts.cpp`

3. **Debug Files**
   - `/tmp/cpp_ccnf_neuron_debug.txt`
   - `/tmp/python_eye_response_maps.txt`
   - `/tmp/cpp_eye_response_maps.txt`

---

## Estimated Impact

If we fix the response map computation:
- Response maps will have proper structure (10-50x improvement)
- Mean-shifts will have correct magnitude and direction
- Eye refinement will IMPROVE results instead of degrading them
- Expected final error: <1.0px (vs current 3.13px)
