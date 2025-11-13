# Mean-Shift Bug Summary - Critical Findings

**Date**: 2025-11-10
**Status**: 🔴 CRITICAL BUGS FOUND
**Impact**: Explains convergence failure (final update 3.88 vs target <0.005)

## Quick Summary

Found **TWO CRITICAL BUGS** in PyCLNF's mean-shift computation that prevent convergence:

1. **Bug #1**: dx, dy computed in wrong coordinate system (IMAGE vs REFERENCE)
2. **Bug #2**: Mean-shift transform uses incorrect matrix multiplication

Both bugs cause mean-shift vectors to be computed incorrectly, explaining why optimization diverges.

---

## Bug #1: Wrong Coordinate System for dx, dy

### Location
`pyclnf/core/optimizer.py`, lines 284-286 in `_compute_mean_shift()`

### The Problem

**PyCLNF computes**:
```python
lm_x, lm_y = landmarks_2d[landmark_idx]  # IMAGE coordinates
dx_frac = lm_x - int(lm_x)  # Fractional part in IMAGE space
dy_frac = lm_y - int(lm_y)

# Uses these IMAGE-space fractional offsets with REFERENCE-space response map
ms_ref_x, ms_ref_y = self._kde_mean_shift(
    response_map, dx_frac + center, dy_frac + center, a
)
```

**OpenFace computes** (`LandmarkDetectorModel.cpp` lines 1068-1075):
```cpp
// Compute displacement in IMAGE space
cv::Mat_<float> offsets =
    (current_shape_2D - base_shape_2D) * cv::Mat(sim_img_to_ref).t();

// Transform to REFERENCE space and add center
dxs = offsets.col(0) + (resp_size-1)/2;
dys = offsets.col(1) + (resp_size-1)/2;
```

### Why This Matters

The response map is computed in **REFERENCE coordinates** (after warping via `cv2.warpAffine`). The KDE kernel must be centered at the correct location **within the REFERENCE-space response map**.

- **PyCLNF**: Centers KDE using IMAGE-space fractional offset (wrong!)
- **OpenFace**: Centers KDE using REFERENCE-space displacement (correct!)

This causes the KDE kernel to weight **the wrong pixels** in the response map, producing incorrect mean-shift vectors.

### Example Impact

Consider:
- Landmark at IMAGE coordinates: `(100.5, 100.5)`
- Similarity transform scale: 2.0
- Displacement from reference: `(0.5, 0.5)` in IMAGE space

**PyCLNF uses**: `dx = 0.5 + center` (IMAGE-space fractional part)
**OpenFace uses**: `dx = 1.0 + center` (REFERENCE-space displacement = 0.5 × 2.0)

Result: KDE kernel centered at **wrong location**, off by a factor of the scale.

---

## Bug #2: Wrong Transform Back to Image Space

### Location
`pyclnf/core/optimizer.py`, lines 296-301 in `_compute_mean_shift()`

### The Problem

**PyCLNF computes**:
```python
a_mat = sim_ref_to_img[0, 0]
b_mat = sim_ref_to_img[1, 0]

ms_x = a_mat * ms_ref_x + b_mat * ms_ref_y   # ❌ Wrong sign!
ms_y = -b_mat * ms_ref_x + a_mat * ms_ref_y  # ❌ Wrong sign!
```

**OpenFace computes** (`LandmarkDetectorModel.cpp` lines 1080-1083):
```cpp
// Matrix multiplication: mean_shifts_2D * sim_ref_to_img.T
mean_shifts_2D = mean_shifts_2D * cv::Mat(sim_ref_to_img).t();
```

Which expands to:
```
[ms_img_x]   [a  -b] [ms_ref_x]
[ms_img_y] = [b   a] [ms_ref_y]
```

Or:
```
ms_img_x = a * ms_ref_x - b * ms_ref_y  (Note: MINUS)
ms_img_y = b * ms_ref_x + a * ms_ref_y  (Note: PLUS)
```

### Why This Matters

Even if the mean-shift is computed correctly in REFERENCE space, it must be transformed back to IMAGE space using proper 2D rotation matrix multiplication. The incorrect signs cause the mean-shift vector to be applied in the **wrong direction**.

---

## How These Bugs Cause Convergence Failure

### Normal Convergence (OpenFace)
1. Compute response map in REFERENCE coordinates ✓
2. Center KDE at correct REFERENCE-space position ✓
3. Compute mean-shift in REFERENCE coordinates ✓
4. Transform mean-shift to IMAGE coordinates correctly ✓
5. Apply parameter updates → shape moves toward target ✓
6. Repeat until convergence (final update <0.005) ✓

### Broken Convergence (PyCLNF with bugs)
1. Compute response map in REFERENCE coordinates ✓
2. Center KDE at wrong position (IMAGE-space fractional offset) ❌
3. Compute mean-shift from **wrong region** of response map ❌
4. Transform mean-shift to IMAGE coordinates **incorrectly** ❌
5. Apply parameter updates → shape moves in **wrong direction** ❌
6. Never converges (final update = 3.88) ❌

---

## The Fixes

### Fix #1: Compute dx, dy in REFERENCE Coordinates

**Current code** (`optimizer.py` lines 284-291):
```python
dx_frac = lm_x - int(lm_x)
dy_frac = lm_y - int(lm_y)

ms_ref_x, ms_ref_y = self._kde_mean_shift(
    response_map, dx_frac + center, dy_frac + center, a
)
```

**Fixed code**:
```python
# Compute displacement from reference shape in IMAGE coordinates
ref_lm_x, ref_lm_y = reference_shape[landmark_idx]
displacement_img_x = lm_x - ref_lm_x
displacement_img_y = lm_y - ref_lm_y

# Transform to REFERENCE coordinates (rotation + scale only)
a_img_to_ref = sim_img_to_ref[0, 0]
b_img_to_ref = sim_img_to_ref[1, 0]
displacement_ref_x = a_img_to_ref * displacement_img_x - b_img_to_ref * displacement_img_y
displacement_ref_y = b_img_to_ref * displacement_img_x + a_img_to_ref * displacement_img_y

# Position within response map
dx = displacement_ref_x + center
dy = displacement_ref_y + center

ms_ref_x, ms_ref_y = self._kde_mean_shift(response_map, dx, dy, a)
```

**Required changes**:
1. Add `reference_shape` parameter to `_compute_mean_shift()`
2. Compute displacement from reference shape
3. Transform displacement using `sim_img_to_ref` rotation/scale matrix

### Fix #2: Correct Mean-Shift Transform

**Current code** (`optimizer.py` lines 296-301):
```python
ms_x = a_mat * ms_ref_x + b_mat * ms_ref_y
ms_y = -b_mat * ms_ref_x + a_mat * ms_ref_y
```

**Fixed code**:
```python
ms_x = a_mat * ms_ref_x - b_mat * ms_ref_y  # Change + to -
ms_y = b_mat * ms_ref_x + a_mat * ms_ref_y  # Change - to +
```

Or more clearly:
```python
# Apply 2x2 rotation/scale matrix (no translation)
rotation_scale = sim_ref_to_img[:2, :2]
ms_img = rotation_scale @ np.array([ms_ref_x, ms_ref_y])
ms_x, ms_y = ms_img[0], ms_img[1]
```

---

## Reference Shape Computation

For Fix #1, we need `reference_shape`. Looking at the code:

**PyCLNF** (`optimizer.py` line 134):
```python
reference_shape = pdm.get_reference_shape(patch_scaling, params[6:])
```

This is already computed in the `optimize()` method at line 134, but it's not passed to `_compute_mean_shift()`.

**OpenFace**: Uses `base_shape = current_shape` (the shape at the start of optimization).

**Solution**: Pass `reference_shape` to `_compute_mean_shift()`:

```python
# In optimize() method, line 145:
mean_shift = self._compute_mean_shift(
    landmarks_2d, patch_experts, image, pdm, window_size,
    sim_img_to_ref, sim_ref_to_img, sigma_components,
    reference_shape  # ADD THIS
)
```

---

## Expected Impact

After applying both fixes:

### Before (Broken)
- Final update: **3.88** (never converges)
- Mean-shift vectors: Computed from wrong regions, applied in wrong directions
- Result: Optimization diverges or gets stuck

### After (Fixed)
- Final update: **<0.005** (converges successfully)
- Mean-shift vectors: Computed correctly, applied correctly
- Result: Shape converges to correct landmark positions

---

## Confidence Level

**VERY HIGH (95%+)**

The bugs are:
1. **Clear**: Coordinate system mismatch and incorrect matrix multiplication
2. **Well-documented**: OpenFace code shows exactly what should be done
3. **Directly explanatory**: Explains the observed convergence failure (3.88 vs <0.005)

The previous investigation concluded the algorithm was "algorithmically identical" - this was **partially true** (the KDE computation itself is correct), but **missed these critical coordinate system bugs**.

---

## Implementation Priority

**IMMEDIATE**: These bugs are blockers for CLNF convergence. All other optimizations are meaningless until these are fixed.

**Order**:
1. Fix Bug #2 first (simpler - just change two signs)
2. Fix Bug #1 second (requires passing reference_shape parameter)
3. Run convergence tests to verify fixes

---

## Full Implementation

See `/Users/johnwilsoniv/repo/fea_tool/MEAN_SHIFT_COMPARISON.md` for:
- Complete line-by-line comparison
- Detailed mathematical analysis
- Full fixed code for `_compute_mean_shift()` method
- Verification strategy and test cases

---

**Next Steps**: Apply fixes to `pyclnf/core/optimizer.py` and test convergence.
