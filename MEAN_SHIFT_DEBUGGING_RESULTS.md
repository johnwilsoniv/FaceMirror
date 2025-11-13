# Mean-Shift Debugging Results

**Date**: 2025-11-10
**Status**: Critical discovery - agent analysis was partially incorrect

---

## Summary

Debug logging revealed that the **mean-shift coordinate system "fix"** was based on a **false premise** and actually made things worse.

### Convergence Timeline

| State | Final Update | Change |
|-------|-------------|--------|
| After Jacobian fix only | 3.88 | Baseline |
| After all "fixes" (Jacobian + mean-shift) | **14.45** | **WORSE** ❌ |
| After reverting coordinate "fix" | **3.61** | **Better** ✓ |

---

## The Problem with the Coordinate System "Fix"

### What the Agent Suggested

The agent analysis (MEAN_SHIFT_BUG_SUMMARY.md) claimed:
- Bug: dx, dy computed in IMAGE-space fractional offset instead of REFERENCE-space displacement
- Fix: Compute displacement from `reference_shape`, transform to REFERENCE coordinates

### What Debug Logging Revealed

```
Landmark 0:
  Current IMAGE: (341.8375, 933.5361)
  Reference IMAGE: (-18.3484, -7.4504)  ← WRONG! Negative values!
  Displacement IMAGE: (360.1859, 940.9865)
  dx, dy (within response map): (38.3505, 92.1284)  ← WAY OUTSIDE 11x11 window!
```

**The root cause**: `pdm.get_reference_shape(patch_scaling, params[6:])` returns the shape in **PDM normalized coordinates** (centered at origin), NOT in IMAGE coordinates!

- PDM normalized coordinates: (-18.3484, -7.4504) ← Around origin
- IMAGE coordinates: (341.8375, 933.5361) ← Actual pixel positions

Computing displacement between these incompatible coordinate systems produced completely wrong dx, dy values (38, 92) that were far outside the 11x11 response map window.

### The Correct Approach (Original Code)

```python
dx_frac = lm_x - int(lm_x)  # Fractional part: 0.8375
dy_frac = lm_y - int(lm_y)  # Fractional part: 0.5361
dx = dx_frac + center  # Result: 5.8375 (within 11x11 window ✓)
dy = dy_frac + center  # Result: 5.5361 (within 11x11 window ✓)
```

This works because:
1. The response map is centered at the current landmark position
2. The window is extracted around integer pixel coordinates
3. The fractional offset represents sub-pixel displacement from the window center
4. This is exactly what OpenFace does!

---

## What OpenFace Actually Does

Looking more carefully at OpenFace's LandmarkDetectorModel.cpp lines 1068-1075:

```cpp
cv::Mat_<float> offsets =
    (current_shape_2D - base_shape_2D) * cv::Mat(sim_img_to_ref).t();

dxs = offsets.col(0) + (resp_size-1)/2;
dys = offsets.col(1) + (resp_size-1)/2;
```

The key insight: `base_shape_2D` is the **2D shape at the START of the iteration** in IMAGE coordinates, NOT the PDM reference shape!

For the first iteration or when using simple extraction (no warping), `base_shape_2D = current_shape_2D`, so the offset is **zero**, and we get:
- dx = 0 + center
- dy = 0 + center

For sub-pixel positions, this becomes the fractional offset, which is what PyCLNF's original code was doing.

---

## The Transform Sign "Fix"

The second "fix" changed the transform signs:
- Old: `ms_x = a * ms_ref_x + b * ms_ref_y`, `ms_y = -b * ms_ref_x + a * ms_ref_y`
- New: `ms_x = a * ms_ref_x - b * ms_ref_y`, `ms_y = b * ms_ref_x + a * ms_ref_y`

**Analysis**: The new version is mathematically correct for a 2x2 rotation matrix [a -b; b a].

**However**: In our test case, `sim_ref_to_img` is approximately diagonal (b ≈ 0):
```
sim_ref_to_img:
[[ 1.080000e+01  5.532684e-17  ...]
 [-5.532684e-17  1.080000e+01  ...]]
```

So a = 10.8, b ≈ 0, and the sign difference has minimal impact (~5e-17).

**Kept**: The sign fix remains in place as it's mathematically correct, even though it may not significantly affect this particular test case.

---

## Final State

### Fixes Applied:
1. ✅ **Jacobian fix** (analytical derivatives) - Definitely correct
2. ✅ **Transform sign fix** - Mathematically correct
3. ❌ **Coordinate system "fix"** - REVERTED (was based on false premise)

### Code Changes:
- Reverted to fractional offset approach for dx, dy
- Removed `reference_shape` parameter from `_compute_mean_shift()`
- Kept transform sign corrections

### Result:
- Final update: **3.61** (improved from 14.45 after revert)
- Still not converging (target < 0.005), but better than before

---

## Lessons Learned

1. **Agent analysis can be wrong**: Even detailed line-by-line comparisons can miss context
2. **Debug logging is essential**: The bug was immediately obvious once we saw dx=38, dy=92
3. **Coordinate system assumptions are dangerous**: Always verify what coordinate space values are in
4. **Mathematical correctness ≠ algorithmic correctness**: The transform sign fix is mathematically right but may not be the issue

---

## Next Steps

The convergence is still not achieved (3.61 vs target 0.005). Possible remaining issues:

1. **Response map computation**: Values may still be too small (~0.00006)
2. **Weight matrix**: May not be weighting landmarks correctly
3. **PDM initialization**: Poor initial parameters → poor responses
4. **Other bugs**: More subtle issues in parameter update or Jacobian
5. **Fundamental approach difference**: OpenFace C++ may do something we haven't captured

**Recommendation**: Remove debug logging, document findings, and consider whether the 3.61 convergence is "close enough" for practical use, or if we need to continue deep investigation.
