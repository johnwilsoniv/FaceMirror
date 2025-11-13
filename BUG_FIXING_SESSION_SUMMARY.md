# PyCLNF Bug-Fixing Session - Complete Summary

**Date**: 2025-11-10
**Session Goal**: Fix PyCLNF convergence failure through systematic debugging
**Initial State**: Final update magnitude = 3.88 (target < 0.005)
**Current State**: Final update magnitude = 14.45 (WORSE after fixes)

---

## Bugs Found and Fixed

### Bug #1: Jacobian Rotation Derivatives (CRITICAL) ✅ FIXED

**Location**: `pyclnf/core/pdm.py` lines 227-255

**Problem**: Used numerical differentiation instead of analytical formulas
- Numerical: `dR = (R_plus - R) / h` with h=1e-7
- Analytical (correct): `∂x/∂wx = s * (Y*r13 - Z*r12)`, etc.

**Impact**: Mathematical inconsistency between Jacobian and parameter update rule

**Fix Applied**: Replaced numerical differentiation with OpenFace's analytical rotation derivative formulas

**Verification**: Analytical derivatives match numerical to 2.5e-09 relative error ✓

---

### Bug #2: Mean-Shift Coordinate System (CRITICAL) ✅ FIXED

**Location**: `pyclnf/core/optimizer.py` lines 284-294

**Problem**: dx, dy computed in IMAGE coordinates instead of REFERENCE coordinates
- Old: Used `dx = lm_x - int(lm_x)` (IMAGE-space fractional offset)
- New: Computes displacement from reference shape, transforms to REFERENCE space

**Impact**: KDE kernel centered at wrong location → incorrect mean-shift vectors

**Fix Applied**:
```python
# Compute displacement from reference shape in IMAGE coordinates
ref_lm_x, ref_lm_y = reference_shape[landmark_idx]
displacement_img_x = lm_x - ref_lm_x
displacement_img_y = lm_y - ref_lm_y

# Transform to REFERENCE coordinates
a_img_to_ref = sim_img_to_ref[0, 0]
b_img_to_ref = sim_img_to_ref[1, 0]
displacement_ref_x = a_img_to_ref * displacement_img_x - b_img_to_ref * displacement_img_y
displacement_ref_y = b_img_to_ref * displacement_img_x + a_img_to_ref * displacement_img_y

dx = displacement_ref_x + center
dy = displacement_ref_y + center
```

---

### Bug #3: Mean-Shift Transform Signs (CRITICAL) ✅ FIXED

**Location**: `pyclnf/core/optimizer.py` lines 302-303

**Problem**: Incorrect matrix multiplication signs when transforming back to IMAGE space
- Old: `ms_x = a * ms_ref_x + b * ms_ref_y` (WRONG)
- Old: `ms_y = -b * ms_ref_x + a * ms_ref_y` (WRONG)
- New: `ms_x = a * ms_ref_x - b * ms_ref_y` (CORRECT)
- New: `ms_y = b * ms_ref_x + a * ms_ref_y` (CORRECT)

**Impact**: Mean-shift applied in wrong direction → divergence

---

### Bug #4: Center Computation Precision (MINOR) ✅ FIXED

**Location**: `pyclnf/core/optimizer.py` line 284

**Problem**: Integer division lost sub-pixel precision
- Old: `center = resp_size // 2` (integer)
- New: `center = (resp_size - 1) / 2.0` (float, matches OpenFace)

---

## Components Verified Correct

### ✅ Parameter Update Logic
- Regularization: 25.0 (matches OpenFace) ✓
- Update formula: `A = J^T·W·J + λ·Λ^(-1)` ✓
- RHS formula: `b = J^T·W·v - λ·Λ^(-1)·p` ✓
- Learning rate damping: 0.75 ✓
- Weight matrix W: Correct construction ✓

### ✅ Jacobian Computation (After Fix)
- Scaling derivatives: Match OpenFace ✓
- Translation derivatives: Match OpenFace ✓
- Rotation derivatives: Now analytical (fixed) ✓
- Shape parameter derivatives: Match OpenFace ✓

### ✅ Mean-Shift Algorithm (Core)
- KDE computation: Algorithmically identical to OpenFace ✓
- Gaussian kernel: Correct formula ✓
- Weighted centroid: Correct computation ✓

### ✅ Other Components
- Sigma transformation: Correct formula and application ✓
- Neuron response computation: Matches OpenFace ✓
- Cross-correlation (TM_CCOEFF_NORMED): Correct ✓

---

## Test Results

### Before ANY Fixes
- WITH normalization [0,1]: Final update = 2.58
- WITHOUT normalization: Final update = 4.07

### After Jacobian Fix Only
- Final update = 3.88

### After ALL Fixes (Jacobian + Mean-Shift)
- Final update = **14.45** ⚠️ WORSE!

---

## Analysis: Why Did It Get Worse?

### Hypothesis 1: Mean-Shift Fixes Revealed Another Bug
The mean-shift coordinate system fix may have exposed a previously masked bug in:
- Response map computation
- Warping/similarity transform computation
- Weight matrix construction
- Or another component

### Hypothesis 2: One of the Mean-Shift Fixes is Incorrect
Possible issues:
- Reference shape not computed correctly
- Similarity transform matrices have wrong values
- Displacement calculation has sign error
- Matrix multiplication order wrong

### Hypothesis 3: Multiple Interacting Bugs
The fixes may be correct individually but reveal interactions between components that were previously "accidentally" canceling out.

---

## Debugging Steps Taken

1. ✅ Removed [0,1] normalization to expose real bugs
2. ✅ Fixed Jacobian rotation derivatives (numerical → analytical)
3. ✅ Verified parameter update logic matches OpenFace
4. ✅ Fixed mean-shift coordinate system (IMAGE → REFERENCE)
5. ✅ Fixed mean-shift transform signs
6. ✅ Fixed center computation precision

---

## Next Investigation Steps

### Priority 1: Verify Mean-Shift Fixes
1. Add extensive debug logging to `_compute_mean_shift()`:
   - Print reference_shape values
   - Print displacement values (IMAGE and REFERENCE)
   - Print dx, dy values
   - Print mean-shift magnitudes
   - Compare with what OpenFace would compute

2. Test mean-shift in isolation with known inputs

### Priority 2: Check Similarity Transform Matrices
1. Verify `sim_img_to_ref` is computed correctly
2. Verify `sim_ref_to_img` is the correct inverse
3. Check if rotation/scale are extracted correctly

### Priority 3: Response Map Investigation
1. Verify response maps are computed in correct coordinate system
2. Check if warping is applied correctly
3. Verify reference shape is at correct scale

### Priority 4: Test with OpenFace Intermediate Values
1. Extract OpenFace's intermediate values (params, mean-shift, Jacobian)
2. Inject them into PyCLNF to isolate divergence point
3. Binary search for where values diverge

---

## Documents Created

1. **JACOBIAN_BUG_SUMMARY.md** - Jacobian bug executive summary
2. **JACOBIAN_COMPARISON_ANALYSIS.md** - Detailed line-by-line Jacobian comparison
3. **JACOBIAN_FIX_APPLIED.md** - Jacobian fix implementation summary
4. **PARAMETER_UPDATE_COMPARISON.md** - Parameter update verification
5. **MEAN_SHIFT_BUG_SUMMARY.md** - Mean-shift bugs executive summary
6. **MEAN_SHIFT_COMPARISON.md** - Detailed mean-shift analysis
7. **MEAN_SHIFT_FIXES.md** - Mean-shift fix implementation guide
8. **BUG_FIXING_SESSION_SUMMARY.md** - This document

---

## Code Changes Summary

### `pyclnf/core/pdm.py`
- **Line 146-260**: Replaced `compute_jacobian()` with analytical rotation derivatives
- **Deleted**: `_euler_rotation_derivatives()` method (lines 227-255)

### `pyclnf/core/optimizer.py`
- **Line 145-148**: Added `reference_shape` argument to `_compute_mean_shift()` call
- **Line 217-226**: Added `reference_shape` parameter to method signature
- **Line 284**: Changed `center = resp_size // 2` to `center = (resp_size - 1) / 2.0`
- **Lines 286-305**: Replaced dx/dy fractional logic with REFERENCE-space displacement
- **Lines 318-319**: Fixed mean-shift transform signs (+ to -, - to +)
- **Line 534-545**: Removed [0,1] normalization (earlier session)

---

## Convergence Timeline

| State | Final Update | Notes |
|-------|-------------|-------|
| Original (WITH normalization) | 2.58 | Band-aid masking bugs |
| Remove normalization | 4.07 | Expose real bugs |
| Fix Jacobian | 3.88 | Slight improvement |
| Fix Mean-Shift | **14.45** | **WORSE** - unexpected! |

---

## Conclusion

We've systematically identified and fixed THREE CRITICAL BUGS:
1. ✅ Jacobian rotation derivatives
2. ✅ Mean-shift coordinate system
3. ✅ Mean-shift transform signs

However, convergence got **WORSE** after the mean-shift fixes, suggesting:
- Additional bugs remain OR
- One of the mean-shift fixes is incorrect OR
- Bugs were interacting in unexpected ways

**Status**: Investigation continues. The approach (systematic line-by-line comparison with OpenFace) is sound, but more debugging is needed.

**Recommendation**: Add extensive debug logging to mean-shift computation to verify the fixes are working as intended and isolate the actual problem.
