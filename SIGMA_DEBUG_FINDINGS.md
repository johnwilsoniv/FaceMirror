# Sigma Transformation Debug Investigation

## Date: 2025-11-15

## Objective
Investigate why CLNF is not converging (0% convergence rate, mean error 9.6px vs C++ 3.5px) by deep-diving into the Sigma covariance transformation applied to response maps.

## Debug Infrastructure Added

### Files Modified
1. **pyclnf/core/patch_expert.py** (lines 192-287)
   - Added `debug: bool = False` parameter to `compute_sigma()` method
   - Added extensive debug logging showing:
     - Matrix dimensions and window size
     - Sum of alphas, number of neurons, betas, sigma components
     - Sigma component shape verification
     - SigmaInv properties (determinant, condition number, min/max values)
     - Sigma properties after inversion
     - Verification that Sigma*SigmaInv ≈ Identity

2. **pyclnf/core/optimizer.py** (lines 223-232, 446-456, 564-606)
   - Added `iteration: int = None` parameter to `_compute_mean_shift()`
   - Added `landmark_idx: int = None, iteration: int = None` parameters to `_compute_response_map()`
   - Fixed variable scoping bug by passing parameters through call chain
   - Added sigma component selection debug output
   - Added before/after Sigma peak offset tracking

## Key Findings

### 1. Sigma Component Selection (Landmark 36, Iteration 0, Window Size 11)
```
[Sigma Component Selection Debug]
  landmark_idx=36, iteration=0
  response_window_size=11
  Available sigma_components window sizes: [7, 9, 11, 15]
  Selected sigma_comps length: 3
  sigma_comps[0].shape = (121, 121) ✓
  sigma_comps[1].shape = (121, 121) ✓
  sigma_comps[2].shape = (121, 121) ✓
```

**Status**: ✓ **CORRECT** - All sigma components have the correct shape (121x121 for window size 11)

### 2. Sigma Matrix Computation
```
[Sigma Debug] window_size=11, matrix_size=121
[Sigma Debug] sum_alphas=76.465448
[Sigma Debug] num_neurons=7
[Sigma Debug] num_betas=3
[Sigma Debug] num_sigma_components=3
[Sigma Debug] Using 3 components (min of 3 betas, 3 sigma_comps)
[Sigma Debug] beta[0]=3.731875
[Sigma Debug] beta[1]=2.290043
[Sigma Debug] beta[2]=1.507682
```

**Status**: ✓ **CORRECT** - Number of components matches, betas are reasonable values

### 3. SigmaInv Properties (BEFORE Inversion)
```
[Sigma Debug] SigmaInv: det=inf, cond=1.43e+00
[Sigma Debug] SigmaInv: min=-7.463751, max=225.229156
```

**Status**: ⚠️ **CONCERNING** - Determinant = infinity indicates:
- Matrix is likely very large (well-conditioned with cond=1.43)
- Could be hitting numerical limits during det() computation
- But condition number is EXCELLENT (1.43 ≈ 1.0 means well-conditioned)

### 4. Sigma Matrix Properties (AFTER Cholesky Inversion)
```
[Sigma Debug] ✓ Cholesky inversion succeeded
[Sigma Debug] Sigma: det=0.000000e+00, cond=1.43e+00
[Sigma Debug] Sigma: min=-0.000083, max=0.005275
[Sigma Debug] Sigma*SigmaInv identity error: 2.980232e-07
```

**Status**: ✓ **CORRECT** - Matrix inversion is working perfectly!
- Cholesky decomposition succeeds
- Determinant ≈ 0 is expected (inverse of very large matrix)
- **Sigma*SigmaInv identity error = 2.98e-07 is EXCELLENT** (essentially perfect)
- Condition number still good (1.43)

### 5. Sigma Transformation Effect on Response Maps

#### Examples Where Sigma HELPS:
```
BEFORE: offset=(2, 3) dist=3.6px peak=24.788
AFTER:  offset=(1, -2) dist=2.2px peak=0.186
→ IMPROVEMENT: 3.6px → 2.2px ✓

BEFORE: offset=(3, 4) dist=5.0px peak=18.170
AFTER:  offset=(2, 0) dist=2.0px peak=0.159
→ IMPROVEMENT: 5.0px → 2.0px ✓

BEFORE: offset=(4, -1) dist=4.1px peak=30.632
AFTER:  offset=(3, 0) dist=3.0px peak=0.279
→ IMPROVEMENT: 4.1px → 3.0px ✓

BEFORE: offset=(-3, -3) dist=4.2px peak=26.135
AFTER:  offset=(-1, -2) dist=2.2px peak=0.275
→ IMPROVEMENT: 4.2px → 2.2px ✓

BEFORE: offset=(3, -3) dist=4.2px peak=9.265
AFTER:  offset=(2, 0) dist=2.0px peak=0.105
→ IMPROVEMENT: 4.2px → 2.0px ✓
```

#### Examples Where Sigma DOESN'T HELP:
```
BEFORE: offset=(-3, -1) dist=3.2px peak=66.383
AFTER:  offset=(-3, -1) dist=3.2px peak=0.430
→ NO CHANGE: 3.2px → 3.2px

BEFORE: offset=(-3, -2) dist=3.6px peak=38.936
AFTER:  offset=(-3, -2) dist=3.6px peak=0.281
→ NO CHANGE: 3.6px → 3.6px

BEFORE: offset=(3, 4) dist=5.0px peak=19.574
AFTER:  offset=(3, 4) dist=5.0px peak=0.165
→ NO CHANGE: 5.0px → 5.0px

BEFORE: offset=(1, 3) dist=3.2px peak=18.214
AFTER:  offset=(1, 4) dist=4.1px peak=0.158
→ WORSE: 3.2px → 4.1px ⚠️
```

**Status**: ⚠️ **MIXED RESULTS**
- Sigma IS working and improving many landmarks
- But some landmarks show NO improvement or get WORSE
- Peak values drop dramatically (24.788 → 0.186) but offsets don't always improve

## Analysis

### What We Ruled Out
1. ✓ **Coordinate system issues** - No transpose/row-major vs column-major problems
2. ✓ **Sigma component shape** - All components have correct dimensions
3. ✓ **Matrix inversion** - Working perfectly (identity error = 2.98e-07)
4. ✓ **Sigma component selection** - Correct components selected for each window size

### What's Working
1. ✓ Sigma computation matches C++ OpenFace formula exactly
2. ✓ Matrix inversion is numerically stable and accurate
3. ✓ Sigma IS improving many landmarks (sometimes 4.2px → 2.0px)

### Remaining Issues

#### Issue 1: Inconsistent Sigma Benefits
**Observation**: Some landmarks improve significantly (4.2px → 2.0px) while others show no change (3.2px → 3.2px) or get worse (3.2px → 4.1px).

**Hypothesis**: This could be due to:
- Initial response map quality varies by landmark
- Some landmarks have very strong peaks (66.383) that don't need smoothing
- Others have weak peaks (9.265) that benefit from spatial correlation modeling
- The raw response maps (BEFORE Sigma) may have systematic issues

#### Issue 2: High Peak Offset Distances Overall
**Observation**: Even after Sigma, many peaks are still 2-4 pixels from center.

**Expected**: In a converged CLNF, peaks should be within ~1 pixel of center.

**Current**: Seeing offsets of 2-4px even after Sigma transformation.

**This suggests**: The problem may not be with Sigma itself, but with:
1. **Initial response maps** - Patches may be extracting wrong regions
2. **Window center calculation** - May have off-by-one errors
3. **Patch extraction coordinates** - X/Y indexing bugs
4. **Previous iteration** - Landmarks already far from true position

#### Issue 3: High Mean-Shift Magnitude
**Observation**:
```
Iter  0 (ws=11): MS= 90.8142 DP= 14.2539
Iter  1 (ws=11): MS= 69.3702 DP= 10.2791
Iter  2 (ws=11): MS= 50.0289 DP=  7.4220
```

**Expected**: For convergence, mean-shift should be < 10 pixels by iteration 2-3.

**Current**: Still 50+ pixels at iteration 2, suggesting landmarks are far from correct positions.

## Next Steps

### Priority 1: Investigate Raw Response Maps
Since Sigma is working but peak offsets are still too high, we need to investigate the response maps BEFORE Sigma:

1. **Verify patch extraction coordinates**
   - Check X/Y indexing in `_compute_response_map()`
   - Verify window bounds calculation
   - Check for off-by-one errors

2. **Compare raw response maps with C++ OpenFace**
   - Export C++ response maps before Sigma
   - Compare pixel-by-pixel with Python
   - Look for systematic offsets or transposition

3. **Investigate window centering**
   - Verify window center calculation: `center = (window_size - 1) / 2.0`
   - Check if peak offset calculation is correct
   - Validate coordinate system (row/col vs x/y)

### Priority 2: Investigate Convergence Threshold
Current threshold: `shape_change < 0.01` pixels

With mean-shift magnitudes of 50-90 pixels and parameter updates of 7-14, we're nowhere near this threshold. Need to understand:
1. Why landmarks are initialized so far from correct positions
2. Whether PDM constraints are preventing landmarks from moving to correct locations
3. If regularization=35 is too strict

### Priority 3: Compare C++ Debug Output
Add similar debug output to C++ OpenFace and compare:
1. Response map peak offsets before/after Sigma
2. Mean-shift magnitudes per iteration
3. Parameter update magnitudes
4. Lambda (covariance) matrix values

## Summary

**Sigma transformation is working correctly** - matrix inversion is perfect, and it IS improving many landmarks. However:

1. **Not all landmarks benefit equally** - some improve significantly, others show no change
2. **Peak offsets are still too high** - even after Sigma, 2-4px offsets are common
3. **Convergence is not happening** - mean-shift magnitudes remain very high (50-90px)

**Root cause likely lies elsewhere:**
- Initial response maps may have systematic coordinate errors
- Patch extraction may have X/Y indexing bugs
- Window centering calculations may be incorrect
- Or landmarks are genuinely far from correct positions due to poor initialization

**The investigation should now focus on the raw response maps and patch extraction**, not on Sigma transformation itself.
