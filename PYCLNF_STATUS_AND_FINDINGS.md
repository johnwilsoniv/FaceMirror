# PyCLNF Implementation Status and Findings

**Date**: November 10, 2025
**Status**: Sigma transformation implemented but convergence issues remain

---

## Executive Summary

We successfully implemented CCNF Sigma spatial correlation transformation for Py CLNF, matching the OpenFace C++ implementation exactly. However, PyCLNF still fails to converge properly (final update ~2.5-4.0 instead of <0.005).

**Key Finding**: The normalization paradox reveals a deeper underlying issue - both normalized and unnormalized versions fail to converge, suggesting the root problem lies elsewhere in the optimization pipeline.

---

## What We Implemented

### 1. Sigma Transformation (✅ Complete & Correct)

**Components**:
- Exported 12 sigma component matrices from OpenFace C++ (4 window sizes × 3 components)
- Added `load_sigma_components()` to openface_loader.py
- Implemented `compute_sigma()` in CCNFPatchExpert with correct OpenFace formula
- Integrated transformation into optimizer._compute_response_map()

**Validation**:
- Formula matches OpenFace C++ exactly: `Sigma = inv(2*(sum_alphas*I + sum(beta*sigma_comp)))`
- Sigma matrices are symmetric and positive definite
- Transformation applies correctly without errors
- Dimension handling fixed (uses response_window_size, not patch_width)

### 2. Code Verification (✅ Matches OpenFace C++)

**Verified Components**:
- ✅ Neuron response computation (lines 166-173 in patch_expert.py)
- ✅ Cross-correlation (TM_CCOEFF_NORMED)
- ✅ Sigma computation (lines 214-244 in patch_expert.py)
- ✅ Sigma transformation application (lines 516-532 in optimizer.py)
- ✅ Negative value removal (lines 536-538 in optimizer.py)

---

## The Normalization Paradox

### OpenFace C++ Behavior

```cpp
// CCNF_patch_expert.cpp lines 406-413
double min;
minMaxIdx(response, &min, 0);
if(min < 0) {
    response = response - min;
}
// NO FURTHER NORMALIZATION - returns here
```

**Key Point**: OpenFace C++ does NOT normalize response maps to [0,1] range.

### PyCLNF Behavior

```python
# optimizer.py lines 536-545
min_val = response_map.min()
if min_val < 0:
    response_map = response_map - min_val

max_val = response_map.max()
if max_val > 0:
    response_map = response_map / max_val  # Extra step not in OpenFace!
```

**PyCLNF adds normalization** to [0,1] range.

### Empirical Test Results

**WITH Normalization** (current PyCLNF):
- Final update: 2.58
- Converged: False
- All response maps have max=1.0

**WITHOUT Normalization** (matching OpenFace C++):
- Final update: 4.07 (WORSE!)
- Converged: False
- Response maps retain original magnitudes (e.g., 0.024, 0.036)

### The Paradox

1. **Theory says**: Normalization destroys relative confidence information → should hurt convergence
2. **Practice shows**: Removing normalization makes convergence WORSE
3. **Conclusion**: Both versions fail because there's a more fundamental bug elsewhere

---

## Why Sigma Showed No Improvement

### Test Results

**With Sigma Transformation**:
- Final update: 2.58
- Iterations: 80 (max)

**Without Sigma Transformation**:
- Final update: 2.51
- Iterations: 80 (max)

**Difference**: Effectively no improvement (~3% worse with sigma)

### Analysis

The sigma transformation works correctly but has no effect because:

1. **Normalization undoes attenuation**: Sigma reduces response magnitudes by ~160x, but [0,1] normalization immediately rescales everything back
2. **Both versions fail to converge**: The underlying optimization issue affects sigma and non-sigma versions equally
3. **Relative information preserved**: Even with normalization, the SPATIAL STRUCTURE of responses (peak locations) is preserved, just not magnitudes

---

## Unverified Components (Likely Bug Location)

### 1. Mean-Shift Computation ⚠️

**Location**: `pyclnf/core/optimizer.py` lines 223-407 (`_compute_mean_shift` method)

**What it does**:
- Computes KDE weighted mean-shift for each landmark
- Uses Gaussian kernel with sigma parameter
- Weights samples by response map values

**Potential issues**:
- KDE bandwidth calculation may be incorrect
- Weight normalization might be wrong
- Peak finding in response map could be off
- Gaussian kernel application might differ from OpenFace

**Not yet compared with**: OpenFace C++ CLNF.cpp mean-shift implementation

### 2. Jacobian Computation ⚠️

**Location**: `pyclnf/core/pdm.py` (`compute_jacobian` method - if it exists)

**What it does**:
- Computes derivatives of landmark positions w.r.t. PDM parameters
- Used to convert mean-shift updates into parameter updates

**Potential issues**:
- Jacobian formula might be incorrect
- Sign errors in derivatives
- Missing scaling factors
- Rotation/translation Jacobian could be wrong

**Not yet compared with**: OpenFace C++ PDM.cpp Jacobian computation

### 3. Parameter Update Step ⚠️

**Location**: `pyclnf/core/optimizer.py` lines 100-180 (`optimize` method)

**What it does**:
- Converts mean-shift vectors to parameter updates
- Applies regularization
- Updates PDM parameters

**Potential issues**:
- Update formula might be incorrect
- Regularization application could be wrong
- Step size/learning rate missing or incorrect
- Parameter bounds not enforced

**Not yet compared with**: OpenFace C++ CLNF.cpp optimization loop

### 4. PDM Parameter Initialization ⚠️

**Location**: `pyclnf/core/pdm.py` (`init_params` method)

**What it does**:
- Initializes PDM parameters from bounding box
- Sets scale, translation, rotation, shape params

**Potential issues**:
- Scale calculation from bbox might be wrong
- Translation offset could be incorrect
- Initial shape params might not be zeros
- Coordinate system mismatch (OpenFace uses different conventions)

**Not yet compared with**: OpenFace C++ PDM.cpp initialization

---

## Key Technical Details

### Sigma Transformation Properties

From diagnostic testing (landmark 30, window_size=11):

**Raw Response** (before sigma):
- Min: 0.0000042
- Max: 3.8919
- Dynamic range: ~1000x

**After Sigma Transformation**:
- Min: -0.0012
- Max: 0.0238
- Attenuation: ~160x (3.89 → 0.024)

**After Normalization** (PyCLNF):
- Min: 0.0
- Max: 1.0
- Relative structure preserved, but magnitudes lost

### Sigma Matrix Properties

For window_size=11 (121×121 matrix):
- Diagonal mean: 0.006257
- Condition number: 1.90
- Eigenvalue range: [5.0e-3, 9.5e-3]
- Positive definite: Yes
- Symmetric: Yes

**Effect**: Acts as a smoothing/correlation filter that:
- Reduces response magnitudes significantly (~160x)
- Models spatial dependencies between response map elements
- Makes response peaks broader and smoother

---

## Sigma Component Files

**Location**: `/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/sigma_components/`

**Files**:
```
window_sizes.npy              # [7, 9, 11, 15]
sigma_w7_c0.npy              # 49×49 matrix
sigma_w7_c1.npy              # 49×49 matrix
sigma_w7_c2.npy              # 49×49 matrix
sigma_w9_c0.npy              # 81×81 matrix
sigma_w9_c1.npy              # 81×81 matrix
sigma_w9_c2.npy              # 81×81 matrix
sigma_w11_c0.npy             # 121×121 matrix
sigma_w11_c1.npy             # 121×121 matrix
sigma_w11_c2.npy             # 121×121 matrix
sigma_w15_c0.npy             # 225×225 matrix
sigma_w15_c1.npy             # 225×225 matrix
sigma_w15_c2.npy             # 225×225 matrix
```

All matrices verified as:
- Symmetric
- Positive definite
- Correct dimensions (window_size² × window_size²)

---

## Research Findings

From web search on CLNF/NU-RLMS:

> "NU-RLMS (Non-uniform Regularised Landmark Mean-Shift) performs better on more reliable response maps"

**Implication**: The optimizer relies on response map quality. If response maps are unreliable (noisy peaks, weak signals), convergence will fail regardless of normalization or sigma transformation.

**Hypothesis**: PyCLNF's response maps may be fundamentally unreliable due to:
1. Incorrect mean-shift computation
2. Wrong Jacobian preventing parameter updates from improving fit
3. Poor initialization causing optimization to start far from solution
4. Missing preprocessing/calibration steps

---

## Next Steps

### Recommended Investigation Order

1. **Compare Mean-Shift Computation** (HIGH PRIORITY)
   - Read OpenFace C++ CLNF.cpp mean-shift implementation
   - Line-by-line comparison with PyCLNF
   - Check KDE bandwidth, kernel application, peak finding
   - **Rationale**: Mean-shift directly affects parameter updates

2. **Verify Jacobian Computation** (HIGH PRIORITY)
   - Read OpenFace C++ PDM.cpp Jacobian methods
   - Compare with PyCLNF implementation
   - Test with known parameter values
   - **Rationale**: Wrong Jacobian prevents convergence regardless of response quality

3. **Check Parameter Initialization** (MEDIUM PRIORITY)
   - Compare bbox → params conversion with OpenFace
   - Verify coordinate system conventions
   - Test with known good initialization
   - **Rationale**: Poor initialization makes convergence harder but shouldn't prevent it entirely

4. **Validate Parameter Update Step** (MEDIUM PRIORITY)
   - Compare update formula with OpenFace
   - Check regularization application
   - Verify parameter bounds
   - **Rationale**: Wrong update formula causes divergence

### Alternative Approach

**Use OpenFace C++ Directly**:
- PyCLNF was experimental/educational
- OpenFace C++ is proven, optimized, and works
- Could wrap OpenFace C++ with Python bindings (pybind11)
- Would eliminate all convergence issues immediately
- Faster performance (C++ vs Python)

**Trade-offs**:
- Loses pure Python benefit
- Adds C++ dependency
- But gains: correctness, speed, maintenance

---

## Files Modified in This Session

1. `pyclnf/models/openface_loader.py` - Added `load_sigma_components()`
2. `pyclnf/core/patch_expert.py` - Added `compute_sigma()` method and sigma loading
3. `pyclnf/core/optimizer.py` - Integrated sigma transformation
4. `pyclnf/clnf.py` - Pass sigma_components to optimizer

5. `export_sigma_components.cpp` - C++ tool to export sigma from OpenFace (in `/tmp/sigma_export/`)

6. `test_sigma_convergence.py` - Comparison test for sigma impact
7. `SIGMA_IMPLEMENTATION_SUMMARY.md` - Technical documentation
8. `RESPONSE_MAP_INVESTIGATION_REPORT.md` - Agent investigation results (incorrect hypothesis)
9. `diagnose_response_maps.py` - Diagnostic script
10. `PYCLNF_STATUS_AND_FINDINGS.md` - This document

---

## Conclusion

**Sigma Transformation**: ✅ Implemented correctly, matches OpenFace C++ exactly

**Convergence Issue**: ❌ Still present, root cause likely in mean-shift, Jacobian, or parameter update logic

**Normalization Paradox**: Reveals that both approaches fail, suggesting deeper bug

**Recommendation**: Either fix mean-shift/Jacobian after detailed comparison with OpenFace C++, or switch to using OpenFace C++ directly for production use.

The sigma implementation work was not wasted - we now have:
- Complete understanding of CCNF spatial correlation modeling
- Verified components (neurons, cross-correlation, sigma computation)
- Diagnostic tools and test scripts
- Clear identification of remaining unverified components

This provides a solid foundation for either completing PyCLNF or documenting why OpenFace C++ is the better choice.
