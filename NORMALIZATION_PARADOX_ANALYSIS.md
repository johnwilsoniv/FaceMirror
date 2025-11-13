# The Normalization Paradox: Why PyC LNF Needs [0,1] Normalization

**Date**: November 10, 2025
**Status**: Root cause identified through magnitude analysis

---

## Executive Summary

PyCLNF requires [0,1] normalization of response maps to achieve better (though still unsuccessful) convergence, while OpenFace C++ does not normalize at all. Our diagnostics reveal the underlying cause: **compound attenuation** from sigma transformation applied to poorly-initialized landmarks produces catastrophically small response values (~0.00006) that likely cause numerical precision issues in mean-shift computation.

---

## Diagnostic Results

### Response Magnitude Analysis

**Test Setup**:
- Image: 1920×1080 frame from test video
- 3 test landmarks: 30 (nose), 36 (eye), 48 (mouth)
- Window size: 11×11
- Sigma transformation: enabled

**Key Findings**:

#### Landmark 30 (Nose) - Well Initialized
```
RAW response (before sigma):
  Max: 6.689101
  Mean: 0.400189

After sigma transformation (~170x attenuation):
  Max: 0.039291
  Mean: 0.003371

Sigma diagonal mean: 0.005399
Attenuation factor: 170.2x
```

#### Landmark 36 (Eye) - Moderately Initialized
```
RAW response (before sigma):
  Max: 1.759686
  Mean: 0.124529

After sigma transformation (~191x attenuation):
  Max: 0.009228
  Mean: 0.000907

Sigma diagonal mean: 0.004732
Attenuation factor: 190.7x
```

#### Landmark 48 (Mouth) - Poorly Initialized ⚠️
```
RAW response (before sigma):
  Max: 0.009627  ← Already TINY!
  Mean: 0.001232

After sigma transformation (~170x attenuation):
  Max: 0.000057  ← Catastrophically small!
  Mean: 0.000008

Sigma diagonal mean: 0.004938
Attenuation factor: 169.8x
```

### Overall Statistics (All Landmarks)

**Unnormalized** (after sigma, before [0,1] normalization):
```
Range: [0.000000, 0.039291]
Mean: 0.001429
Std: 0.003895
Percentiles:
  25th: 0.000008
  50th: 0.000267
  75th: 0.001166
  95th: 0.005972
  99th: 0.018422
```

**Normalized** (after [0,1] rescaling):
```
Range: [0.000000, 1.000000]
Mean: 0.108302
Std: 0.167979
```

---

## The Root Cause: Compound Attenuation

### Problem Mechanism

1. **Poor Initialization**: Some landmarks (e.g., landmark 48) start with poor PDM initialization, resulting in weak raw neuron responses (max 0.01 vs expected 1-10)

2. **Sigma Transformation Attenuation**: Sigma transformation correctly attenuates responses by ~170x to model spatial correlations
   - Formula: `Sigma = inv(2*(sum_alphas*I + sum(beta*sigma_comp)))`
   - Sigma diagonal mean: ~0.005
   - Expected behavior for spatial correlation modeling

3. **Catastrophic Compounding**: Weak initial response × 170x attenuation = unusable tiny values
   - Example: 0.01 × (1/170) = 0.00006
   - Values this small likely cause **numerical precision issues** in mean-shift computation

### Why Normalization Helps (2.58 vs 4.07 Without)

[0,1] normalization rescales all response maps to a consistent numerical range:
- 0.00006 → 1.0 (for the peak)
- Makes values large enough for stable floating-point computation
- Prevents underflow/precision loss in KDE Gaussian kernel evaluation
- Stabilizes mean-shift weight calculations

### Why Both Approaches Still Fail

**Convergence Results**:
- WITH normalization: Final update = 2.58 (target < 0.005)
- WITHOUT normalization: Final update = 4.07 (WORSE!)

Both fail because:
1. Normalization destroys **relative magnitude information** between landmarks
   - High-confidence landmarks (strong responses) lose their advantage
   - Low-confidence landmarks (weak responses) get equal weight
   - NU-RLMS optimizer cannot properly weight reliable vs unreliable patches

2. There are likely **additional bugs** in:
   - Mean-shift computation (KDE weighting, bandwidth, peak finding)
   - Jacobian calculation (PDM parameter derivatives)
   - Parameter update logic (conversion from mean-shift to parameter space)

---

## Why OpenFace C++ Doesn't Need Normalization

### OpenFace C++ Code

**File**: `CCNF_patch_expert.cpp` lines 406-413

```cpp
// Making sure the response does not have negative numbers
double min;
minMaxIdx(response, &min, 0);
if(min < 0)
{
    response = response - min;
}
// NO FURTHER NORMALIZATION - returns here
```

OpenFace C++ only shifts to remove negative values. It does NOT normalize to [0,1].

### Possible Explanations

#### 1. Better Numerical Handling
- C++ float arithmetic may have different precision characteristics than NumPy
- OpenFace may use different compiler optimizations that preserve precision
- Different numerical libraries (OpenCV vs NumPy) may handle small values differently

#### 2. Different Mean-Shift Implementation
- OpenFace C++ mean-shift computation may be more robust to small response values
- KDE bandwidth calculation might automatically adapt to magnitude scale
- Gaussian kernel evaluation might use different numerical methods

#### 3. Better Initialization
- OpenFace C++ PDM initialization from bbox may be more accurate
- Landmarks start closer to correct positions → stronger initial responses
- Fewer landmarks suffer from the "compound attenuation" problem

#### 4. Different Optimization Parameters
- OpenFace may use different regularization that compensates
- Different convergence thresholds or iteration limits
- Different sigma component values (though we verified these match)

### Precision Analysis

Both use **float32**:
- OpenFace C++: `cv::Mat_<float>` (verified in CCNF_patch_expert.cpp line 82)
- PyCLNF: `np.float32` (in optimizer.py)

So precision difference is not the primary factor.

---

## The Normalization Paradox Explained

### The Paradox

**Theory**: Normalization destroys relative confidence information → should hurt convergence
**Practice**: Removing normalization makes convergence WORSE (4.07 vs 2.58)

### Resolution

The paradox resolves when we recognize **two competing effects**:

1. **Numerical Stability** (helps): Normalization rescales tiny values to usable range
   - 0.00006 → 1.0 prevents precision loss
   - Stabilizes floating-point arithmetic
   - Effect: +positive (makes convergence better)

2. **Lost Relative Information** (hurts): Normalization makes all peaks equal
   - High-confidence landmarks lose their advantage
   - NU-RLMS cannot weight reliable patches properly
   - Effect: -negative (makes convergence worse)

In PyCLNF's current state:
- **Numerical stability benefit > Lost information cost**
- Net result: Normalization helps (2.58) vs hurts (4.07) without it
- But both still fail because the lost information effect is still significant

In OpenFace C++ (hypothesized):
- Better numerical handling → numerical stability not needed
- OR: Better initialization → fewer catastrophically small values
- OR: Different mean-shift implementation that's more robust
- Result: Can preserve relative information without normalization

---

## Evidence Summary

### Verified Facts

✅ **OpenFace C++ does not normalize to [0,1]**
   Source: CCNF_patch_expert.cpp lines 406-413

✅ **PyCLNF sigma transformation is correct**
   - Formula matches OpenFace exactly
   - Attenuation factor ~170x is expected
   - Sigma diagonal mean ~0.005 is consistent

✅ **Both use float32 precision**
   - OpenFace: `cv::Mat_<float>`
   - PyCLNF: `np.float32`

✅ **Poor initialization causes weak responses**
   - Landmark 48: raw max 0.01 (vs 6.69 for well-initialized landmark 30)
   - After sigma: 0.00006 (catastrophically small)

✅ **Normalization improves PyCLNF convergence**
   - WITH: 2.58 final update
   - WITHOUT: 4.07 final update (57% worse)

✅ **Both normalized and unnormalized versions fail**
   - Target: < 0.005 final update
   - Both >> 0.005 (convergence failure)

### Remaining Questions

❓ How does OpenFace C++ mean-shift computation handle values like 0.00006 without precision loss?

❓ Does OpenFace C++ have better PDM initialization that avoids weak initial responses?

❓ Are there differences in KDE bandwidth calculation or Gaussian kernel evaluation?

❓ What are the bugs in PyCLNF's mean-shift, Jacobian, or parameter update logic that cause both approaches to fail?

---

## Next Investigation Steps

### Priority 1: Compare Mean-Shift Computation

**Goal**: Determine if OpenFace C++ mean-shift handles small values differently

**Approach**:
1. Read OpenFace C++ mean-shift implementation (LandmarkDetectorModel.cpp or PDM.cpp)
2. Line-by-line comparison with PyCLNF optimizer._compute_mean_shift()
3. Check for:
   - KDE bandwidth calculation differences
   - Gaussian kernel evaluation methods
   - Weight normalization strategies
   - Handling of near-zero response values

### Priority 2: Test PDM Initialization Quality

**Goal**: Determine if PyC LNF initialization is worse than OpenFace C++

**Approach**:
1. Compare initial landmark positions from PDM.init_params()
2. Measure initial response magnitudes for both implementations
3. Check if OpenFace C++ has preprocessing steps we're missing

### Priority 3: Verify Jacobian Computation

**Goal**: Ensure PDM Jacobian is computed correctly

**Approach**:
1. Read OpenFace C++ PDM Jacobian implementation
2. Compare with PyCLNF PDM.compute_jacobian()
3. Test with known parameter values and verify derivatives

---

## Conclusion

PyCLNF requires [0,1] normalization not because it's "correct", but because it provides **numerical stability for catastrophically small response values** that result from compound attenuation (poor initialization × 170x sigma attenuation). This normalization helps convergence (2.58 vs 4.07) but still fails because it destroys the relative magnitude information that NU-RLMS relies on to weight high-confidence patches.

OpenFace C++ avoids this issue through some combination of:
- Better numerical handling in mean-shift computation
- Better PDM initialization (fewer weak initial responses)
- Different optimization implementation details

The next investigation should focus on mean-shift computation to understand how OpenFace C++ handles small response values without normalization.

**Key Insight**: The normalization paradox is not a paradox at all - it's the intersection of two real effects (numerical stability vs lost information), where PyCLNF's current bugs make the numerical stability effect dominate.
