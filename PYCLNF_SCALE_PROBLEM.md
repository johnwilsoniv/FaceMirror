# PyCLNF Scale Convergence Problem

## Executive Summary

PyCLNF (Pure Python CLNF implementation) is converging to a **29% smaller scale** than OpenFace C++, even when initialized with the **exact same bounding box**. This causes landmarks to be vertically compressed to 78% of their correct size.

## Background

We are implementing a pure Python version of OpenFace's CLNF (Constrained Local Neural Fields) landmark detector to replace the C++ implementation. The Python version uses:
- Same PDM (Point Distribution Model) from `main_clnf_general.txt`
- Same CCNF patch experts from `ccnf_patches_*_general.dat`
- Same NU-RLMS optimization algorithm from the 2013 ICCV paper

## The Problem

### Test Case: IMG_0433.MOV Frame 50

**OpenFace C++ Results:**
```
Detection bbox: (286.5, 686.2, 425.8, 412.6)
Final parameters:
  scale: 2.799
  rotation: wx=-0.059, wy=0.002, wz=-0.079
  translation: tx=510.2, ty=875.2
Landmark range: y=[691.4, 1087.6]  (height: 396px)
```

**PyCLNF Results (using EXACT same bbox):**
```
Detection bbox: (286.5, 686.2, 425.8, 412.6)  ← IDENTICAL INPUT
Final parameters:
  scale: 2.164  ← 29% SMALLER
  rotation: wx=0.034, wy=0.003, wz=0.005
  translation: tx=499.9, ty=904.2
Landmark range: y=[763.4, 1072.4]  (height: 309px - only 78% of correct)
Converged: False (hit max 40 iterations)
```

### Visual Symptoms

1. **Landmarks track facial features** but are too "tight" - compressed vertically
2. **Not converging** - hitting max iterations trying to adjust
3. **Vertical position appears off** - compressed model sits higher in bbox than it should

## What We've Ruled Out

### ✅ Face Detector Differences
- **TESTED**: Used OpenFace C++'s exact detected bbox
- **RESULT**: Problem persists - scale still 29% smaller
- **CONCLUSION**: Bbox initialization is NOT the root cause

### ✅ PDM Initialization (CalcParams)
- **TESTED**: Initial parameters match bbox center correctly
- Python: tx=502.2, ty=904.4 (matches bbox center)
- **CONCLUSION**: PDM.init_params() is working correctly

### ✅ Missing Weight Multiplier
- **FIXED**: Added missing weight multiplier `w=5.0` in optimizer (line 88)
- Paper Equation 24: `W = w · diag(c1,...,cn,c1,...,cn)` where w=5-7
- **RESULT**: Iterations doubled from 20→40, but scale still wrong
- **CONCLUSION**: This was a real bug but not the scale issue

### ✅ Wrong Sigma Parameter
- **FIXED**: Changed sigma from 1.5 → 2.0 for in-the-wild data
- **CONCLUSION**: Helped convergence but didn't fix scale

### ✅ Insufficient Iterations
- **FIXED**: Increased max_iterations from 5 → 10 per window (20→40 total)
- **RESULT**: Still not converging, hitting max iterations
- **CONCLUSION**: More iterations alone won't fix it

## What We Know

### PDM Initialization is Correct
```python
# Initial params (before optimization)
PyCLNF: scale=2.090, tx=502.2, ty=904.4
- Landmarks centered at bbox center: ✓
- Mean shape centered at origin: ✓ (y_mean = 0.0)
```

### Optimization Dynamics are Wrong
The optimizer is **shrinking** the model instead of growing it:
- Initial scale: 2.090
- Final scale: 2.164 (grew only 3.5%)
- **Should be**: 2.799 (33% larger than initial)

The optimization appears to be:
1. Getting trapped in a local minimum at small scale
2. Unable to grow the model to match facial features
3. Patch responses or regularization preventing scale increase

## The Mystery

**Given identical inputs (same bbox, same PDM, same patch experts), why does the NU-RLMS optimizer converge to a different scale?**

Possible causes to investigate:

### 1. Patch Expert Response Evaluation
```python
# pyclnf/core/patch_expert.py: evaluate()
```
- Are we computing patch responses correctly?
- Bilinear interpolation accuracy?
- Response normalization?
- Could patch responses be **systematically lower** at larger scales?

### 2. NU-RLMS Jacobian Computation
```python
# pyclnf/core/optimizer.py: compute_jacobian()
```
- Finite difference step size correct?
- Partial derivatives w.r.t. scale parameter accurate?
- Could numerical instability bias against scale increases?

### 3. Regularization Balance
```python
# optimizer.py line 88-95
W = self.weight_multiplier * np.diag(np.repeat(weights, 2))
delta = (W @ patch_responses_flat) - (self.regularization * current_params)
```
- Is regularization term too strong, preventing scale growth?
- Weight multiplier correct (w=5)?
- Regularization coefficient correct (r=25)?

### 4. Multi-Window Optimization
The CLNF paper uses coarse-to-fine optimization with multiple window sizes:
- Window sizes: [25, 17, 11] (in `pyclnf/clnf.py`)
- Each window runs up to 10 iterations
- **Could the window size schedule affect scale convergence?**

### 5. Similarity Transform (sim_img_to_ref)
```python
# pdm.py: params_to_landmarks_2d()
```
- The PDM applies a similarity transform to project 3D→2D
- OpenFace uses orthographic projection
- Could there be a subtle difference in how scale is applied?

## Key Code Locations

### pyclnf/core/optimizer.py
- Line 37-60: `__init__` - Parameters (regularization, sigma, weight_multiplier)
- Line 62-108: `optimize_step()` - Single NU-RLMS iteration
- Line 88-95: **CRITICAL** - Weight matrix and delta computation

### pyclnf/core/patch_expert.py
- Line 73-129: `evaluate()` - Computes patch responses
- Uses bilinear interpolation to sample response map

### pyclnf/core/pdm.py
- Line 302-347: `init_params()` - Bbox → initial parameters
- Line 198-232: `params_to_landmarks_2d()` - Parameters → 2D landmarks
- Uses similarity transform with scale, rotation, translation

### pyclnf/clnf.py
- Line 110-158: `fit()` - Main CLNF fitting loop
- Orchestrates multi-window optimization

## Diagnostic Observations

### Scale Evolution During Optimization
Need to track: **How does scale change across iterations?**
- Initial: 2.090
- After 40 iterations: 2.164 (only +3.5%)
- Target: 2.799 (+33% from initial)

**Question**: Is the scale:
1. Growing slowly and would converge with more iterations?
2. Stuck in local minimum?
3. Being actively pulled down by patch responses?

### Patch Response Magnitudes
**Question**: Are patch response values comparable to OpenFace?
- What's the typical response magnitude?
- Do responses get weaker at larger scales?
- Could response normalization be wrong?

### Regularization Dominance
```python
delta = (W @ patch_responses) - (regularization * params)
```
- If `regularization * params` dominates, optimizer resists change
- regularization=25 might be too strong?
- weight_multiplier=5 might be too weak?

## What Another LLM Should Investigate

### Priority 1: Add Detailed Logging
Instrument the optimizer to track **per-iteration**:
```python
# For each iteration:
- Current scale value
- Patch response magnitude: ||W @ patch_responses||
- Regularization magnitude: ||regularization * params||
- Delta magnitude: ||delta||
- Jacobian w.r.t. scale: J[0, :]
- Scale update: delta_params[0]
```

This will reveal:
- Is scale trying to grow but being stopped?
- Which term (data vs prior) is dominating?
- Is the Jacobian w.r.t. scale computed correctly?

### Priority 2: Compare Patch Responses Directly
Extract patch responses from both implementations:
- OpenFace C++ at scale=2.799
- PyCLNF at scale=2.799 (forced)
- **Are the response values identical?**

If different → patch expert evaluation bug
If identical → optimizer bug

### Priority 3: Test Jacobian Accuracy
Compute Jacobian with finite differences:
```python
# Analytical Jacobian (our implementation)
J_analytical = compute_jacobian(...)

# Numerical Jacobian (ground truth)
J_numerical = finite_difference_jacobian(...)

# Compare
error = ||J_analytical - J_numerical||
```

### Priority 4: Test with Fixed Scale
Force scale=2.799 and optimize only rotation/translation:
- Do other parameters converge to OpenFace values?
- Are landmarks positioned correctly?

This isolates: Is scale the ONLY issue or are other params also wrong?

## Critical Questions

1. **Why does the optimizer prefer smaller scale?**
   - Lower regularization penalty?
   - Better patch response fit?
   - Numerical artifact?

2. **Is the scale parameter treated differently in OpenFace C++?**
   - Different parameterization?
   - Logarithmic vs linear?
   - Special handling?

3. **Could there be a units/scaling issue?**
   - PDM mean shape units?
   - Patch expert response scales?
   - Coordinate system differences?

4. **Is the initialization actually different?**
   - We confirmed bbox center matches
   - But is the initial scale=2.090 correct?
   - What does OpenFace C++ use as initial scale?

## Files for Reference

### Test Data
- Video: `Patient Data/Normal Cohort/IMG_0433.MOV`
- Frame: 50
- OpenFace bbox: (286.5, 686.2, 425.8, 412.6)

### Test Scripts
- `test_openface_bbox.py` - Tests PyCLNF with OpenFace's bbox
- `debug_vertical_offset.py` - Analyzes PDM initialization
- `compare_pyclnf_vs_cpp.py` - Side-by-side comparison

### Modified OpenFace C++
Added debug output to extract bbox and params:
```cpp
// LandmarkDetectorFunc.cpp line 323-325
std::cout << "DEBUG_BBOX: " << bounding_box.x << "," << bounding_box.y << ","
          << bounding_box.width << "," << bounding_box.height << std::endl;

// LandmarkDetectorFunc.cpp line 353-359
std::cout << "DEBUG_PARAMS: scale=" << clnf_model.params_global[0]
          << " wx=" << clnf_model.params_global[1]
          << " wy=" << clnf_model.params_global[2]
          << " wz=" << clnf_model.params_global[3]
          << " tx=" << clnf_model.params_global[4]
          << " ty=" << clnf_model.params_global[5] << std::endl;
```

## Bottom Line

We have a **29% scale convergence discrepancy** that persists even with identical initialization. The NU-RLMS optimizer is converging to a local minimum with too-small scale. We need to understand:

1. **What's different in the optimization dynamics?**
2. **Is it a bug in patch response evaluation, Jacobian computation, or regularization balance?**
3. **Or is there a fundamental difference in how OpenFace C++ handles scale that we're missing?**

The good news: Landmarks ARE tracking features, just compressed. The optimizer is working, just converging to the wrong answer. This suggests the bug is subtle - likely a numerical issue or missing parameter rather than a gross implementation error.
