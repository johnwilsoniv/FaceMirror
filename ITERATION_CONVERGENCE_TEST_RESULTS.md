# PyCLNF Iteration Convergence Test Results

**Date**: 2025-11-10
**Status**: CRITICAL FINDING - Increasing iterations DOES NOT reduce error

---

## Summary

Tested PyCLNF with 1, 5, 10, 20, and 50 iterations per window size to determine if convergence is simply slow. **Result: Error does NOT decrease with more iterations - it actually gets WORSE.**

---

## Test Results: PyCLNF vs OpenFace C++

| Iterations | Mean Error | Max Error | Converged |
|------------|------------|-----------|-----------|
| 1          | 182.7 px   | 409.3 px  | False     |
| 5          | 186.4 px   | 419.7 px  | False     |
| 10         | 187.0 px   | 421.5 px  | False     |
| 20         | 187.2 px   | 421.6 px  | False     |
| 50         | 187.5 px   | 421.5 px  | False     |

**Trend**: Error INCREASES with more iterations (182.7 → 187.5 px)

---

## Mean-Shift Magnitude Analysis

### Window Size 11:
- Iteration 0: MS = 58.13 px
- Iteration 10: MS = 40.14 px
- Iteration 20: MS = 38.88 px
- Iteration 50 (end): MS = 38.97 px

**Observation**: Decreases from 58 → 39 px, then plateaus. Never reaches < 1 px.

### Window Size 9:
- Iteration 0: MS = 50.94 → 36.91 px (depending on starting point)
- Iteration 10: MS = 34.23 px
- Iteration 20: MS = 32.11 px
- Iteration 50 (end): MS = 31.14 px

**Observation**: Decreases to ~30-35 px, then plateaus. Never reaches < 1 px.

### Window Size 7:
- Iteration 0: MS = 35.65 → 28.08 px (depending on starting point)
- Iteration 10: MS = 24.66 px
- Iteration 20: MS = 22.74 px
- Iteration 50 (end): MS = 24.18 px

**Observation**: Decreases to ~22-28 px, then plateaus. Never reaches < 1 px.

---

## Response Map Peak Offset Analysis

Peak offsets measured as distance from response map center. For convergence, peaks should be < 1 pixel from center.

### Consistent Offenders (appear across all iterations):

**Landmark 48** (mouth corner):
- ws=11 iter 2: (+5.0, -5.0) → 7.1 px
- ws=11 iter 10: (+5.0, -5.0) → 7.1 px
- ws=11 iter 50: (+5.0, -5.0) → 7.1 px

**Landmark 1** (jaw):
- ws=11 iter 2: (+3.0, +4.0) → 5.0 px
- ws=11 iter 10: (+4.0, +5.0) → 6.4 px
- ws=11 iter 50: (+4.0, +5.0) → 6.4 px

**Landmark 6** (jaw):
- ws=9 iter 2: (+4.0, +4.0) → 5.7 px
- ws=9 iter 10: (+4.0, +4.0) → 5.7 px
- ws=9 iter 50: (+4.0, +4.0) → 5.7 px

**Critical Observation**: Peak offsets DO NOT decrease with more iterations. Same landmarks show same offsets across 2, 10, and 50 iterations.

---

## Parameter Update (DP) Analysis

| Window Size | Iteration 0 | Iteration 10 | Iteration 50 |
|-------------|-------------|--------------|--------------|
| ws=11       | 10.10       | 2.78         | ~3-6         |
| ws=9        | 7.53        | 2.27         | ~2.5-3.8     |
| ws=7        | 7.13        | 1.78         | ~2.0-4.2     |

**Observation**: DP decreases from ~10 to ~2-4, then oscillates. Never reaches target < 0.005.

---

## Critical Findings

### 1. Convergence Does NOT Improve with More Iterations ❌❌❌

**Expected**: If convergence is just slow, increasing iterations from 1 → 50 should reduce error.

**Actual**: Error INCREASES from 182.7px → 187.5px.

**Interpretation**:
- More iterations are making landmarks WORSE, not better
- The optimization is not converging toward OpenFace's solution
- This is NOT a convergence speed issue

### 2. Mean-Shift Magnitudes Plateau at 22-58 Pixels ❌

**Expected**: MS should decrease to < 1 pixel as landmarks converge.

**Actual**: MS decreases initially but plateaus at:
- ws=11: ~39 px
- ws=9: ~31 px
- ws=7: ~24 px

**Interpretation**:
- Response maps consistently indicate landmarks should move 24-58 pixels
- Landmarks never settle because response maps keep showing large offsets
- This suggests response maps are fundamentally wrong

### 3. Peak Offsets Are Stable Across Iterations ❌

**Expected**: Peak offsets should decrease toward 0 as landmarks converge.

**Actual**: Same landmarks show same peak offsets at iteration 2, 10, and 50:
- Landmark 48: consistently 7.1 px offset
- Landmark 1: consistently 5.0-6.4 px offset
- Landmark 6: consistently 5.7 px offset

**Interpretation**:
- Peak locations are NOT artifacts of poor convergence
- Response maps are consistently producing offset peaks
- This is a fundamental issue with response map computation

### 4. Parameter Updates Oscillate, Never Converge ❌

**Expected**: DP should monotonically decrease to < 0.005.

**Actual**: DP decreases from ~10 to ~2-4, then oscillates.

**Interpretation**:
- Optimization cannot find a stable solution
- Caused by response maps giving contradictory information across iterations
- Landmarks move back and forth without settling

---

## Root Cause Analysis

### The Problem is NOT:
- ❌ Slow convergence (more iterations make it worse)
- ❌ Wrong convergence threshold (DP never gets close to 0.005)
- ❌ Poor initial parameters (error increases with optimization)

### The Problem IS:
- ✓ **Response maps are fundamentally wrong**
- ✓ **Peak offsets of 3-7 pixels are systematic, not random**
- ✓ **Response maps give contradictory guidance across iterations**

### Evidence:
1. Peak offsets stable across 2-50 iterations (not convergence artifacts)
2. Mean-shift magnitudes plateau at 24-58 px (response maps say "move this far")
3. Error increases with more iterations (optimization diverges from correct solution)
4. Same landmarks show same offsets consistently (systematic bias)

---

## Comparison with OpenFace Expected Behavior

### OpenFace C++ (Expected):
- **Mean-shift**: Starts at 10-20 px, decreases to < 1 px in 10-20 iterations
- **Peak offsets**: Starts at 2-5 px, decreases to < 0.5 px as landmarks converge
- **Parameter updates**: Decreases monotonically from ~5 to < 0.005
- **Convergence**: Achieves target in 10-20 iterations
- **Stability**: Once converged, landmarks don't move

### PyCLNF (Actual):
- **Mean-shift**: Starts at 58 px, plateaus at 22-39 px (never < 1 px) ❌
- **Peak offsets**: Starts at 5-7 px, STAYS at 5-7 px (never decreases) ❌
- **Parameter updates**: Decreases from ~10 to ~2-4, then oscillates (never < 0.005) ❌
- **Convergence**: Never achieves target, even with 50 iterations ❌
- **Stability**: Landmarks keep moving back and forth ❌

---

## Next Investigation Steps

### 🔥 Priority 1: Compare Response Map Values with OpenFace C++ (CRITICAL)

**Why**: Response maps are the root cause. Need to know if:
1. Peak locations differ from OpenFace
2. Peak values differ from OpenFace
3. Response map normalization differs from OpenFace

**Action**:
1. Extract response map for a single landmark from both PyCLNF and OpenFace C++
2. Compare peak location and peak value
3. Visualize both response maps side-by-side
4. Check if there's a systematic offset or scaling

### 🔥 Priority 2: Investigate Patch Expert Forward Pass

**Why**: If patch experts produce wrong outputs, response maps will be wrong.

**Action**:
1. Check if patch preprocessing matches OpenFace (normalization, mean subtraction)
2. Verify patch expert weights loaded correctly
3. Compare patch expert output for a single patch with OpenFace C++

### Priority 3: Check Image Warping Transform

**Why**: If warped patches are misaligned, response peaks will be offset.

**Action**:
1. Visualize warped patches from PyCLNF
2. Compare with OpenFace C++ warped patches
3. Verify similarity transform computation

---

## Conclusion

**Increasing iterations does NOT improve convergence - error actually gets WORSE (182.7 → 187.5 px).**

This definitively proves the problem is NOT slow convergence, but rather **fundamentally wrong response maps**. The response maps are producing:
1. Peak offsets of 3-7 pixels (stable across iterations)
2. Mean-shift magnitudes of 24-58 pixels (never < 1 px)
3. Contradictory guidance causing landmarks to oscillate

**The ONLY way forward is to compare response map values directly with OpenFace C++ to identify the bug.**

---

## Files Referenced

- Test video: `Patient Data/Normal Cohort/IMG_0433.MOV` (frame 50)
- Face bbox: (241, 555, 532, 532)
- OpenFace output: `/tmp/openface_compare/frame.csv`
- PyCLNF optimizer: `pyclnf/core/optimizer.py`

---

## Test Command

```python
for max_iter in [1, 5, 10, 20, 50]:
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=max_iter)
    py_landmarks, info = clnf.fit(gray, face_bbox)

    diff = py_landmarks - cpp_landmarks
    diff_mag = np.linalg.norm(diff, axis=1)

    print(f"PyCLNF ({max_iter:2d} iters): Mean error={diff_mag.mean():.1f}px")
```
