# Investigation Plan: Understanding Why C++ Gets Upright Faces

## Current Status

We have completed systematic analysis and research. Here's what we know:

### Rotation Measurements (Current State)

| Configuration | Mean Angle | Stability (std) | Expression Sensitive? |
|---------------|------------|-----------------|----------------------|
| C++ (reference) | ~0° | ~0° | No |
| Python + 24 points (with eyes) | -5° | 4.51° | Yes (6.4° jump) |
| Python + 16 points (no eyes) | +30° | 1.47° | No |

### Key Insight from Research

**MOST LIKELY ROOT CAUSE: SVD Sign Ambiguity**

- SVD decomposition has arbitrary column signs in U and V matrices
- numpy.linalg.svd and OpenCV cv::SVD may choose different signs
- This propagates through Kabsch algorithm → different rotation matrices
- Explains why different point sets (24 vs 16) give different rotations
- Explains why each approach is internally consistent

---

## The SVD Sign Hypothesis Explained

### What Is Sign Ambiguity?

In SVD: `A = U × S × V^T`

But also valid: `A = (-U) × S × (-V)^T` (flipping both)

Or even: `A = U' × S × V'^T` where U' and V' have some columns flipped

**All mathematically correct, but produce different results in Kabsch!**

### How It Affects Kabsch

```
Rotation = V^T × correction_matrix × U^T
```

If U or V columns have different signs:
- Rotation matrix will be different
- Could be off by 5°, 30°, even 180°
- But still consistent for same input

### Why Our Observations Fit This

1. **Python with eyes = -5°**
   - numpy SVD chooses signs1 for 24-point matrix
   - Consistent across frames (same signs each time)
   - But different from OpenCV's choice

2. **Python without eyes = +30°**
   - numpy SVD chooses signs2 for 16-point matrix (different matrix!)
   - Consistent across frames
   - But different from both OpenCV and 24-point case

3. **C++ = 0°**
   - OpenCV SVD chooses signs3
   - This is the "correct" orientation
   - Or rather, the reference we want to match

4. **Expression sensitivity with eyes**
   - Eye closure changes the input matrix
   - Changed matrix → SVD might choose different signs
   - Without eyes: matrix more stable → signs more stable

---

## Three Possible Explanations

### Explanation 1: SVD Sign Differences (Likelihood: 70%)

**numpy and OpenCV choose different signs for U/V columns**

Evidence:
- ✓ Well-documented issue
- ✓ Explains all observations
- ✓ Explains both 5° and 30° offsets
- ✓ Explains expression sensitivity

Test: Use cv2.SVDecomp instead of np.linalg.svd

### Explanation 2: Hidden Rotation in C++ (Likelihood: 20%)

**C++ applies additional rotation we haven't found**

Evidence:
- ⚠️ We've checked thoroughly
- ⚠️ No rotation found after AlignFace
- ⚠️ PDM mean shape is used as-is

Test: Add debug output to C++ to print actual rotation matrix

### Explanation 3: Different Rigid Points at Runtime (Likelihood: 10%)

**C++ uses different rigid points than source code shows**

Evidence:
- ⚠️ Source code clearly shows 24 indices
- ⚠️ No runtime configuration found
- ⚠️ Would be very unusual

Test: Print actual rigid points used in C++ at runtime

---

## Proposed Investigation Steps (No Implementation Yet)

### Phase 1: Verify SVD Hypothesis (CRITICAL - Do This First)

**Question:** Does using OpenCV's SVD in Python match C++ output?

**What to check:**
1. Does cv2.SVDecomp exist and work with our matrices?
2. If we replace numpy SVD with OpenCV SVD, do rotations match?
3. Do U/S/V values differ between numpy and OpenCV SVD?

**Expected outcome:**
- If rotations match → SVD sign ambiguity confirmed!
- If rotations still differ → Need to investigate other causes

**Estimated time:** 30 minutes to test

---

### Phase 2: Compare SVD Outputs Directly

**Question:** How exactly do numpy and OpenCV SVD outputs differ?

**What to compare:**
For same input matrix (src.T @ dst):

| Component | numpy | OpenCV | Difference? |
|-----------|-------|--------|-------------|
| Singular values (S) | ? | ? | Should be nearly identical |
| U matrix | ? | ? | May have sign differences |
| V^T matrix | ? | ? | May have sign differences |
| U column 0 signs | ? | ? | Check each column |
| U column 1 signs | ? | ? | |
| V column 0 signs | ? | ? | |
| V column 1 signs | ? | ? | |

**Expected outcome:**
- S values: ~identical (within 10^-6)
- U/V: Some columns may have opposite signs

**Estimated time:** 15 minutes to test

---

### Phase 3: Visual Inspection Plan

**For your visual inspection, please check these specific things:**

#### Test 1: C++ Baseline
Look at C++ aligned faces for frames: 1, 493, 617, 863

Questions:
1. Are eyes perfectly horizontal? (rotation = 0°)
2. Or approximately horizontal? (rotation within ±2°)
3. Is there ANY visible rotation difference between frames?
4. Does frame 617 (eyes closed) look identical to 493 (eyes open)?

#### Test 2: Python Comparison
Compare Python (current, with eyes) to C++ for same frames:

Questions:
1. Visual estimate: How many degrees is Python rotated vs C++?
2. Is the rotation consistent across all frames?
3. Does frame 617 look different from 493 in Python?
4. Other than rotation, does alignment quality look similar?

#### Test 3: Stability Check
Line up all 4 frames side by side (C++):

Question:
1. If you overlay them, would face orientation be identical?
2. Can you see ANY rotation variation?

---

### Phase 4: If SVD Doesn't Explain It

**Alternative investigations (only if Phase 1 fails):**

1. **Check PDM mean shape usage**
   - Add debug to print C++ mean_shape after loading
   - Compare to Python's loaded mean_shape
   - Look for any rotation applied to mean_shape

2. **Check rigid point extraction**
   - Add debug to print actual rigid points in C++
   - Verify they match source code (24 indices)
   - Check if any filtering happens

3. **Check for coordinate transforms**
   - Search for rotation matrices outside AlignFace
   - Look for "canonical pose" or "reference frame" transforms
   - Check CLNF landmark detector output

4. **Test with synthetic data**
   - Create perfectly upright face landmarks
   - Process through both C++ and Python
   - See if Python still produces rotation

---

## Decision Tree

```
START: Why does C++ get upright faces?
  │
  ├─ Test: Use cv2.SVDecomp in Python
  │   │
  │   ├─ Result: Rotations match C++
  │   │   └─ SOLUTION: Use OpenCV SVD in Python (DONE!)
  │   │
  │   └─ Result: Rotations still differ
  │       └─ Continue to Phase 4 (alternative investigations)
  │
  └─ [If needed] Test: Compare SVD outputs
      └─ [If needed] Test: Debug C++ runtime
          └─ [If needed] Test: Synthetic data
```

---

## What We Need From You

### 1. Visual Inspection Feedback

Please look at these comparison images and tell us:
- `cpp_expression_comparison.png` - Are all 3 C++ faces perfectly upright?
- How many degrees (roughly) is Python tilted vs C++?
- Any other observations about alignment quality?

### 2. Priority Decision

Which should we investigate first?
- **Option A:** Test SVD hypothesis (quick, high likelihood)
- **Option B:** Deep dive into C++ code (slow, lower likelihood)
- **Option C:** Accept stable offset and apply correction

### 3. Success Criteria Confirmation

Confirm these are the right metrics:
- ✓ Rotation angle (target: ~0° like C++)
- ✓ Rotation stability (target: std < 2°)
- ✓ Expression invariance (no jump on eye closure)
- ✗ NOT pixel correlation (we're ignoring this now)

---

## Summary

**Most Likely Problem:** numpy SVD and OpenCV SVD choose different signs for U/V matrices

**Quick Test:** Replace np.linalg.svd with cv2.SVDecomp in Python

**Expected Result:** Rotations will match C++ exactly

**If It Works:** Problem solved! Pure Python using cv2.SVDecomp

**If It Doesn't:** Continue systematic investigation per Phase 4

**Current Status:** Waiting for your visual inspection feedback and decision on which path to investigate first

---

## Files Created

1. `SYSTEMATIC_ROOT_CAUSE_ANALYSIS.md` - Complete list of possible causes
2. `SVD_RESEARCH_FINDINGS.md` - Research on numpy vs OpenCV SVD
3. `INVESTIGATION_PLAN_NO_CODE_YET.md` - This document

**No code changes made yet per your request.**
