# Dependency Analysis and Path Forward

## Your Questions Answered

### 1. Binary Wrapper Dependencies (vs pyfhog)

**Short answer:** More complex than pyfhog, but still feasible.

#### AlignFace Dependencies:
```
AlignFace requires:
├── OpenCV (cv::warpAffine, cv::Mat) ✓ Already have
├── PDM class (mean_shape from LandmarkDetector) ⚠️ Complex
├── RotationHelpers.h (Kabsch algorithm) ✓ Header-only
└── dlib (indirect, through LandmarkDetector) ⚠️ Large dependency
```

#### pyfhog Dependencies (for comparison):
```
pyfhog required:
├── OpenCV (basic Mat operations) ✓
└── FHOGFeatures (self-contained) ✓ Simple
```

**Complexity comparison:**
- **pyfhog:** Simple - single feature extraction function, minimal dependencies
- **AlignFace:** Medium - needs PDM object (which has dlib dependencies)
- **Landmark Detector:** HIGH - Would need entire CLNF model (PDM, patch experts, SVR models, dlib, OpenBLAS)

**Portability:**
- pyfhog: **Excellent** - compiles easily, minimal deps
- AlignFace wrapper: **Good** - needs OpenCV + dlib + OpenBLAS
- Landmark detector: **Poor** - heavyweight, many platform-specific builds

### 2. What Are We Missing?

You're absolutely right - there IS something we're missing! The tests reveal:

**Key Finding:** Rotation correction gives us:
- ✓ Better stability (std = 2.32° vs 4.51° with eyes)
- ✗ Lower correlation (r = 0.555 vs 0.765 with eyes)

**BUT** - I think we're using the WRONG correction angle! Let me test something...

The correction angle of -30.98° was computed to make the rotation ~0°, but maybe we should correct to match C++ output directly, not to 0°.

### 3. Landmark Detector Wrapping

**Short answer:** Not recommended - very complex.

The CLNF landmark detector requires:
- PDM (Point Distribution Model) - 3D face model
- Patch Experts - SVR/SVM models for each landmark
- Model files - Multiple large binary files
- dlib - Face detection, shape prediction
- OpenBLAS - Linear algebra
- OpenMP - Parallel processing

**Estimated effort:**
- AlignFace wrapper: 2-3 days
- Landmark detector wrapper: 2-3 weeks

**Alternative:** Use ONNX STAR detector (which you already have!)

## Test Results Summary

| Method | Correlation | Stability | Notes |
|--------|-------------|-----------|-------|
| With eyes (24 pts) | r = 0.765 | std = 4.51° | Original |
| No eyes raw (16 pts) | r = 0.439 | std = 2.32° | Terrible correlation |
| No eyes + correction | r = 0.555 | std = 2.32° | Better but still low |
| C++ reference | r = 1.000 | std = ~0° | Target |

## New Hypothesis: Optimal Correction

The issue is: **We don't know what angle C++ actually produces!**

Let me test measuring C++'s actual rotation angle from the aligned faces themselves...

## Proposed Next Steps

### Option 1: Find Optimal Correction (Quick Test - 1 hour)

1. Measure C++ output's actual rotation angle
2. Compute optimal correction to match it
3. Test if this improves correlation

**If this works:** Pure Python solution!

### Option 2: C++ Wrapper (Medium effort - 2-3 days)

Dependencies needed:
- OpenCV 4.x (already have)
- dlib (can include in wheel)
- OpenBLAS (can include in wheel)
- PDM file (bundle with package)

**Portability:** Good - Can create wheels for macOS/Linux/Windows

### Option 3: Hybrid Approach

- Use current Python with eyes (r=0.765)
- Test if it's sufficient for AU prediction
- Fall back to C++ wrapper only if needed

## What I Recommend Testing Next

Let me try ONE more thing before deciding: **measure the actual rotation in C++ aligned faces** and compute the optimal correction from that. This could give us a pure Python solution!

```python
# Pseudo-code:
for each_frame:
    cpp_aligned = load_cpp_aligned_face()

    # Measure actual rotation in C++ output
    # (e.g., by detecting face angle or using specific landmarks)
    cpp_angle = measure_face_rotation(cpp_aligned)

    # Our rotation without eyes
    python_angle = kabsch_rotation_no_eyes(frame)

    # Optimal correction
    correction = cpp_angle - python_angle
```

If the optimal correction is constant, we WIN!

## Bottom Line

**Your instinct is correct** - the rotation without eyes being "very stable" is a huge clue. We may just need the RIGHT correction angle, not the one I computed (which targets 0° rather than C++ output).

**Give me 30 minutes to test the optimal correction approach before we commit to the C++ wrapper.**
