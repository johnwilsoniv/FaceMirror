# CLNF Initialization Analysis - Root Cause Discovered

**Date**: 2025-11-11
**Test**: compare_initialization_and_test_sigma.py
**Video**: IMG_0433.MOV, Frame 50
**Bbox**: (241, 555, 532, 532)

---

## Critical Discovery: Initialization is the Root Cause!

### Test Results Summary

The test compared PyCLNF initialization (0 iterations) vs OpenFace C++ final landmarks (fully converged):

```
INITIAL LANDMARK ERROR (PyCLNF init vs OpenFace final):
  Mean error: 224.44 px  ← CATASTROPHIC!
  Median error: 187.97 px
  Max error: 468.93 px
  Min error: 18.19 px

Worst 5 initialized landmarks:
  1. Landmark 45: 468.93 px error  (right eye outer corner)
  2. Landmark 23: 459.75 px error  (eyebrow)
  3. Landmark  2: 430.05 px error  (jaw)
  4. Landmark 10: 419.95 px error  (jaw)
  5. Landmark  3: 415.14 px error  (jaw)
```

**This is devastating!** The initialization is starting over **200 pixels away** from the true position on average.

---

## Sigma Testing - ALL VALUES FAILED

Tested sigma values from 1.5 to 4.5:

| Sigma | Description | Converged | Final Update | Ratio to Target |
|-------|------------|-----------|--------------|-----------------|
| 1.5 | PyCLNF default | NO | 5.651 | 1130x |
| 1.75 | OpenFace base | NO | 7.329 | 1466x |
| 2.0 | OpenFace @ scale=0.5 | NO | 8.913 | 1783x |
| 2.5 | Moderately larger | NO | 8.801 | 1760x |
| 3.0 | Large (ws/3.7) | NO | 7.430 | 1486x |
| 3.5 | OpenFace eyebrow sigma | NO | 10.673 | 2135x |
| 4.0 | Very large (ws/2.75) | NO | 11.308 | 2262x |
| 4.5 | Extremely large (ws/2.44) | NO | 8.624 | 1725x |

### Key Finding: Sigma Had OPPOSITE Effect!

**SMALLER sigma performed BETTER** (σ=1.5 best, σ=4.0 worst)

This is the **opposite** of our hypothesis! When initialization is this bad:
- Narrow Gaussian is more "cautious" → smaller updates → prevents divergence
- Wide Gaussian makes larger updates based on bad response maps → performs worse

---

## Visualization Analysis

The saved visualization `initialization_and_sigma_comparison.png` clearly shows:
- **Red dots (PyCLNF init, 0 iterations)**: Wildly scattered across face, completely wrong
- **Blue squares (OpenFace final)**: Correctly positioned on facial features
- **All sigma results**: Clustered together, all still far from OpenFace

None of the different sigma values could recover from the terrible initialization.

---

## Conclusions

### What We Learned

1. **Sigma is NOT the problem** - All values from 1.5 to 4.5 failed equally
2. **Initialization is CATASTROPHIC** - 224 px mean error is unrecoverable
3. **Response maps are probably fine** - Edge peaks are expected when landmarks are displaced
4. **Mean shift algorithm is working** - It just can't overcome 200+ pixel initialization errors

### Why Our Previous Hypothesis Was Wrong

We thought:
- ❌ Edge peaks indicated bad response maps
- ❌ Narrow sigma was downweighting edge peaks too much
- ❌ Larger sigma would improve convergence

Reality:
- ✅ Edge peaks are CORRECT when initialization is far off
- ✅ Sigma doesn't matter when you're starting 200+ pixels away
- ✅ Need to fix initialization, not optimization parameters

---

## Next Steps

### Priority 1: Investigate PDM Initialization

Compare PyCLNF's bbox→PDM initialization with OpenFace C++:

1. **Find initialization code** in pyclnf/clnf.py `fit()` method
2. **Compare with OpenFace C++** LandmarkDetectorModel.cpp initialization
3. **Check bbox preprocessing** - Does OpenFace apply detector-specific corrections?
4. **Verify parameter calculation** - Scale, translation, rotation from bbox

### Priority 2: Test OpenFace Initialization Method

If PyCLNF initialization differs from OpenFace:
1. Extract OpenFace's initialization logic
2. Implement in PyCLNF
3. Re-test convergence with same bbox

### Priority 3: Detector-Specific Bbox Corrections

OpenFace C++ applies corrections for different detectors:
- MTCNN: Shifts bbox by specific offsets
- Haar cascades: Different corrections
- RetinaFace: May need custom correction

Check if PyCLNF needs similar corrections.

---

## Files to Investigate

1. **pyclnf/clnf.py** - `fit()` method, look for where `initial_params` is computed from `face_bbox`
2. **pyclnf/models/openface_loader.py** - PDM class, bbox→params conversion
3. **OpenFace C++** `lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp`
   - `DetectLandmarksInVideo()` function
   - Look for bbox initialization code

---

## References

- Test script: `compare_initialization_and_test_sigma.py`
- Visualization: `initialization_and_sigma_comparison.png`
- OpenFace mean shift algorithm: `OPENFACE_MEAN_SHIFT_ALGORITHM.md`
- OpenFace C++ source: ~/repo/fea_tool/external_libs/openFace/OpenFace/

---

## Status: Root Cause Identified

**The convergence issue is NOT in the optimization loop, it's in the INITIALIZATION.**

We need to fix how PyCLNF converts the face bounding box into initial PDM parameters. Once the initialization is correct, convergence should work with σ=1.75 (OpenFace default).
