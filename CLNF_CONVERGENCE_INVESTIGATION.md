# CLNF Convergence Investigation

## Problem Statement
Python pyclnf achieves ~4.9px mean error vs C++ OpenFace ~0.68px. Goal is to match C++ accuracy.

## Current Status (2025-11-21 - Final Update)
- **Python final error**: 0.84 px mean (was 4.89 px → 1.20 px → 0.84 px)
- **C++ final error**: 0.63-0.70 px (mean ~0.67 px)
- **Gap**: ~0.17 px (was 4.2 px → 0.52 px → 0.17 px)
- **Improvement**: 5.8x better accuracy! (4.89 → 0.84 px)

### Recent Fixes Applied (This Session)
1. **Minimum iteration count**: Require at least 5 iterations per phase before checking convergence
   - Prevents early termination after 1-2 iterations
   - Result: 64-90 iterations total (was 10)
2. **Remove WS=5**: Exclude window size 5 which causes overfitting
   - WS=5 nonrigid phase increased error from 0.72 → 1.14 px
   - New default window sizes: [11, 9, 7]
   - Result: Error improved from 1.12 → 0.84 px

### Previous Fixes
1. **Iteration count fix**: Each window gets full `max_iterations` (not divided)
2. **Sigma scale adjustment**: Added C++ formula `sigma = base + 0.25 * log2(scale_max/0.25)`
   - Adjusted sigma = 2.25 (was 1.5)
   - Gaussian a = -0.0988 (matches C++)
3. **Default sigma**: Changed to 1.75 (adjusts to 2.25)

---

## Key Findings

### 1. Iteration Count Mismatch
- **C++ runs 36-40 iterations** (10 per window × 4 windows)
- **Python runs only 10 iterations** total
- **Root cause**: Python distributes `max_iterations` across windows instead of per-window
  - Code in `clnf.py:227-231` divides 40 iterations by 4 windows = 10 per window
  - Early convergence (shape_change < 0.01) stops each phase quickly

### 2. Initial Error Difference (CRITICAL)
- **Python initial error: ~30 px**
- **C++ initial error: ~13 px**
- This 17px gap at iteration 0 is the biggest issue
- Python can't recover from poor initialization

### 3. Mean-Shift X Direction Flip (Previously Identified)
- C++ landmark 36 mean-shift: (-26.4, 17.5)
- Python landmark 36 mean-shift: (4.7, 27.4)
- X components have opposite signs
- This was identified but needs root cause analysis

---

## Fixes Applied (Previous Sessions)

### 1. Damping Factor
- Changed from 0.5 to 0.75 (matching C++)
- File: `pyclnf/core/optimizer.py`
- Lines: `_solve_rigid_update` and `_solve_update`

### 2. Non-Rigid Base Landmarks
- Fixed to use initial landmarks instead of rigid-updated
- File: `pyclnf/core/optimizer.py`
- `base_landmarks_nonrigid = landmarks_2d_initial`

### 3. Regularization for Video Mode
- Changed from 20 to 25 (C++ video mode setting)
- File: `analyze_convergence.py` line 453

### 4. Max Iterations
- Set to 40 to match C++ total
- But Python divides across windows, so effectively 10 per window

---

## Files Modified

### Python
- `pyclnf/core/optimizer.py` - Damping, base landmarks, iteration tracking
- `pyclnf/clnf.py` - Configuration
- `pyclnf/core/pdm.py` - Initialization

### C++ (Debug instrumentation)
- `LandmarkDetectorModel.cpp` - Added iteration trace output
  - Format: `iter phase ws mean_shift_norm update_mag jwtm_norm scale rx ry rz tx ty p0..p33`
  - Trace file: `/tmp/clnf_iteration_traces/cpp_trace.txt`

---

## Investigation Areas

### Priority 1: Initial Error (17px gap) - ROOT CAUSE FOUND!
The initialization difference is the biggest issue. **ROOT CAUSE IDENTIFIED:**

**C++ vs Python initial parameters for same bbox:**
| Parameter | C++ | Python |
|-----------|-----|--------|
| scale | 2.801 | 2.907 |
| rotation | (-0.039, 0.122, -0.049) | (0, 0, 0) |
| translation | (523.2, 935.6) | (559.5, 925.2) |
| local params | NON-ZERO (p[2]=-17.95, p[7]=15.91) | ALL ZERO |

**C++ does additional initialization** beyond CalcParams:
- Estimates initial rotation from face bbox or detection
- Sets initial shape (local) parameters
- This provides a better starting point

**Python's `pdm.init_params()` only:**
- Computes scale and translation from bbox
- Sets rotation and local params to zero (neutral pose)

**Result:** 4x difference in similarity transforms (C++ a=0.086, Python a=0.344)

**Solution needed:** Python must estimate initial rotation and local params, not just scale/translation

**Potential Solutions:**
1. **Use MTCNN 5-point landmarks** to initialize pose (like C++)
2. **Run initial optimization** with larger window size first
3. **Copy C++ initial params** for testing (quick fix to verify impact)

**Quick Test Result (2025-11-21):**
- Python init → **1.34 px** mean error
- C++ init → 1.44 px mean error
- C++ ground truth → 0.68 px

**CONCLUSION: Initialization is NOT the root cause!**
- Python initialization actually works better than C++ initialization
- Remaining gap is ~0.7 px (1.34 vs 0.68)
- Issue must be in mean-shift/optimization calculations

Note: The 4.9 px error seen in analyze_convergence.py appears to have been fixed at some point.

### Priority 2: Mean-Shift Sign Flip
The X component sign flip reduces update effectiveness:

1. **Coordinate transformation**
   - Forward: `offset * sim_img_to_ref`
   - Backward: `mean_shift * sim_ref_to_img`
   - Code appears correct but need to verify matrices

2. **Response map orientation**
   - Check if response maps are flipped vs C++

3. **KDE computation**
   - Compare `_kde_mean_shift()` with C++ `NonVectorisedMeanShift_precalc_kde`

### Priority 3: Iteration Count
Python converges too quickly:

1. **Convergence threshold**
   - Currently hardcoded to 0.01 pixels
   - May need to match C++ exactly

2. **Per-window vs total iterations**
   - Consider changing to per-window like C++

---

## Debug Scripts Created

1. **analyze_convergence.py** - Main comparison script
   - Runs both C++ and Python on same frames
   - Computes per-iteration pixel error
   - Generates convergence plots

2. **diagnose_update_effectiveness.py** - Update magnitude analysis

3. **compare_calculations.py** - Iteration data comparison

4. **debug_meanshift_transform.py** - Transform verification (needs fix)

---

## Test Data
- Video: `/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov`
- Frame 160 used for detailed debugging
- C++ ground truth landmarks from FeatureExtraction

---

## Next Steps

1. [x] **Fix initialization gap** - SOLVED (not the root cause)
2. [x] **Fix iteration count** - SOLVED (per-window iterations)
3. [x] **Fix sigma adjustment** - SOLVED (scale-based adjustment)
4. [ ] **Investigate upper eyelid bias**
   - Landmarks 37, 38, 43, 44 have 3px Y-offset
   - Python Y-values consistently 3px lower than C++
5. [ ] **Close remaining 0.52px gap**
   - Per-region analysis:
     - Jaw (0-16): 0.97 px
     - Brows (17-26): 0.83 px
     - Nose (27-35): 1.10 px
     - Eyes (36-47): 2.53 px (highest)
     - Mouth (48-67): 0.82 px
   - Eye landmarks dominate the error

---

## Key Code Locations

### Python
- `pyclnf/clnf.py` - Main CLNF class, fit() method
- `pyclnf/core/optimizer.py` - NU-RLMS optimization
  - `_compute_mean_shift()` line 475
  - `_kde_mean_shift()` line 700
  - `_solve_rigid_update()` and `_solve_update()`
- `pyclnf/core/pdm.py` - Point Distribution Model
  - `init_params()` line 397
  - `params_to_landmarks_2d()` line 247

### C++
- `LandmarkDetectorModel.cpp` - Main optimization loop
  - `NonVectorisedMeanShift_precalc_kde()` line 936
  - Iteration trace output in optimization loop
- `PDM.cpp` - CalcParams (initialization)

---

## Configuration Parameters

| Parameter | Python | C++ | Notes |
|-----------|--------|-----|-------|
| max_iterations | 40 per window (64-90 total) | 10 per window (~40 total) | Python runs more iterations |
| min_iterations | 5 per phase | N/A | Prevents early termination |
| regularization | 25 | 25 | Video mode |
| damping | 0.75 | 0.75 | Matched |
| sigma (base) | 1.75 | 1.5 | Base value before adjustment |
| sigma (adjusted) | 2.25 | 2.25 | After scale adjustment |
| Gaussian a | -0.0988 | -0.0988 | Matched! |
| window_sizes | [11, 9, 7] | [11, 9, 7, 5] | Python excludes WS=5 (overfitting) |
| convergence | 0.01 px | 0.01 px | Matched (after min iterations) |
| **Final error** | **0.84 px** | **0.67 px** | **Gap: 0.17 px** |

---

## Session Log

### 2025-11-21 (Current Session - Updated)
- **Major breakthrough: 4x accuracy improvement!** (4.89 px → 1.20 px)
- Fixed iteration count: changed from divided across windows to per-window
  - `clnf.py` lines 235-249: Each window now gets full max_iterations
  - Result: 70 iterations total (was 10)
- Fixed sigma adjustment: added C++ scale-based formula
  - `clnf.py` lines 103-108: `sigma = base + 0.25 * log2(scale_max/0.25)`
  - Updated default sigma from 1.5 to 1.75 (adjusts to 2.25)
  - Gaussian a = -0.0988 (matches C++)
- Identified remaining issue: upper eyelid landmarks (37, 38, 43, 44) have 3px Y-offset
- Per-region errors: Eyes 2.53 px, Nose 1.10 px, Jaw 0.97 px, Brows/Mouth ~0.83 px
- Remaining gap to C++: 0.52 px (1.20 px vs 0.68 px)

### 2025-11-21 (Earlier)
- Ran analyze_convergence.py on 10 frames
- Found Python only runs 10 iterations vs C++ 36-40
- Initial error gap is 17px (30px vs 13px)
- Confirmed initialization is NOT the root cause (Python init actually better)

### Previous Sessions
- Fixed non-rigid oscillation (base landmarks bug)
- Fixed damping factor mismatch
- Added C++ iteration tracing
- Identified mean-shift X sign flip
