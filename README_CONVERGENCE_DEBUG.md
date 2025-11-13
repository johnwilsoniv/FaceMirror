# OpenFace Convergence Bug: Complete Analysis Package

## Documents Generated

This package contains complete analysis of the OpenFace mean-shift to PDM parameter update pipeline:

1. **OPENFACE_MEANSHIFT_DEBUG.md** (14 KB)
   - Comprehensive 6-section breakdown
   - Mean-shift computation pipeline
   - NU_RLMS optimization loop structure
   - Jacobian computation details
   - Parameter update application
   - Shape rendering from parameters
   - Debugging checklist with 5 potential bug categories
   - Critical equations to verify
   - Key file locations and line numbers

2. **MEANSHIFT_QUICK_REFERENCE.md** (6.7 KB)
   - Executive summary of data flow
   - Critical equations (mean-shift, Jacobian projection, parameter update)
   - 4 common failure modes with root causes and debug code
   - Validation checklist (9 items)
   - 3 hypotheses for the bug

3. **INSTRUMENTATION_POINTS.md** (8.4 KB)
   - 9 strategic instrumentation locations in the code
   - Copy-paste ready debug statements
   - Expected output pattern for working case
   - Red flags to watch for

---

## The Bug in One Sentence

**Landmarks aren't moving despite correct response maps because the mean-shift vectors aren't being correctly projected through the Jacobian into parameter space, or the parameter updates aren't translating to 2D landmark motion.**

---

## 3-Step Debugging Process

### Step 1: Verify Mean-Shift Computation (5 min)
Add instrumentation from **INSTRUMENTATION_POINTS.md Location 1-2**

Expected: `mean_shifts` should be non-zero and point toward response peaks

If fails: Check coordinate space transformation (sim_img_to_ref / sim_ref_to_img)

### Step 2: Verify Jacobian-Parameter Projection (5 min)
Add instrumentation from **INSTRUMENTATION_POINTS.md Location 3-5**

Expected: 
- Jacobian should be full-rank (zero rows = invisible landmarks)
- Hessian condition number < 1e6
- param_update should be non-zero and mixed-sign

If fails: Check Jacobian computation or weight matrix

### Step 3: Verify Parameter-to-Landmark Mapping (5 min)
Add instrumentation from **INSTRUMENTATION_POINTS.md Location 6**

Expected: CalcShape2D should show > 1 pixel movement per iteration

If fails: Bug is in UpdateModelParameters or CalcShape2D

---

## Most Likely Root Cause

Based on the code analysis, the bug is probably in one of these (in order of probability):

1. **Coordinate transformation error** (40% probability)
   - `sim_ref_to_img` is transposed or inverted
   - Mean-shifts in wrong units/scale for Jacobian
   - Check: Location 2 instrumentation will show > 2x magnitude change

2. **Jacobian-visibility mismatch** (35% probability)
   - Jacobian rows are zeroed for invisible landmarks
   - But weight matrix isn't zeroed consistently
   - Causes dimension/rank mismatch
   - Check: Location 3 will show > 0 zero rows

3. **Hessian singularity** (20% probability)
   - Weight matrix has very small values
   - Regularization is too strong
   - Cholesky solver produces garbage
   - Check: Location 5 will show cond_num > 1e6

4. **Parameter update application bug** (5% probability)
   - UpdateModelParameters has rotation handling bug
   - CalcShape2D isn't using updated parameters
   - Check: Location 6 will show shape_change = 0

---

## Quick Verification Checklist

Before diving into C++ code, verify:

```
[ ] Response maps are non-zero and have clear peaks
[ ] Peaks are near where landmarks should be
[ ] Mean-shift vectors point from current→peak (not backwards)
[ ] Jacobian is not all zeros (rows > 0, cols > 0)
[ ] Weight matrix is not all zeros
[ ] Hessian is not singular (condition number < 1e10)
[ ] Parameter updates are nonzero
[ ] CalcShape2D moves landmarks after parameter update
```

---

## Key Equations

### Mean-Shift (Response-Map Space)
```
mean_shift[i] = E[position | response] - current_position
              = sum(response * KDE * position) / sum(response * KDE) - dx
```

### Parameter Update (Gauss-Newton)
```
(J^T W J + λI) * delta_p = J^T W * mean_shifts
delta_p = (Hessian)^-1 * (J^T W * mean_shifts)
```

### New Landmarks
```
new_landmarks = CalcShape2D(params_local + delta_p[6:], params_global + delta_p[0:5])
```

---

## File Locations (Absolute Paths)

| Component | File | Lines |
|-----------|------|-------|
| Mean-shift | `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp` | 820-935 |
| NU_RLMS loop | LandmarkDetectorModel.cpp | 990-1191 |
| **CRITICAL:** Jacobian projection | LandmarkDetectorModel.cpp | 1107 |
| **CRITICAL:** Parameter update | LandmarkDetectorModel.cpp | 1131 |
| Jacobian computation | `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/PDM.cpp` | 346-450 |
| **CRITICAL:** UpdateModelParameters | PDM.cpp | 454-506 |
| Shape rendering | PDM.cpp | 159-188 |

---

## How to Use This Package

1. **Start with MEANSHIFT_QUICK_REFERENCE.md**
   - Understand the data flow
   - Identify which of the 4 failure modes matches your symptoms

2. **Go to INSTRUMENTATION_POINTS.md**
   - Pick the relevant location (1-9)
   - Copy-paste the debug code
   - Recompile and run

3. **Analyze the output**
   - Compare against "Expected Output Pattern"
   - Look for red flags
   - Identify which step breaks

4. **Use OPENFACE_MEANSHIFT_DEBUG.md**
   - Section on "Potential Bugs - Debugging Checklist"
   - Detailed explanation of that step
   - Equations to verify

---

## Example Debugging Session

```bash
# 1. Add instrumentation to LandmarkDetectorModel.cpp at locations 1, 3, 5, 6
# 2. Recompile
cd /path/to/build
cmake ..
make -j4

# 3. Run with output to file
./landmark_detector video.mp4 2>&1 | tee debug.log

# 4. Check output
grep "ITERATION 0" debug.log -A 20

# Expected to see:
# - Mean-shift norm is large (e.g., 15.3)
# - Jacobian has no zero rows
# - Hessian condition number < 100
# - param_update is nonzero
# - Shape change is > 1 pixel

# If any of these fail, trace through that step
```

---

## Summary of Root Cause by Symptom

**Symptom:** Mean-shift correct, but landmarks don't move
→ Check INSTRUMENTATION_POINTS Location 6

**Symptom:** Response maps correct, no mean-shift
→ Check INSTRUMENTATION_POINTS Location 1-2

**Symptom:** Mean-shift exists but parameter updates are zero
→ Check INSTRUMENTATION_POINTS Location 3-5

**Symptom:** Parameter updates nonzero but Jacobian projection weird
→ Check Location 4 and review OPENFACE_MEANSHIFT_DEBUG.md Section 4

---

## Next Steps

1. Apply instrumentation from INSTRUMENTATION_POINTS.md
2. Run one test case
3. Look at output and find which step fails
4. Read corresponding section in OPENFACE_MEANSHIFT_DEBUG.md
5. Fix the identified bug
6. Repeat until convergence works

Good luck! This should be fixable in under an hour once you pinpoint the failure point.

