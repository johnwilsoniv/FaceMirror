# Eye Refinement Debug Status

## Current Status (2024-11-22)

### Main CLNF: ✅ WORKING
- **Mean error: 1.24 px** (without eye refinement)
- Fixed window sizes to [11, 9, 7, 5] matching C++
- Runs ~66 iterations, converges well

### Eye Refinement: ❌ DEGRADING ACCURACY
- **Degrades overall by 0.06 px** (1.24 → 1.30 px)
- **Degrades eye landmarks by 0.36 px** (2.39 → 2.75 px)
- Eye model has 0 sigma components (by design, not a bug)

---

## Debug Checkpoints Needed

We need to find where C++ and Python diverge. Here are the key comparison points:

### 1. Eye Model Initialization
**Input from main CLNF:**
- [ ] Main model landmarks (36-41 for left eye)
- [ ] Main model rotation params (rx, ry, rz)
- [ ] Main model scale

**Questions:**
- Are we passing the same input landmarks to eye model?
- Is the rotation/scale transfer correct?

### 2. Eye Model PDM Initialization
**C++ does:**
```cpp
// From FaceAnalyser/FaceAnalyser.cpp
part.model.pdm.CalcParams(part.model.params_global, bounding_box, part.model.params_local);
```

**Python should match:**
- [ ] Bounding box calculation from 6 input landmarks
- [ ] Initial params_global (scale, rot, tx, ty)
- [ ] Initial params_local (should be zeros)

### 3. Initial 28 Eye Landmarks
After PDM initialization, compare the 28 eye landmark positions:
- [ ] C++ initial 28 landmarks
- [ ] Python initial 28 landmarks
- [ ] Error at this stage

### 4. Response Maps
For each of the 28 eye landmarks:
- [ ] Patch extraction location
- [ ] Response map values
- [ ] Mean-shift vectors

### 5. Optimization Loop
- [ ] Number of iterations
- [ ] Convergence behavior
- [ ] Final 28 landmarks

### 6. Mapping Back to Main Model
The 28 eye landmarks map back to main model landmarks 36-41:
```
Main 36 → Eye 8
Main 37 → Eye 10
Main 38 → Eye 12
Main 39 → Eye 14
Main 40 → Eye 16
Main 41 → Eye 18
```
- [ ] Is this mapping correct?
- [ ] Are we using the right indices?

---

## Files to Compare

### C++ Debug Output (need to add)
- `/tmp/cpp_eye_init_debug.txt` - Eye model initialization
- `/tmp/cpp_eye_response_maps.txt` - Response maps for eye landmarks

### Python Debug Output (existing)
- `/tmp/python_eye_init_debug.txt` - Already created
- Eye patch expert debug in `eye_patch_expert.py`

---

## Known Values to Match

### From Previous Debug Session

**C++ Eye Initialization (left eye):**
```
Model name: left_eye_28
Input landmarks (from main model):
  Eye_8 (main 36): (399.33, 825.12)  <- Need to verify this matches Python
  Eye_10 (main 37): (417.83, 806.29)
  Eye_12 (main 38): (444.60, 804.63)
  Eye_14 (main 39): (469.48, 821.30)
  Eye_16 (main 40): (445.70, 828.04)
  Eye_18 (main 41): (418.64, 831.27)

Fitted params_global:
  scale: 3.355560
  rot: (-0.129031, 0.168620, -0.091544)
  tx, ty: (433.60, 817.99)
```

**Python Eye Initialization:**
- [ ] Need to capture and compare

---

## Next Steps

### Immediate Actions:
1. **Add debug output to Python eye model initialization**
   - Capture input landmarks (main model 36-41)
   - Capture fitted params_global
   - Capture initial 28 landmarks

2. **Compare with C++ debug output**
   - Run C++ with same image/bbox
   - Extract eye initialization values
   - Find first point of divergence

3. **Create side-by-side comparison**
   - Input landmarks match?
   - Params_global match?
   - Initial 28 landmarks match?

### Code Locations:

**Python eye refinement entry point:**
```python
# pyclnf/clnf.py line 333
landmarks = self.eye_model.refine_eye_landmarks(gray, landmarks, 'left', main_rotation, main_scale)
```

**Python eye model initialization:**
```python
# pyclnf/core/eye_patch_expert.py - HierarchicalEyeModel.refine_eye_landmarks()
```

**C++ eye refinement:**
```cpp
// LandmarkDetector/src/LandmarkDetectorFunc.cpp
// DetectLandmarksInImage() calls hierarchical part models
```

---

## Hypotheses for Degradation

### Hypothesis 1: Input Landmark Mismatch
The landmarks passed to eye model differ between C++ and Python.
- Python main CLNF has 1.24 px error
- This error propagates to eye model input

### Hypothesis 2: Params Initialization
The CalcParams for eye model differs:
- Rotation inheritance
- Scale calculation
- Translation computation

### Hypothesis 3: Optimization Differences
Eye model optimization converges differently:
- Different regularization
- Different convergence criteria
- Different number of iterations

### Hypothesis 4: Mapping Error
The 28→6 landmark mapping is incorrect:
- Wrong indices
- Wrong interpolation/selection

---

## Test Commands

### Run Python with eye refinement debug:
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
env PYTHONPATH="pyclnf:pymtcnn:." python3 -c "
from pyclnf import CLNF
import cv2
video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
ret, frame = video.read()
video.release()
bbox = [345.858, 726.593, 394.655, 423.916]
clnf = CLNF(use_eye_refinement=True)
landmarks = clnf.fit(frame, bbox)[0]
print('Done - check /tmp for debug files')
"
```

### Run C++ with eye debug enabled:
```bash
/path/to/FeatureExtraction -f "video.mov" ...
# Check /tmp/cpp_eye_* files
```

---

## Progress Log

### 2024-11-22 Session 1
- [x] Fixed main CLNF to 1.24 px error
- [x] Confirmed eye model has 0 sigma components (expected)
- [x] Identified eye refinement degrades by 0.36 px
- [x] Captured Python eye initialization debug

### Current Python Eye Model Values:
```
Pre-refinement eye landmarks (main model):
  36: (398.7335, 825.5589)
  37: (417.7131, 805.7572)
  38: (445.3858, 804.2058)
  39: (470.9699, 822.4481)
  40: (446.2884, 828.3587)
  41: (418.0464, 832.1575)

Fitted params_global:
  scale: 3.488194
  rot: (-0.149446, 0.188008, -0.076481)
  tx, ty: (433.863586, 817.991652)

Post-refinement:
  36: (398.5318, 825.4911)  delta=0.21 px
  37: (417.1515, 806.3016)  delta=0.78 px
  38: (446.3131, 804.4960)  delta=0.97 px
  39: (470.2698, 821.8138)  delta=0.94 px
  40: (446.2949, 829.0155)  delta=0.66 px
  41: (417.5910, 831.2065)  delta=1.05 px
```

### Key Parameters to Check:

**CRITICAL FINDING: Window Size Mismatch!**

C++ uses MULTIPLE window sizes for eye_28 model:
```cpp
windows_large = [3, 5, 9]  // Three hierarchical passes!
reg_factor = 0.5
sigma = 1.0
```

Python only uses:
```python
window_size = 3  # Single window - MISSING 5 and 9!
max_iterations = 5
reg_factor = 0.5  # ✓ matches
sigma = 1.0       # ✓ matches
```

**This is likely the main cause of degradation!**
Python needs to implement hierarchical window sizes [3, 5, 9] for the eye model,
similar to how main CLNF uses [11, 9, 7, 5].

### TODO: Get C++ eye init values for comparison
Need to run C++ FeatureExtraction with eye model debug enabled to get:
- C++ input landmarks (main 36-41)
- C++ fitted params_global
- C++ initial 28 landmarks
- C++ window_size

### 2024-11-22 Session 2 - Window Size Investigation

**Implemented hierarchical windows [3, 5, 9] to match C++:**

Results:
- Single window [3]: Degrades by **0.36 px**
- Hierarchical [3, 5, 9]: Degrades by **0.59 px** (WORSE!)
- Hierarchical [9, 5, 3]: Degrades by **0.58 px** (WORSE!)

**Conclusion: Window sizes are NOT the cause of degradation.**

The issue must be in:
1. Response map computation
2. Mean-shift calculation
3. Parameter update formulas
4. Eye PDM initialization/projection

Reverted to single window [3] for now.

### Next Steps
- [ ] Compare eye response maps (C++ vs Python) for same input
- [ ] Compare mean-shift vectors
- [ ] Compare Jacobian computation for eye model
- [ ] Check if eye PDM projection differs

### 2024-11-22 Session 3 - Iteration Count Investigation

**CRITICAL FINDING: Iteration Count Does NOT Affect Accuracy!**

Discovered Python was only running 10 iterations vs C++ 37-40 iterations. Fixed by increasing minimum iterations before convergence check from 5 to 10.

**Results:**
- Before fix: 10 iterations → 1.24 px error
- After fix: 83 iterations → 1.24 px error

**Conclusion: Iteration count is NOT the cause of the accuracy gap.**

Python converges to the same result in ~10 iterations. Additional iterations don't improve accuracy. The issue is in the **quality of updates**, not the **quantity**.

**Current Configuration:**
```python
# Main CLNF
window_sizes = [11, 9, 7, 5]  # ✓ matches C++
max_iterations = 40           # Now runs 83 total iterations

# Eye Model
window_sizes = [9, 5, 3]      # Hierarchical (currently degrades accuracy)
```

**Key Insight:**
The remaining 1.24 px error (vs C++ 0.63 px) must be caused by:
1. **Response map computation** - different patch extraction or CNN output
2. **Mean-shift calculation** - weighted mean computation
3. **Parameter update** - Jacobian or regularization differences
4. **Model initialization** - CalcParams differences

The analyze_convergence.py script shows Python error plateauing at 4.75 px while C++ continues improving to 0.63 px. This suggests Python's updates are moving landmarks in wrong directions.

### Priority Debug Actions
1. **Compare response maps** - Extract patch at same location, compare CNN output
2. **Compare mean-shift vectors** - Same response map should give same mean-shift
3. **Compare parameter updates** - Track delta_p for same mean-shift input
4. **Focus on eye initialization** - Python scale=3.488 vs C++ scale=3.356 (4% difference)
