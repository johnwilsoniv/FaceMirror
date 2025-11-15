# CLNF Step-by-Step Divergence Investigation Plan

**Date:** 2025-11-15
**Objective:** Systematically compare Python CLNF vs C++ OpenFace to find exactly where they diverge

---

## Overview

This investigation will add matching debug statements at critical stages of both Python and C++ CLNF implementations to enable direct comparison and pinpoint divergence.

**Debug Output Format:**
- **Python:** `[PY][STAGE] description: values`
- **C++:** `[CPP][STAGE] description: values`

**Test Setup:**
- **Image:** `patient1_frame1.jpg`
- **BBox:** `(296, 778, 405, 407)` from known good result
- **Tracked Landmarks:** 36, 48, 30, 8 (left eye, mouth, nose, jaw)

---

## Phase 1: Identify Comparison Points

### 1.1 Initialization (Before Iteration 0)

**PDM Initialization from bbox:**
```
[INIT] BBox (x,y,w,h): (296, 778, 405, 407)
[INIT] Params shape: (40,)
[INIT] Params_local (first 5): [...]
[INIT] Params_global (scale): X.XXXXXX
[INIT] Params_global (rotation): [wx, wy, wz]
[INIT] Params_global (translation): [tx, ty]
[INIT] Landmark_36: (x, y)
[INIT] Landmark_48: (x, y)
[INIT] Landmark_30: (x, y)
[INIT] Landmark_8: (x, y)
```

**What to compare:**
- Do both implementations produce identical initial parameters?
- Are landmark positions identical after PDM transformation?

---

### 1.2 Iteration Start (Each iteration, each window size)

```
[ITER{N}_WS{W}] Computing response maps...
[ITER{N}_WS{W}] Landmark_36: (x, y)
[ITER{N}_WS{W}] Landmark_48: (x, y)
[ITER{N}_WS{W}] Landmark_30: (x, y)
[ITER{N}_WS{W}] Landmark_8: (x, y)
```

**What to compare:**
- Are landmark positions identical at iteration start?
- Do they drift between iterations?

---

### 1.3 Response Map Stage (For each tracked landmark)

```
[ITER{N}_WS{W}_LM{L}] Window bounds: x=[x_min, x_max] y=[y_min, y_max]
[ITER{N}_WS{W}_LM{L}] Response_RAW: peak=(row,col) val=X.XXXXXX offset=(dx,dy)
[ITER{N}_WS{W}_LM{L}] Response_SIGMA: peak=(row,col) val=X.XXXXXX offset=(dx,dy)
[ITER{N}_WS{W}_LM{L}] Mean-shift: (dx, dy) sum_weights=X.XXXXXX
```

**What to compare:**
- Window bounds (same patch extraction region?)
- Raw response peak location and value (before Sigma)
- Sigma-transformed response peak location and value
- Mean-shift vector magnitude and direction

**Key Questions:**
1. Are raw responses identical? → Tests patch extraction and evaluation
2. Are Sigma-transformed responses identical? → Tests matrix computation
3. Are mean-shift vectors identical? → Tests Gaussian weighting

---

### 1.4 Jacobian Construction

```
[ITER{N}_WS{W}] Jacobian shape: (136, 40)
[ITER{N}_WS{W}] Jacobian_36 (first 6): [...]
[ITER{N}_WS{W}] JtJ condition number: X.XXXX
```

**What to compare:**
- Jacobian matrix dimensions
- Individual Jacobian entries for tracked landmarks
- J^T * J condition number (numerical stability check)

---

### 1.5 Weight Matrix W

```
[ITER{N}_WS{W}] Weight_36: X.XXXXXX
[ITER{N}_WS{W}] Weight_48: X.XXXXXX
[ITER{N}_WS{W}] W_mean: X.XXXXXX
```

**What to compare:**
- Individual confidence weights
- Weight matrix statistics

---

### 1.6 Covariance Matrix Lambda

```
[ITER{N}_WS{W}] Lambda regularization: 35
[ITER{N}_WS{W}] (JtWJ + Lambda) condition: X.XXXX
[ITER{N}_WS{W}] (JtWJ + Lambda) determinant: X.XXXX
```

**What to compare:**
- Regularization parameter (should be 35 in both)
- Matrix conditioning
- Numerical stability indicators

---

### 1.7 Parameter Update

```
[ITER{N}_WS{W}] Delta_p magnitude: X.XXXXXX
[ITER{N}_WS{W}] Delta_p (first 5): [...]
[ITER{N}_WS{W}] Updated params (first 5): [...]
```

**What to compare:**
- Parameter update vector magnitude
- Individual parameter changes
- Updated parameter values after manifold-aware rotation update

---

### 1.8 Iteration End

```
[ITER{N}_WS{W}] Updated Landmark_36: (x, y)
[ITER{N}_WS{W}] Shape change: X.XXXX
[ITER{N}_WS{W}] Convergence: {True|False}
```

**What to compare:**
- Final landmark positions after iteration
- Shape change magnitude
- Convergence decision

---

## Phase 2: C++ Debug Points

### Files to Modify

**Primary Files:**
1. **LandmarkDetectorModel.cpp**
   - Location: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp`
   - Function: `bool LandmarkDetector::DetectLandmarksInVideo(...)` (~line 1000-1100)
   - Add debug at: initialization, iteration loop, convergence check

2. **CCNF_patch_expert.cpp**
   - Location: `.../LandmarkDetector/src/CCNF_patch_expert.cpp`
   - Function: `void CCNF_patch_expert::Response(...)` (~line 380-420)
   - Add debug at: response map computation, Sigma transformation

3. **PDM.cpp**
   - Location: `.../LandmarkDetector/src/PDM.cpp`
   - Function: `void PDM::CalcParams(...)` (initialization)
   - Add debug at: parameter computation from bbox

### Debug Code Pattern

```cpp
// At initialization
std::cout << "[CPP][INIT] BBox (x,y,w,h): ("
          << bbox.x << ", " << bbox.y << ", "
          << bbox.width << ", " << bbox.height << ")" << std::endl;

// For landmarks
for(int i : {36, 48, 30, 8}) {
    std::cout << "[CPP][INIT] Landmark_" << i << ": ("
              << detected_landmarks.at<double>(i) << ", "
              << detected_landmarks.at<double>(i + n_points) << ")" << std::endl;
}

// In iteration loop
std::cout << "[CPP][ITER" << iteration << "_WS" << window_size << "] "
          << "Computing response maps..." << std::endl;

// For response maps (landmark 36 example)
if(landmark_id == 36) {
    cv::Point peak_loc;
    double peak_val;
    cv::minMaxLoc(response_map, NULL, &peak_val, NULL, &peak_loc);
    int center = window_size / 2;
    std::cout << "[CPP][ITER" << iteration << "_WS" << window_size << "_LM36] "
              << "Response_RAW: peak=(" << peak_loc.y << "," << peak_loc.x << ") "
              << "val=" << std::fixed << std::setprecision(6) << peak_val << " "
              << "offset=(" << (peak_loc.x - center) << "," << (peak_loc.y - center) << ")"
              << std::endl;
}
```

### Compilation

```bash
cd /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build
cmake ..
make -j8
```

---

## Phase 3: Python Debug Points

### Implementation

**File:** `debug_clnf_stepwise.py` (already created)

**Key Components:**
- `InstrumentedNURLMSOptimizer` class that wraps standard optimizer
- Debug output at every comparison point
- Tracks landmarks: 36, 48, 30, 8

**Execution:**
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
env PYTHONPATH="." python3 debug_clnf_stepwise.py > python_debug.log 2>&1
```

---

## Phase 4: Execution Strategy

### Step 1: Run C++ with Debug

```bash
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3

/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction \
    -f calibration_frames/patient1_frame1.jpg \
    -out_dir /tmp \
    > cpp_debug.log 2>&1
```

### Step 2: Run Python with Debug

```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
env PYTHONPATH="." python3 debug_clnf_stepwise.py > python_debug.log 2>&1
```

### Step 3: Compare Logs

Use comparison script (Phase 5) to parse and compare outputs.

---

## Phase 5: Comparison Strategy

### Comparison Script: `compare_clnf_debug.py`

**Features:**
1. Parse both log files
2. Extract values at each comparison point
3. Compute differences
4. Report first significant divergence

**Tolerance Thresholds:**
- **Positions:** 0.1 pixels
- **Values:** 1% relative difference
- **Vectors:** 0.1 magnitude difference

### Output Format

```
================================================================================
CLNF Debug Comparison: Python vs C++
================================================================================

[INIT] Checking initialization...
  ✓ BBox: Python=(296, 778, 405, 407) C++=(296, 778, 405, 407) MATCH
  ✓ Landmark_36: Python=(369.2341, 854.1234) C++=(369.2341, 854.1234) diff=0.0000px
  ✓ Landmark_48: Python=(491.5678, 1027.2412) C++=(491.5678, 1027.2412) diff=0.0000px
  ✓ Landmark_30: Python=(485.9395, 939.8551) C++=(485.9395, 939.8551) diff=0.0000px
  ✓ Landmark_8: Python=(485.6989, 1173.7400) C++=(485.6989, 1173.7400) diff=0.0000px

[ITER0_WS11] Checking iteration 0, window size 11...
  ✓ Landmark_36: Python=(369.2341, 854.1234) C++=(369.2341, 854.1234) diff=0.0000px

[ITER0_WS11_LM36] Checking response map for landmark 36...
  ✓ Window bounds: Python=x=[364, 375] y=[849, 860] C++=x=[364, 375] y=[849, 860] MATCH
  ✓ Response_RAW peak: Python=(3, 8) C++=(3, 8) MATCH
  ✓ Response_RAW value: Python=24.788234 C++=24.788234 diff=0.00%
  ✗ Response_SIGMA peak: Python=(1, -2) C++=(2, -2) DIFFER by 1px ⚠️
  ✗ Response_SIGMA value: Python=0.186234 C++=0.192145 diff=3.17% ⚠️

  ^--- FIRST DIVERGENCE FOUND at ITER0_WS11_LM36 Response_SIGMA

  Likely cause: Sigma transformation implementation difference
  - Check matrix reshaping order (row-major vs column-major)
  - Check Sigma component indexing
  - Check matrix multiplication order

DIVERGENCE POINT: [ITER0_WS11_LM36] Response_SIGMA
```

---

## Phase 6: Detailed Breakdown Table

| Stage | Component | C++ Value | Python Value | Difference | Status |
|-------|-----------|-----------|--------------|------------|--------|
| INIT | Landmark 36 | (369.23, 854.12) | (369.23, 854.12) | 0.00px | ✓ |
| ITER0 WS11 | Response RAW peak | (3, 8) | (3, 8) | match | ✓ |
| ITER0 WS11 | Response RAW val | 24.788234 | 24.788234 | 0.00% | ✓ |
| ITER0 WS11 | Response SIGMA peak | (2, -2) | (1, -2) | 1px | ✗ |
| ITER0 WS11 | Response SIGMA val | 0.192145 | 0.186234 | 3.17% | ✗ |
| ITER0 WS11 | Mean-shift | (2.3, -1.9) | (2.1, -1.8) | 0.22px | ✗ |

---

## Phase 7: Focus Areas Based on Known Issues

**Priority Investigation Order:**

1. **Initialization** ✓ LIKELY CORRECT
   - We've verified initial landmarks match between implementations
   - PDM parameter calculation appears consistent

2. **Response Maps** 🔍 INVESTIGATE HERE
   - Raw responses: Check patch extraction coordinates
   - Sigma transformation: Check matrix operations
   - Mean-shift: Check Gaussian weighting formula

3. **Jacobian Construction** ⏸️ INVESTIGATE IF ABOVE PASSES
   - PDM derivatives
   - Rotation parameter handling

4. **Weight Matrix** ⏸️ INVESTIGATE IF ABOVE PASSES
   - Confidence calculation
   - Diagonal matrix construction

5. **Parameter Update** ⏸️ INVESTIGATE IF ABOVE PASSES
   - NU-RLMS formula
   - Rotation manifold update

---

## Phase 8: Specific Debug Values to Track

### For Landmark 36 (Left Eye Corner)

```
================================================================================
LANDMARK 36 DETAILED TRACE
================================================================================

[INIT]
  bbox: (296, 778, 405, 407)
  params_local (shape coefficients):
    [0]: 0.000000
    [1]: 0.000000
    [2]: 0.000000
    [3]: 0.000000
    [4]: 0.000000
  params_global:
    scale (s): 2.751336
    rotation (wx, wy, wz): (-0.001320, -0.007691, -0.000588)
    translation (tx, ty): (487.259, 959.878)
  landmark_36: (369.2341, 854.1234)

[ITER0, WS=11]
  landmark_36_start: (369.2341, 854.1234)

  window_bounds:
    x: [364, 375] (center=369)
    y: [849, 860] (center=854)

  patch_expert:
    width: 19
    height: 19
    num_neurons: 7

  response_map_raw (11x11):
    peak_location: (row=3, col=8)
    peak_value: 24.788234
    peak_offset_from_center: (dx=3, dy=-2)
    top3_peaks: [(3,8,24.78), (4,8,24.21), (3,7,23.95)]

  sigma_transformation:
    sigma_components_used: [0, 1, 2]
    Sigma matrix: (121, 121)

  response_map_sigma (11x11):
    peak_location: (row=1, col=-2)  <-- CHECK THIS
    peak_value: 0.186234            <-- CHECK THIS
    peak_offset_from_center: (dx=1, dy=-2)

  mean_shift:
    sigma_param: 1.5
    gaussian_center: 5.0 (=(11-1)/2)
    computed_ms: (2.1, -1.8)        <-- CHECK THIS
    sum_weights: 145.234

  jacobian_36:
    J[72:74, 0:6]: [[...], [...]]   (2 rows for x,y; 6 cols for global params)

  weight_36: 0.987654

  after_update:
    delta_p magnitude: 14.2539
    landmark_36_end: (371.3456, 852.3241)
```

---

## Key Questions This Plan Will Answer

1. **Do we initialize identically?**
   - PDM parameters from bbox ✓
   - Initial landmark positions ✓

2. **Are response maps identical before Sigma?**
   - Patch extraction coordinates 🔍
   - Patch evaluation (CCNF forward pass) 🔍

3. **Is Sigma transformation identical?**
   - Matrix computation ⚠️ SUSPECT
   - Component selection ⚠️ SUSPECT
   - Reshape order (row/column-major) ⚠️ SUSPECT

4. **Is mean-shift calculation identical?**
   - Gaussian weighting 🔍
   - Summation order 🔍

5. **Is Jacobian construction identical?**
   - PDM derivatives ⏸️
   - Numerical differentiation ⏸️

6. **Is weight matrix identical?**
   - Confidence calculation ⏸️

7. **Is parameter update identical?**
   - NU-RLMS formula ⏸️
   - Rotation manifold handling ⏸️

---

## Deliverables

1. **cpp_debug_instrumentation.patch**
   - C++ code changes for LandmarkDetectorModel.cpp, CCNF_patch_expert.cpp, PDM.cpp

2. **debug_clnf_stepwise.py** ✓ CREATED
   - Python instrumented debug script

3. **compare_clnf_debug.py**
   - Automated comparison script

4. **cpp_debug.log**
   - C++ debug output

5. **python_debug.log**
   - Python debug output

6. **DIVERGENCE_REPORT.md**
   - Final findings and root cause analysis

---

## Success Criteria

**Investigation succeeds when we can answer:**
1. At which exact stage do Python and C++ diverge?
2. What is the numerical difference at divergence point?
3. What is the root cause (indexing bug, transpose, formula error)?
4. Can we fix the Python implementation to match C++?

**Expected outcome:**
```
DIVERGENCE FOUND at [ITER0_WS11_LM36] Response_SIGMA

Root Cause: Response map reshape order
- C++ uses column-major (Fortran-style) reshape
- Python uses row-major (C-style) reshape
- Fix: Change response_map.reshape(-1, 1) to use order='F'
```

---

## Timeline

1. **C++ Instrumentation:** 2-3 hours
2. **Run & Collect Logs:** 5 minutes
3. **Comparison Script:** 1-2 hours
4. **Analysis & Fix:** 1-4 hours

**Total estimated time:** 4-9 hours
