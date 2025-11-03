# CLNF Diagnostic System: C++ vs Python Comparison

## Overview

This diagnostic system provides comprehensive tools to compare C++ OpenFace CLNF with Python CLNF step-by-step to identify where they diverge.

## Goal

Extract initial landmarks from C++ OpenFace, then run both C++ and Python CLNF on the same initialization to see exactly where they differ iteration-by-iteration.

## System Components

### 1. `clnf_debug_logger.py` - Detailed Iteration Logging

**Purpose:** Enhanced CLNF detector that captures iteration-by-iteration diagnostics.

**Features:**
- Wraps standard CLNF detector with debug logging
- Captures for each iteration:
  - Landmark positions (68, 2)
  - PDM parameters (n_modes,)
  - Response map statistics (min, max, mean, std per landmark)
  - Mean shift targets
  - Residual norm
  - Jacobian condition number
  - Parameter update magnitude
  - Convergence metrics

**Usage:**
```python
from clnf_debug_logger import DebugCLNFDetector, print_iteration_summary

# Initialize with debug logging
clnf = DebugCLNFDetector(
    model_dir=weights_dir / 'clnf',
    max_iterations=10,
    convergence_threshold=0.01
)

# Run refinement
refined_landmarks, converged, num_iters = clnf.refine_landmarks(
    frame, initial_landmarks, scale_idx=2, regularization=0.5
)

# Get debug history
history = clnf.get_debug_history()

# Print iteration summaries
for i in range(len(history)):
    print_iteration_summary(history, i)
```

**Diagnostic Output Per Iteration:**
```
===============================================================================
ITERATION 0
===============================================================================
  Average movement: 2.5432 pixels
  Converged: False

  PDM Parameters:
    Scale: 125.3456
    Translation: [512.34, 384.67]
    Params shape: (34,)
    Params range: [-2.3456, 1.8901]
    Params mean: 0.0234, std: 0.8765

  Landmarks:
    X range: [450.12, 574.89]
    Y range: [320.45, 495.23]
    Centroid: [512.34, 407.84]

  Response Map Statistics:
    Mean response: 0.234567 (±0.045612)
    Max response: 0.876543 (±0.123456)

  Mean Shift Targets:
    Average shift: 2.5432 pixels
    Max shift: 5.6789 pixels
    Min shift: 0.1234 pixels

  Jacobian:
    Shape: (136, 34)
    Condition number: 1.23e+05

  Parameter Update:
    Delta params norm: 0.012345
    Delta params range: [-0.003456, 0.002789]

  Residual:
    Norm: 0.123456
    Mean: 0.001234
```

### 2. `compare_clnf_cpp_vs_python.py` - Main Comparison Script

**Purpose:** Compare C++ OpenFace CLNF vs Python CLNF on same test frames.

**Workflow:**
1. Load OpenFace C++ landmarks from existing CSV (ground truth)
2. Extract Python initialization using RetinaFace + PFLD
3. Run Python CLNF with debug logging
4. Compare iteration-by-iteration
5. Generate visualization and diagnostic report

**Usage:**
```bash
python3 compare_clnf_cpp_vs_python.py
```

**Outputs:**
- `/tmp/clnf_diagnostic_results/IMG_9330_frame100_comparison.jpg` - Visualization
- `/tmp/clnf_diagnostic_results/IMG_9330_frame100_diagnostic.json` - Detailed JSON report

**Test Cases:**
- IMG_9330 frame 100 (known CLNF failure case)
- IMG_8401 frame 100 (if available)

### 3. `compare_clnf_simple.py` - Simplified Comparison

**Purpose:** Simplified version using existing PyFaceAU detector infrastructure.

**Advantages:**
- Reuses tested detector initialization code
- Avoids ONNX/CoreML initialization issues
- Faster execution

**Usage:**
```bash
python3 compare_clnf_simple.py
```

### 4. `analyze_clnf_divergence.py` - Divergence Analysis

**Purpose:** Analyze diagnostic JSON files to identify specific bug locations.

**Features:**
- Identifies convergence issues:
  - Stalled convergence (not decreasing)
  - Divergence (increasing error)
  - Weak response maps
  - Ill-conditioned Jacobian
  - Tiny parameter updates
- Suggests specific bug locations with debug steps
- Generates convergence plots

**Usage:**
```bash
python3 analyze_clnf_divergence.py
```

**Output:**
- Analysis report with suspected bug locations
- Convergence trajectory plots
- Specific debug recommendations

### 5. `extract_openface_initial_landmarks.py` - C++ Initialization Extraction

**Purpose:** Extract OpenFace C++ landmarks for single frames to use as initialization reference.

**Usage:**
```bash
python3 extract_openface_initial_landmarks.py
```

**Outputs:**
- `/tmp/clnf_diagnostic_data/IMG_9330_frame100_openface_init.npz`
  - Contains: frame, landmarks, OpenFace metadata

## Possible Divergence Points

The diagnostic system helps identify where Python CLNF diverges from C++ OpenFace:

### 1. Response Map Computation
**File:** `pyfaceau/clnf/cen_patch_experts.py`
**Function:** `CENPatchExperts.response()`, `CENPatchExpert.response()`
**Symptoms:**
- Very weak response maps (mean < 0.01)
- All-zero responses
**Possible Causes:**
- Patch extraction bounds incorrect (coordinate system issue)
- Contrast normalization failing
- `im2col_bias` creating wrong patch layout
- Neural network weights not loaded correctly
**Debug Steps:**
```python
# Print extracted patch before/after contrast normalization
print(f"Patch range: [{patch.min()}, {patch.max()}]")
print(f"Normalized range: [{normalized.min()}, {normalized.max()}]")

# Compare patch extraction coordinates with OpenFace C++
print(f"Extraction bounds: {extraction_bounds}")
print(f"Landmark position: {landmarks[lm_idx]}")

# Verify response map output
print(f"Response shape: {response.shape}")
print(f"Response range: [{response.min()}, {response.max()}]")
```

### 2. Mean Shift Calculation
**File:** `pyfaceau/clnf/nu_rlms.py`
**Function:** `NURLMSOptimizer._mean_shift_targets()`
**Symptoms:**
- Convergence stalled (movement not decreasing)
- Landmarks not moving toward better positions
**Possible Causes:**
- Coordinate transformation from response map to image is wrong
- `extraction_bounds` not accounting for image boundary clamping
- Weighted mean calculation incorrect
- Gaussian smoothing destroying peaks
**Debug Steps:**
```python
# Visualize response maps
import matplotlib.pyplot as plt
plt.imshow(response, cmap='hot')
plt.title(f"Response map for landmark {lm_idx}")
plt.colorbar()
plt.show()

# Print target vs current
print(f"Current landmark: {landmarks[lm_idx]}")
print(f"Target landmark: {target_landmarks[lm_idx]}")
print(f"Shift: {target_landmarks[lm_idx] - landmarks[lm_idx]}")

# Verify extraction bounds
print(f"Extraction bounds (x1, y1, x2, y2): {extraction_bounds[lm_idx]}")
print(f"Response map center in image coords: ({x1 + mean_x}, {y1 + mean_y})")
```

### 3. Jacobian Computation
**File:** `pyfaceau/clnf/nu_rlms.py`
**Function:** `NURLMSOptimizer._compute_jacobian()`
**Symptoms:**
- Ill-conditioned Jacobian (condition number > 1e10)
- Numerical instability
**Possible Causes:**
- PDM eigenvectors not scaled correctly
- Scale factor too small or too large
- Wrong coordinate system
**Debug Steps:**
```python
# Print scale/translation
print(f"Scale: {scale}")
print(f"Translation: {translation}")

# Check PDM eigenvector magnitudes
print(f"Eigenvector norms: {np.linalg.norm(pdm.eigenvectors, axis=0)}")

# Compare jacobian singular values
U, S, Vt = np.linalg.svd(jacobian)
print(f"Singular values: {S}")
print(f"Condition number: {S[0] / S[-1]}")
```

### 4. Parameter Update
**File:** `pyfaceau/clnf/nu_rlms.py`
**Function:** `NURLMSOptimizer._solve_regularized_ls()`
**Symptoms:**
- Divergence (error increasing)
- Parameters exploding or vanishing
**Possible Causes:**
- Regularization too weak
- Residual sign flipped
- Jacobian transposed incorrectly
- Parameter clamping not working
**Debug Steps:**
```python
# Print parameter update
print(f"Delta params norm: {np.linalg.norm(delta_params)}")
print(f"Delta params: {delta_params}")

# Verify JtJ matrix
JtJ = jacobian.T @ jacobian
print(f"JtJ condition: {np.linalg.cond(JtJ)}")
print(f"JtJ eigenvalues: {np.linalg.eigvals(JtJ)}")

# Check parameter clamping
params_before = params.copy()
params_after = pdm.clamp_params(params, n_std=3.0)
print(f"Params clamped: {np.sum(params_before != params_after)} / {len(params)}")
```

### 5. PDM Parameter Estimation
**File:** `pyfaceau/clnf/pdm.py`
**Function:** `PointDistributionModel.landmarks_to_params_2d()`
**Symptoms:**
- Poor initialization (landmarks way off)
- Scale/translation estimation wrong
**Possible Causes:**
- Coordinate system mismatch (image vs normalized)
- Wrong depth prior
- Alignment algorithm incorrect
**Debug Steps:**
```python
# Check initial parameter estimation
params_init, scale_init, trans_init = pdm.landmarks_to_params_2d(initial_landmarks)
print(f"Initial params: {params_init}")
print(f"Initial scale: {scale_init}")
print(f"Initial translation: {trans_init}")

# Reconstruct landmarks from parameters
landmarks_reconstructed = pdm.params_to_landmarks_2d(params_init, scale_init, trans_init)
error = np.linalg.norm(landmarks_reconstructed - initial_landmarks, axis=1).mean()
print(f"Reconstruction error: {error:.2f} pixels")
```

## Expected Workflow

1. **Run Comparison:**
   ```bash
   python3 compare_clnf_simple.py
   ```

2. **Analyze Results:**
   ```bash
   python3 analyze_clnf_divergence.py
   ```

3. **Review Outputs:**
   - Check visualization: `/tmp/clnf_diagnostic_results/IMG_9330_frame100_comparison.jpg`
   - Review JSON: `/tmp/clnf_diagnostic_results/IMG_9330_frame100_diagnostic.json`
   - See convergence plot: `/tmp/clnf_diagnostic_results/IMG_9330_frame100_diagnostic.png`

4. **Identify Bug:**
   - Look at analysis report for suspected bug locations
   - Focus on highest priority issues first (weak responses, stalled convergence)
   - Follow debug steps to narrow down exact location

5. **Fix and Retest:**
   - Apply fix to suspected location
   - Re-run comparison to verify improvement
   - Check that CLNF error decreases toward OpenFace C++ level

## Known Issues

### CoreML/ONNX Segfault
**Symptom:** Script exits with code 139 (segmentation fault)
**Cause:** CoreML compilation or ONNX Runtime issue on first load
**Workaround:**
1. Models should be pre-compiled (cached) after first run
2. Use `compare_clnf_simple.py` which handles this better
3. If persistent, disable CoreML acceleration (CPU fallback)

### CLNF Not Improving
**Expected:** CLNF should improve upon PFLD initialization
**Current:** In some cases, CLNF degrades accuracy (e.g., IMG_9330: 45px → 387px)
**Root Cause:** One or more of the divergence points listed above
**Fix:** Use this diagnostic system to identify and fix the specific issue

## Success Criteria

Python CLNF is working correctly when:
1. **Converges:** Optimization converges within 5-10 iterations
2. **Improves:** Final error < initial error (CLNF better than PFLD)
3. **Matches OpenFace:** Final error close to OpenFace C++ (~10-20px typical)
4. **Response Maps:** Strong responses (mean > 0.1, max > 0.5)
5. **Stability:** Jacobian well-conditioned (< 1e8), no divergence

## Next Steps

1. Fix the segfault issue (likely ONNX/CoreML conflict)
2. Run diagnostic on IMG_9330 frame 100 successfully
3. Identify specific divergence point
4. Implement fix in Python CLNF
5. Validate fix improves accuracy to r > 0.92 correlation

## References

- OpenFace C++ source: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/`
- Python CLNF: `/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/pyfaceau/clnf/`
- Test videos: `/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/`
- OpenFace CSV outputs: `/tmp/openface_test_9330_rotated/`, `/tmp/openface_test_8401_rotated/`
