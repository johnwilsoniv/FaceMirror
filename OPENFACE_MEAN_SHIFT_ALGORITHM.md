# OpenFace Mean Shift Algorithm - Pseudocode and Analysis

## Source

Based on OpenFace C++ implementation:
- File: `LandmarkDetectorModel.cpp`
- Function: `NonVectorisedMeanShift_precalc_kde` (lines 820-935)
- Called from: CLNF optimization loop (line 1077)

---

## Algorithm Pseudocode

```python
def mean_shift_update(patch_expert_responses, current_landmarks, pdm, window_size, sigma):
    """
    Compute mean shift updates for all landmarks using KDE (Kernel Density Estimation).

    Args:
        patch_expert_responses: List of response maps (window_size × window_size) for each landmark
        current_landmarks: Current landmark positions in IMAGE coordinates
        pdm: Point Distribution Model (provides expected landmark positions)
        window_size: Size of search window (e.g., 11)
        sigma: Gaussian kernel bandwidth for KDE

    Returns:
        mean_shifts: Update vectors (delta_x, delta_y) for each landmark
    """

    n_landmarks = len(current_landmarks)
    mean_shifts = np.zeros((n_landmarks, 2))

    # Precompute KDE Gaussian kernel weights for all possible (dx, dy) offsets
    # This is an optimization - compute once and reuse
    kde_kernel = precompute_kde_kernel(window_size, sigma)

    for i in range(n_landmarks):
        # 1. Get expected landmark position from PDM (in response map coordinates)
        #    This is NOT the center of the response map!
        #    It's the position where the PDM predicts the landmark should be
        expected_offset = pdm.get_landmark_offset(i, current_landmarks)
        dx = expected_offset.x + (window_size - 1) / 2
        dy = expected_offset.y + (window_size - 1) / 2

        # Clamp to valid range
        dx = clamp(dx, 0, window_size - 0.1)
        dy = clamp(dy, 0, window_size - 0.1)

        # 2. Select appropriate KDE kernel row based on (dx, dy)
        #    Uses nearest neighbor lookup with step_size=0.1
        kernel_row = kde_kernel[int(dx/0.1), int(dy/0.1)]

        # 3. Compute weighted mean position in response map
        response_map = patch_expert_responses[i]  # window_size × window_size

        weighted_sum_x = 0.0
        weighted_sum_y = 0.0
        total_weight = 0.0

        for row in range(window_size):
            for col in range(window_size):
                # KDE weight: Gaussian centered at (dx, dy)
                kde_weight = kernel_row[row * window_size + col]

                # Combined weight: response × KDE
                combined_weight = response_map[row, col] * kde_weight

                weighted_sum_x += combined_weight * col
                weighted_sum_y += combined_weight * row
                total_weight += combined_weight

        # 4. Compute mean shift
        #    This is the KEY step: weighted mean MINUS expected position
        weighted_mean_x = weighted_sum_x / total_weight
        weighted_mean_y = weighted_sum_y / total_weight

        mean_shift_x = weighted_mean_x - dx
        mean_shift_y = weighted_mean_y - dy

        mean_shifts[i] = (mean_shift_x, mean_shift_y)

    return mean_shifts


def precompute_kde_kernel(window_size, sigma):
    """
    Precompute KDE Gaussian kernel for all possible (dx, dy) positions.

    For efficiency, OpenFace discretizes the continuous (dx, dy) space
    with step_size=0.1, resulting in a lookup table.
    """
    step_size = 0.1
    n_steps = int(window_size / step_size)

    # Each row corresponds to one (dx, dy) position
    # Each column corresponds to one (row, col) position in the response map
    kde_kernel = np.zeros((n_steps * n_steps, window_size * window_size))

    a = -0.5 / (sigma * sigma)  # Gaussian coefficient

    idx = 0
    for dx_step in range(n_steps):
        dx = dx_step * step_size
        for dy_step in range(n_steps):
            dy = dy_step * step_size

            # Compute Gaussian weights for this (dx, dy) position
            for row in range(window_size):
                for col in range(window_size):
                    # Squared distance from (dx, dy) to (col, row)
                    dist_sq = (dy - row)**2 + (dx - col)**2

                    # Gaussian kernel: exp(-0.5 × dist² / sigma²)
                    weight = np.exp(a * dist_sq)

                    kde_kernel[idx, row * window_size + col] = weight

            idx += 1

    return kde_kernel
```

---

## Critical Understanding: Edge Peaks are CORRECT!

### Common Misconception
❌ "Response map peaks should always be at the CENTER (5,5) of the 11×11 window"
❌ "Edge peaks indicate a bug in response map computation"

### Correct Understanding
✅ **Response map peaks indicate where the TRUE landmark position is**
✅ **Edge peaks mean the landmark is DISPLACED from the expected PDM position**
✅ **The mean shift algorithm EXPECTS and HANDLES edge peaks correctly**

### Example Scenario

Given:
- Response map: 11×11 grid
- Peak at position (10, 3) [near right edge]
- Expected position (dx, dy) = (5, 5) [center, from PDM]

**What this means:**
1. The PDM predicts the landmark should be at the CENTER of the search window
2. But the patch expert detects high response at position (10, 3)
3. This means the TRUE landmark is ~5 pixels to the RIGHT of where PDM expects
4. Mean shift computes: `msx = 10 - 5 = +5`, `msy = 3 - 5 = -2`
5. Update: Move landmark +5 pixels right, -2 pixels up

**This is CORRECT behavior!** Edge peaks are the algorithm working as intended.

---

## Why Convergence Still Fails

If edge peaks are correct, why doesn't the algorithm converge?

### Root Cause 1: Gaussian Kernel Too Narrow

```python
# Current PyCLNF setting
sigma = 1.75
window_size = 11

# Gaussian weight at edge peak (5 pixels from expected position)
dist = 5
weight = exp(-0.5 × dist² / sigma²)
      = exp(-0.5 × 25 / 3.0625)
      = exp(-4.08)
      = 0.017  # Only 1.7% weight!
```

**Problem:** The Gaussian kernel heavily DOWNWEIGHTS the edge peaks!
- Even though the peak is at (10, 3), it gets only 1.7% weight
- The weighted mean gets "pulled back" toward the center
- Result: Tiny updates (~0.3 pixels instead of ~5 pixels)

**Solution:** Use larger sigma that scales with window_size
```python
sigma = window_size / 2.5  # For ws=11: sigma=4.4
# Weight at 5px: exp(-0.5 × 25 / 19.36) = exp(-0.65) = 0.52 (52% weight!)
```

### Root Cause 2: Poor Initialization

If response maps consistently show edge peaks across ALL landmarks:
- The initial PDM shape is FAR from the true face shape
- This could be due to:
  1. **Bbox quality**: Face detector provides inaccurate bounding box
  2. **PDM initialization**: Poor conversion from bbox to PDM parameters
  3. **Scale/rotation mismatch**: PDM initialized with wrong scale or rotation

**Investigation needed:**
1. Compare initial landmark positions with ground truth
2. Check if PDM initialization matches OpenFace C++ exactly
3. Test with different bbox sources (RetinaFace vs MTCNN)

---

## OpenFace C++ vs PyCLNF Comparison

### Similarities ✓
- Both compute weighted mean using response × KDE weights
- Both use mean shift = weighted_mean - expected_position
- Both use same KDE Gaussian kernel formula

### Differences to Investigate
1. **Sigma parameter:**
   - OpenFace C++: Uses `a = -0.5 / (sigma²)` where sigma may be dynamic
   - PyCLNF: Currently uses fixed sigma=1.5 (may be too small)

2. **Window size handling:**
   - OpenFace C++: Uses [11, 9, 7] with different iteration budgets
   - PyCLNF: Fixed after recent bug fix

3. **PDM initialization:**
   - OpenFace C++: Has detector-specific bbox corrections (MTCNN, Haar, etc.)
   - PyCLNF: Uses generic bbox→PDM conversion

---

## Next Steps

1. **Test larger sigma values** (window_size / 2.5 to window_size / 2)
   - Expected: Much better convergence with same response maps
   - File: `test_sigma_fix.py` already created

2. **Verify PDM initialization** matches OpenFace C++
   - Compare initial landmarks between PyCLNF and C++
   - Check if bbox preprocessing is needed

3. **If still not converging:** Compare individual patch expert responses
   - Run `compare_patch_expert_responses.py` to test patch experts
   - Verify response maps match between PyCLNF and C++ on same patches

---

## References

- OpenFace implementation: `lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp`
- CCNF paper: Baltrusaitis et al., "Constrained Local Neural Fields for Robust Facial Landmark Detection in the Wild", ICCV Workshop 2013
- Mean shift: https://en.wikipedia.org/wiki/Mean_shift
