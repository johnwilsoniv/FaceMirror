# Left Eye Landmark Bias - Debug Investigation

## Problem Summary

The Python CLNF eye refinement produces a systematic left eye bias:
- **Left eye error**: 1.42px (worst region)
- **Right eye error**: 0.84px
- **Target**: <0.5px mean error to match C++ OpenFace

The bias manifests as an "eye widening" effect where both eyes are pushed outward:
- Left eye: -1.05px X bias (pushed left)
- Right eye: +0.71px X bias (pushed right)

## Root Cause Analysis

### Phase 1: Bias Source Identification ✅

**Confirmed**: Eye refinement causes the bias, not the main model.
- Main model alone: left -0.17px, right -0.44px (good)
- After eye refinement: left -1.05px, right +0.71px (bad)

### Phase 2: Debugging Details ✅

#### Key Finding: Opposite Refinement Directions

**C++ Left Eye Refinement** (LM36 = outer corner):
```
delta = (+0.47, +2.59) → moves RIGHT and DOWN
```

**Python Left Eye Refinement** (LM36 = outer corner):
```
delta = (-0.57, +3.92) → moves LEFT and DOWN
```

The X component is **opposite**! Python moves the outer corner leftward (widening the eye) while C++ moves it rightward (toward the correct position).

#### Raw Mean-Shift Values Match

C++ Eye_8 (outer corner):
```
msx = -0.17655391
msy = 0.24534798
```

Python Eye_8:
```
raw ms = (-0.156793, 0.250842)
```

These are nearly identical, confirming the raw KDE computation is correct.

#### Transform Values

Python sim_ref_to_img matrix:
```
[[3.245628,  0.373258]
 [-0.373258, 3.245628]]
```

After transformation, mean-shift becomes:
```
Reference: (-0.17, 0.25)
Image:     (-0.44, 0.88)
```

The negative X persists through the transform, causing the leftward movement.

## Technical Details

### Eye PDM Reference Shape

Left eye mean shape (28 landmarks):
- **Landmark 8** (outer corner): X = -11.21, Y = 0.96
- **Landmark 14** (inner corner): X = +10.15, Y = 2.35

Convention: Negative X = leftward (outer), Positive X = rightward (inner)

### Landmark Mappings

From `main_clnf_general.txt`:
```
Left eye:  36→8, 37→10, 38→12, 39→14, 40→16, 41→18
Right eye: 42→8, 43→10, 44→12, 45→14, 46→16, 47→18
```

Main model indices:
- 36 = left outer, 37 = left upper outer, 38 = left upper inner, 39 = left inner
- 42 = right inner, 43 = right upper inner, 44 = right upper outer, 45 = right outer

**Key insight**: For LEFT eye, Eye_8 = outer corner. For RIGHT eye, Eye_8 = inner corner (mirrored).

### Python Mapping (Correct)

```python
LEFT_EYE_MAPPING = {
    36: 8,   # Outer corner
    37: 10,  # Upper outer
    38: 12,  # Upper inner
    39: 14,  # Inner corner
    40: 16,  # Lower inner
    41: 18,  # Lower outer
}

RIGHT_EYE_MAPPING = {
    42: 8,   # Inner corner
    43: 10,  # Upper inner
    44: 12,  # Upper outer
    45: 14,  # Outer corner
    46: 16,  # Lower outer
    47: 18,  # Lower inner
}
```

## Configuration

Current eye refinement settings:
- `max_iterations`: 10 per window (×3 windows = 30 total)
- `convergence_threshold`: 0.01
- `window_sizes`: [11, 9, 7] (WS=5 disabled - causes worse accuracy)
- `sigma`: 1.75
- `reg_factor`: 0.5

## Hypotheses for Root Cause

### 1. Mean-Shift Direction Interpretation (HIGH)
The mean-shift vector in reference space may need sign adjustment when transformed to image space. The negative X mean-shift correctly indicates "move toward outer corner in reference", but after transform it should become positive X in image space for the left eye.

### 2. Jacobian Sign Convention (MEDIUM)
The Jacobian computed by `EyePDM.compute_jacobian()` may have incorrect signs for some derivatives, causing the solver to produce opposite parameter updates.

### 3. Transform Inversion (LOW)
The `sim_ref_to_img = np.linalg.inv(sim_img_to_ref)` may not produce the correct inverse for the 2x2 rotation-scale matrix.

### 4. Reference Shape Orientation (LOW)
The Python eye PDM export may have different orientation conventions than C++.

## Files Involved

- `pyclnf/core/eye_patch_expert.py` - Eye refinement implementation
  - `align_shapes_with_scale()` - Similarity transform computation
  - `_compute_sim_transforms()` - Transform matrix creation
  - `_compute_eye_mean_shift_with_offsets()` - KDE mean-shift computation
  - `_solve_eye_update()` / `_solve_eye_update_rigid()` - Parameter solving

- `pyclnf/core/eye_pdm.py` - Eye PDM model
  - `params_to_landmarks_2d()` - Parameter to landmark conversion
  - `compute_jacobian()` - Jacobian matrix computation

- `pyclnf/models/exported_eye_pdm_left/` - Left eye PDM data
- `pyclnf/models/exported_eye_pdm_right/` - Right eye PDM data

## Debug Output Files

- `/tmp/python_eye_model_debug.txt` - Pre/post refinement landmarks
- `/tmp/python_eye_model_detailed.txt` - Per-iteration mean-shifts
- `/tmp/python_eye_raw_meanshift.txt` - Raw vs transformed mean-shifts
- `/tmp/cpp_eye_model_debug.txt` - C++ refinement output
- `/tmp/cpp_eye_lm8_meanshift.txt` - C++ KDE computation

## Next Steps (Phase 3)

1. **Verify mean-shift direction convention**
   - Check if the mean-shift should be negated when applying to image space
   - Compare with C++ code in `LandmarkDetectorModel.cpp`

2. **Test isolated parameter update**
   - Create unit test with known mean-shift
   - Verify Jacobian produces correct landmark movement

3. **Compare C++ vs Python transforms**
   - Add debug output of C++ sim_ref_to_img matrix values
   - Verify they match Python computation

4. **Check the actual C++ mean-shift application**
   - Trace C++ code path from mean-shift to parameter update
   - Look for any sign flips or negations

## Code Snippets

### Mean-shift transformation (eye_patch_expert.py:661-662)
```python
mean_shift_2D = mean_shift.reshape(-1, 2)  # (28, 2)
mean_shift_2D = mean_shift_2D @ sim_ref_to_img.T
```

### Similarity transform computation (eye_patch_expert.py:520-522)
```python
sim_img_to_ref = align_shapes_with_scale(image_shape, reference_shape)
sim_ref_to_img = np.linalg.inv(sim_img_to_ref)
```

### Parameter update solving (eye_patch_expert.py:1324-1330)
```python
A = J.T @ W @ J + reg * Lambda_inv
b = J.T @ W @ mean_shift - reg * Lambda_inv @ params
delta_p = np.linalg.solve(A, b)
```

## Verification Test

Simple test confirms basic direction is correct:
```python
# Mean-shift of +1 in X should move landmark right
ms[16] = 1.0  # Eye landmark 8, X component
# Result: dx = +0.6524 (correct!)
```

This suggests the issue is not in the solver itself, but in how the mean-shift is computed or transformed in the actual refinement flow.
