# Mean-Shift Bug Fixes - Implementation Guide

**Quick reference for fixing the two critical bugs in `pyclnf/core/optimizer.py`**

---

## Fix #1: Pass reference_shape to _compute_mean_shift()

### Change 1: Update method call (line ~145)

**Before**:
```python
mean_shift = self._compute_mean_shift(
    landmarks_2d, patch_experts, image, pdm, window_size,
    sim_img_to_ref, sim_ref_to_img, sigma_components
)
```

**After**:
```python
mean_shift = self._compute_mean_shift(
    landmarks_2d, patch_experts, image, pdm, window_size,
    sim_img_to_ref, sim_ref_to_img, sigma_components,
    reference_shape  # ADD THIS - computed at line 134
)
```

### Change 2: Update method signature (line ~216)

**Before**:
```python
def _compute_mean_shift(self,
                       landmarks_2d: np.ndarray,
                       patch_experts: dict,
                       image: np.ndarray,
                       pdm,
                       window_size: int = 11,
                       sim_img_to_ref: np.ndarray = None,
                       sim_ref_to_img: np.ndarray = None,
                       sigma_components: dict = None) -> np.ndarray:
```

**After**:
```python
def _compute_mean_shift(self,
                       landmarks_2d: np.ndarray,
                       patch_experts: dict,
                       image: np.ndarray,
                       pdm,
                       window_size: int = 11,
                       sim_img_to_ref: np.ndarray = None,
                       sim_ref_to_img: np.ndarray = None,
                       sigma_components: dict = None,
                       reference_shape: np.ndarray = None) -> np.ndarray:  # ADD THIS
```

---

## Fix #2: Compute dx, dy in REFERENCE coordinates

### Change 3: Replace dx_frac, dy_frac computation (lines ~284-291)

**Before**:
```python
# For fractional positions, we need sub-pixel offset
dx_frac = lm_x - int(lm_x)
dy_frac = lm_y - int(lm_y)

# Compute KDE mean-shift using OpenFace's algorithm
# Result is in REFERENCE coordinates if warping was used
ms_ref_x, ms_ref_y = self._kde_mean_shift(
    response_map, dx_frac + center, dy_frac + center, a
)
```

**After**:
```python
# Compute position within response map in REFERENCE coordinates
if use_warping and reference_shape is not None:
    # Compute displacement from reference shape in IMAGE coordinates
    ref_lm_x, ref_lm_y = reference_shape[landmark_idx]
    displacement_img_x = lm_x - ref_lm_x
    displacement_img_y = lm_y - ref_lm_y

    # Transform displacement to REFERENCE coordinates (rotation + scale only)
    a_img_to_ref = sim_img_to_ref[0, 0]
    b_img_to_ref = sim_img_to_ref[1, 0]
    displacement_ref_x = a_img_to_ref * displacement_img_x - b_img_to_ref * displacement_img_y
    displacement_ref_y = b_img_to_ref * displacement_img_x + a_img_to_ref * displacement_img_y

    # Add center offset to get position within response map
    dx = displacement_ref_x + center
    dy = displacement_ref_y + center
else:
    # No warping: use fractional part of IMAGE coordinates
    dx = (lm_x - int(lm_x)) + center
    dy = (lm_y - int(lm_y)) + center

# Compute KDE mean-shift using OpenFace's algorithm
# Result is in REFERENCE coordinates if warping was used
ms_ref_x, ms_ref_y = self._kde_mean_shift(
    response_map, dx, dy, a
)
```

**Note**: Changed `center` computation from `resp_size // 2` (integer) to `(resp_size - 1) / 2.0` (float) for sub-pixel precision - see Change 5 below.

---

## Fix #3: Correct mean-shift transform signs

### Change 4: Fix matrix multiplication (lines ~296-301)

**Before**:
```python
if use_warping:
    # Transform mean-shift from REFERENCE back to IMAGE coordinates
    # Extract scale and rotation from sim_ref_to_img
    a_mat = sim_ref_to_img[0, 0]
    b_mat = sim_ref_to_img[1, 0]
    # Mean-shift transforms as: ms_img = R * ms_ref (no translation)
    ms_x = a_mat * ms_ref_x + b_mat * ms_ref_y  # ❌ WRONG SIGN
    ms_y = -b_mat * ms_ref_x + a_mat * ms_ref_y  # ❌ WRONG SIGN
else:
    ms_x = ms_ref_x
    ms_y = ms_ref_y
```

**After**:
```python
if use_warping:
    # Transform mean-shift from REFERENCE back to IMAGE coordinates
    # Apply 2x2 rotation/scale matrix: [a -b; b a]
    a_mat = sim_ref_to_img[0, 0]
    b_mat = sim_ref_to_img[1, 0]
    ms_x = a_mat * ms_ref_x - b_mat * ms_ref_y  # Fixed: + to -
    ms_y = b_mat * ms_ref_x + a_mat * ms_ref_y  # Fixed: - to +
else:
    ms_x = ms_ref_x
    ms_y = ms_ref_y
```

**Alternative (more readable)**:
```python
if use_warping:
    # Transform mean-shift from REFERENCE back to IMAGE coordinates
    # Apply 2x2 rotation/scale matrix (no translation)
    rotation_scale = sim_ref_to_img[:2, :2]
    ms_img = rotation_scale @ np.array([ms_ref_x, ms_ref_y])
    ms_x, ms_y = ms_img[0], ms_img[1]
else:
    ms_x = ms_ref_x
    ms_y = ms_ref_y
```

---

## Fix #4: Use float for center (sub-pixel precision)

### Change 5: Update center computation (line ~282)

**Before**:
```python
center = resp_size // 2  # Integer division
```

**After**:
```python
center = (resp_size - 1) / 2.0  # Float, matches OpenFace
```

**Why**: OpenFace uses `(resp_size - 1) / 2` which is float division in C++. This provides sub-pixel precision for centering the KDE kernel.

---

## Summary of All Changes

1. **Line ~145**: Add `reference_shape` argument to `_compute_mean_shift()` call
2. **Line ~216**: Add `reference_shape` parameter to method signature
3. **Line ~282**: Change `center = resp_size // 2` to `center = (resp_size - 1) / 2.0`
4. **Lines ~284-291**: Replace dx_frac/dy_frac logic with proper REFERENCE-space computation
5. **Lines ~296-301**: Fix mean-shift transform signs (+ to -, - to +)

---

## Testing After Fixes

Run these checks to verify the fixes work:

### 1. Basic convergence test
```python
python test_current_implementation.py
```

Expected: Final update magnitude should be **<0.01** instead of **3.88**.

### 2. Check mean-shift vectors
Add debug output before/after fixes:
```python
print(f"Mean-shift magnitude: {np.linalg.norm(mean_shift)}")
```

Expected behavior:
- **Iteration 0**: Large magnitude (e.g., 5-10 pixels)
- **Iteration 5**: Medium magnitude (e.g., 0.5-2 pixels)
- **Iteration 10**: Small magnitude (e.g., <0.01 pixels)

### 3. Visual verification
Run landmark detection on test images and verify:
- Landmarks converge to correct positions
- No jittering or divergence
- Matches OpenFace C++ output

---

## Diff Preview

For quick copy-paste, here's the complete diff:

```diff
--- a/pyclnf/core/optimizer.py
+++ b/pyclnf/core/optimizer.py
@@ -142,7 +142,8 @@ class NURLMSOptimizer:

             # 5. Compute mean-shift vector from patch responses
             mean_shift = self._compute_mean_shift(
-                landmarks_2d, patch_experts, image, pdm, window_size,
-                sim_img_to_ref, sim_ref_to_img, sigma_components
+                landmarks_2d, patch_experts, image, pdm,
+                window_size, sim_img_to_ref, sim_ref_to_img,
+                sigma_components, reference_shape
             )

@@ -219,7 +220,8 @@ class NURLMSOptimizer:
                            window_size: int = 11,
                            sim_img_to_ref: np.ndarray = None,
                            sim_ref_to_img: np.ndarray = None,
-                           sigma_components: dict = None) -> np.ndarray:
+                           sigma_components: dict = None,
+                           reference_shape: np.ndarray = None) -> np.ndarray:
         """
         Compute mean-shift vector from patch expert responses using KDE.

@@ -279,17 +281,34 @@ class NURLMSOptimizer:

             # Current offset within response map
             resp_size = response_map.shape[0]
-            center = resp_size // 2
-
-            # For fractional positions, we need sub-pixel offset
-            dx_frac = lm_x - int(lm_x)
-            dy_frac = lm_y - int(lm_y)
-
-            # Compute KDE mean-shift using OpenFace's algorithm
-            # Result is in REFERENCE coordinates if warping was used
+            center = (resp_size - 1) / 2.0
+
+            # Compute position within response map in REFERENCE coordinates
+            if use_warping and reference_shape is not None:
+                # Compute displacement from reference shape in IMAGE coordinates
+                ref_lm_x, ref_lm_y = reference_shape[landmark_idx]
+                displacement_img_x = lm_x - ref_lm_x
+                displacement_img_y = lm_y - ref_lm_y
+
+                # Transform displacement to REFERENCE coordinates
+                a_img_to_ref = sim_img_to_ref[0, 0]
+                b_img_to_ref = sim_img_to_ref[1, 0]
+                displacement_ref_x = (a_img_to_ref * displacement_img_x -
+                                     b_img_to_ref * displacement_img_y)
+                displacement_ref_y = (b_img_to_ref * displacement_img_x +
+                                     a_img_to_ref * displacement_img_y)
+
+                dx = displacement_ref_x + center
+                dy = displacement_ref_y + center
+            else:
+                # No warping: use fractional part of IMAGE coordinates
+                dx = (lm_x - int(lm_x)) + center
+                dy = (lm_y - int(lm_y)) + center
+
+            # Compute KDE mean-shift
             ms_ref_x, ms_ref_y = self._kde_mean_shift(
-                response_map, dx_frac + center, dy_frac + center, a
+                response_map, dx, dy, a
             )

             if use_warping:
-                # Transform mean-shift from REFERENCE back to IMAGE coordinates
-                a_mat = sim_ref_to_img[0, 0]
-                b_mat = sim_ref_to_img[1, 0]
-                ms_x = a_mat * ms_ref_x + b_mat * ms_ref_y
-                ms_y = -b_mat * ms_ref_x + a_mat * ms_ref_y
+                # Transform back to IMAGE coordinates
+                rotation_scale = sim_ref_to_img[:2, :2]
+                ms_img = rotation_scale @ np.array([ms_ref_x, ms_ref_y])
+                ms_x, ms_y = ms_img[0], ms_img[1]
             else:
```

---

## Notes

1. **Backwards compatibility**: The `reference_shape` parameter is optional (defaults to None), so existing calls without it will still work (but won't benefit from the fix).

2. **No warping case**: When `use_warping=False` (direct extraction without warping), the original fractional offset logic is preserved for backwards compatibility.

3. **Alternative implementation**: Instead of extracting `a_mat` and `b_mat`, you can use direct matrix multiplication `sim_ref_to_img[:2, :2] @ [ms_ref_x, ms_ref_y]` for cleaner code.

4. **Validation**: Consider adding an assertion to verify `reference_shape` is provided when `use_warping=True`:
   ```python
   if use_warping:
       assert reference_shape is not None, "reference_shape required for warping mode"
   ```

---

**Ready to implement!** Apply these changes to `/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/core/optimizer.py` and test.
