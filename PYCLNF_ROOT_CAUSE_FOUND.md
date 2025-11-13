# PyCLNF Scale Problem: ROOT CAUSE IDENTIFIED

## Executive Summary

The 29% scale convergence discrepancy is caused by **missing reference shape transformation**. OpenFace evaluates patch responses in a CANONICAL COORDINATE SYSTEM defined by `patch_scaling`, but PyCLNF evaluates them directly in image coordinates at the wrong scale.

## The Critical Discovery

### OpenFace C++ (CORRECT)

```cpp
// 1. Create reference shape at FIXED scale (patch_scaling)
cv::Vec6f global_ref(patch_scaling[scale], 0, 0, 0, 0, 0);
pdm.CalcShape2D(reference_shape, params_local, global_ref);

// 2. Compute similarity transforms between image and reference
sim_img_to_ref = AlignShapesWithScale(image_shape_2D, reference_shape_2D);
sim_ref_to_img = sim_img_to_ref.inv();

// 3. Pass transforms to Response computation
patch_experts.Response(patch_expert_responses, sim_ref_to_img, sim_img_to_ref,
                      grayscale_image, pdm, params_global, params_local, window_size, scale);
```

**Key insight**: Patch experts were TRAINED on images normalized to specific scales (0.25, 0.35, 0.5). The reference shape creates a canonical coordinate system at the training scale, ensuring patches are evaluated at the correct scale regardless of the current model's scale parameter.

### PyCLNF (WRONG)

```python
# 1. Get current landmark positions (no reference shape!)
landmarks_2d = pdm.params_to_landmarks_2d(params)  # Uses current scale!

# 2. Evaluate patches directly in image coordinates (WRONG SCALE!)
response_map = patch_expert.evaluate(image, lm_x, lm_y, window_size)
```

**Problem**: Patches are evaluated at whatever scale the current model happens to be at, not at the training scale. This produces incorrect patch responses, leading to wrong mean-shifts and ultimately wrong scale convergence.

## Why This Causes 29% Scale Error

The patch experts expect to see features at a specific size:
- **0.25 scale patches**: Trained on very coarse features (downsampled 4x)
- **0.35 scale patches**: Trained on medium features
- **0.50 scale patches**: Trained on fine features (downsampled 2x)

Without the reference shape transformation:
1. Current model at scale=2.1 produces landmarks at certain pixel distances
2. Patches evaluate features at those pixel distances DIRECTLY
3. Features appear LARGER than they should to the patch experts (wrong scale)
4. Patch responses are computed incorrectly
5. Mean-shifts push model in wrong direction
6. Optimizer converges to wrong scale (too small by 29%)

With the reference shape transformation:
1. Reference shape created at patch_scaling=0.25 (fixed)
2. Similarity transform computed from current landmarks → reference landmarks
3. Image patches WARPED into reference coordinate system
4. Features appear at CORRECT scale to patch experts (normalized)
5. Patch responses computed correctly
6. Optimizer converges to correct scale

## What Was Already Fixed

✅ Multi-scale patch expert loading (0.25, 0.35, 0.5)
✅ Window-to-scale mapping
✅ Passing correct scale to patch expert retrieval
✅ Adjusted regularization based on patch scale

These fixes were NECESSARY but NOT SUFFICIENT. We're loading the right patch experts but using them wrong!

## What Still Needs To Be Implemented

### 1. Reference Shape Generation (PDM)

Add method to generate reference shape at fixed scale:

```python
# pyclnf/core/pdm.py
def get_reference_shape(self, patch_scaling: float, params_local: np.ndarray = None) -> np.ndarray:
    """
    Generate reference shape at fixed scale for patch evaluation.

    Args:
        patch_scaling: Fixed scale for reference shape (0.25, 0.35, or 0.5)
        params_local: Local shape parameters (default: zeros)

    Returns:
        reference_shape: 2D landmarks at reference scale, shape (n_points, 2)
    """
    if params_local is None:
        params_local = np.zeros(self.n_modes)

    # Create reference global params: [scale, tx, ty, wx, wy, wz]
    # Scale = patch_scaling, rotation = 0, translation = 0
    global_ref = np.array([patch_scaling, 0, 0, 0, 0, 0])

    # Concatenate global and local params
    ref_params = np.concatenate([global_ref, params_local])

    # Generate 2D shape
    reference_shape = self.params_to_landmarks_2d(ref_params)

    return reference_shape
```

### 2. Similarity Transform Computation (Utilities)

Add function to compute similarity transform (OpenFace's AlignShapesWithScale):

```python
# pyclnf/core/utils.py
def align_shapes_with_scale(src_shape: np.ndarray, dst_shape: np.ndarray) -> np.ndarray:
    """
    Compute similarity transform (scale + rotation + translation) from src to dst.

    This matches OpenFace's Utilities::AlignShapesWithScale function.

    Args:
        src_shape: Source landmarks, shape (n_points, 2)
        dst_shape: Destination landmarks, shape (n_points, 2)

    Returns:
        transform: 2x3 similarity transform matrix [a -b tx; b a ty]
                  where (a,b) encode scale+rotation, (tx,ty) is translation
    """
    # Center shapes
    src_mean = src_shape.mean(axis=0)
    dst_mean = dst_shape.mean(axis=0)

    src_centered = src_shape - src_mean
    dst_centered = dst_shape - dst_mean

    # Compute scale
    src_scale = np.sqrt((src_centered ** 2).sum())
    dst_scale = np.sqrt((dst_centered ** 2).sum())

    src_norm = src_centered / (src_scale + 1e-8)
    dst_norm = dst_centered / (dst_scale + 1e-8)

    # Compute rotation
    a = (src_norm * dst_norm).sum()
    b = (src_norm[:, 0] * dst_norm[:, 1] - src_norm[:, 1] * dst_norm[:, 0]).sum()

    scale = dst_scale / (src_scale + 1e-8)

    # Build transform matrix
    transform = np.array([
        [scale * a, -scale * b, dst_mean[0] - scale * (a * src_mean[0] - b * src_mean[1])],
        [scale * b,  scale * a, dst_mean[1] - scale * (b * src_mean[0] + a * src_mean[1])]
    ])

    return transform
```

### 3. Modified Optimizer (Use Reference Coordinates)

Update optimizer to accept patch_scaling and use reference transform:

```python
# pyclnf/core/optimizer.py: optimize()
def optimize(self,
             pdm,
             initial_params: np.ndarray,
             patch_experts: dict,
             image: np.ndarray,
             weights: Optional[np.ndarray] = None,
             window_size: int = 11,
             patch_scaling: float = 0.25) -> Tuple[np.ndarray, dict]:  # NEW PARAMETER
    """
    Args:
        patch_scaling: Fixed scale for reference shape (must match patch experts)
    """
    params = initial_params.copy()

    for iteration in range(self.max_iterations):
        # 1. Get current landmarks in image coordinates
        landmarks_image = pdm.params_to_landmarks_2d(params)

        # 2. Get reference shape at patch_scaling
        reference_shape = pdm.get_reference_shape(patch_scaling, params[6:])

        # 3. Compute similarity transforms
        sim_img_to_ref = align_shapes_with_scale(landmarks_image, reference_shape)
        sim_ref_to_img = np.linalg.inv(np.vstack([sim_img_to_ref, [0, 0, 1]]))[:2, :]

        # 4. Compute mean-shift in reference coordinates
        mean_shift_ref = self._compute_mean_shift_with_transform(
            landmarks_image, reference_shape, sim_img_to_ref, sim_ref_to_img,
            patch_experts, image, window_size
        )

        # 5. Transform mean-shift back to image coordinates
        # mean_shift_image = apply_transform(mean_shift_ref, sim_ref_to_img)

        # ... rest of optimization
```

### 4. Modified Mean-Shift Computation

Update mean-shift to use reference coordinates:

```python
def _compute_mean_shift_with_transform(self,
                                       landmarks_image: np.ndarray,
                                       landmarks_ref: np.ndarray,
                                       sim_img_to_ref: np.ndarray,
                                       sim_ref_to_img: np.ndarray,
                                       patch_experts: dict,
                                       image: np.ndarray,
                                       window_size: int) -> np.ndarray:
    """
    Compute mean-shift in reference coordinates.

    Key difference from current implementation:
    - Landmark positions are in REFERENCE coordinates
    - Patches are sampled from IMAGE using sim_ref_to_img transform
    - Mean-shift is computed in REFERENCE coordinates
    """
    n_points = landmarks_ref.shape[0]
    mean_shift = np.zeros(2 * n_points)

    for landmark_idx, patch_expert in patch_experts.items():
        # Reference landmark position
        ref_x, ref_y = landmarks_ref[landmark_idx]

        # Map reference position to image position
        img_pos = apply_transform_point([ref_x, ref_y], sim_ref_to_img)

        # Sample patch from image at image position
        # Evaluate response in REFERENCE coordinates
        response_map = patch_expert.evaluate(image, img_pos[0], img_pos[1], window_size)

        # Compute KDE mean-shift in reference coordinates
        # ... (rest of KDE logic)
```

## Implementation Complexity

This is a SIGNIFICANT architectural change requiring:

1. **New PDM method**: `get_reference_shape()`
2. **New utility function**: `align_shapes_with_scale()`
3. **Modified optimizer interface**: Add `patch_scaling` parameter
4. **Modified mean-shift computation**: Use reference coordinates
5. **Coordinate transform utilities**: Apply transforms to points/vectors

Estimated implementation: ~300-400 lines of new/modified code

## Alternative: Simpler Approximation?

Could we approximate the reference transform with a simple scale adjustment?

**NO** - The reference transform is a FULL SIMILARITY TRANSFORM (scale + rotation + translation), not just scale. The rotation component is critical for handling non-frontal poses. A simple scale adjustment would be incorrect.

## Next Steps

1. Implement `get_reference_shape()` in PDM
2. Implement `align_shapes_with_scale()` in utils
3. Add coordinate transform utilities
4. Modify optimizer to accept `patch_scaling`
5. Modify mean-shift to use reference coordinates
6. Update CLNF.fit() to pass `patch_scaling` to optimizer
7. Test with OpenFace bbox and verify scale convergence

## Expected Outcome

After implementing reference shape transformation:
- **Before**: scale=2.134 (23.7% too small)
- **After**: scale=2.799 ± 0.05 (matches OpenFace C++)

This should fix the scale convergence problem completely.
