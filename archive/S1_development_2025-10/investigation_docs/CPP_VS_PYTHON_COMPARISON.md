# C++ vs Python Alignment Implementation Comparison

## Current Python Implementation Analysis

### Step-by-Step Comparison

| Step | C++ Implementation | Current Python Implementation | Match? |
|------|-------------------|-------------------------------|---------|
| **1. Load PDM & Scale** | `pdm.mean_shape * 0.7` | `pdm.mean_shape * 0.7` | ✓ YES |
| **2. Extract 2D coords** | First 136 values (2/3 of 204) | First 136 values | ✓ YES |
| **3. Reshape format** | `reshape(1, 2).t()` → (68×2) | Direct `[:136].reshape(68, 2)` | ✓ YES (equivalent) |
| **4. Extract rigid points** | 24 rigid indices | 24 rigid indices | ✓ YES |
| **5. Mean normalize** | Subtract mean of each column | Subtract mean of each axis | ✓ YES |
| **6. Compute RMS scale** | `sqrt(sum(all²) / n)` | `sqrt(sum(all²) / n)` | ✓ YES |
| **7. Normalize by scale** | Divide by RMS | Divide by RMS | ✓ YES |
| **8. Kabsch SVD** | `svd(src.t() * dst)` | `svd(src.T @ dst)` | ✓ YES |
| **9. Reflection check** | `det(V^T × U^T)` | `det(Vt.T @ U.T)` | ✓ YES |
| **10. Rotation matrix** | `R = V^T × corr × U^T` | `R = Vt.T @ corr @ U.T` | ✓ YES |
| **11. Scale × Rotation** | `s * R` | `scale * R` | ✓ YES |
| **12. Matrix usage** | Use directly (NO transpose) | **Using `.T` (TRANSPOSED)** | ❌ **NO** |
| **13. Pose transform** | `T' = scale_rot × [tx, ty]` | **REMOVED** (using `[tx*scale, ty*scale]`) | ❌ **NO** |
| **14. Translation** | `-T'[0] + 56`, `-T'[1] + 56` | `-T[0] + 56 + 2`, `-T[1] + 56 - 2` | ❌ **NO** |
| **15. warpAffine** | INTER_LINEAR | INTER_LINEAR | ✓ YES |

## Critical Differences Found

### Difference #1: Matrix Transpose
**C++ Code:**
```cpp
// Line 127
cv::Matx22f scale_rot_matrix = Utilities::AlignShapesWithScale(source_landmarks, destination_landmarks);

// Lines 130-133: Use directly
warp_matrix(0,0) = scale_rot_matrix(0,0);
warp_matrix(0,1) = scale_rot_matrix(0,1);
warp_matrix(1,0) = scale_rot_matrix(1,0);
warp_matrix(1,1) = scale_rot_matrix(1,1);
```

**Python Code (WRONG):**
```python
# Line 104
scale_rot_matrix = self._align_shapes_with_scale(source_rigid, dest_rigid).T  # ← TRANSPOSE!

# Line 107
warp_matrix = self._build_warp_matrix_no_rotation(scale_factor, pose_tx, pose_ty)
```

**Impact:** Transposing inverts the rotation direction!

### Difference #2: Pose Translation Transform
**C++ Code:**
```cpp
// Lines 135-139
float tx = params_global[4];
float ty = params_global[5];

cv::Vec2f T(tx, ty);
T = scale_rot_matrix * T;  // ← Transform through scale-rotation!
```

**Python Code (WRONG):**
```python
# Lines 205-206
T = np.array([pose_tx * scale, pose_ty * scale])  # ← Only scaled, NOT rotated!
warp_matrix[0, 2] = -T[0] + self.output_width / 2 + 2
warp_matrix[1, 2] = -T[1] + self.output_height / 2 - 2
```

**Impact:** Translation not being transformed through rotation causes misalignment!

### Difference #3: Empirical Shift Corrections
**C++ Code:**
```cpp
// Lines 142-143
warp_matrix(0,2) = -T(0) + out_width/2;
warp_matrix(1,2) = -T(1) + out_height/2;
```

**Python Code (WRONG):**
```python
warp_matrix[0, 2] = -T[0] + self.output_width / 2 + 2  # ← Added +2!
warp_matrix[1, 2] = -T[1] + self.output_height / 2 - 2  # ← Added -2!
```

**Impact:** These arbitrary shifts throw off the alignment!

## Root Cause of Progressive Rotation

The combination of these three errors creates a compounding problem:

1. **Transpose Error:** Inverts rotation direction
2. **Translation Error:** Pose is not transformed through rotation, so as face moves, translation is increasingly wrong
3. **Shift Error:** Adds consistent offset that compounds with translation error

As the face moves across frames (changing `pose_tx`, `pose_ty`), the incorrect translation computation causes progressively worse rotation alignment.

## Correct Python Implementation

Here's what the Python code SHOULD be:

```python
def align_face(self, image, landmarks_68, pose_tx, pose_ty):
    # Extract rigid points
    source_rigid = self._extract_rigid_points(landmarks_68)
    dest_rigid = self._extract_rigid_points(self.reference_shape)

    # Compute scale-rotation (DO NOT TRANSPOSE!)
    scale_rot_matrix = self._align_shapes_with_scale(source_rigid, dest_rigid)

    # Build warp matrix
    warp_matrix = self._build_warp_matrix(scale_rot_matrix, pose_tx, pose_ty)

    # Apply transformation
    aligned_face = cv2.warpAffine(image, warp_matrix, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned_face

def _build_warp_matrix(self, scale_rot, pose_tx, pose_ty):
    warp_matrix = np.zeros((2, 3), dtype=np.float32)

    # Copy scale-rotation directly (NO TRANSPOSE!)
    warp_matrix[:2, :2] = scale_rot

    # Transform translation through scale-rotation
    T = scale_rot @ np.array([pose_tx, pose_ty], dtype=np.float32)

    # Simple centering (NO EMPIRICAL SHIFTS!)
    warp_matrix[0, 2] = -T[0] + self.output_width / 2
    warp_matrix[1, 2] = -T[1] + self.output_height / 2

    return warp_matrix
```

## Expected Impact of Fixes

1. **Remove transpose:** Will restore correct rotation direction
2. **Transform pose through scale-rotation:** Will correctly handle face movement across frames
3. **Remove empirical shifts:** Will eliminate arbitrary offsets

These fixes should eliminate the progressive rotation drift and produce consistently upright faces matching C++ output.
