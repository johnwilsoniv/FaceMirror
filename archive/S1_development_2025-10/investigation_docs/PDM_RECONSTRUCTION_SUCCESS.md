# PDM Reconstruction Success - Full Understanding Achieved

**Date:** 2025-10-29
**Status:** ✅ COMPLETED

## Achievement

We have **perfectly replicated OpenFace C++ PDM reconstruction** in Python with RMSE < 0.1 pixels.

## Results

```
Frame    1: RMSE X=0.035 px, Y=0.047 px  ✓ Perfect match
Frame  493: RMSE X=0.045 px, Y=0.037 px  ✓ Perfect match
Frame  617: RMSE X=0.039 px, Y=0.038 px  ✓ Perfect match (eyes closed!)
Frame  863: RMSE X=0.060 px, Y=0.066 px  ✓ Perfect match
```

## Implementation

### CalcShape3D
```python
shape_3d = mean_shape + principal_components @ params_local
```
- `mean_shape`: (204, 1) - PDM canonical face
- `principal_components`: (204, 34) - PCA basis vectors
- `params_local`: (34, 1) - Expression/identity parameters from CSV (p_0...p_33)
- Result: 3D face shape with expression deformation

### CalcShape2D
```python
# Build 3D rotation matrix from Euler angles (XYZ convention)
R = euler_to_rotation_matrix(rx, ry, rz)

# Apply rotation and scale
rotated = scale * (R @ shape_3d_points.T).T

# Project to 2D (weak perspective)
out_x = rotated[:, 0] + tx
out_y = rotated[:, 1] + ty
```

- Applies params_global: [scale, rx, ry, rz, tx, ty]
- 3D rotation → 2D projection → translation
- Result: 2D landmarks matching CSV exactly

## Key Findings

### 1. CSV Landmarks ARE PDM-Reconstructed
- **Confirmed by code trace:** FeatureExtraction.cpp → face_model.detected_landmarks → CalcShape2D()
- **Confirmed by perfect reconstruction:** RMSE < 0.1 pixels across all test frames
- Expression deformation is included via params_local

### 2. Rotation Convention is XYZ Euler Angles
- Verified by matching CSV landmarks
- Order: Rx (pitch) → Ry (yaw) → Rz (roll)
- All params are in **radians** (not degrees)

### 3. PDM Format Understanding
PDM stores coordinates as: `[x0...x67, y0...y67, z0...z67]`
NOT interleaved: `[x0,y0,z0, x1,y1,z1, ...]`

This is critical for correct reshaping.

### 4. Expression is Encoded in params_local
The CSV landmarks include expression deformation:
- Neutral expression: params_local ≈ 0
- Eyes closed (frame 617): params_local ≠ 0
- Still achieves perfect reconstruction

## Implications for Alignment Problem

**We now know:**
1. ✅ CSV landmarks are PDM-reconstructed (not raw detections)
2. ✅ They've been rotated by params_global 3D rotation matrix
3. ✅ Expression is already factored in via params_local
4. ✅ We can perfectly replicate the C++ reconstruction

**The remaining mystery:**
- C++ AlignFace produces upright faces (~0° rotation)
- Our Python AlignFace produces tilted faces (-8° to +2°)
- Both use Kabsch alignment of CSV landmarks → PDM mean_shape

**Next investigation:**
What does C++ AlignFace do differently that produces ~0° rotation?

Possible hypotheses:
1. C++ applies some coordinate transform we're missing
2. C++ uses a different reference shape (not raw mean_shape)
3. C++ incorporates params_global rotation in a way we don't
4. There's a post-processing rotation correction

## Code Files

- `full_pdm_reconstruction.py` - Complete PDM reconstruction implementation
- `pdm_parser.py` - PDM file parser (already existing)
- Test data: `of22_validation/IMG_0942_left_mirrored.csv`

## Next Steps

1. ✅ PDM reconstruction complete
2. 🔄 Investigate C++ AlignFace rotation difference
3. ⏸️ Test alternative alignment approaches if needed
4. ⏸️ Implement pybind11 wrapper if pure Python proves infeasible
