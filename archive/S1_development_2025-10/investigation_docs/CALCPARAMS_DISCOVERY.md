# Critical Discovery: C++ CalcParams() Function

## The Finding

C++ does NOT use the pose parameters (p_tx, p_ty, p_rx, p_ry, p_rz) directly from the CSV for alignment!

Instead, it **recalculates** `params_global` using:

```cpp
pdm.CalcParams(params_global, params_local, detected_landmarks);
```

**Location:** FaceAnalyser.cpp line 240 (before AlignFace is called)

## CalcParams Function Signature

```cpp
void PDM::CalcParams(
    cv::Vec6f& out_params_global,
    cv::Mat_<float>& out_params_local,
    const cv::Mat_<float>& landmark_locations,
    const cv::Vec3f rotation = cv::Vec3f(0.0f)  // ← DEFAULT: (0, 0, 0)!
);
```

**Key insight:** The rotation parameter defaults to **(0, 0, 0)** when not provided!

## What CalcParams Actually Does

Based on PDM.cpp lines 508-587:

1. **Filters invisible landmarks** (where x coordinate = 0)
2. **Computes bounding box** from visible landmarks
3. **Calculates scale** from bounding box: `scaling = ((width / model_bbox.width) + (height / model_bbox.height)) / 2.0`
4. **Uses provided rotation** (defaults to 0,0,0): `rotation_init = rotation`
5. **Computes translation** from centroid: `translation = ((min_x + max_x) / 2, (min_y + max_y) / 2)`
6. **Optimizes params** to fit PDM model to landmarks

## Impact on AlignFace

AlignFace receives `params_global` which contains:
- `params_global[0]`: scale (recomputed)
- `params_global[1]`: rx = 0 (from default rotation)
- `params_global[2]`: ry = 0 (from default rotation)
- `params_global[3]`: rz = 0 (from default rotation)
- `params_global[4]`: tx (recomputed from centroid)
- `params_global[5]`: ty (recomputed from centroid)

**But AlignFace only uses `params_global[4]` and `params_global[5]`** (tx, ty) from Face_utils.cpp lines 135-136!

The rotation components [1:4] are **never used** in the 2D AlignFace function.

## Why This Matters

### What We Thought:
- C++ uses p_tx, p_ty directly from CSV
- CSV values are accurate pose parameters

### What Actually Happens:
- C++ recalculates tx, ty from landmark centroid
- CSV rotation params (p_rx, p_ry, p_rz) are ignored
- Translation is based purely on bounding box center

## Test Results

Tested recalculating tx, ty from landmark centroid vs using CSV values:

| Method | Frame 1 Correlation |
|--------|-------------------|
| CSV tx, ty | 0.750547 |
| Recalculated tx, ty | 0.745576 |

**No improvement** - suggesting translation calculation difference is not the main issue.

## Remaining Mystery

If C++ also uses:
- ✓ Same PDM file (rotated 45°)
- ✓ Same Kabsch algorithm
- ✓ Same rigid points
- ✓ Rotation from Kabsch (not from params_global)

**Why does C++ produce upright faces while Python produces ~5° tilted faces?**

## Hypotheses to Investigate

1. **PDM coordinate space transformation**: Maybe there's a coordinate system conversion we're missing
2. **Landmark preprocessing**: Maybe landmarks are rotated/normalized before AlignFace
3. **Different PDM file in practice**: Maybe production C++ uses a different PDM
4. **Post-processing rotation correction**: Maybe there's a step after AlignFace we haven't found

## Next Steps

1. Search for any coordinate system transformations before AlignFace is called
2. Check if landmarks undergo any preprocessing
3. Verify which PDM file is actually loaded by C++ at runtime
4. Look for post-processing after AlignFace returns

## Key Code Locations

- **CalcParams call**: `FaceAnalyser.cpp:240`
- **CalcParams implementation**: `PDM.cpp:508-587`
- **CalcParams declaration**: `PDM.h` (shows default rotation parameter)
- **AlignFace usage of params_global**: `Face_utils.cpp:135-136` (only tx, ty used)
