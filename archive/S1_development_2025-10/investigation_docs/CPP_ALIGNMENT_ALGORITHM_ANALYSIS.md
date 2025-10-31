# OpenFace C++ Face Alignment Algorithm - Step-by-Step Analysis

## Function: `AlignFace()`
**Location:** `Face_utils.cpp` lines 110-146

## Input Parameters
```cpp
void AlignFace(
    cv::Mat& aligned_face,              // Output: 112×112 aligned face
    const cv::Mat& frame,                // Input: original video frame
    const cv::Mat_<float>& detected_landmarks,  // Input: 68 landmarks (136,1) format [x0,y0,x1,y1,...,x67,y67]
    cv::Vec6f params_global,             // Input: [scale, rx, ry, rz, tx, ty]
    const LandmarkDetector::PDM& pdm,    // Input: Point Distribution Model
    bool rigid,                          // Input: true = use only rigid points
    double sim_scale,                    // Input: 0.7 for AU analysis
    int out_width,                       // Input: 112
    int out_height                       // Input: 112
)
```

## Step-by-Step Algorithm

### Step 1: Prepare Reference Shape
```cpp
// Line 113
cv::Mat_<float> similarity_normalised_shape = pdm.mean_shape * sim_scale;
```
- Takes PDM mean shape (204×1 matrix with [x0,y0,x1,y1,...,x67,y67,z0,z1,...,z67])
- Scales by `sim_scale = 0.7`
- **Result:** Scaled mean shape

### Step 2: Discard Z Component
```cpp
// Line 116
similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();
```
- Takes first 136 rows (first 2/3 of 204)
- This extracts only X and Y coordinates, discarding Z
- **Format:** (136×1) matrix: [x0,y0,x1,y1,...,x67,y67]

### Step 3: Reshape to (N×2) Format
```cpp
// Lines 118-119
cv::Mat_<float> source_landmarks = detected_landmarks.reshape(1, 2).t();
cv::Mat_<float> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();
```

**Critical reshaping logic:**
- `reshape(1, 2)` changes (136×1) → (2×68)
  - Row 0: [x0, x1, x2, ..., x67]
  - Row 1: [y0, y1, y2, ..., y67]
- `.t()` transposes (2×68) → (68×2)
  - Row 0: [x0, y0]
  - Row 1: [x1, y1]
  - ...
  - Row 67: [x67, y67]

**Result:** Both `source_landmarks` and `destination_landmarks` are now (68×2) matrices.

### Step 4: Extract Rigid Points (if rigid=true)
```cpp
// Lines 122-125
if(rigid)
{
    extract_rigid_points(source_landmarks, destination_landmarks);
}
```

The `extract_rigid_points` function (lines 46-107) filters to 24 rigid landmark indices:
```
{1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47}
```

**After this step:**
- `source_landmarks`: (24×2) if rigid=true, else (68×2)
- `destination_landmarks`: (24×2) if rigid=true, else (68×2)

### Step 5: Compute Scale-Rotation Matrix
```cpp
// Line 127
cv::Matx22f scale_rot_matrix = Utilities::AlignShapesWithScale(source_landmarks, destination_landmarks);
```

Calls `AlignShapesWithScale()` from `RotationHelpers.h` (lines 195-241):

#### Sub-step 5a: Mean Normalize
```cpp
// Lines 200-212
float mean_src_x = cv::mean(src.col(0))[0];
float mean_src_y = cv::mean(src.col(1))[0];
float mean_dst_x = cv::mean(dst.col(0))[0];
float mean_dst_y = cv::mean(dst.col(1))[0];

cv::Mat_<float> src_mean_normed = src.clone();
src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

cv::Mat_<float> dst_mean_normed = dst.clone();
dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;
```

#### Sub-step 5b: Compute RMS Scales
```cpp
// Lines 215-222
cv::Mat src_sq;
cv::pow(src_mean_normed, 2, src_sq);

cv::Mat dst_sq;
cv::pow(dst_mean_normed, 2, dst_sq);

float s_src = sqrt(cv::sum(src_sq)[0] / n);
float s_dst = sqrt(cv::sum(dst_sq)[0] / n);
```

**Formula:** `s = sqrt(sum(x²+y²) / n)` for all points

#### Sub-step 5c: Normalize by Scale
```cpp
// Lines 224-227
src_mean_normed = src_mean_normed / s_src;
dst_mean_normed = dst_mean_normed / s_dst;

float s = s_dst / s_src;
```

#### Sub-step 5d: Kabsch Rotation
```cpp
// Line 230
cv::Matx22f R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);
```

Calls `AlignShapesKabsch2D()` (lines 168-191):

```cpp
// Line 171
cv::SVD svd(align_from.t() * align_to);

// Lines 175-185: Prevent reflection
double d = cv::determinant(svd.vt.t() * svd.u.t());

cv::Matx22f corr = cv::Matx22f::eye();
if (d > 0)
    corr(1, 1) = 1;
else
    corr(1, 1) = -1;

// Line 188
cv::Matx22f R;
cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);
```

**Formula:** `R = V^T × corr × U^T` where SVD = U·Σ·V^T

#### Sub-step 5e: Return Scaled Rotation
```cpp
// Lines 232-233
cv::Matx22f A;
cv::Mat(s * R).copyTo(A);
return A;
```

**Result:** `scale_rot_matrix` is a (2×2) matrix = scale × rotation

### Step 6: Build Warp Matrix
```cpp
// Lines 128-133
cv::Matx23f warp_matrix;

warp_matrix(0,0) = scale_rot_matrix(0,0);
warp_matrix(0,1) = scale_rot_matrix(0,1);
warp_matrix(1,0) = scale_rot_matrix(1,0);
warp_matrix(1,1) = scale_rot_matrix(1,1);
```

Copies (2×2) scale-rotation matrix into first 2 columns of (2×3) warp matrix.

### Step 7: Extract Translation from params_global
```cpp
// Lines 135-136
float tx = params_global[4];
float ty = params_global[5];
```

**params_global format:** [scale, rx, ry, rz, tx, ty]
- `tx = params_global[4]` = p_tx from CSV
- `ty = params_global[5]` = p_ty from CSV

### Step 8: Transform Translation Through Scale-Rotation
```cpp
// Lines 138-139
cv::Vec2f T(tx, ty);
T = scale_rot_matrix * T;
```

**Critical:** Translation is transformed through the scale-rotation matrix!

### Step 9: Compute Final Translation for Centering
```cpp
// Lines 142-143
warp_matrix(0,2) = -T(0) + out_width/2;
warp_matrix(1,2) = -T(1) + out_height/2;
```

**Formula:**
- `warp_matrix[0,2] = -T_transformed[0] + 56`
- `warp_matrix[1,2] = -T_transformed[1] + 56`

### Step 10: Apply Affine Transformation
```cpp
// Line 145
cv::warpAffine(frame, aligned_face, warp_matrix, cv::Size(out_width, out_height), cv::INTER_LINEAR);
```

Uses OpenCV's `warpAffine` with bilinear interpolation.

## Summary of Matrix Operations

**Final warp matrix structure:**
```
[scale×R[0,0]  scale×R[0,1]  -T_x' + 56]
[scale×R[1,0]  scale×R[1,1]  -T_y' + 56]
```

Where:
- `R` is the 2D rotation matrix from Kabsch
- `scale = s_dst / s_src`
- `T' = scale×R × [tx, ty]`

## Key Implementation Details

1. **Coordinate format:** PDM stores as [all X, all Y, all Z] NOT interleaved
2. **Reshaping:** Uses `reshape(1,2).t()` to convert (136,1) → (68,2)
3. **Rigid points:** 24 specific indices for stable alignment
4. **Scale computation:** RMS-based, computed AFTER mean normalization
5. **Rotation:** Uses Kabsch algorithm with SVD
6. **NO transpose:** C++ does NOT transpose the scale-rotation matrix
7. **Translation transform:** Pose translation IS transformed through scale-rotation
8. **Centering:** Simple offset of `out_width/2` and `out_height/2`
9. **No -0.5 offset:** C++ does NOT use sub-pixel centering offsets
10. **No empirical shifts:** C++ does NOT add correction offsets like +2 or -2

## Critical Difference to Note

The C++ code:
- **DOES** compute rotation using Kabsch
- **DOES NOT** transpose the result
- **DOES** transform pose translation through scale-rotation matrix
- **DOES NOT** add empirical shift corrections

This means the rotation IS being computed and IS expected to vary per frame based on the actual landmark positions. The question is: why does C++ produce upright faces despite computing rotation?
