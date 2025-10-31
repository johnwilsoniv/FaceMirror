# Face Alignment Problem: Python vs C++ Rotation Discrepancy

## Problem Statement

We have implemented OpenFace 2.2 face alignment in Python, attempting to exactly replicate the C++ implementation. Despite verifying every algorithmic step matches, Python produces faces with variable rotation (-8.79° to +2.17°) while C++ produces consistently upright faces (~0°).

**Critical Issue:** The rotation variation in Python is expression-sensitive. When eyes close (frame 617), rotation jumps by 6.4° relative to eyes-open frames.

## Visual Evidence

### C++ Output (Reference)
Three frames showing expression invariance:
- Frame 493: Eyes open, perfectly upright
- Frame 617: Eyes closed, perfectly upright (identical orientation to 493)
- Frame 863: Eyes open, perfectly upright (identical orientation to 493)

**Observation:** All three C++ faces have identical orientation despite expression change.

### Python Output (Our Implementation)
Same three frames with Python alignment:
- Frame 493: -4.27° rotation (slight CCW tilt)
- Frame 617: +2.17° rotation (slight CW tilt) - **6.44° difference from frame 493!**
- Frame 863: -2.89° rotation (slight CCW tilt)

**Problem:** Python rotation varies with expression, particularly when eyes close.

### Additional Python Frame
- Frame 1: -8.79° rotation (more pronounced CCW tilt)

**Range:** Python rotation varies across an 11° range (-8.79° to +2.17°)

## Input Data Confidence

### Inputs to Both C++ and Python (Confidence: 100% - Verified Identical)

Both implementations receive identical inputs from the same OpenFace 2.2 CSV file:

1. **68 Facial Landmarks** (x_0 to x_67, y_0 to y_67)
   - Format: Pixel coordinates in image space
   - Source: CLNF landmark detector output
   - Example Frame 1: x_0=316.8, y_0=720.4, ..., x_67=654.4, y_67=837.4
   - **Verification:** We read the same CSV file for both comparisons
   - **Confidence:** 100% identical

2. **Pose Translation** (p_tx, p_ty)
   - Format: Float values representing face center
   - Source: PDM CalcParams output
   - Example Frame 1: p_tx=523.1, p_ty=916.1
   - **Verification:** Same CSV row used for both
   - **Confidence:** 100% identical

3. **Input Image Frame**
   - Format: BGR image, 1080×1920 pixels
   - Source: Same MP4 video file
   - **Verification:** We read from the same video file at the same frame numbers
   - **Confidence:** 100% identical

4. **PDM Reference File** (In-the-wild_aligned_PDM_68.txt)
   - Content: Mean shape coordinates + principal components
   - Source: OpenFace model directory
   - **Verification:** Both load the same file from disk
   - **Confidence:** 100% identical - we verified the file path and contents

5. **Configuration Parameters**
   - sim_scale: 0.7 (scaling factor for reference shape)
   - output_size: 112×112 pixels
   - rigid alignment: true (use 24 rigid points)
   - **Verification:** Hardcoded to match C++ defaults
   - **Confidence:** 100% identical

### Processing Parameters (Confidence: 99% - Algorithmically Identical)

Both implementations use:

1. **Rigid Point Indices** (24 landmarks)
   ```
   [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35,
    36, 39, 40, 41, 42, 45, 46, 47]
   ```
   - Includes 8 eye landmarks: 36, 39, 40, 41, 42, 45, 46, 47
   - **Source:** OpenFace C++ source code (Face_utils.cpp line 45-107)
   - **Verification:** Extracted directly from C++ code
   - **Confidence:** 100% these are the indices C++ source code uses

2. **Kabsch Algorithm** (for computing rotation)
   - Mean normalization: Subtract centroid
   - RMS scaling: Normalize by sqrt(sum(points²)/n)
   - SVD computation: `U, S, Vt = SVD(src.T @ dst)`
   - Reflection check: determinant of `Vt.T @ U.T`
   - Rotation matrix: `R = Vt.T @ correction @ U.T`
   - Final transform: `scale * R`
   - **Verification:** Implemented step-by-step from C++ (RotationHelpers.h)
   - **Confidence:** 100% algorithm matches

3. **Warp Matrix Construction**
   ```python
   warp_matrix[0:2, 0:2] = scale_rotation_matrix  # NO transpose
   T = scale_rotation_matrix @ [pose_tx, pose_ty]  # Transform pose
   warp_matrix[0, 2] = -T[0] + output_width/2
   warp_matrix[1, 2] = -T[1] + output_height/2
   ```
   - **Source:** Face_utils.cpp lines 127-143
   - **Verification:** Matches C++ exactly
   - **Confidence:** 100% implementation matches

4. **Interpolation Method**
   - Both use: `cv2.INTER_LINEAR` / `cv::INTER_LINEAR`
   - **Confidence:** 100% identical

## Algorithm Verification

### What We've Verified Matches C++

1. ✓ **PDM Loading**
   - Verified mean_shape values match
   - Verified reshape to (68, 2) format
   - Verified sim_scale multiplication

2. ✓ **Rigid Point Extraction**
   - Verified indices match C++ source code
   - Verified array indexing produces correct subset

3. ✓ **Kabsch Algorithm**
   - Verified mean normalization
   - Verified RMS scaling computation
   - Verified SVD input matrix (src.T @ dst)
   - Verified reflection detection
   - Verified rotation matrix construction
   - **Tested:** numpy.linalg.svd vs cv2.SVDecomp produce identical rotation angles

4. ✓ **Warp Matrix Construction**
   - Verified NO transpose of rotation matrix
   - Verified pose translation transformation
   - Verified centering computation

5. ✓ **No Hidden Preprocessing**
   - Checked: No coordinate transforms before AlignFace
   - Checked: No landmark rotation/normalization
   - Checked: No visibility filtering

6. ✓ **No Hidden Post-Processing**
   - Checked: No rotation correction after alignment
   - Checked: No additional warping steps

### What We've Ruled Out

1. ✗ **SVD Implementation Differences**
   - Tested numpy.linalg.svd vs cv2.SVDecomp
   - Result: Identical rotation angles despite U/V sign differences
   - Conclusion: Not the root cause

2. ✗ **Matrix Operation Differences**
   - Tested src.T @ dst vs alternatives
   - Result: Same values
   - Conclusion: Not the root cause

3. ✗ **PDM De-rotation Theory**
   - Tested rotating reference shape to be upright
   - Result: Made alignment worse (-128° rotation)
   - Conclusion: Not the solution

4. ✗ **Temporal Smoothing**
   - Searched C++ code for median filters, smoothing
   - Found: Only used for HOG features, not alignment
   - Conclusion: Not the root cause

5. ✗ **Different Rigid Points at Runtime**
   - Verified C++ source shows 24 indices
   - No configuration flags found
   - Conclusion: Likely uses same 24 points

## Key Observations

### Observation 1: Expression Sensitivity in Python

**Frame Comparison:**
- Frame 493 (eyes open): -4.27° rotation
- Frame 617 (eyes closed): +2.17° rotation
- **Difference:** 6.44° jump when eyes close!

**In C++:**
- Frame 493 (eyes open): ~0° rotation
- Frame 617 (eyes closed): ~0° rotation
- **Difference:** No change (expression invariant)

**Significance:** The 8 eye landmarks in the rigid point set affect Python but NOT C++, despite both using the same 24 rigid indices.

### Observation 2: Removing Eye Landmarks

When we exclude the 8 eye landmarks (use only 16 rigid points):

**Python Results:**
- Frame 493: +30.98° rotation
- Frame 617: +32.12° rotation
- Frame 863: +32.74° rotation
- **Stability:** std = 1.47° (excellent!)
- **Expression sensitivity:** None (only 1.14° variation)

**Interpretation:**
- Without eyes: Python is very stable (like C++)
- But rotation is wrong (~+30° vs 0°)
- This suggests eye landmarks are needed for correct magnitude
- But somehow cause expression sensitivity in Python, not C++

### Observation 3: PDM Mean Shape is Rotated

The PDM reference shape is rotated ~120° CCW:
- Nose-to-chin vector: 119.52° from vertical
- Eye axis: -130.18° from horizontal

**Question:** If both Python and C++ use this rotated reference, why do they produce different output rotations?

### Observation 4: C++ Output Characteristics

Measured from C++ aligned faces:
- All faces have eyes horizontal (gradient ~0°)
- Facial feature lines are vertical (angle ~0°)
- Minimal pixel differences between expression changes
- Perfect orientation consistency across all frames

**Conclusion:** C++ definitively produces upright (0°) faces, not aligned to the rotated PDM.

## Code Comparison

### C++ Implementation (Face_utils.cpp lines 109-146)

```cpp
void AlignFace(cv::Mat& aligned_face, const cv::Mat& frame,
               const cv::Mat_<float>& detected_landmarks,
               cv::Vec6f params_global, const LandmarkDetector::PDM& pdm,
               bool rigid, double sim_scale, int out_width, int out_height)
{
    // Scale mean shape
    cv::Mat_<float> similarity_normalised_shape = pdm.mean_shape * sim_scale;

    // Discard Z (keep first 136 values = X,Y coords)
    similarity_normalised_shape = similarity_normalised_shape(
        cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)
    ).clone();

    // Reshape to (68, 2)
    cv::Mat_<float> source_landmarks = detected_landmarks.reshape(1, 2).t();
    cv::Mat_<float> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();

    // Extract rigid points if requested
    if(rigid)
    {
        extract_rigid_points(source_landmarks, destination_landmarks);
    }

    // Compute similarity transform (scale + rotation)
    cv::Matx22f scale_rot_matrix = Utilities::AlignShapesWithScale(
        source_landmarks, destination_landmarks
    );

    // Build 2×3 warp matrix
    cv::Matx23f warp_matrix;
    warp_matrix(0,0) = scale_rot_matrix(0,0);
    warp_matrix(0,1) = scale_rot_matrix(0,1);
    warp_matrix(1,0) = scale_rot_matrix(1,0);
    warp_matrix(1,1) = scale_rot_matrix(1,1);

    // Transform pose translation
    float tx = params_global[4];
    float ty = params_global[5];
    cv::Vec2f T(tx, ty);
    T = scale_rot_matrix * T;

    // Center in output image
    warp_matrix(0,2) = -T(0) + out_width/2;
    warp_matrix(1,2) = -T(1) + out_height/2;

    // Apply warp
    cv::warpAffine(frame, aligned_face, warp_matrix,
                   cv::Size(out_width, out_height), cv::INTER_LINEAR);
}
```

### Python Implementation (openface22_face_aligner.py)

```python
def align_face(self, image: np.ndarray, landmarks_68: np.ndarray,
               pose_tx: float, pose_ty: float) -> np.ndarray:
    """Align face to canonical 112×112 reference frame"""

    # Ensure landmarks are (68, 2) shape
    if landmarks_68.shape == (136,):
        landmarks_68 = landmarks_68.reshape(68, 2)

    # Extract rigid points from both source and destination
    source_rigid = self._extract_rigid_points(landmarks_68)
    dest_rigid = self._extract_rigid_points(self.reference_shape)

    # Compute similarity transform (scale + rotation)
    scale_rot_matrix = self._align_shapes_with_scale(source_rigid, dest_rigid)

    # Build 2×3 affine warp matrix
    warp_matrix = self._build_warp_matrix(scale_rot_matrix, pose_tx, pose_ty)

    # Apply affine transformation
    aligned_face = cv2.warpAffine(
        image,
        warp_matrix,
        (self.output_width, self.output_height),
        flags=cv2.INTER_LINEAR
    )

    return aligned_face

def _align_shapes_with_scale(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute similarity transform (scale + rotation) using Kabsch"""
    n = src.shape[0]

    # Mean normalize
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)

    # RMS scale
    s_src = np.sqrt(np.sum(src_centered ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)

    # Normalize by scale
    src_norm = src_centered / s_src
    dst_norm = dst_centered / s_dst

    # Kabsch SVD
    U, S, Vt = np.linalg.svd(src_norm.T @ dst_norm)

    # Check for reflection
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    if d > 0:
        corr[1, 1] = 1
    else:
        corr[1, 1] = -1

    # Rotation matrix
    R = Vt.T @ corr @ U.T

    # Return scaled rotation
    scale = s_dst / s_src
    return scale * R

def _build_warp_matrix(self, scale_rot: np.ndarray,
                       pose_tx: float, pose_ty: float) -> np.ndarray:
    """Build 2×3 affine warp matrix"""
    warp_matrix = np.zeros((2, 3), dtype=np.float32)

    # Copy scale-rotation to first 2×2 block (NO transpose)
    warp_matrix[:2, :2] = scale_rot

    # Transform pose translation through scale-rotation
    T = np.array([pose_tx, pose_ty], dtype=np.float32)
    T_transformed = scale_rot @ T

    # Translation for centering
    warp_matrix[0, 2] = -T_transformed[0] + self.output_width / 2
    warp_matrix[1, 2] = -T_transformed[1] + self.output_height / 2

    return warp_matrix
```

**Assessment:** Python implementation appears to exactly match C++ logic.

## Measurements

### Rotation Angles by Frame (Python)

| Frame | Eyes State | Rotation | Notes |
|-------|-----------|----------|-------|
| 1 | Open | -8.79° | Largest CCW tilt |
| 493 | Open | -4.27° | Moderate CCW tilt |
| 617 | **Closed** | **+2.17°** | **Flips to CW tilt!** |
| 863 | Open | -2.89° | Moderate CCW tilt |

**Range:** 11° total variation
**Expression sensitivity:** 6.44° jump when eyes close
**Stability (std):** 4.51° (poor)

### Rotation Angles by Frame (C++)

| Frame | Eyes State | Rotation | Notes |
|-------|-----------|----------|-------|
| 493 | Open | ~0° | Upright |
| 617 | **Closed** | **~0°** | **Still upright** |
| 863 | Open | ~0° | Upright |

**Range:** ~0° (minimal variation)
**Expression sensitivity:** None
**Stability (std):** ~0° (excellent)

### Without Eye Landmarks (Python - 16 rigid points)

| Frame | Eyes State | Rotation | Notes |
|-------|-----------|----------|-------|
| 1 | Open | 27.55° | Consistent tilt |
| 493 | Open | 30.98° | Consistent tilt |
| 617 | Closed | 32.12° | Barely changes! |
| 863 | Open | 32.74° | Consistent tilt |

**Range:** 5.19° variation
**Expression sensitivity:** 1.14° (minimal!)
**Stability (std):** 1.47° (excellent!)
**Problem:** Wrong absolute rotation (~+30° vs 0°)

## Hypotheses Under Consideration

### Hypothesis 1: C++ Uses Different Rigid Points at Runtime

**Reasoning:**
- Source code shows 24 rigid indices including 8 eye landmarks
- But C++ behavior suggests eye landmarks don't affect rotation
- Perhaps runtime uses different points than source code indicates

**Evidence For:**
- C++ is expression-invariant despite eye landmarks in rigid set
- Python WITH eyes is expression-sensitive
- Python WITHOUT eyes is expression-invariant (like C++)

**Evidence Against:**
- We found no configuration or conditional logic in C++
- Source code clearly shows all 24 indices being used

**Confidence:** 30% - Possible but seems unlikely

### Hypothesis 2: C++ Applies Hidden Reference Frame Transform

**Reasoning:**
- PDM mean shape is rotated ~120° CCW
- C++ outputs upright faces (~0°)
- Perhaps there's a coordinate transform we're missing

**Evidence For:**
- There must be SOMETHING that makes C++ output upright
- PDM rotation doesn't match output rotation

**Evidence Against:**
- We checked entire AlignFace function - no transforms found
- No rotations applied to mean_shape after loading

**Confidence:** 40% - Most likely remaining possibility

### Hypothesis 3: Landmark Preprocessing Difference

**Reasoning:**
- Perhaps CLNF detector outputs landmarks differently
- Or CSV values don't match runtime values
- Or there's filtering we haven't found

**Evidence For:**
- Would explain why identical algorithm gives different results
- Could explain expression sensitivity

**Evidence Against:**
- We read the same CSV C++ generates
- No preprocessing found before AlignFace call
- Landmark values are in expected ranges

**Confidence:** 20% - Unlikely but not impossible

### Hypothesis 4: Float Precision Accumulation

**Reasoning:**
- C++ uses float32, Python uses float32 (but internally float64)
- Precision differences could accumulate

**Evidence For:**
- Known issue with numerical algorithms
- Could cause systematic bias

**Evidence Against:**
- Would expect small differences (~0.1°), not 11° range
- Would expect consistent offset, not variable rotation

**Confidence:** 10% - Very unlikely to cause this magnitude

## Questions for External Review

1. **Is there any mechanism in Kabsch algorithm where same inputs but different internal precision could produce 5-10° rotation differences?**

2. **Could the PDM coordinate system have an intended "canonical orientation" that requires transformation we're missing?**

3. **How could C++ use the same 24 rigid indices (including 8 eye landmarks) yet be immune to eye closure, while Python is affected?**

4. **Is there any OpenCV vs numpy difference in matrix operations (transpose, multiplication, etc.) that could cause rotation differences?**

5. **Could there be a reference frame or coordinate system convention we're violating?**

6. **Is there anything in the OpenFace 2.2 C++ codebase outside of AlignFace() that could affect face rotation?**

7. **Could the rigid point extraction in C++ have hidden logic not visible in extract_rigid_points()?**

## Files Available for Review

### Code Files
- `openface22_face_aligner.py` - Python implementation
- `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp` - C++ implementation
- `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/Utilities/include/RotationHelpers.h` - C++ Kabsch implementation

### Data Files
- `In-the-wild_aligned_PDM_68.txt` - PDM reference model (identical for both)
- `of22_validation/IMG_0942_left_mirrored.csv` - Landmark and pose data (identical input)
- Video: `/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4`

### Visual Comparisons
- `PYTHON_VS_CPP_GRID.png` - Side-by-side comparison (4 frames)
- `cpp_expression_comparison.png` - C++ expression invariance proof (3 frames)

### Analysis Documents
- `SYSTEMATIC_ROOT_CAUSE_ANALYSIS.md` - Complete investigation log
- `SVD_RESEARCH_FINDINGS.md` - Research on SVD differences
- `CPP_ALIGNMENT_ALGORITHM_ANALYSIS.md` - C++ code breakdown
- `CPP_VS_PYTHON_COMPARISON.md` - Algorithm comparison

## What We Need

**Primary Goal:** Understand why C++ produces upright (0°) faces while Python produces variable rotation (-8.79° to +2.17°) despite identical inputs and algorithm.

**Success Criteria:**
- Identify the root cause of the rotation discrepancy
- Explain why C++ is expression-invariant while Python is expression-sensitive
- Provide a solution that makes Python match C++ behavior

## Current Status

- ✓ Algorithm verified to match C++ step-by-step
- ✓ Inputs verified identical
- ✓ SVD implementation tested (not the cause)
- ✓ Expression sensitivity identified as key difference
- ✗ Root cause still unknown
- ✗ Solution not yet found

**We are stuck and need fresh perspective to identify what we're missing.**
