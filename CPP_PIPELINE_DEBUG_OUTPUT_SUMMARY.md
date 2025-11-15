# C++ OpenFace Pipeline Debug Output Summary

## Status: ✅ SUFFICIENT FOR VALIDATION

The C++ OpenFace binary provides comprehensive output through **CSV files** and **binary files** that contain all pipeline stage results needed for validation against PyFaceAU.

## Output Files Generated

### 1. **CSV File** (`patient1_frame1.csv`)
**Contains:** Complete pipeline results per frame

**Columns include:**
- `frame`, `face_id`, `timestamp` - Frame metadata
- `confidence`, `success` - Detection confidence and success flag
- `gaze_0_x`, `gaze_0_y`, `gaze_0_z` - Left eye gaze vector
- `gaze_1_x`, `gaze_1_y`, `gaze_1_z` - Right eye gaze vector
- `gaze_angle_x`, `gaze_angle_y` - Gaze angles
- `eye_lmk_x_0` through `eye_lmk_x_55` - Eye landmarks (56 points, 2D x-coords)
- `eye_lmk_y_0` through `eye_lmk_y_55` - Eye landmarks (56 points, 2D y-coords)
- `eye_lmk_X_0` through `eye_lmk_X_55` - Eye landmarks (56 points, 3D X-coords)
- `eye_lmk_Y_0` through `eye_lmk_Y_55` - Eye landmarks (56 points, 3D Y-coords)
- `eye_lmk_Z_0` through `eye_lmk_Z_55` - Eye landmarks (56 points, 3D Z-coords)
- **`pose_Tx`, `pose_Ty`, `pose_Tz`** - Translation (comparable to PyFaceAU tx, ty)
- **`pose_Rx`, `pose_Ry`, `pose_Rz`** - Rotation (comparable to PyFaceAU rx, ry, rz)
- **`x_0` through `x_67`** - 68-point landmarks (2D x-coords)
- **`y_0` through `y_67`** - 68-point landmarks (2D y-coords)
- **`X_0` through `X_67`** - 68-point landmarks (3D X-coords)
- **`Y_0` through `Y_67`** - 68-point landmarks (3D Y-coords)
- **`Z_0` through `Z_67`** - 68-point landmarks (3D Z-coords)
- **`p_scale`** - Face scale (comparable to PyFaceAU scale)
- **`p_rx`, `p_ry`, `p_rz`** - Pose rotation (alternative representation)
- **`p_tx`, `p_ty`** - Pose translation (alternative representation)
- **`p_0` through `p_33`** - Shape parameters (PDM coefficients)
- **`AU01_r` through `AU45_r`** - AU regression values
- **`AU01_c` through `AU45_c`** - AU classification values (presence/absence)

### 2. **HOG File** (`patient1_frame1.hog`)
**Contains:** Binary HOG features extracted from aligned face
**Size:** ~17KB per frame
**Comparable to:** PyFaceAU HOG extraction step

### 3. **Aligned Face Directory** (`patient1_frame1_aligned/`)
**Contains:** Aligned face images
**Comparable to:** PyFaceAU face alignment step

### 4. **Details File** (`patient1_frame1_of_details.txt`)
**Contains:** Processing configuration metadata
```
Input:/Users/johnwilsoniv/Documents/SplitFace Open3/calibration_frames/patient1_frame1.jpg
Camera parameters:1421.88,1421.88,540,960
Gaze: 1
AUs: 1
Landmarks 2D: 1
Landmarks 3D: 1
Pose: 1
Shape parameters: 1
```

### 5. **MTCNN 5-Point Landmarks** (`/tmp/mtcnn_debug.csv`)
**Contains:** Frame-by-frame MTCNN output with 5-point landmarks from ONet
**Format:** CSV with columns:
- `frame` - Frame number
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` - Bounding box from MTCNN
- `confidence` - Face detection confidence
- `onet_size` - Size of ONet output tensor
- `lm1_x`, `lm1_y` through `lm5_x`, `lm5_y` - 5-point landmarks from MTCNN ONet

**Location in code:** `FaceDetectorMTCNN.cpp:1394-1511`
**Comparable to:** PyMTCNN ONet 5-point landmark output

### 6. **Initial 68-Point Landmarks** (stdout)
**Output format:**
```
DEBUG_INIT_PARAMS: scale=X wx=X wy=X wz=X tx=X ty=X
DEBUG_INIT_LANDMARKS: x0,y0,x1,y1,...,x67,y67
```

**Contains:** Initial landmarks calculated from bbox BEFORE CLNF refinement
- `DEBUG_INIT_PARAMS` - Initial PDM parameters (scale, rotation, translation) calculated from face bbox
- `DEBUG_INIT_LANDMARKS` - Initial 68-point landmarks generated from PDM using bbox-derived parameters

**Location in code:** `LandmarkDetectorFunc.cpp:344-355`
**Comparable to:** PyFaceAU initial landmarks before CLNF refinement

### 7. **MTCNN Stage Debug Files** (already documented in CPP_MTCNN_DEBUG_OUTPUT_SUMMARY.md)
```
/tmp/cpp_pnet_all_boxes.txt
/tmp/cpp_rnet_scores.txt
/tmp/cpp_rnet_output.txt
/tmp/cpp_before_onet.txt
/tmp/cpp_onet_input.bin
```

## Comparison with PyFaceAU Debug Mode

| Component | PyFaceAU Debug Output | C++ OpenFace Output | Validation Strategy |
|-----------|----------------------|---------------------|---------------------|
| **Face Detection (MTCNN)** | Box count, bbox coords, timing | `/tmp/cpp_*.txt` files + stdout | ✅ Compare box counts and coords |
| **MTCNN 5-Point Landmarks** | 5-point landmarks from ONet | `/tmp/mtcnn_debug.csv`: `lm1_x`-`lm5_y` | ✅ Compare 5-point landmark positions |
| **Initial 68-Point Landmarks** | Initial landmarks from bbox before CLNF | stdout: `DEBUG_INIT_LANDMARKS` | ✅ Compare initial landmark positions |
| **Initial PDM Parameters** | Initial scale, rotation, translation | stdout: `DEBUG_INIT_PARAMS` | ✅ Compare initial pose parameters |
| **Final Landmark Detection** | 68-point landmarks after CLNF, timing | CSV: `x_0`-`x_67`, `y_0`-`y_67` | ✅ Compare final landmark positions (euclidean distance) |
| **Pose Estimation** | scale, rx, ry, rz, tx, ty, timing | CSV: `p_scale`, `pose_Rx`, `pose_Ry`, `pose_Rz`, `pose_Tx`, `pose_Ty` | ✅ Compare pose parameters |
| **Face Alignment** | Aligned face shape, timing | `_aligned/` directory (images) | ✅ Visual comparison (optional) |
| **HOG Extraction** | HOG features shape, timing | `.hog` binary file | ✅ Compare HOG feature vectors (if needed) |
| **Geometric Features** | Geom features shape, timing | CSV: Shape params `p_0`-`p_33` | ✅ Compare PDM coefficients |
| **Running Median** | Median shape, update status, timing | Not explicitly output | ⚠️ Cannot validate directly |
| **AU Prediction** | AU count, timing | CSV: `AU01_r`-`AU45_r` | ✅ Compare AU regression values (correlation) |

## Key Differences

### ✅ Advantages of C++ Output
1. **Complete CSV format** - Easy to parse and analyze
2. **All stages in one file** - Convenient for validation
3. **Binary HOG files** - Exact feature comparison possible
4. **Aligned face images** - Visual verification available

### ⚠️ Limitations
1. **No per-component timing** - C++ doesn't output individual stage timing
2. **No running median tracking** - Internal state not exposed
3. **No explicit CLNF refinement flag** - Can't tell if CLNF was used (but results are comparable)

## Validation Approach

### For comprehensive Python vs C++ validation:

1. **MTCNN Stage Comparison:**
   - Parse `/tmp/cpp_pnet_all_boxes.txt`, `/tmp/cpp_rnet_output.txt`, `/tmp/mtcnn_debug.csv`
   - Compare with PyMTCNN debug output
   - Metrics: Box count, IoU, coordinate differences
   - **Visualizations:** Overlay bboxes with 5-point MTCNN landmarks (C++ green, Python blue)

2. **Initial Landmark Accuracy (Pre-CLNF):**
   - Parse stdout `DEBUG_INIT_LANDMARKS` for initial 68-point landmarks from bbox
   - Compare with PyFaceAU initial landmarks before CLNF refinement
   - Metrics: Mean euclidean distance, max error
   - **Visualizations:** Overlay initial 68-point landmarks (C++ green, Python blue)

3. **Final Landmark Accuracy (Post-CLNF):**
   - Parse CSV columns `x_0`-`x_67`, `y_0`-`y_67`
   - Compare with PyFaceAU final landmark output after CLNF
   - Metrics: Mean euclidean distance, max error
   - **Visualizations:** Overlay final 68-point landmarks (C++ green, Python blue)

4. **Pose Estimation:**
   - Parse CSV: `p_scale`, `pose_Rx`, `pose_Ry`, `pose_Rz`, `pose_Tx`, `pose_Ty`
   - Compare with PyFaceAU pose output
   - Metrics: Scale difference, rotation angle difference, translation difference

5. **AU Correlation:**
   - Parse CSV: `AU01_r` through `AU45_r`
   - Compare with PyFaceAU AU output
   - Metrics: Pearson correlation, RMSE

## Conclusion

✅ **The C++ OpenFace binary provides SUFFICIENT output for comprehensive validation:**
- CSV file contains all stage results (landmarks, pose, AUs)
- MTCNN debug files provide detection stage details
- HOG and aligned face files available for deep-dive analysis

⚠️ **The C++ output does NOT provide:**
- Per-component timing (not needed for accuracy validation)
- Running median tracking state (internal implementation detail)

**This is sufficient to validate >95% accuracy target across all comparable stages.**
