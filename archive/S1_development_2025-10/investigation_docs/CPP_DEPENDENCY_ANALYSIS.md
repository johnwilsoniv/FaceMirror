# C++ Dependency Analysis - Current Status

**Date:** 2025-10-29

## If Current Approach Works (Inverse CSV p_rz)

### What We've Implemented in Pure Python

✅ **Face Detection (RetinaFace ONNX)**
- No C++ dependency
- Using `onnx_retinaface_detector.py`
- ONNX Runtime backend

✅ **Landmark Detection (STAR ONNX)**
- No C++ dependency
- Using `onnx_star_detector.py`
- ONNX Runtime backend

✅ **Face Alignment**
- Pure Python implementation
- `openface22_face_aligner.py`
- Uses CSV params (tx, ty, rz) from OpenFace
- Kabsch algorithm for scale computation
- Inverse p_rz for head pose correction

✅ **Triangulation Masking**
- Pure Python implementation
- `triangulation_parser.py`
- Masks out neck/ears/background

❌ **HOG Feature Extraction**
- Currently using C++ FeatureExtraction binary
- Location: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction`
- **This is our main C++ dependency**

✅ **AU Model Loading**
- Pure Python implementation
- `openface22_model_parser.py`
- Parses binary SVR models

✅ **AU Prediction (SVR)**
- Pure Python implementation
- `openface22_au_predictor.py`
- Uses scikit-learn SVR

### Remaining C++ Dependencies

**1. Initial CSV Generation**
- Need OpenFace FeatureExtraction once to create CSV with:
  - 68 landmarks (x_0...x_67, y_0...y_67)
  - Pose parameters (p_tx, p_ty, p_rz)
  - Confidence scores
- **Status:** One-time preprocessing, acceptable

**2. HOG Feature Extraction** ⚠️
- Current AU prediction pipeline uses C++ HOG
- C++ extracts 5596-dimensional HOG features
- **Options:**
  a. Implement HOG in Python (skimage.feature.hog)
  b. Continue using C++ binary (current approach)
  c. Use different features (requires model retraining)

**3. CalcParams for p_rz** (if needed)
- CSV already contains p_rz from first PDM fitting
- Using inverse of CSV p_rz works reasonably well
- **Status:** May not need CalcParams if AU predictions are good enough

### Pure Python HOG Implementation Status

**From agent search findings:**
- No OpenFace-compatible Python HOG implementation exists in codebase
- scikit-image HOG has different parameters
- Would need to match C++ exactly:
  - Cell size: 8×8 pixels
  - Block size: 2×2 cells
  - 9 orientation bins
  - L2-Hys normalization
  - Specific scanning window approach

**Estimated effort:** 1-2 days to implement and validate

### Decision Tree

```
Current approach AU predictions match C++?
├─ YES → Pure Python except CSV preprocessing
│         Dependencies: OpenFace binary (one-time CSV generation)
│         Production: Can distribute Python-only pipeline
│
├─ NO, but close (r > 0.9) → Try Python HOG implementation
│         Effort: 1-2 days
│         Risk: Medium (HOG is well-documented)
│
└─ NO, poor results (r < 0.9) → Need full C++ integration
          Options:
          1. Fix OpenFace build (CalcParams binary)
          2. Use C++ FeatureExtraction throughout
          3. Retrain AU models on Python features
```

### Minimal C++ Dependency Scenario

**Best case (if current approach works):**

1. **Preprocessing (one-time):**
   ```bash
   # Use C++ OpenFace to generate CSV with landmarks + pose
   ./FeatureExtraction -f video.mp4 -out_dir output/
   ```

2. **Runtime (pure Python):**
   ```python
   # Read CSV
   df = pd.read_csv('output/video.csv')

   # For each frame:
   #   - Load frame from video
   #   - Get landmarks + pose from CSV
   #   - Align face (Python)
   #   - Extract HOG (Python - to be implemented)
   #   - Predict AUs (Python)
   ```

**Production deployment:**
- Ship: Python code + CSV files + ONNX models + SVR models
- No C++ runtime dependency
- All processing in Python

**Size estimate:**
- Python packages: ~200 MB (NumPy, OpenCV, scikit-learn, ONNX Runtime)
- Models: ~50 MB (ONNX + SVR)
- CSV files: ~1-5 MB per video
- **Total: ~250-300 MB**

### Current Bottleneck: HOG

**The HOG dependency is significant because:**

1. **Feature dimensionality:** 5596 features per frame
2. **Critical for AU prediction:** SVR models trained on HOG
3. **Must match C++ exactly:** Different HOG = different predictions

**Python HOG Implementation Path:**

```python
# Pseudo-code for OpenFace-compatible HOG
def extract_hog_openface(aligned_face_112x112):
    # Convert to grayscale
    gray = cv2.cvtColor(aligned_face_112x112, cv2.COLOR_BGR2GRAY)

    # HOG parameters matching OpenFace
    cell_size = (8, 8)
    block_size = (2, 2)  # In cells
    block_stride = (1, 1)  # In cells
    nbins = 9

    # Use skimage or custom implementation
    features = hog(
        gray,
        orientations=nbins,
        pixels_per_cell=cell_size,
        cells_per_block=block_size,
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )

    return features
```

**Validation approach:**
1. Extract HOG from same aligned face in C++ and Python
2. Compare feature vectors element-wise
3. Ensure correlation r > 0.999
4. Test AU predictions match

### Summary

**If current alignment works:**
- **CSV generation:** One-time C++ dependency (acceptable)
- **HOG extraction:** Current blocker (1-2 days to implement in Python)
- **Everything else:** Pure Python ✅

**If current alignment doesn't work:**
- Need CalcParams → Either fix C++ build or implement in Python (3-4 weeks)
- Or accept systematic error and tune empirically

**Recommendation:** Test AU predictions first with current alignment + C++ HOG. If results are good (r > 0.9), invest in Python HOG. If results are poor, revisit CalcParams approach.
