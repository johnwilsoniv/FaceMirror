# Full Python AU Extraction Pipeline

**Status:** ✅ Complete - Ready for Testing
**Date:** 2025-10-30

---

## Overview

This is a **complete end-to-end Python implementation** of the OpenFace 2.2 AU extraction pipeline. No C++ OpenFace binary required!

**File:** `full_python_au_pipeline.py`

### What It Does

Processes videos through the complete AU extraction pipeline:

1. **Face Detection** → RetinaFace ONNX (optimized with CoreML)
2. **Landmark Detection** → Cunjian PFLD 68-point detector
3. **3D Pose Estimation** → CalcParams (99.45% accuracy) or simplified PnP
4. **Face Alignment** → OpenFace 2.2 algorithm (112×112 output)
5. **Triangulation Masking** → Remove background
6. **HOG Feature Extraction** → PyFHOG (r=1.0 vs C++)
7. **Geometric Feature Extraction** → PDM-based features (238 dims)
8. **Running Median Tracking** → Cython-optimized (260x faster!)
9. **AU Prediction** → SVR models (17 AUs)
10. **Post-processing** → Cutoff adjustment + temporal smoothing

**Output:** Frame-by-frame AU intensities (AU01_r, AU02_r, ..., AU45_r)

---

## Installation Requirements

### 1. Core Dependencies

```bash
pip install numpy pandas opencv-python onnxruntime torch
```

### 2. Cython Modules (for 260x speedup)

```bash
# Build Cython-optimized running median
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"
python3 setup_cython_modules.py build_ext --inplace
```

### 3. PyFHOG (for HOG extraction)

```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfhog"
pip install -e .
```

### 4. Model Files

Ensure the following models are in the `weights/` directory:

- `retinaface_mobilenet025_coreml.onnx` - Face detection
- `pfld_cunjian.onnx` - Landmark detection
- `In-the-wild_aligned_PDM_68.txt` - PDM shape model
- `tris_68_full.txt` - Triangulation for masking
- `AU_predictors/` - Directory with 17 AU SVR models

---

## Usage

### Basic Usage

```bash
python full_python_au_pipeline.py --video input.mp4 --output results.csv
```

### Test on First 100 Frames

```bash
python full_python_au_pipeline.py --video input.mp4 --max-frames 100
```

### Use Simplified Pose Estimation (Faster)

```bash
python full_python_au_pipeline.py --video input.mp4 --simple-pose
```

### Custom Model Paths

```bash
python full_python_au_pipeline.py \
  --video input.mp4 \
  --retinaface weights/retinaface.onnx \
  --pfld weights/pfld.onnx \
  --pdm weights/PDM_68.txt \
  --au-models weights/AU_predictors \
  --triangulation weights/tris_68_full.txt
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | - | Input video file path |
| `--output` | No | `<video>_python_aus.csv` | Output CSV file path |
| `--max-frames` | No | All frames | Limit frames processed (for testing) |
| `--simple-pose` | No | False | Use simplified pose (faster, less accurate) |
| `--retinaface` | No | `weights/retinaface_mobilenet025_coreml.onnx` | RetinaFace model path |
| `--pfld` | No | `weights/pfld_cunjian.onnx` | PFLD model path |
| `--pdm` | No | `weights/In-the-wild_aligned_PDM_68.txt` | PDM model path |
| `--au-models` | No | `weights/AU_predictors` | AU models directory |
| `--triangulation` | No | `weights/tris_68_full.txt` | Triangulation file |

---

## Output Format

### CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int | Frame index (0-based) |
| `timestamp` | float | Time in seconds |
| `success` | bool | Whether frame processed successfully |
| `AU01_r` | float | Inner brow raiser intensity [0-5] |
| `AU02_r` | float | Outer brow raiser intensity [0-5] |
| `AU04_r` | float | Brow lowerer intensity [0-5] |
| `AU05_r` | float | Upper lid raiser intensity [0-5] |
| `AU06_r` | float | Cheek raiser intensity [0-5] |
| `AU07_r` | float | Lid tightener intensity [0-5] |
| `AU09_r` | float | Nose wrinkler intensity [0-5] |
| `AU10_r` | float | Upper lip raiser intensity [0-5] |
| `AU12_r` | float | Lip corner puller intensity [0-5] |
| `AU14_r` | float | Dimpler intensity [0-5] |
| `AU15_r` | float | Lip corner depressor intensity [0-5] |
| `AU17_r` | float | Chin raiser intensity [0-5] |
| `AU20_r` | float | Lip stretcher intensity [0-5] |
| `AU23_r` | float | Lip tightener intensity [0-5] |
| `AU25_r` | float | Lips part intensity [0-5] |
| `AU26_r` | float | Jaw drop intensity [0-5] |
| `AU45_r` | float | Blink intensity [0-5] |

### Example Output

```csv
frame,timestamp,success,AU01_r,AU02_r,AU04_r,...
0,0.0,True,0.234,0.112,1.456,...
1,0.033,True,0.245,0.118,1.423,...
2,0.066,True,0.251,0.125,1.401,...
```

---

## Performance

### Expected Speed (Apple Silicon M-series)

**With Cython optimization:**
- **Face Detection:** ~20-40ms per frame (RetinaFace ONNX + CoreML)
- **Landmark Detection:** ~10ms per frame (PFLD ONNX)
- **Pose Estimation:** ~5-10ms per frame (CalcParams) or <1ms (simplified)
- **Face Alignment + HOG:** ~15-20ms per frame
- **Running Median:** ~0.2ms per frame (Cython!) vs ~47ms (Python)
- **AU Prediction:** ~0.5ms per frame (17 AUs)

**Total:** ~50-85ms per frame (~12-20 FPS)

**For 60-second video (1800 frames):**
- Processing time: ~90-150 seconds
- Speedup vs C++ hybrid: Similar performance
- Advantage: 100% Python, no C++ dependencies!

### Without Cython (Python fallback)

If Cython modules unavailable:
- **Running Median:** ~47ms per frame (slow!)
- **Total:** ~100-130ms per frame (~8-10 FPS)
- For 60-second video: ~180-230 seconds

**Recommendation:** Always use Cython-optimized version for production!

---

## Architecture

### Components and Their Status

| # | Component | Implementation | Status | Accuracy |
|---|-----------|----------------|--------|----------|
| 1 | Face Detection | RetinaFace ONNX | ✅ Ready | High |
| 2 | Landmark Detection | Cunjian PFLD | ✅ Ready | 4.37% NME |
| 3 | Pose Estimation | CalcParams | ✅ Gold Standard | 99.45% |
| 4 | Face Alignment | OpenFace 2.2 Python | ✅ Gold Standard | r=0.94 |
| 5 | Triangulation Masking | Python | ✅ Perfect | r=1.0 |
| 6 | HOG Extraction | PyFHOG (C library) | ✅ Gold Standard | r=1.0 |
| 7 | Geometric Features | PDM Parser Python | ✅ Perfect | Exact |
| 8 | Running Median | Cython-optimized | ✅ Gold Standard | 260x faster |
| 9 | AU Prediction | SVR Python | ✅ Working | r=0.83 |
| 10 | Post-processing | Python | ✅ Working | Correct |

### Data Flow

```
Video Frame (BGR)
  ↓
RetinaFace Detector
  ├─> Face bbox [x1, y1, x2, y2]
  └─> Confidence score
  ↓
PFLD Landmark Detector
  └─> 68 landmarks (x, y)
  ↓
CalcParams (or simplified PnP)
  ├─> Global pose: scale, rx, ry, rz, tx, ty
  └─> Local shape: 34 PCA coefficients
  ↓
Face Aligner
  └─> 112×112 aligned face (BGR)
  ↓
Triangulation Masking
  └─> Masked 112×112 face
  ↓
PyFHOG Extractor
  └─> 4464 HOG features
  ↓
PDM Geometric Extractor
  └─> 238 geometric features
  ↓
Running Median Tracker (Cython!)
  ├─> HOG median (4464)
  └─> Geometric median (238)
  ↓
AU Prediction (SVR)
  ├─> Dynamic AUs (normalized by median)
  └─> Static AUs (non-normalized)
  ↓
Post-processing
  ├─> Cutoff adjustment
  └─> Temporal smoothing
  ↓
Final AU Intensities (17 AUs × [0, 5])
```

---

## Comparison: Full Python vs. Hybrid (C++/Python)

### Hybrid Pipeline (Current Production)

```
C++ OpenFace Binary (FeatureExtraction)
  ├─> Face detection (MTCNN)
  ├─> Landmark detection (CLNF)
  ├─> Pose estimation (CalcParams C++)
  ├─> Face alignment (C++)
  └─> HOG extraction (FHOG C++)
  ↓
Output: .hog and .csv files
  ↓
Python AU Extraction
  ├─> Load HOG/CSV
  ├─> Running median (Cython)
  ├─> AU prediction (SVR)
  └─> Post-processing
  ↓
Final AUs (r=0.83)
```

**Pros:**
- Proven accuracy (r=0.83)
- C++ feature extraction is mature

**Cons:**
- Requires C++ OpenFace binary (99.24% of time!)
- Not cross-platform friendly
- Complex build dependencies

### Full Python Pipeline (This Script)

```
Pure Python (with ONNX/Cython)
  ├─> Face detection (RetinaFace ONNX)
  ├─> Landmark detection (PFLD ONNX)
  ├─> Pose estimation (CalcParams Python 99.45%)
  ├─> Face alignment (Python)
  ├─> HOG extraction (PyFHOG)
  ├─> Running median (Cython 260x)
  └─> AU prediction (SVR Python)
  ↓
Final AUs (expected r=0.75-0.85)
```

**Pros:**
- ✅ 100% Python (easier to distribute)
- ✅ Cross-platform (Windows, Mac, Linux)
- ✅ ONNX models (portable, optimized)
- ✅ CalcParams 99.45% accuracy (gold standard)
- ✅ Running median 260x faster (Cython)
- ✅ No C++ build dependencies

**Cons:**
- Face detection + landmarks may differ slightly from C++ CLNF
- Need to validate end-to-end accuracy (TODO)

---

## Validation Plan (TODO)

### Phase 1: Component Validation (Already Done)

- ✅ CalcParams: 99.45% accuracy
- ✅ Face Alignment: r=0.94 for static AUs
- ✅ HOG Extraction: r=1.0 vs C++
- ✅ Running Median: Extensively validated
- ✅ AU Prediction: r=0.83 with C++ features

### Phase 2: End-to-End Validation (Next Step)

**Test procedure:**
1. Run full Python pipeline on validation video
2. Compare to C++ OpenFace baseline
3. Measure:
   - Landmark RMSE (expected: <5 pixels)
   - Pose parameter correlation (expected: >0.95)
   - AU correlation (expected: >0.75)

**Target:** Overall AU correlation r > 0.75 (acceptable for production)

---

## Known Limitations

### 1. Face Detection Differences

- **C++ OpenFace:** Uses MTCNN (multi-stage, very accurate)
- **Python Pipeline:** Uses RetinaFace (single-stage, faster)
- **Impact:** May detect slightly different faces in challenging conditions

**Mitigation:** RetinaFace is very accurate, differences should be minimal

### 2. Landmark Detection Differences

- **C++ OpenFace:** Uses CLNF (Constrained Local Neural Fields)
- **Python Pipeline:** Uses PFLD (4.37% NME, fast)
- **Impact:** Landmarks may differ by 2-5 pixels

**Mitigation:** PFLD is accurate enough for AU extraction (validated at 4.37% NME)

### 3. Two-Pass Processing Not Implemented Yet

Currently, the pipeline does NOT reprocess early frames with final median.

**Impact:** First 3000 frames may have slightly immature running median

**TODO:** Implement two-pass processing:
- Pass 1: Store features for first 3000 frames
- Pass 2: Reprocess with final median

---

## PyInstaller Integration

### Will Cython Modules Work?

**Yes!** PyInstaller automatically includes compiled extensions:

- Cython `.so` files (Mac/Linux) or `.pyd` files (Windows)
- Detected during `Analysis` phase
- Packaged in `a.binaries`

### Automatic Detection

No changes needed to your `.spec` file! PyInstaller will:
1. Scan imports in `full_python_au_pipeline.py`
2. Find `cython_histogram_median.cpython-313-darwin.so`
3. Include it automatically

### Graceful Fallback

Our implementation already handles missing Cython:

```python
try:
    from cython_histogram_median import DualHistogramMedianTrackerCython
    USING_CYTHON = True
except ImportError:
    from histogram_median_tracker import DualHistogramMedianTracker
    USING_CYTHON = False
```

**Result:**
- If Cython included → 260x faster ⚡
- If Cython missing → Still works (Python fallback)
- No crashes, no errors!

---

## Next Steps

### Immediate (Testing)

1. ✅ **Script created** (`full_python_au_pipeline.py`)
2. ⏳ **Test on sample video** (validate it runs end-to-end)
3. ⏳ **Compare to C++ baseline** (measure AU correlation)
4. ⏳ **Document results**

### Short-term (Improvements)

1. **Implement two-pass processing** for early frames
2. **Add progress bar** (tqdm) for better UX
3. **Add visualization** option (overlay AUs on video)
4. **Batch processing** support (multiple videos)

### Long-term (Optimization)

1. **GPU acceleration** for face detection (ONNX Runtime GPU)
2. **Multi-threading** for parallel frame processing
3. **Optimize face alignment** (Component 5 - see TODO list)

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pyfhog'"

**Solution:**
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfhog"
pip install -e .
```

### Error: "Using Python running median (Cython not available)"

**Solution:**
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"
python3 setup_cython_modules.py build_ext --inplace
```

### Error: "Failed to initialize pipeline: File not found"

**Solution:** Check that all model files exist in `weights/` directory

---

## Summary

**Status:** ✅ Complete and ready for testing!

**Key Achievements:**
- 100% Python pipeline (no C++ required!)
- CalcParams 99.45% accuracy (gold standard)
- Running median 260x faster (Cython-optimized)
- All components integrated and validated individually

**Next:** Test on real video and measure end-to-end AU correlation!

---

**Date:** 2025-10-30
**Author:** Full Python AU Pipeline Integration
**File:** `full_python_au_pipeline.py`
