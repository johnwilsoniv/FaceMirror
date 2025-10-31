# OpenFace 2.2 Python Migration Analysis

## Executive Summary

**Current Situation:**
- OpenFace 2.2 provides clinically validated AU detection results
- OpenFace 3.0 has incorrect AU mapping, weak correlations, and inversions
- OpenFace 2.2 is a compiled C++ application with non-Python dependencies

**Challenge:**
Remove non-Python dependencies (particularly dlib) while maintaining OF2.2's excellent AU detection quality.

---

## OpenFace 2.2 Architecture

### Current Stack

```
┌─────────────────────────────────────────┐
│  FeatureExtraction (C++ Binary)         │
│  ┌───────────────────────────────────┐  │
│  │ 1. Face Detection                 │  │
│  │    - dlib HOG detector OR         │  │
│  │    - Haar cascades (OpenCV)       │  │
│  ├───────────────────────────────────┤  │
│  │ 2. Facial Landmark Detection      │  │
│  │    - CLNF (Constrained Local      │  │
│  │      Neural Fields) - Custom C++  │  │
│  │    - 68-point landmarks           │  │
│  ├───────────────────────────────────┤  │
│  │ 3. Head Pose Estimation           │  │
│  │    - From landmarks               │  │
│  ├───────────────────────────────────┤  │
│  │ 4. Face Alignment                 │  │
│  │    - Crop aligned face patches    │  │
│  ├───────────────────────────────────┤  │
│  │ 5. HOG Feature Extraction         │  │
│  │    - Histogram of Oriented        │  │
│  │      Gradients from aligned faces │  │
│  ├───────────────────────────────────┤  │
│  │ 6. AU Prediction                  │  │
│  │    - SVMs trained on HOG features │  │
│  │    - Separate models for 17 AUs   │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Dependencies
- **OpenCV** (C++/Python available) ✓
- **dlib** (C++ library, Python bindings available) ⚠️
- **Boost** (C++ only) ❌
- **TBB (Threading Building Blocks)** (C++ only) ❌
- **OpenBLAS** (C++ only) ❌
- **Trained model files** (.dat, .txt) ✓

---

## The dlib Question

### What dlib Provides in OF2.2

dlib's role in OpenFace 2.2 is **primarily for face detection**, specifically:
- `frontal_face_detector` - HOG-based face detector
- Optional: Alternative to OpenCV's Haar cascade detector

**Important:** OpenFace 2.2 uses its **own CLNF system for landmark detection**, NOT dlib's landmark detector!

### Can We Remove dlib?

**Short Answer:** Partially yes, but it won't eliminate the C++ dependency problem.

**Why:** OpenFace 2.2 can use OpenCV's Haar cascades instead of dlib for face detection. However:
1. FeatureExtraction is still a C++ binary (requires compilation)
2. The AU extraction logic is in C++ (not portable to Python)
3. CLNF landmark detector is C++ only
4. Removing dlib alone doesn't make OpenFace 2.2 pure Python

---

## Python-Based Landmark Detector Options

### Option 1: face-alignment (Recommended)
**GitHub:** https://github.com/1adrianb/face-alignment

**Pros:**
- Pure Python/PyTorch
- 68-point landmarks (dlib-compatible format)
- 2D and 3D landmark detection
- State-of-the-art accuracy
- Well-maintained

**Cons:**
- Different model than OF2.2's CLNF
- May produce slightly different landmarks
- Requires PyTorch

**Installation:**
```bash
pip install face-alignment
```

**Usage:**
```python
import face_alignment
from face_alignment import FaceAlignment, LandmarksType

# Initialize
fa = FaceAlignment(LandmarksType.TWO_D, device='cpu')

# Detect landmarks
preds = fa.get_landmarks(image)
landmarks_68 = preds[0]  # (68, 2) array
```

### Option 2: MediaPipe Face Mesh
**Pros:**
- Google-maintained
- Very fast (optimized for mobile)
- 468 landmarks (more detailed than 68)
- Pure Python

**Cons:**
- Different landmark format (not 68-point compatible)
- Would require remapping to 68-point format
- Not directly compatible with OF2.2's expectations

### Option 3: OpenFace 3.0's STAR Detector
**Pros:**
- Already integrated in your codebase
- 98-point landmarks
- ONNX-accelerated version available

**Cons:**
- Not 68-point format (would need conversion)
- Part of the OF3 system we're trying to move away from

---

## Migration Strategies

### Strategy 1: Hybrid Approach (Easiest)
**Keep OF2.2 binary, replace face detection only**

```python
import face_alignment
import subprocess

# Detect faces with Python
fa = FaceAlignment(LandmarksType.TWO_D, device='cpu')
preds = fa.get_landmarks(image)

# Get bounding box from landmarks
if preds is not None:
    x_min, y_min = preds[0].min(axis=0)
    x_max, y_max = preds[0].max(axis=0)

    # Pass bounding box to OpenFace 2.2
    subprocess.run([
        "FeatureExtraction",
        "-f", video_path,
        "-bbox", f"{x_min},{y_min},{x_max},{y_max}"
    ])
```

**Result:**
- Removes dlib dependency
- Still requires OF2.2 C++ binary
- Maintains AU extraction quality

**Effort:** Low (1-2 days)

---

### Strategy 2: Pure Python AU Detector (Complex)
**Replicate OF2.2's AU extraction in Python**

**Requirements:**
1. Extract OF2.2's trained SVM models
2. Implement HOG feature extraction in Python
3. Implement face alignment in Python
4. Apply SVM models to features

**Challenges:**
- OF2.2's SVM models are in C++ binary format
- Would need to reverse-engineer model structure
- Complex calibration and normalization steps
- High risk of accuracy loss

**Effort:** Very High (4-8 weeks)

---

### Strategy 3: Retrain on py-feat (Alternative System)
**Use py-feat library for pure Python AU detection**

**py-feat Features:**
- Pure Python/PyTorch
- Pre-trained AU models
- Active development
- Research-backed

**Installation:**
```bash
pip install py-feat
```

**Challenges:**
- Different AU detection models than OF2.2
- Would need clinical revalidation
- May not match OF2.2's performance
- Your S3 models are trained on OF2.2 outputs

**Effort:** High (2-4 weeks + validation)

---

### Strategy 4: OpenFace 2.2 Python Port (Massive Undertaking)
**Create full Python reimplementation**

**Components to Port:**
- CLNF landmark detector
- Head pose estimation
- Face alignment
- HOG extraction
- SVM AU predictors

**Challenges:**
- Thousands of lines of C++ code
- Complex mathematical operations
- Would need exact replication for compatibility
- Ongoing maintenance burden

**Effort:** Extremely High (3-6 months)

---

## Recommendation

### Immediate Solution: Fix OpenFace 3.0 Mapping

**Why:**
1. Already pure Python
2. ONNX-accelerated (10-20x faster)
3. Just needs corrected AU mapping
4. Modern deep learning approach

**Required Work:**
1. Apply corrected AU mapping from correlation analysis ✓ (Already done!)
2. Handle inverted outputs (negation)
3. Validate against ground truth
4. Potentially retrain S3 models on corrected OF3 outputs

**Corrected Mapping:**
```python
self.of3_au_mapping = {
    0: 'AU25_r',  # r=-0.488 (INVERTED - negate values)
    1: 'AU12_r',  # r=0.248 ✓
    2: 'AU15_r',  # r=0.206 ✓
    3: 'AU01_r',  # r=0.000 (no valid data - use NaN)
    4: 'AU20_r',  # r=0.350 ✓
    5: 'AU02_r',  # r=0.000 (no valid data - use NaN)
    6: 'AU06_r',  # r=-0.326 (INVERTED - negate values)
    7: 'AU04_r',  # r=-0.139 (INVERTED - negate values)
}
```

**Effort:** Low (1-2 days)

---

### Long-Term Solution: If OF3 Still Problematic

**Hybrid Approach with face-alignment:**
1. Use face-alignment for Python-based face/landmark detection
2. Keep OF2.2 FeatureExtraction for AU extraction
3. Provides partial Python migration
4. Maintains clinical validation

**Implementation:**
```python
import face_alignment
import subprocess
import pandas as pd
from pathlib import Path

class OpenFace22WithPythonDetection:
    def __init__(self, of22_binary_path):
        self.of22_binary = of22_binary_path
        self.face_detector = FaceAlignment(
            LandmarksType.TWO_D,
            device='cpu',
            face_detector='sfd'  # SFD is pure Python
        )

    def process_video(self, video_path, output_csv):
        """Process video with Python face detection + OF2.2 AU extraction"""
        # Pre-detect faces with Python
        faces = self.detect_faces_python(video_path)

        # Run OF2.2 with face hints
        self.run_openface22(video_path, output_csv, face_hints=faces)

        return pd.read_csv(output_csv)
```

**Pros:**
- Removes dlib dependency
- Maintains OF2.2 AU quality
- Incremental migration path

**Cons:**
- Still requires C++ binary
- Doesn't fully solve dependency problem

---

## Decision Matrix

| Strategy | Effort | Risk | Python % | Maintains Quality |
|----------|--------|------|----------|-------------------|
| Fix OF3 mapping | Low | Low | 100% | TBD (needs validation) |
| Hybrid (face-alignment + OF2.2) | Low | Low | 60% | Yes |
| Pure Python AU detector | Very High | High | 100% | No (needs retraining) |
| py-feat | High | Medium | 100% | No (needs validation) |
| Full OF2.2 port | Extremely High | High | 100% | Maybe |

---

## Next Steps

### Recommended Path:

**Phase 1: Fix OpenFace 3.0 (1-2 days)**
1. ✅ Apply corrected AU mapping (already identified)
2. Handle inverted AUs (negate indices 0, 6, 7)
3. Set indices 3, 5 to NaN (no valid data)
4. Re-run validation tests
5. Compare corrected OF3 with OF2.2

**Phase 2: If OF3 Works (Best Case)**
- Continue using pure Python OF3
- Consider retraining S3 models on OF3 outputs
- Full Python stack achieved!

**Phase 3: If OF3 Still Problematic (Fallback)**
- Implement hybrid approach with face-alignment
- Keep OF2.2 for AU extraction
- Partial Python migration
- Maintain clinical validation

---

## Conclusion

**The dlib question is a red herring.** The real issue is that OpenFace 2.2 is a C++ application, and removing dlib alone won't make it pure Python.

**The best path forward is:**
1. Fix the OpenFace 3.0 AU mapping (we've already identified the correct mapping!)
2. Validate that corrected OF3 matches OF2.2's clinical performance
3. If successful, you get a pure Python, ONNX-accelerated AU detector
4. If not, fall back to hybrid approach with partial Python migration

**Start with fixing OF3 - it's the lowest effort, lowest risk path to a pure Python solution.**
