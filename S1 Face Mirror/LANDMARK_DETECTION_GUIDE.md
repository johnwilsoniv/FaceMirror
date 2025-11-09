# S1 Face Mirror - Landmark Detection Implementation Summary

## Overview

The S1 Face Mirror application implements a complete face detection and landmark extraction pipeline using **PyFaceAU** - a pure Python implementation that replaces the OpenFace 3.0 C++ backend.

**Key Characteristic**: Uses a **68-point facial landmark system** with temporal smoothing and optional CLNF refinement for improved accuracy.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN PIPELINE                             │
│                   (main.py - line 763)                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              StableFaceSplitter                              │
│            (face_splitter.py, lines 13-91)                  │
│  - Encapsulates detection, mirroring, and processing        │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    ┌─────────┬─────────┬─────────┐
    │ Landmark│ Face    │ Video   │
    │ Detector│ Mirror  │Processor│
    └─────────┴─────────┴─────────┘
```

---

## 1. Landmark Detection Component

### Primary Class
**`PyFaceAU68LandmarkDetector`** (`pyfaceau_detector.py`, lines 24-604)

#### Initialization Parameters (lines 36-49)
```python
def __init__(self, debug_mode=False, device='cpu', model_dir=None,
             skip_redetection=False, skip_face_detection=False,
             use_clnf_refinement=True):
```

#### Key Methods

| Method | Purpose | Location |
|--------|---------|----------|
| `get_face_mesh(frame)` | Main detection entry point - returns 68-point landmarks with temporal smoothing | Lines 270-426 |
| `get_facial_midline(landmarks)` | Calculates anatomical midline (glabella & chin) from landmarks | Lines 428-471 |
| `calculate_head_pose(landmarks)` | Estimates head yaw angle for quality control | Lines 473-543 |
| `check_landmark_quality(landmarks)` | Validates landmark spatial distribution (clustering, asymmetry) | Lines 186-245 |
| `reset_tracking_history()` | Clears temporal history between videos | Lines 172-184 |

---

## 2. Face Detection Models

### RetinaFace Detector
**Component**: `ONNXRetinaFaceDetector` (from pyfaceau)
**Location**: `pyfaceau_detector.py`, lines 66-79

```python
self.face_detector = ONNXRetinaFaceDetector(
    str(model_dir / 'retinaface_mobilenet025_coreml.onnx'),
    use_coreml=True,              # Hardware acceleration on Apple Silicon
    confidence_threshold=0.5,
    nms_threshold=0.4
)
```

**Model File**: `/weights/retinaface_mobilenet025_coreml.onnx` (1.73 MB)
- **Framework**: ONNX with CoreML acceleration
- **Purpose**: Detects face bounding boxes in image
- **Hardware**: Auto-accelerated via CoreML on Apple Silicon, ONNX on Intel

**Detection Flow**:
1. Called in `get_face_mesh()` only on first frame or when redetection needed
2. Returns detection array `[x1, y1, x2, y2, confidence, ...]`
3. Cached in `self.cached_bbox` to avoid redundant detection

---

## 3. Landmark Detection Models

### PFLD (Cunjian) 68-Point Detector
**Component**: `CunjianPFLDDetector` (from pyfaceau)
**Location**: `pyfaceau_detector.py`, lines 81-87

```python
self.landmark_detector = CunjianPFLDDetector(
    str(model_dir / 'pfld_cunjian.onnx'),
    use_coreml=True
)
```

**Model File**: `/weights/pfld_cunjian.onnx` (2.99 MB)
- **Framework**: ONNX with CoreML acceleration
- **Accuracy**: 4.37% NME (Normalized Mean Error)
- **Output**: 68-point coordinates (dlib/PFLD standard indexing)

**Landmark Indices (dlib standard)**:
- **Jaw (0-16)**: Points around the jawline
- **Eyebrows (17-26)**: Left and right eyebrows
- **Nose (27-35)**: Nose tip and nostrils
- **Eyes (36-47)**: Eye corners and pupils
- **Mouth (48-67)**: Lips and mouth corners
- **Key points**:
  - **Index 8**: Chin center (used for anatomical midline)
  - **Index 21**: Left medial eyebrow (for glabella)
  - **Index 22**: Right medial eyebrow (for glabella)

---

## 4. Landmark Refinement (CLNF)

### Two CLNF Implementations

#### A. TargetedCLNFRefiner (Fast SVR-based)
**Component**: `TargetedCLNFRefiner` (from pyfaceau)
**Location**: `pyfaceau_detector.py`, lines 90-103

```python
self.clnf_refiner = TargetedCLNFRefiner(
    patch_expert_file=str(model_dir / 'svr_patches_0.25_general.txt'),
    search_window=3,
    pdm=None,              # No shape constraint enforcement
    enforce_pdm=False
)
```

**Model File**: `/weights/svr_patches_0.25_general.txt` (1.13 MB)
- **Type**: SVR (Support Vector Regression) patch experts
- **Speed**: Real-time (1-2ms per frame)
- **Purpose**: Refine PFLD landmarks for improved accuracy
- **Applied in**: `get_face_mesh()`, line 337

#### B. CLNFDetector (Fallback - OpenFace Quality)
**Component**: `CLNFDetector` (from pyfaceau)
**Location**: `pyfaceau_detector.py`, lines 107-124

```python
self.clnf_fallback = CLNFDetector(
    model_dir=clnf_dir,
    max_iterations=5,           # Match OpenFace default
    convergence_threshold=0.01
)
```

**Model Directory**: `/weights/clnf/` (contains patch experts and shape models)
- **Type**: Optimization-based refinement
- **Speed**: ~2-5 FPS (much slower)
- **Trigger**: Activated automatically for poor-quality landmarks (surgical markings, severe paralysis)
- **Applied in**: `get_face_mesh()`, lines 350-374

**Quality Detection Criteria** (lines 186-245):
- **Clustering**: >75% landmarks on one side (indicates poor detection)
- **Poor Distribution**: Standard deviation below 25% of bounding box
- **Asymmetry**: Beyond expected natural variation

---

## 5. Temporal Smoothing

### 5-Frame History Buffer
**Location**: `pyfaceau_detector.py`, lines 135-141

```python
self.landmarks_history = []          # Store last 5 frames
self.glabella_history = []
self.chin_history = []
self.yaw_history = []
self.frame_quality_history = []
self.history_size = 5
```

### Smoothing Algorithm
**Location**: `get_face_mesh()`, lines 376-390

```python
# Weighted average (more weight to recent frames)
weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
weights = weights / np.sum(weights)

smoothed_points = np.zeros_like(landmarks_68, dtype=np.float32)
for pts, w in zip(self.landmarks_history, weights):
    smoothed_points += pts * w
```

**Result**: Smooths temporal jitter while maintaining responsiveness (higher weight on recent frames)

---

## 6. Anatomical Midline Calculation

### Glabella & Chin Points
**Method**: `get_facial_midline()` (lines 428-471)

```python
# Get medial eyebrow points
left_medial_brow = landmarks[21]      # Left eyebrow inner corner
right_medial_brow = landmarks[22]     # Right eyebrow inner corner

# Calculate glabella (midpoint between brows)
glabella = (left_medial_brow + right_medial_brow) / 2

# Chin center
chin = landmarks[8]
```

**Temporal Smoothing** (lines 458-469):
- Maintains 5-frame history of glabella and chin
- Averages history for stable midline (only for calculation)
- Critical for consistent mirroring across frames

---

## 7. Head Pose Estimation

### Head Yaw Calculation
**Method**: `calculate_head_pose()` (lines 473-543)

**Algorithm**:
1. Uses symmetric landmark pairs to estimate rotation
2. Landmark pairs: Eyes (36,45), Inner eyes (39,42), Eyebrows (17,26), Mouth (48,54), Jaw (1,15), Cheeks (4,12)
3. Calculates signed distance from center for each pair
4. Weights eye landmarks 2x (more reliable)
5. Returns weighted average yaw in degrees

**Quality Control**:
- Used in `create_debug_frame()` to display head rotation warnings
- Flags frame as "EXCESSIVE HEAD ROTATION" if |yaw| > 5.0°
- "ACCEPTABLE" if 3° < |yaw| ≤ 5°
- "OPTIMAL" if |yaw| ≤ 3°

---

## 8. Data Flow for Landmark Detection

```
VIDEO FRAME
    │
    ▼
get_face_mesh(frame)  ◄─── Main entry point
    │
    ├─ Frame count increment (line 282)
    │
    ├─ Face Detection (if needed)
    │  └─ RetinaFace.detect_faces() ──► cached_bbox
    │
    ├─ Landmark Detection
    │  └─ PFLD.detect_landmarks(frame, bbox) ──► landmarks_68 (68, 2)
    │
    ├─ Optional: SVR-based CLNF Refinement
    │  └─ TargetedCLNFRefiner.refine_landmarks() ──► refined_landmarks_68
    │
    ├─ Quality Check
    │  └─ check_landmark_quality()
    │     └─ If POOR + clnf_fallback available:
    │        └─ CLNFDetector.refine_landmarks() ──► final_landmarks_68
    │
    ├─ Temporal Smoothing
    │  ├─ Add to landmarks_history (max 5 frames)
    │  └─ Weighted average ──► smoothed_points (68, 2)
    │
    ├─ Anatomical Midline Calculation
    │  └─ get_facial_midline(smoothed_points) ──► (glabella, chin)
    │
    ├─ Head Pose Estimation
    │  └─ calculate_head_pose(smoothed_points) ──► yaw (degrees)
    │
    └─ RETURN: smoothed_points (68x2 array), None
```

---

## 9. Integration Points

### In VideoProcessor (`video_processor.py`)
**Frame Processing Loop** (lines 203-238)

```python
def _process_frame_batch(self, frame_data):
    frame_index, frame = frame_data
    
    # Get face landmarks
    landmarks, _ = self.landmark_detector.get_face_mesh(frame)
    
    # Create mirrored faces
    if landmarks is not None:
        right_face, left_face = self.face_mirror.create_mirrored_faces(frame, landmarks)
        debug_frame = self.face_mirror.create_debug_frame(frame, landmarks)
```

### In FaceMirror (`face_mirror.py`)
**Uses landmarks for**:
1. **Midline calculation** (line 22): `glabella, chin = self.landmark_detector.get_facial_midline(landmarks)`
2. **Head pose** (line 163): `yaw = self.landmark_detector.calculate_head_pose(landmarks)`
3. **Anatomical reflection** (lines 12-106): Mirrors face along calculated midline

### In Main Processing (`main.py`)
**Initialization** (line 763):
```python
from pyfaceau_detector import PyFaceAU68LandmarkDetector
# ... passed to StableFaceSplitter
```

---

## 10. Key Configuration Points

### Model Loading (`config.py`)
- **NUM_THREADS**: Number of parallel frame processing threads (default: 6)
- **BATCH_SIZE**: Frames processed per batch (default: 100)
- **CONFIDENCE_THRESHOLD**: Face detection threshold (default: 0.5)
- **NMS_THRESHOLD**: Duplicate face suppression (default: 0.4)

### Model Directory (`config_paths.py`)
```python
weights_dir = Path(__file__).parent / 'weights'
```

### Performance Settings
- **GC_THRESHOLD**: Garbage collection optimization (higher = faster)
- **OMP_NUM_THREADS**: CPU parallelization (set to 2 for stability)

---

## 11. Files to Modify for Landmark Changes

### Priority 1: Core Detection
| File | Purpose | Key Methods |
|------|---------|-------------|
| `pyfaceau_detector.py` | Main detector class | `get_face_mesh()`, `get_facial_midline()`, `calculate_head_pose()` |
| `pyfaceau_detector.py` | Landmark quality | `check_landmark_quality()` |

### Priority 2: Usage & Integration
| File | Purpose | Key Methods |
|------|---------|-------------|
| `video_processor.py` | Frame processing loop | `_process_frame_batch()` |
| `face_mirror.py` | Landmark application | `create_mirrored_faces()`, `create_debug_frame()` |
| `face_splitter.py` | Pipeline initialization | `__init__()` |

### Priority 3: Fallback & Refinement
| File | Purpose | Key Methods |
|------|---------|-------------|
| `pyfaceau_detector.py` | CLNF fallback logic | Lines 350-374 in `get_face_mesh()` |

---

## 12. Models Summary

| Model | File | Size | Purpose | Framework |
|-------|------|------|---------|-----------|
| RetinaFace | `retinaface_mobilenet025_coreml.onnx` | 1.73 MB | Face detection | ONNX + CoreML |
| PFLD 68-point | `pfld_cunjian.onnx` | 2.99 MB | Landmark detection | ONNX + CoreML |
| SVR Patches | `svr_patches_0.25_general.txt` | 1.13 MB | CLNF refinement | SVR |
| CLNF Models | `clnf/` directory | Variable | OpenFace-quality refinement | PyTorch |
| 3D Shape | `In-the-wild_aligned_PDM_68.txt` | 69 KB | Shape constraints | Text-based |

---

## 13. Current Limitations & Notes

### Strengths
✓ Pure Python implementation (no C++ dependencies)
✓ CoreML acceleration on Apple Silicon (5-20x faster than CPU)
✓ 68-point landmarks (sufficient for most facial analysis)
✓ Automatic fallback to OpenFace-quality CLNF for difficult cases
✓ Temporal smoothing reduces jitter
✓ Real-time performance on Apple Silicon (~14-28 FPS)

### Potential Improvements
- CLNF refinement is optional (disabled by default would speed up ~2-3x)
- Could use additional shape models for specific pathologies
- Head pose estimation uses heuristics (could be replaced with 3D model)
- Landmarks are 2D only (3D available in 1k3d68.onnx but not currently used)

---

## 14. Related Documentation

- **PFLD Paper**: "PFLD: A Practical Facial Landmark Detector" (dlib-compatible)
- **CLNF Reference**: OpenFace 2.2 C++ implementation
- **PyFaceAU**: Pure Python implementation of OpenFace landmark detection
- **dlib indexing**: Standard 68-point facial landmark convention

---

**Document Generated**: S1 Face Mirror Landmark Detection Analysis
**System**: Apple Silicon macOS, PyFaceAU Backend
**Version**: 1.0.0
