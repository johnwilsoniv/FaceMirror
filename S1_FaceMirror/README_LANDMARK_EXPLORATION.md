# S1 Face Mirror - Landmark Detection Documentation

## Overview

This directory contains comprehensive documentation about how facial landmark detection is implemented in the S1 Face Mirror application.

## Documentation Index

### 1. LANDMARK_DETECTION_GUIDE.md (PRIMARY - 500+ lines)
**Best for: Deep technical understanding**

Comprehensive technical reference covering:
- Complete architecture overview
- Detailed component descriptions (RetinaFace, PFLD, CLNF)
- 68-point landmark system and indexing
- Temporal smoothing mechanism
- Anatomical midline calculation
- Head pose estimation
- Data flow diagrams
- Model specifications and loading order
- Performance characteristics
- Configuration options
- Files to modify for changes
- Complete model summary table

**Start here if:** You need to understand how everything works together

### 2. QUICK_REFERENCE.md (LOOKUP - 300+ lines)
**Best for: Quick lookups while coding**

Quick reference guide with:
- File summary (primary/secondary/tertiary)
- Visual pipeline diagram
- Key methods with signatures
- Landmark index reference
- Performance table
- Configuration change examples
- Troubleshooting guide
- Model loading order
- Testing code snippets

**Start here if:** You need to find something quickly or debug an issue

### 3. FILE_RELATIONSHIPS.md (DEPENDENCY MAP - 400+ lines)
**Best for: Understanding code organization**

Visual dependency maps showing:
- Architecture diagram
- Complete data flow diagram
- File details table
- Landmark propagation path
- Model loading dependency chain
- Configuration dependencies
- Quality feedback loop
- Error handling paths
- Critical sections for modification

**Start here if:** You're trying to understand how files connect

### 4. LANDMARK_DETECTION_EXPLORATION_SUMMARY.txt (SUMMARY - 300+ lines)
**Best for: Executive overview**

High-level summary covering:
- Executive summary with key stats
- Core files identified (primary/secondary/tertiary)
- Models and frameworks overview
- Landmark system description
- Data flow summary
- Temporal smoothing mechanism
- Quality control mechanisms
- Performance characteristics
- Configuration options
- Key functions to modify
- Integration points
- Conclusions and recommendations

**Start here if:** You want a one-document overview

---

## Quick Navigation

### I need to...

#### Understand the overall architecture
- Start: README_LANDMARK_EXPLORATION.md (this file)
- Then: LANDMARK_DETECTION_GUIDE.md (Section 1)
- Reference: FILE_RELATIONSHIPS.md (Architecture Diagram)

#### Modify landmark detection
- Start: QUICK_REFERENCE.md (Key Methods to Know)
- Reference: LANDMARK_DETECTION_GUIDE.md (Section 1, 8)
- File to edit: `pyfaceau_detector.py` lines 270-426

#### Calculate anatomical midline differently
- Start: QUICK_REFERENCE.md (Landmark Indices)
- Reference: LANDMARK_DETECTION_GUIDE.md (Section 6)
- File to edit: `pyfaceau_detector.py` lines 428-471

#### Change mirroring algorithm
- Start: FILE_RELATIONSHIPS.md (Data Flow Diagram)
- Reference: LANDMARK_DETECTION_GUIDE.md (Section 9)
- File to edit: `face_mirror.py` lines 12-106

#### Debug poor landmark quality
- Start: QUICK_REFERENCE.md (Troubleshooting Guide)
- Reference: LANDMARK_DETECTION_GUIDE.md (Section 4, 13)
- File to examine: `pyfaceau_detector.py` lines 186-245

#### Speed up processing
- Start: QUICK_REFERENCE.md (Configuration Changes)
- Reference: LANDMARK_DETECTION_EXPLORATION_SUMMARY.txt (Configuration)
- File to modify: `face_splitter.py` line 37 (disable CLNF)

#### Understand performance
- Start: QUICK_REFERENCE.md (Performance Considerations)
- Reference: LANDMARK_DETECTION_GUIDE.md (Section 13)
- Summary: LANDMARK_DETECTION_EXPLORATION_SUMMARY.txt (Performance section)

---

## File Structure Overview

### Core Detection
```
pyfaceau_detector.py (605 lines) - PRIMARY FILE
├── PyFaceAU68LandmarkDetector.__init__() - Load models
├── get_face_mesh() ⭐ - Main detection entry point
├── get_facial_midline() - Calculate glabella & chin
├── calculate_head_pose() - Head yaw estimation
├── check_landmark_quality() - Validate landmarks
└── reset_tracking_history() - Clear temporal state
```

### Pipeline Integration
```
face_splitter.py (91 lines)
video_processor.py (635 lines)
face_mirror.py (185 lines)
```

### Configuration & Support
```
config.py - Settings
config_paths.py - Model paths
main.py - Application entry
```

---

## Key Concepts Quick Reference

### 68-Point Landmarks
- **Source**: PFLD detector (dlib standard indexing)
- **Index 8**: Chin center (anatomical anchor)
- **Index 21**: Left medial eyebrow (for glabella)
- **Index 22**: Right medial eyebrow (for glabella)
- **Format**: (68, 2) numpy array of [x, y] coordinates

### Temporal Smoothing
- **Window**: 5 frames
- **Method**: Weighted average (recent frames weighted higher)
- **Benefit**: Reduces jitter while maintaining responsiveness

### Anatomical Midline
- **Glabella**: Point between eyebrows ((landmark[21] + landmark[22]) / 2)
- **Chin**: Center chin point (landmark[8])
- **Direction**: Vector from glabella to chin
- **Use**: Reflection axis for face mirroring

### Quality Checks
1. **Clustering**: >75% landmarks on one side = poor
2. **Distribution**: Too concentrated = poor
3. **Fallback**: If poor, try OpenFace-quality CLNF (slow)

### Performance
- **Typical**: 10-15ms per frame (~70 FPS possible)
- **Actual**: 14-28 FPS (limited by video I/O, not detection)
- **Models**: ~10 MB total
- **Memory**: 5-10 MB for temporal history

---

## Common Tasks

### To Test Landmark Detection
```python
from pyfaceau_detector import PyFaceAU68LandmarkDetector
import cv2

detector = PyFaceAU68LandmarkDetector(debug_mode=True)
frame = cv2.imread("image.jpg")
landmarks, _ = detector.get_face_mesh(frame)
print(landmarks.shape)  # Should be (68, 2)
```

### To Check Landmark Quality
```python
is_poor, reason = detector.check_landmark_quality(landmarks)
print(f"Quality: {reason}")  # 'clustering', 'poor_distribution', or 'ok'
```

### To Get Anatomical Midline
```python
glabella, chin = detector.get_facial_midline(landmarks)
print(f"Glabella: {glabella}, Chin: {chin}")
```

### To Get Head Pose
```python
yaw = detector.calculate_head_pose(landmarks)
print(f"Head yaw: {yaw} degrees")
# Optimal: |yaw| <= 3°
# Acceptable: 3° < |yaw| <= 5°
# Excessive: |yaw| > 5°
```

---

## Model Files Used

| Model | File | Size | Purpose |
|-------|------|------|---------|
| RetinaFace | `retinaface_mobilenet025_coreml.onnx` | 1.73 MB | Face detection |
| PFLD 68-point | `pfld_cunjian.onnx` | 2.99 MB | Landmark detection |
| SVR CLNF | `svr_patches_0.25_general.txt` | 1.13 MB | Fast refinement |
| CLNF fallback | `clnf/` directory | Variable | Slow refinement |

All located in: `/weights/` directory

---

## Key Statistics

### Landmark Detection
- **Detectors**: RetinaFace (faces) + PFLD (68 points)
- **Refinement**: Optional SVR-based CLNF (default on)
- **Fallback**: OpenFace-quality CLNF (for poor frames)
- **Smoothing**: 5-frame weighted average
- **Output**: 68-point coordinates (numpy array)

### Performance
- **Face detection**: ~30ms (frame 0 only, cached)
- **PFLD landmarks**: ~5-10ms per frame
- **SVR refinement**: ~1-2ms per frame
- **Temporal smoothing**: <1ms per frame
- **CLNF fallback**: ~200-500ms (only poor frames)
- **Total typical**: 10-15ms per frame

### Real-world Performance
- **Theoretical max**: ~70-100 FPS
- **Actual speed**: ~14-28 FPS on Apple Silicon
- **Bottleneck**: Video I/O and writing, not detection

---

## File Modification Guide

### Priority 1: Core Detection Changes
- **File**: `pyfaceau_detector.py`
- **Methods**: `get_face_mesh()`, `get_facial_midline()`, `calculate_head_pose()`
- **Impact**: Fundamental changes to detection/calculation

### Priority 2: Integration Changes
- **File**: `video_processor.py` or `face_mirror.py`
- **Methods**: `_process_frame_batch()`, `create_mirrored_faces()`
- **Impact**: How landmarks are used

### Priority 3: Pipeline Changes
- **File**: `face_splitter.py`
- **Method**: `__init__()`
- **Impact**: Component initialization and configuration

---

## Document Versions & Updates

- **Created**: 2025-11-04
- **System**: Apple Silicon macOS
- **Python**: 3.8+
- **Backend**: PyFaceAU (pure Python, no C++ dependencies)

---

## Additional Resources

### Within S1 Face Mirror
- `pyfaceau_detector.py` - Source code with inline comments
- `config.py` - Configuration options
- Model files in `weights/` directory

### External References
- PFLD Paper: "PFLD: A Practical Facial Landmark Detector"
- CLNF: OpenFace 2.2 C++ implementation
- dlib: 68-point facial landmark standard
- PyFaceAU: Pure Python implementation documentation

---

## Summary

This exploration discovered that S1 Face Mirror uses a well-architected system for facial landmark detection:

1. **RetinaFace** detects the face bounding box
2. **PFLD** extracts 68-point landmarks
3. **SVR-based CLNF** refines them (optional)
4. **Fallback CLNF** handles poor-quality cases
5. **Temporal smoothing** reduces jitter
6. **Anatomical midline** calculation enables accurate mirroring

The system is production-ready with:
- Real-time performance on Apple Silicon
- Automatic quality detection and fallback mechanisms
- Graceful error handling
- Modular, well-documented codebase

For detailed information on any aspect, refer to the appropriate documentation listed at the top of this file.

---

**For questions or clarifications, refer to:**
- LANDMARK_DETECTION_GUIDE.md for technical details
- QUICK_REFERENCE.md for quick lookups
- FILE_RELATIONSHIPS.md for code organization
