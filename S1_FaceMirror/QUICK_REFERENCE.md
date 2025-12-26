# S1 Face Mirror - Landmark Detection Quick Reference

## Files Summary

### Core Detection (PRIMARY FILES TO MODIFY)
```
pyfaceau_detector.py (605 lines)
├── PyFaceAU68LandmarkDetector class (main detector)
│   ├── __init__() - Load models (PFLD + CLNF)
│   ├── get_face_mesh() ⭐ - Main detection entry point
│   ├── get_facial_midline() - Calculate glabella & chin
│   ├── calculate_head_pose() - Estimate head yaw
│   ├── check_landmark_quality() - Validate landmarks
│   └── reset_tracking_history() - Clear history between videos
└── [from pyfaceau package]
    ├── ONNXRetinaFaceDetector - Face detection
    ├── CunjianPFLDDetector - 68-point landmarks
    ├── TargetedCLNFRefiner - Fast refinement
    └── CLNFDetector - Slow refinement (fallback)
```

### Pipeline Integration (SECONDARY FILES)
```
face_splitter.py (91 lines)
└── StableFaceSplitter - Pipeline orchestrator
    ├── Creates PyFaceAU68LandmarkDetector
    ├── Creates FaceMirror
    └── Creates VideoProcessor

video_processor.py (635 lines)
└── VideoProcessor.process_video()
    └── Calls: detector.get_face_mesh() -> landmarks -> mirror

face_mirror.py (185 lines)
├── create_mirrored_faces() - Uses landmarks + midline
├── create_debug_frame() - Uses landmarks + head pose
└── [Depends on: get_facial_midline(), calculate_head_pose()]
```

### Supporting Files (TERTIARY)
```
config.py - Settings (NUM_THREADS, BATCH_SIZE, etc.)
config_paths.py - Model directory paths
main.py - Application entry point (line 763)
openface_integration.py - AU extraction wrapper
```

---

## Landmark Detection Pipeline

```
INPUT: Video Frame
    │
    ├─► get_face_mesh(frame)
    │   │
    │   ├─ Face Detection (Frame 0 only)
    │   │  └─ RetinaFace.detect_faces()
    │   │     └─ Bounding box → cached_bbox
    │   │
    │   ├─ Landmark Detection
    │   │  └─ PFLD.detect_landmarks(frame, bbox)
    │   │     └─ 68 points (68x2 array)
    │   │
    │   ├─ Refinement (Optional)
    │   │  └─ TargetedCLNFRefiner.refine()
    │   │     └─ Improved landmarks
    │   │
    │   ├─ Quality Check
    │   │  └─ check_landmark_quality()
    │   │     └─ If POOR + fallback exists:
    │   │        └─ CLNFDetector.refine() (slow)
    │   │
    │   ├─ Temporal Smoothing
    │   │  └─ 5-frame weighted average
    │   │     └─ Smooth landmarks
    │   │
    │   └─ RETURN: smoothed_points (68x2)
    │
    ├─ get_facial_midline(landmarks)
    │  └─ RETURN: (glabella, chin)
    │
    ├─ calculate_head_pose(landmarks)
    │  └─ RETURN: yaw (degrees)
    │
    └─► create_mirrored_faces(frame, landmarks)
        └─ Use landmarks + midline to mirror

OUTPUT: Mirrored frames (left + right)
```

---

## Key Methods to Know

### Main Detection
```python
landmarks, _ = detector.get_face_mesh(frame)
# Returns: (68, 2) array of [x, y] coordinates, None
```

### Anatomical Midline (for mirroring)
```python
glabella, chin = detector.get_facial_midline(landmarks)
# glabella: point between eyebrows (landmarks 21 & 22)
# chin: center chin point (landmark 8)
```

### Head Quality Check
```python
yaw = detector.calculate_head_pose(landmarks)
# Returns: head yaw in degrees
# Optimal: |yaw| ≤ 3°
# Acceptable: 3° < |yaw| ≤ 5°
# Excessive: |yaw| > 5°
```

### Landmark Quality
```python
is_poor, reason = detector.check_landmark_quality(landmarks)
# is_poor: bool
# reason: str ('clustering', 'poor_distribution', 'ok')
```

---

## Landmark Indices (68-point dlib standard)

```
           Eyebrows (17-26)
          /                \
    Eyes (36-47)      Nose (27-35)
         /                    \
    Jaw (0-16) ← 8 (Chin)      
                             Mouth (48-67)

CRITICAL FOR MIRRORING:
├─ Index 8: Chin center (used as anchor)
├─ Index 21: Left medial eyebrow (glabella calc)
└─ Index 22: Right medial eyebrow (glabella calc)
```

---

## Performance Considerations

| Operation | Time/Frame | Impact |
|-----------|-----------|--------|
| RetinaFace detection | ~30ms | Only frame 0 (cached after) |
| PFLD landmarks | ~5-10ms | Every frame |
| SVR CLNF refinement | ~1-2ms | Every frame (default on) |
| Fallback CLNF | ~200-500ms | Only poor quality frames |
| Temporal smoothing | <1ms | Every frame |
| **Total (typical)** | **10-15ms** | **~70-100 FPS possible** |

**Actual FPS**: ~14-28 FPS on Apple Silicon (limited by video I/O, not detection)

---

## Configuration Changes

### To Speed Up (2-3x faster)
In `face_splitter.py` line 37:
```python
use_clnf_refinement=False  # Disable SVR refinement
```

### To Use Only First Frame Detection
In `face_splitter.py` line 35:
```python
skip_redetection=True  # Reuse bbox for all frames
```

### To Skip Face Detection Entirely (risky)
In `face_splitter.py` line 36:
```python
skip_face_detection=True  # Use centered default bbox
```

### To Debug Landmarks
In `main.py` or `face_splitter.py`:
```python
debug_mode=True  # Print detailed debug info
```

---

## Troubleshooting Guide

### Poor Landmark Quality
**Symptom**: Mirroring looks distorted
**Cause**: PFLD detected landmarks poorly
**Solution**: Check `check_landmark_quality()` output
- If clustering: Face detection failed
- If poor distribution: Extreme head rotation

### Head Rotation Warnings
**Symptom**: Debug frame shows "EXCESSIVE HEAD ROTATION"
**Cause**: |yaw| > 5°
**Solution**: Ensure face is ~straight-on (within 3-5 degrees)

### Performance Slow
**Symptom**: <10 FPS processing
**Cause**: CLNF fallback activated for poor frames
**Solution**: Check `poor_quality_frames` list
- May indicate systematic landmark issues
- Consider improving input video quality

### Temporal Jitter
**Symptom**: Landmarks jump frame to frame
**Cause**: History buffer not fully initialized
**Solution**: First 5 frames will be less smooth (by design)

---

## Model Loading Order

1. **Face Detection** (RetinaFace)
   - File: `retinaface_mobilenet025_coreml.onnx` (1.73 MB)
   - Use: First frame only (cached)

2. **Landmark Detection** (PFLD)
   - File: `pfld_cunjian.onnx` (2.99 MB)
   - Use: Every frame

3. **Refinement** (SVR CLNF)
   - File: `svr_patches_0.25_general.txt` (1.13 MB)
   - Use: Every frame (default)

4. **Fallback** (CLNF - loaded on-demand)
   - Directory: `clnf/`
   - Use: Poor quality frames only

---

## Testing & Validation

### To Test Landmark Detection Only
```python
from pyfaceau_detector import PyFaceAU68LandmarkDetector
detector = PyFaceAU68LandmarkDetector(debug_mode=True)
landmarks, _ = detector.get_face_mesh(frame)
print(landmarks.shape)  # Should be (68, 2)
```

### To Check Landmark Quality
```python
is_poor, reason = detector.check_landmark_quality(landmarks)
print(f"Quality: {reason}")
```

### To Verify Midline
```python
glabella, chin = detector.get_facial_midline(landmarks)
print(f"Glabella: {glabella}, Chin: {chin}")
```

---

## Related Source Files

**This directory**: `/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/`

Key files:
- `pyfaceau_detector.py` - Main implementation (EDIT THIS)
- `video_processor.py` - Frame loop (calls detector)
- `face_mirror.py` - Uses landmarks for mirroring
- `face_splitter.py` - Pipeline initialization
- `config.py` - Settings

Model files:
- `weights/pfld_cunjian.onnx` - Landmark model
- `weights/retinaface_mobilenet025_coreml.onnx` - Face detector
- `weights/svr_patches_0.25_general.txt` - CLNF refinement

---

**Last Updated**: 2025-11-04
**System**: Apple Silicon macOS
**Python Version**: 3.8+
