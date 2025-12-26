# S1 Face Mirror - File Dependency Map

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         main.py (Application Entry)                      │
│                          Line 763: Initialize                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     face_splitter.py (Pipeline)                          │
│              StableFaceSplitter.__init__() [Lines 16-56]                │
│  Creates: detector, face_mirror, video_processor                        │
└──┬──────────────────────────────────────┬───────────────┬────────────────┘
   │                                      │               │
   │ (Creates)                            │               │
   ▼                                      ▼               ▼
┌──────────────────────┐    ┌──────────────────────┐  ┌──────────────────┐
│  pyfaceau_detector   │    │   face_mirror.py     │  │ video_processor  │
│  PyFaceAU68Landmark  │    │   FaceMirror         │  │ VideoProcessor   │
│  Detector            │    │   (Uses landmarks)   │  │ (Uses detector   │
│ [LINES 24-604]       │    │ [LINES 5-185]        │  │  & mirror)       │
└──────┬───────────────┘    └──────────┬───────────┘  │ [LINES 180-635]  │
       │                               │              └──┬────────────────┘
       │ (Imports from)                │                 │
       │                               │ (Calls)         │ (Calls)
       │                               │                 │
       └───┬───────────────────────────┴─────────────────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  pyfaceau [EXTERNAL PACKAGE]                   │
    │  ├─ ONNXRetinaFaceDetector                     │
    │  ├─ CunjianPFLDDetector                        │
    │  ├─ TargetedCLNFRefiner                        │
    │  └─ CLNFDetector                              │
    └─────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ VIDEO FRAME                                                              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ VideoProcessor._process_frame_batch()  [Lines 203-238]                  │
│ ├─ Calls: landmarks, _ = detector.get_face_mesh(frame)                 │
│ ├─ Calls: right_face, left_face = face_mirror.create_mirrored_faces()  │
│ └─ Calls: debug_frame = face_mirror.create_debug_frame()               │
└──────┬─────────────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────┐
       │                                                 │
       ▼                                                 ▼
   DETECTOR FLOW                                    MIRROR FLOW
   ════════════════════════════════════════════════════════════════════

   get_face_mesh() [270-426]                    FaceMirror.create_mirrored_faces()
   │                                            [Lines 12-106]
   ├─ RetinaFace.detect_faces()                 │
   │  └─ [Frame 0 only, cached]                 ├─ get_facial_midline()
   │     └─ cached_bbox                         │  ├─ Read landmark 21 (L brow)
   │                                            │  ├─ Read landmark 22 (R brow)
   ├─ PFLD.detect_landmarks()                   │  ├─ Read landmark 8 (chin)
   │  └─ [Every frame]                          │  └─ RETURN: (glabella, chin)
   │     └─ landmarks_68 (68x2)                 │
   │                                            └─ Use midline for reflection
   ├─ TargetedCLNFRefiner.refine()
   │  └─ [Every frame, optional]
   │     └─ refined_landmarks_68
   │
   ├─ check_landmark_quality()
   │  ├─ Check clustering (<75% on one side)
   │  ├─ Check distribution (std > 25% bbox)
   │  └─ If POOR + fallback available:
   │     └─ CLNFDetector.refine() [Slow]
   │
   ├─ Temporal smoothing [376-390]
   │  ├─ Add to landmarks_history (max 5)
   │  └─ Weighted average
   │     └─ smoothed_points (68x2)
   │
   └─ RETURN: smoothed_points

   FaceMirror.create_debug_frame() [108-185]
   │
   ├─ get_facial_midline()
   │  └─ Draw midline on frame
   │
   └─ calculate_head_pose()
      └─ Analyze head yaw
         └─ Display rotation warning
```

---

## File Details Table

### Primary Detection Files

| File | Lines | Class/Function | Purpose | Called By |
|------|-------|-----------------|---------|-----------|
| `pyfaceau_detector.py` | 1-604 | PyFaceAU68LandmarkDetector | Main detector | face_splitter, video_processor |
| | 24-149 | `__init__()` | Load all models | face_splitter |
| | 270-426 | `get_face_mesh()` | **Main entry point** | video_processor |
| | 428-471 | `get_facial_midline()` | Calculate midline | face_mirror |
| | 473-543 | `calculate_head_pose()` | Head yaw | face_mirror |
| | 186-245 | `check_landmark_quality()` | Validate landmarks | get_face_mesh() |
| | 172-184 | `reset_tracking_history()` | Clear history | video_processor |

### Integration Files

| File | Lines | Class/Function | Purpose | Calls |
|------|-------|-----------------|---------|-------|
| `face_splitter.py` | 1-91 | StableFaceSplitter | Pipeline init | PyFaceAU68LandmarkDetector, FaceMirror, VideoProcessor |
| | 16-56 | `__init__()` | Create components | - |
| | 84-86 | `process_video()` | Start processing | video_processor |
| `video_processor.py` | 1-635 | VideoProcessor | Frame batch processor | landmark_detector |
| | 203-238 | `_process_frame_batch()` | **Main loop** | detector.get_face_mesh() |
| | 240-635 | `process_video()` | Video orchestration | _process_frame_batch() |
| `face_mirror.py` | 1-185 | FaceMirror | Apply mirroring | landmark_detector |
| | 12-106 | `create_mirrored_faces()` | Do reflection | get_facial_midline() |
| | 108-185 | `create_debug_frame()` | Debug visual | get_facial_midline(), calculate_head_pose() |

### Supporting Files

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 1-158 | Settings & environment setup |
| `config_paths.py` | 1-100 | Model directory paths |
| `main.py` | 1-1000+ | Application entry & workflow |
| `openface_integration.py` | 1-142 | AU extraction wrapper |

---

## Landmark Propagation Path

```
[Frame]
    │
    ▼
VideoProcessor._process_frame_batch()
    │
    ├─ landmarks = detector.get_face_mesh(frame)
    │                          │
    │                          ├─ RetinaFace.detect_faces()
    │                          ├─ PFLD.detect_landmarks()
    │                          ├─ TargetedCLNFRefiner.refine()
    │                          ├─ check_landmark_quality()
    │                          └─ Temporal smoothing
    │                             └─ RETURNS: landmarks (68x2)
    │
    ├─ right_face, left_face = face_mirror.create_mirrored_faces(frame, landmarks)
    │                                            │
    │                                            ├─ glabella, chin = detector.get_facial_midline(landmarks)
    │                                            │                              │
    │                                            │                              ├─ Read landmark 21
    │                                            │                              ├─ Read landmark 22
    │                                            │                              └─ Read landmark 8
    │                                            │
    │                                            └─ Calculate reflection matrix
    │                                               RETURNS: right_face, left_face
    │
    └─ debug_frame = face_mirror.create_debug_frame(frame, landmarks)
                                                               │
                                                               ├─ glabella, chin = detector.get_facial_midline(landmarks)
                                                               └─ yaw = detector.calculate_head_pose(landmarks)
                                                                  RETURNS: debug_frame
```

---

## Model Loading Dependency Chain

```
face_splitter.py.__init__()
    │
    ├─ Create PyFaceAU68LandmarkDetector
    │  │
    │  ├─ Load ONNXRetinaFaceDetector
    │  │  └─ Model: weights/retinaface_mobilenet025_coreml.onnx
    │  │
    │  ├─ Load CunjianPFLDDetector
    │  │  └─ Model: weights/pfld_cunjian.onnx
    │  │
    │  ├─ Load TargetedCLNFRefiner
    │  │  └─ Model: weights/svr_patches_0.25_general.txt
    │  │
    │  └─ Load CLNFDetector (fallback, on-demand)
    │     └─ Directory: weights/clnf/
    │
    ├─ Create FaceMirror(landmark_detector)
    │  └─ Stores reference to detector
    │
    └─ Create VideoProcessor(landmark_detector, face_mirror)
       └─ Stores references to both
```

---

## Configuration Dependencies

```
main.py
    │
    ├─ Import config
    │  └─ Apply settings (NUM_THREADS, BATCH_SIZE, OMP_NUM_THREADS, etc.)
    │
    ├─ Import config_paths
    │  └─ Find model directory (weights/)
    │
    └─ Import StableFaceSplitter
       └─ Reads: config.NUM_THREADS, model paths from config_paths
```

---

## Landmark Quality Feedback Loop

```
get_face_mesh()
    │
    ├─ Detect/Refine landmarks
    │
    ├─ check_landmark_quality()
    │  │
    │  └─ If POOR:
    │     │
    │     ├─ Add to poor_quality_frames[]
    │     │
    │     └─ If clnf_fallback available:
    │        │
    │        ├─ CLNFDetector.refine()
    │        │  (slower, ~200-500ms)
    │        │
    │        └─ RETURN: refined landmarks
    │
    └─ If GOOD:
       └─ RETURN: original/refined landmarks

Temporal History:
    │
    ├─ landmarks_history[] (max 5 frames)
    ├─ glabella_history[] (max 5 frames)
    ├─ chin_history[] (max 5 frames)
    └─ yaw_history[] (max 5 frames)
       │
       └─ Used for smoothing & stability calculation
```

---

## Error Handling Paths

```
get_face_mesh(frame)
    │
    ├─ Face detection fails
    │  └─ Return (None, None)
    │     └─ Handled in _process_frame_batch(): use frame.copy()
    │
    ├─ Landmark detection fails
    │  ├─ If last_landmarks exists:
    │  │  └─ Return last_landmarks.copy()
    │  └─ Else:
    │     └─ Return (None, None)
    │
    └─ CLNF refinement fails
       └─ Continue with unrefined landmarks
          (quality check may catch it)

All failures caught in:
    _process_frame_batch() [Lines 236-238]
    └─ Return (frame_index, frame.copy(), frame.copy(), frame.copy())
```

---

## Critical Sections for Modification

### 1. Landmark Detection
- **File**: `pyfaceau_detector.py`
- **Method**: `get_face_mesh()` [Lines 270-426]
- **Impact**: Changes how landmarks are extracted

### 2. Midline Calculation
- **File**: `pyfaceau_detector.py`
- **Method**: `get_facial_midline()` [Lines 428-471]
- **Impact**: Changes how face is mirrored

### 3. Frame Processing
- **File**: `video_processor.py`
- **Method**: `_process_frame_batch()` [Lines 203-238]
- **Impact**: Changes how frames are processed

### 4. Mirroring Algorithm
- **File**: `face_mirror.py`
- **Method**: `create_mirrored_faces()` [Lines 12-106]
- **Impact**: Changes mirror output quality

---

**Directory**: `/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/`
**Last Updated**: 2025-11-04
