# MTCNN vs RetinaFace Raw Detector Comparison

## Overview

This document compares the raw face detection outputs from:
- **OpenFace MTCNN** (used in OpenFace 2.2)
- **RetinaFace** (used in our PyFaceAU pipeline)

Both detectors produce:
- Bounding box (bbox)
- 5 facial landmarks: left eye, right eye, nose, left mouth corner, right mouth corner

## Test Videos

Tested on 4 videos from the patient cohort:
- IMG_0441_source.MOV (Paralysis)
- IMG_0452_source.MOV (Normal)
- IMG_0861_source.MOV (Paralysis)
- IMG_0942_source.MOV (Normal)

## Key Findings

### 1. Bbox Sizing Strategy

**MTCNN (Green bbox):**
- Produces **tighter** bboxes focused on core facial features
- Typically crops closer to the actual face boundaries
- More conservative bbox sizing

**RetinaFace (Red bbox):**
- Produces **larger** bboxes with more vertical/horizontal padding
- Includes more context around the face
- More generous bbox sizing (often extends well beyond face)

### 2. Landmark Placement

Both detectors place 5 landmarks at:
1. Left eye center
2. Right eye center
3. Nose tip
4. Left mouth corner
5. Right mouth corner

**Observation:** Landmark placement is remarkably similar between both detectors. Both accurately identify the 5 key facial points.

### 3. Detection Reliability

**MTCNN:**
- Successfully detected all 4 test videos
- Consistent detection across normal and paralysis patients
- No failures observed

**RetinaFace:**
- Successfully detected all 4 test videos
- Consistent detection across normal and paralysis patients
- No failures observed

## Visualizations

Side-by-side comparisons saved in `test_output/`:
- `IMG_0441_source_mtcnn_vs_retinaface.jpg`
- `IMG_0452_source_mtcnn_vs_retinaface.jpg`
- `IMG_0861_source_mtcnn_vs_retinaface.jpg`
- `IMG_0942_source_mtcnn_vs_retinaface.jpg`

Each image shows:
- **Left:** MTCNN detection (green bbox + 5 green landmarks)
- **Right:** RetinaFace detection (red bbox + 5 red landmarks)

## Implications for Pipeline

### OpenFace Pipeline
1. MTCNN detects face → 5 landmarks
2. CLNF refines landmarks → 68 points
3. AU extraction

### PyFaceAU Pipeline
1. RetinaFace detects face → 5 landmarks
2. PFLD expands landmarks → 68 points
3. SVR CLNF refines landmarks
4. AU extraction

**Key Difference:** OpenFace uses CLNF to *refine and expand* 5 points to 68, while PyFaceAU uses PFLD to directly predict 68 points, then refines with CLNF.

## Conclusion

Both MTCNN and RetinaFace are highly capable face detectors with similar landmark accuracy. The main difference is bbox sizing strategy:
- **MTCNN:** Tighter, more conservative
- **RetinaFace:** Larger, more generous padding

For our use case (facial paralysis analysis), RetinaFace's larger bbox may be advantageous as it provides more context for subsequent landmark detection, especially when facial features may be asymmetric or partially occluded.

## Technical Details

**MTCNN Output Format:**
```
bbox: [x, y, width, height]
landmarks: [lm1_x, lm1_y, lm2_x, lm2_y, ..., lm5_x, lm5_y] (normalized 0-1 within bbox)
```

**RetinaFace Output Format:**
```
detection: [x1, y1, x2, y2, conf, lm1_x, lm1_y, lm2_x, lm2_y, ..., lm5_x, lm5_y]
```

## Files Generated

- `compare_mtcnn_vs_retinaface.py` - Comparison script
- `test_output/*_mtcnn_vs_retinaface.jpg` - Side-by-side visualizations
- `/tmp/mtcnn_debug.csv` - MTCNN detection data (per frame)
- Modified OpenFace source: `FaceDetectorMTCNN.cpp` (outputs debug data)
