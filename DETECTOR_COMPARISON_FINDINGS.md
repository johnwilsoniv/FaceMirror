# Face Detector Comparison: Key Findings

## TL;DR

**OpenFace doesn't use MTCNN's 5-point landmarks.** It only uses the bbox and then CLNF refines from scratch to 68 points.

The meaningful comparison is bbox quality, not raw landmark quality.

## What We Discovered

### OpenFace Pipeline (Simplified)
```
Video Frame
  ↓
MTCNN Detector
  ├─ Outputs: bbox + 5 landmarks
  ├─ Uses: bbox only
  └─ Discards: 5 landmarks ❌
  ↓
CLNF Refinement (starting from bbox only)
  ↓
68-point landmarks
```

### Our PyFaceAU Pipeline
```
Video Frame
  ↓
RetinaFace Detector
  ├─ Outputs: bbox + 5 landmarks
  └─ Uses: bbox + 5 landmarks ✓
  ↓
PFLD (uses 5 landmarks for alignment)
  ↓
68-point landmarks
  ↓
SVR CLNF Refinement
  ↓
Final 68-point landmarks
```

## Technical Details

### Why MTCNN Landmarks Appeared "Wrong"

When I modified OpenFace to output MTCNN's raw 5-point landmarks, they appeared misplaced because:

1. **Extracted at wrong stage**: Landmarks were captured from `proposal_boxes_all[k]` (the intermediate proposal)
2. **Wrong bbox association**: Bboxes undergo multiple corrections after landmark extraction:
   - `apply_correction()` - applies ONet bbox regression
   - Empirical adjustments for "tight fit around facial landmarks" (lines 921-924)
   - Non-maximum suppression (which may drop some proposals)
3. **Index mismatch**: The k-th proposal's landmarks don't necessarily match `o_regions[0]` (the final selected face)

### What OpenFace Actually Does

From `FaceDetectorMTCNN.cpp`:
```cpp
// ONet outputs 16 values:
// [0-1]: Face/non-face probability
// [2-5]: Bbox corrections
// [6-15]: 5 facial landmarks (x,y pairs) ← Never used!
```

OpenFace extracts the bbox corrections but **never uses the 5 landmarks**. The bbox is then passed directly to CLNF which estimates 68 landmarks from scratch.

## The Real Comparison

### Bbox Quality

**MTCNN (OpenFace):**
- Tighter bbox focused on core facial features
- Multiple correction stages for "tight fit"
- More conservative sizing

**RetinaFace (PyFaceAU):**
- Larger bbox with generous padding
- Provides more context for landmark detection
- May be better for facial paralysis (asymmetric faces)

### Landmark Estimation Approach

**OpenFace:**
- CLNF alone (estimating 68 points from bbox only)
- No alignment guidance from detector

**PyFaceAU:**
- PFLD uses RetinaFace's 5 landmarks for alignment
- Directly predicts 68 points
- SVR CLNF refinement for final adjustment

## Conclusion

The comparison of "raw MTCNN vs raw RetinaFace landmarks" is **not meaningful** because:

1. OpenFace discards MTCNN's 5-point landmarks
2. The only shared starting point is the bbox
3. Everything after the bbox is a different algorithmic approach

The meaningful comparisons are:
1. **Bbox quality**: MTCNN vs RetinaFace bboxes
2. **Final 68-point quality**: OpenFace CLNF vs PyFaceAU (PFLD + CLNF)

## Next Steps

To properly compare the pipelines, we should:
1. Compare final 68-point landmark outputs (already done in earlier tests)
2. Compare AU extraction accuracy (the actual end goal)
3. Focus on failure modes (which pipeline handles challenging cases better)

The earlier testing showed PyFaceAU with MTCNN fallback achieved 100% detection rate, while C++ OpenFace crashed on rotated videos due to OpenCV 4.12.0 bugs. That remains the most important practical finding.
