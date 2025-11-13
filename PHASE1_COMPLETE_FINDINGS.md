# Phase 1 Complete: RNet Scoring Investigation

## Executive Summary

**ROOT CAUSE IDENTIFIED**: RNet is scoring crops backwards - rejecting the actual 452×451px face (score=0.63) while accepting 40px feature crops (score >0.8).

## Visual Evidence

See files:
- `rnet_crops_inspection/SUMMARY_all_crops_by_score.jpg` - Grid of all 161 RNet inputs sorted by score
- `debug_rnet/resized_24x24.jpg` - The actual face crop resized to 24×24 (scored 0.63)
- `debug_rnet/small_crop1_24x24.jpg` - Small feature crop resized to 24×24 (scored 0.90)

## Detailed Findings

### The Correct Face Crop EXISTS

**Crop #6 from PNet:**
- Location: (330, 672) - Only 81px from gold standard y-coordinate!
- Size: 452×451px (after regression + squaring)
- Visual: Clear, well-framed face with visible features
- **RNet score: 0.6314 < 0.7 threshold → REJECTED**

**Logit breakdown:**
```
logit_not_face: -0.2692
logit_face:      0.2689
difference:     -0.5382
Final score:     0.6314
```

The network barely favors "face" over "not-face".

### Small Feature Crops Score Higher

**Crop #1 from PNet:**
- Location: (888, 783) - Does NOT overlap with actual face
- Size: 40×41px (tight crop of a feature)
- Visual: Partial feature/eye with blue background
- **RNet score: 0.8994 > 0.7 threshold → ACCEPTED**

**Logit breakdown:**
```
logit_not_face: -1.0957
logit_face:      1.0946
difference:      2.1903
Final score:     0.8994
```

The network strongly favors "face".

## Statistics Across All 161 Crops

```
Total crops tested: 161
Score range: [0.0004, 0.8654]
Mean score: 0.0998

Score distribution:
  >= 0.9: 0 crops (0.0%)
  >= 0.8: 2 crops (1.2%)
  >= 0.7: 3 crops (1.9%)  ← These pass to ONet
  >= 0.6: 6 crops (3.7%)
  >= 0.5: 7 crops (4.3%)

Size range: [35, 609] px
Mean size: 92.8 px
```

**The 3 crops that pass RNet (score >0.7):**
1. 40×41px at (888, 783) - score 0.865 - NOT at face location
2. 40×40px at (553, 829) - score 0.822 - 1% IoU with face (feature)
3. 112×112px at (389, 1581) - score 0.729 - NOT at face location

## Why This is Backwards

Previous standalone RNet test achieved **0.94 score on a real face crop**. But now:
- Real face (452×451): score 0.63
- Small features (40px): score 0.86-0.90

Something is different about these crops vs the standalone test.

## Hypotheses for Why RNet Scores Backwards

### Hypothesis A: Resize Artifacts
- 452×451 → 24×24 is a 18.8x downscale
- 40×41 → 24×24 is only 1.7x downscale
- Different resize ratios create different patterns
- Maybe RNet was trained on tighter crops?

### Hypothesis B: Crop Positioning
- The 452×451 box might include too much background
- It's positioned at (330, 672) but gold face is at (331, 753)
- 81px vertical offset - is this cutting off the face?
- Need to visualize the ACTUAL 452×451 crop in original image

### Hypothesis C: Squaring Operation Bug
- The box was "squared" before RNet
- Maybe squaring is adding too much background?
- Original PNet box before squaring might have been better

### Hypothesis D: Missing Rectify After Regression
- C++ applies rectify() after PNet regression (line 1021)
- Python applies _square_bbox() which might differ
- Need to compare C++ rectify() vs Python _square_bbox()

### Hypothesis E: RNet Preprocessing
- Standalone test used correct preprocessing
- But maybe batch processing does something different?
- Check if resize method (INTER_LINEAR) matters

## Previous Misconceptions Corrected

### ❌ Original Diagnosis: "Bbox generation creates boxes 3x too small"
**WRONG** - PNet generates boxes of all sizes including the correct 452px face box.

### ❌ "Missing PNet regression causes tiny boxes"
**PARTIALLY WRONG** - We added regression, but boxes are still scored incorrectly.

### ✅ Actual Problem: "RNet scores large face crops too low"
**CORRECT** - The large face crop exists but RNet rejects it while accepting small feature crops.

## Next Investigation Priority

### Option 1: Check Crop Positioning (HIGH PRIORITY)
Visualize the 452×451 crop overlaid on the original image. Is it actually framing the face correctly, or is there a 81px offset error?

### Option 2: Compare Squaring Logic (HIGH PRIORITY)
C++ rectify() (line 793-813) vs Python _square_bbox():
```python
# Python _square_bbox:
max_side = np.maximum(h, w)
square_bboxes[:, 0] = bboxes[:, 0] + w * 0.5 - max_side * 0.5
square_bboxes[:, 1] = bboxes[:, 1] + h * 0.5 - max_side * 0.5
```

Does this match C++?

### Option 3: Test Different Resize Methods (MEDIUM)
Try cv2.INTER_AREA, INTER_CUBIC instead of INTER_LINEAR

### Option 4: Check RNet Weights (LOW)
We already proved RNet works standalone. Unlikely to be weights.

## Files Created

- `phase1_inspect_rnet_crops.py` - Extracts and scores all 161 RNet inputs
- `rnet_crops_inspection/` - Directory with all crops and visual summary
- `phase1b_debug_rnet_scoring.py` - Detailed analysis of specific crops
- `debug_rnet/` - Directory with problematic crop visualization
- `PHASE1_COMPLETE_FINDINGS.md` - This document

## Recommended Next Step

**Visualize the 452×451 crop on the original image** to confirm it's actually framing the face correctly. If there's an 81px offset bug, that would explain why RNet rejects it!

Command to run:
```python
# Draw the bbox on original image
img_vis = img.copy()
cv2.rectangle(img_vis, (330, 672), (782, 1123), (0, 255, 0), 3)  # Large face
cv2.rectangle(img_vis, (331, 753), (699, 1176), (0, 0, 255), 2)   # Gold standard
cv2.imwrite('bbox_visualization.jpg', img_vis)
```

This will show if the 452×451 box is properly aligned with the actual face.
