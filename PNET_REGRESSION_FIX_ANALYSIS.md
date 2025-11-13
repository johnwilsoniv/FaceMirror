# PNet Regression Fix - Analysis and Findings

## What I Fixed

Added the missing PNet bbox regression application step in `pure_python_mtcnn_v2.py`:

```python
# Line 192: Added after cross-scale NMS
total_boxes = self._apply_bbox_regression(total_boxes)
```

This matches C++ line 1009: `apply_correction(proposal_boxes_all, proposal_corrections_all, false);`

## The Regression Implementation

```python
def _apply_bbox_regression(self, bboxes):
    """Apply bbox regression corrections (matching C++ apply_correction)."""
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i, 0:4]
        dx1, dy1, dx2, dy2 = bboxes[i, 5:9]

        w = x2 - x1
        h = y2 - y1

        # Apply C++ formula (lines 828-832)
        new_x1 = x1 + dx1 * w
        new_y1 = y1 + dy1 * h
        new_x2 = x2 + dx2 * w
        new_y2 = y2 + dy2 * h

        result[i, 0:4] = [new_x1, new_y1, new_x2, new_y2]
```

## What the Regression Actually Does

**CRITICAL FINDING**: The regression values are SMALL FRACTIONAL ADJUSTMENTS, not expansions!

### Example from Test Run:

**Before regression:**
- Box 1: (563.0, 1437.0) 40.0×40.0, reg=[0.162, 0.007, 0.027, 0.054]

**After regression:**
- Box 1: (569.5, 1437.3) 34.6×41.9

**What happened:**
- dx1 = 0.162 → shifts x1 by 0.162 * 40 = 6.5px
- dy1 = 0.007 → shifts y1 by 0.007 * 40 = 0.3px
- dx2 = 0.027 → adjusts x2, changing width by ~6px
- dy2 = 0.054 → adjusts y2, changing height by ~2px

The regression makes **small refinements**, not dramatic expansions from 40px to 400px!

## Test Results After Fix

```
PNet: 161 boxes after cross-scale NMS
PNet: Applied bbox regression to 161 boxes

RNet: Tested 161 faces
  Score range: [0.0004, 0.8654]
  Scores > 0.7: 3

Final detections:
  Face 1: 30.1 × 29.2 px (IoU: 0.6% with C++ gold standard)
  Face 2: 88.4 × 89.8 px (IoU: 0.0%)
  Face 3: 36.4 × 33.8 px (IoU: 0.0%)
```

**Still detecting tiny boxes instead of the expected 368×423px face!**

## Re-evaluating the Previous Diagnosis

The previous diagnosis stated:

> "The bug is in the bbox generation formula that converts PNet probability map coordinates into image coordinates. The boxes being generated are roughly 1/3 the size they should be..."

### This was INCORRECT!

The boxes are the CORRECT size based on PNet's receptive field:
- PNet has a receptive field of 12×12 at each output pixel
- At scale m=0.3, a 12px receptive field corresponds to 40px in the original image
- This is exactly what we're seeing: 40px boxes at scale 0

## The Real Problem

If the regression fix doesn't solve it, and the box sizes are correct, then the issue must be:

### Hypothesis 1: Scale Selection
- Python and C++ might be detecting faces at DIFFERENT scales
- C++ diagnosis showed 315px boxes from scale 0.0381 (designed for 1050px faces)
- But the actual face is only 368px - why is C++ using such a large scale?
- Maybe the scale/threshold logic differs between Python and C++?

### Hypothesis 2: Probability Map Differences
- Python PNet generates high probabilities (0.99) at small scales
- Maybe C++ PNet generates DIFFERENT probabilities at DIFFERENT scales?
- The comprehensive investigation showed PNet outputs look correct
- But maybe there's a subtle difference causing different scale activation patterns?

### Hypothesis 3: Image Preprocessing
- The comprehensive investigation tested RGB/BGR swap, showed standard preprocessing is best
- But maybe there's another preprocessing issue (gamma, color space, etc.)?
- C++ and Python use identical formula: `(img - 127.5) * 0.0078125`

## Next Steps

1. **Compare C++ vs Python PNet outputs scale-by-scale**
   - Not just probability max/mean, but actual box locations and sizes
   - Which scales are generating the winning boxes in each implementation?

2. **Debug C++ bbox generation directly**
   - Why does C++ generate 315px boxes from a 1050px-face scale?
   - Is there additional logic we're missing?

3. **Check if the issue is in RNet/ONet, not PNet**
   - Python RNet accepts small boxes (40-80px) with high scores
   - Maybe there's a bug in how we're cropping/resizing for RNet?
   - Or RNet weights are different?

## Code Changes Made

### pure_python_mtcnn_v2.py

**Lines 190-209: Added PNet regression application**
```python
# Apply PNet bbox regression (matching C++ line 1009)
if debug:
    print(f"\nPNet: Before bbox regression:")
    # ... debug output ...

total_boxes = self._apply_bbox_regression(total_boxes)

if debug:
    print(f"\nPNet: After bbox regression:")
    # ... debug output ...
```

**Lines 458-500: Added _apply_bbox_regression() method**
```python
def _apply_bbox_regression(self, bboxes):
    """Apply bbox regression corrections to PNet boxes."""
    # ... implementation ...
```

## Status

✅ **FIX APPLIED**: PNet bbox regression now matches C++ implementation
❌ **ISSUE PERSISTS**: Still detecting 30-90px boxes instead of 368px face
🔍 **INVESTIGATION CONTINUES**: Root cause is NOT missing bbox regression

The diagnosis was partially correct that regression was missing, but this was NOT the root cause of tiny detections.
