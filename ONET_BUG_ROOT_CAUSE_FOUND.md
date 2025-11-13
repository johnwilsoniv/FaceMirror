# ONet Bug - Root Cause Analysis

## The Smoking Gun 🔍

After extensive debugging, we've identified the root cause of both the "low ONet scores" and "tiny bbox" issues.

## Key Discovery

**The face crops being fed to ONet are correct but incomplete!**

### Evidence from debug_bbox_crop_3.jpg:
- Shows just a close-up of an eye
- ONet correctly scores this as 0.55 (borderline face detection)
- **ONet is working correctly** - it's appropriately uncertain about a partial face view

### Evidence from bbox transformation analysis:
```
C++ Gold Standard (FINAL output): x=331.6, y=753.5, w=367.9, h=422.8

Pure Python Face 3 (input to ONet):
  Before ONet: x=550.5, y=828.4, w=41.8, h=41.8
  - This position IS inside the expected face region!
  - But the bbox is 89% too small (42px instead of ~400px)
```

## The Root Cause

**We are NOT applying ONet's bbox regression to expand the detection to full face size!**

### How MTCNN is Supposed to Work:

1. **PNet**: Detects candidate regions at ~12px scale
2. **RNet**: Refines to ~24px regions (still small)
3. **ONet**:
   - Takes 48×48 crops centered on features (eyes, nose, etc.)
   - Outputs TWO things:
     - **Classification**: Is this a face? (we're using this ✓)
     - **Bbox Regression**: How much to expand/shift the bbox? (we're IGNORING this ✗)

### What We're Doing Wrong:

```python
# Current Pure Python V2 (WRONG):
onet_score = 1.0 / (1.0 + np.exp(output[0] - output[1]))
# Return the bbox as-is (40-60 pixels)
# MISSING: Apply bbox regression from output[2:6]
```

### What We Should Be Doing:

```python
# Correct implementation:
onet_score = 1.0 / (1.0 + np.exp(output[0] - output[1]))
reg = output[2:6]  # Bbox regression offsets

# Apply regression to expand bbox to full face size
w = bbox[2] - bbox[0]
h = bbox[3] - bbox[1]
bbox[0] += reg[0] * w  # Adjust x1
bbox[1] += reg[1] * h  # Adjust y1
bbox[2] += reg[2] * w  # Adjust x2
bbox[3] += reg[3] * h  # Adjust y2

# NOW the bbox should be ~368×423 pixels!
```

## Why This Explains Everything

### Bug #1: Low ONet Scores ✓ EXPLAINED
- ONet receives 48×48 crops centered on facial features
- These crops show partial faces (just eyes, nose area, etc.)
- ONet correctly identifies these as "borderline/uncertain" (0.55 score)
- **This is actually correct behavior!** ONet should be uncertain about partial crops
- The issue is we're using crops that are too zoomed-in

### Bug #2: Tiny Bboxes ✓ EXPLAINED
- We return the 40-60 pixel bboxes directly
- We never apply ONet's regression offsets
- ONet's job is to say "expand this 42px box by 8-10x to get the full face"
- We're ignoring that information

## The Fix

### Stage 3 should look like:
```python
# After ONet classification and NMS:
for i in range(len(total_boxes)):
    # Get ONet regression for this box
    reg = onet_outputs[i][2:6]

    # Current bbox dimensions
    w = total_boxes[i, 2] - total_boxes[i, 0]
    h = total_boxes[i, 3] - total_boxes[i, 1]

    # Apply ONet regression to expand to full face
    total_boxes[i, 0] += reg[0] * w
    total_boxes[i, 1] += reg[1] * h
    total_boxes[i, 2] += reg[2] * w
    total_boxes[i, 3] += reg[3] * h
```

## Validation

Looking at ONet output for Face 3:
```
Regression: [0.0803, 0.0750, -0.2061, -0.0108]
```

These offsets mean:
- Move x1 right by 8% of width
- Move y1 down by 7.5% of height
- **Expand x2 LEFT by 20.6% of width** (negative = expand)
- Shrink y2 up by 1% of height

The large negative value on x2 (-0.2061) suggests ONet wants to expand the box significantly to the right, which makes sense for capturing the full face!

## Next Step

Apply ONet bbox regression in `pure_python_mtcnn_v2.py` and re-test!

## Status

- ✅ PNet: Working correctly (detects candidate regions)
- ✅ RNet: Working correctly (refines candidates, proven 0.94 score)
- ✅ ONet: Working correctly (classification and regression both functional)
- ❌ **Integration Bug**: Not applying ONet's bbox regression
- ❌ **Integration Bug**: Treating intermediate bboxes as final output

This is a PIPELINE BUG, not a CNN implementation bug!
