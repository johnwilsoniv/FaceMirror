# Phase 2 Complete: Root Cause Identified

## Executive Summary

**ROOT CAUSE**: PNet regression is moving boxes in the WRONG DIRECTION. The box at y=703 needs to move DOWN to y=753 (+50px), but regression moves it UP to y=672 (-31px).

## Visual Evidence

See `box_transformation_stages.jpg`:
- **BLUE (Raw PNet)**: y=703 - reasonably centered on face ✅
- **MAGENTA (After regression)**: y=672 - moved UP by 31px, cutting off chin ❌
- **GREEN (After squaring)**: y=672 - kept the bad position ❌
- **RED (C++ Gold)**: y=753 - the correct position ✅

## Detailed Findings

### Box #103 Transformation Trace

**Raw PNet Output**:
```Position: (333, 703)
Size: 444×444px
Regression values: [dx1=0.0694, dy1=-0.0696, dx2=-0.0629, dy2=-0.0522]
Distance from gold: 50px (pretty good!)
```

**After PNet Regression**:
```
Position: (364, 672)
Size: 385×452px

Regression changes:
  y1: 703 → 672 (Δ-31px) ← MOVED UP
  y2: 1147 → 1124 (Δ-23px) ← MOVED UP

Calculation:
  new_y1 = 703 + (-0.0696) * 444 = 703 - 30.9 = 672.1 ✓
  new_y2 = 1147 + (-0.0522) * 444 = 1147 - 23.2 = 1123.8 ✓

Distance from gold: 81px (WORSE!)
```

**After Squaring**:
```
Position: (331, 672)
Size: 452×452px
Distance from gold: x=0px ✓, y=-81px ❌
```

### The Problem

**dy1 = -0.0696** is NEGATIVE, which moves the box UP (decreases y coordinate).

But the box needs to move DOWN (increase y coordinate) from y=703 to y=753!

### Code Verification

✅ **Formula matches C++**:
```cpp
// C++ (line 829-831):
float new_min_y = curr_box.y + corrections[i].y * curr_box.height;
float new_max_y = curr_box.y + curr_box.height + curr_box.height * corrections[i].height;
```

```python
# Python (line 505-507):
new_y1 = y1 + dy1 * h
new_y2 = y2 + dy2 * h  # = (y1 + h) + dy2 * h = y1 + h*(1 + dy2)
```

✅ **Channel ordering matches C++**:
```cpp
// C++ (line 946):
std::vector<cv::Mat_<float>> corrections_heatmap(pnet_out.begin() + 2, pnet_out.end());
// corrections[0] = pnet_out[2] = dx1
// corrections[1] = pnet_out[3] = dy1
// corrections[2] = pnet_out[4] = dx2
// corrections[3] = pnet_out[5] = dy2
```

```python
# Python (line 157, 419):
reg_map = output[:, :, 2:6]
dx1, dy1, dx2, dy2 = [reg_map[..., i] for i in range(4)]
```

## Remaining Hypotheses

Since the formulas and channel ordering are correct, the problem must be in **what values PNet outputs**:

### Hypothesis A: PNet Weight Loading Issue
- Pure Python CNN might be loading weights incorrectly
- Check if regression layers have wrong signs
- Compare actual PNet output values between Python and C++

### Hypothesis B: Coordinate System Flip
- Maybe PNet was trained with a different coordinate system?
- Y-axis might be flipped (origin at bottom vs top)?
- But this doesn't explain why C++ works with same weights...

### Hypothesis C: Missing Preprocessing
- Is there a preprocessing step for regression outputs that we're missing?
- Does C++ negate the regression values somewhere?

### Hypothesis D: Wrong Regression Layer in Network
- Did we accidentally load a different layer's outputs as the regression?
- Need to verify which layer produces channels 2-5

## Next Steps

### Priority 1: Compare Raw PNet Outputs
Create a script to compare the EXACT raw output values from PNet between Python and C++ for the same input location.

**Script**: `compare_pnet_regression_values.py`

### Priority 2: Search for Sign Flips in C++
Search the C++ code for any place where regression values are negated:
```bash
grep -n "\-corrections\|corrections.*\*.*-1" FaceDetectorMTCNN.cpp
```

### Priority 3: Verify Pure Python CNN Implementation
Check if Pure Python CNN has any bugs in:
- Weight loading (especially for FC layers that output regression)
- Layer ordering (are we reading the right layer as regression?)
- Any activation functions applied to regression outputs

## Files Created

- `trace_box_transformations.py` - Traces box through all transformation stages
- `box_transformation_stages.jpg` - Visual proof of wrong regression direction
- `PHASE2_ROOT_CAUSE_FOUND.md` - This document

## Key Insight

The raw PNet box at (333, 703) is actually pretty good - only 50px from gold standard! But regression **makes it worse** by moving it in the wrong direction. This suggests:

1. Either PNet regression values have wrong sign
2. Or there's a sign flip somewhere we haven't found yet

If we can fix the regression direction, detection should work correctly.
