# MTCNN Investigation Summary

## Problem Statement
Pure Python MTCNN detects tiny 30-90px boxes instead of correct 368×423px face.

## Investigation Journey

### Phase 1: RNet Scoring Investigation
**Finding**: RNet rejects large face crop (452×451px, score 0.63) while accepting tiny feature crops (40px, score >0.8)

**Initial hypothesis**: RNet is scoring backwards
**Reality**: RNet is correctly rejecting POORLY-FRAMED crops

### Phase 2: Box Transformation Tracing
**Finding**: PNet regression moves boxes in WRONG direction!

**Evidence from Box #103**:
- Raw PNet: (333, 703) - only 50px from gold y=753 ✅
- After regression: (364, 672) - moved UP 31px, now 81px away ❌
- Regression value dy1=-0.0696 (negative = moves UP)

**Verified**:
- ✅ Regression formula matches C++
- ✅ Channel ordering matches C++
- ✅ No sign flips in C++ code

### Phase 3: Experimental Tests

**Test A: Negate all regression values**
- Result: Only 1 box passes RNet, rejected by ONet
- Conclusion: Simple negation doesn't fix it

**Test B: Skip PNet regression entirely**
- Result: Gets 1 detection but it's a tiny 30×30px box on nose
- Conclusion: Raw boxes aren't better either!

## Current Hypothesis

The problem is **NOT just regression** - it's that **Pure Python CNN PNet produces DIFFERENT outputs than C++ PNet**.

Evidence:
1. Large boxes score poorly even WITHOUT regression
2. Small boxes consistently score well
3. Regression makes things worse but isn't the root cause
4. The 444×444px raw box from Phase 2 trace doesn't appear in Phase 3 test

## Next Steps

**Priority 1**: Compare raw PNet outputs (probabilities AND regression) between C++ and Python for the EXACT same input patch

**Priority 2**: Check if Pure Python CNN has bugs in:
- Weight loading (especially for FC layer that outputs 6 channels)
- PReLU implementation
- Coordinate transformations

**Priority 3**: Consider using ONNX PNet instead of Pure Python CNN to isolate whether the issue is:
- Pure Python CNN implementation bug
- Or MTCNN integration bug

## Files Created

### Investigation Scripts
- `phase1_inspect_rnet_crops.py` - Extracts all 161 RNet inputs
- `phase1b_debug_rnet_scoring.py` - Compares specific crop scoring
- `trace_box_transformations.py` - Traces box through all stages
- `test_negated_regression.py` - Tests negating regression values
- `test_without_pnet_regression.py` - Tests skipping regression

### Documentation
- `PHASE1_COMPLETE_FINDINGS.md` - RNet scoring investigation
- `PHASE2_ROOT_CAUSE_FOUND.md` - Regression direction problem
- `DIAGNOSIS_SUMMARY.md` - This file

### Visualizations
- `bbox_position_visualization.jpg` - Shows 81px vertical offset
- `box_transformation_stages.jpg` - Shows regression moves box UP
- `test_without_pnet_regression_result.jpg` - Shows tiny 30px detection

## Key Insight

**The raw PNet box at (333, 703) is actually pretty good** - only 50px from gold standard. But **Pure Python CNN's regression makes it worse** by moving it in the wrong direction.

However, when we skip regression entirely, we don't get that large box at all - we only get a tiny 30px box! This suggests Pure Python CNN is producing fundamentally different outputs than C++ PNet, not just wrong regression values.

The investigation needs to shift from "fix the regression" to "why does Pure Python CNN produce different outputs?"
