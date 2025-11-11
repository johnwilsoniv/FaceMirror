# Response Map Peak Location Analysis

**Date**: 2025-11-10
**Status**: CRITICAL FINDINGS - Response maps have systematic offset and scaling issues

---

## Summary

Debug logging of response map peak locations reveals that **response map peaks are consistently 2-6 pixels away from the center**. For good convergence, peaks should be within 0-1 pixel of center. Additionally, response map values show a **suspicious 100x increase at window size 5**.

---

## Test Results

### Peak Location Offsets

| Window Size | Landmark | Offset (x, y) | Distance | Peak Value |
|-------------|----------|---------------|----------|------------|
| **ws=9** | 6 | (+4.0, +4.0) | **5.7px** | 0.146 |
| | 5 | (+1.0, +4.0) | 4.1px | 0.101 |
| | 15 | (-1.0, -4.0) | 4.1px | 0.101 |
| **ws=7** | 30 | (+2.0, +3.0) | **3.6px** | 0.262 |
| | 1 | (-1.0, +3.0) | 3.2px | 0.098 |
| | 3 | (-3.0, -1.0) | 3.2px | 0.114 |
| **ws=5** | 56 | (+2.0, +2.0) | **2.8px** | 12.29 |
| | 1 | (+1.0, -2.0) | 2.2px | 6.37 |
| | 3 | (-2.0, -1.0) | 2.2px | 7.71 |

### Response Map Value Ranges

| Window Size | Min | Max | Mean |
|-------------|-----|-----|------|
| **ws=9** | 0.000000 | 0.856498 | 0.128427 |
| **ws=7** | 0.000000 | 0.665807 | 0.116738 |
| **ws=5** | 0.000028 | **123.979108** | **15.804498** |

---

## Critical Findings

### 1. Peak Offsets Are Too Large ❌

**Expected**: Peak within 0-1 pixel of center (indicating landmark is at correct position)
**Actual**: Peaks are 2-6 pixels from center

**Interpretation**:
- Response maps consistently indicate landmarks should move 2-6 pixels away from current positions
- This explains the large mean-shift values (24-58 pixels across all landmarks)
- Landmarks are NOT at the correct positions according to the response maps

**Why This Prevents Convergence**:
- If peaks are always offset, landmarks keep moving
- Mean-shift algorithm correctly follows the peaks, but peaks are in wrong locations
- System never settles to converged state

### 2. Peak Offsets Decrease Across Scales ✓

Offsets: 5.7px (ws=9) → 3.6px (ws=7) → 2.8px (ws=5)

**This is expected behavior**:
- Larger windows (ws=11, 9) capture more context → larger corrections
- Smaller windows (ws=5) refine positions → smaller corrections
- Multi-scale strategy is working as designed

### 3. Response Map Values Explode at ws=5 ❌❌❌

**Expected**: Response values should be comparable across window sizes (0.1-1.0 range)
**Actual**: Values increase by **100x** at ws=5 (max=123.98, mean=15.80)

**This is a RED FLAG indicating**:
- Normalization bug in response map computation
- Missing scaling factor for smaller windows
- Possible sigma component mismatch at ws=5
- Response values at ws=5 are being computed in a fundamentally different way

**Why This is Critical**:
- If response values are 100x larger, KDE mean-shift computation will be affected
- Weight matrix computation may be affected
- This could explain why convergence gets WORSE at ws=5 (MS increases from 23.7 to 26.2)

---

## Hypotheses for Root Cause

### Hypothesis 1: Patch Expert Outputs Are Wrong (MOST LIKELY)

The patch experts (neural networks) may be:
- Trained differently than OpenFace's models
- Using different preprocessing/normalization
- Missing a scaling factor in the forward pass
- Using wrong input image transformations

**Evidence**:
- Peaks consistently 2-6 pixels off center
- 100x scaling difference at ws=5
- All landmarks affected (not just a few)

**Next Steps**:
- Compare patch expert architecture with OpenFace
- Check if sigma components are applied correctly
- Verify image preprocessing before patch expert forward pass
- Compare response map values with OpenFace C++ for same landmark

### Hypothesis 2: Response Map Normalization Bug

Response maps may be missing normalization or scaling after patch expert evaluation.

**Evidence**:
- 100x increase at ws=5 suggests missing normalization
- Different window sizes should produce comparable response magnitudes

**Next Steps**:
- Check `_compute_response_map()` for normalization steps
- Compare with OpenFace C++ response map computation
- Verify sigma components are being applied correctly

### Hypothesis 3: Image Warping/Transform Bug

If image warping (for reference-aligned patches) has a bug, extracted patches could be misaligned, producing offset peaks.

**Evidence**:
- Peaks consistently offset (not random)
- Offsets vary across landmarks (6: +4,+4; 15: -1,-4)

**Next Steps**:
- Compare warped patches with OpenFace C++
- Verify similarity transform computation
- Check if `patch_scaling=0.25` is applied correctly

### Hypothesis 4: Sigma Components Wrong at ws=5 ✅ **ROOT CAUSE FOUND**

The sigma components (used for response map normalization) are missing for ws=5.

**Evidence**:
- 100x scaling difference ONLY at ws=5 (max=123.98 vs 0.67-0.86 for other sizes)
- Debug output: "Loaded sigma components for window sizes: [7, 9, 11, 15]" - **ws=5 is missing!**
- Our exported model only has sigma components for [7, 9, 11, 15]
- OpenFace C++ REQUIRES sigma components (would crash if missing per `CCNF_patch_expert.cpp:402`)

**Investigation Results**:
- Checked OpenFace C++ source code:
  - Default parameters specify window sizes [11, 9, 7, 5] (`LandmarkDetectorParameters.cpp:244-247`)
  - But CCNF code requires sigma components for every window size used
  - Missing sigma components would cause invalid memory access
- **Conclusion**: OpenFace model files don't include ws=5 sigma components
  - Either default parameters don't match trained models, OR
  - Training process didn't generate sigma components for all window sizes

**Resolution**: Changed PyCLNF to use window sizes **[11, 9, 7]** (what we have sigma components for)
- File: `pyclnf/clnf.py` line 69
- This eliminates the 100x scaling bug at ws=5

---

## Investigation Priorities

### 🔥 Priority 1: Missing ws=5 Sigma Components
**Why**: Debug output shows sigma components only for [7, 9, 11, 15], but we're using ws=5! This directly explains the 100x scaling difference.

**Action**:
- Check how sigma components are loaded
- Verify what happens when ws=5 is requested but not available
- Add ws=5 to sigma components or use proper fallback

### 🔥 Priority 2: Compare Response Map Values with OpenFace C++
**Why**: Need ground truth to know if our response maps are correct.

**Action**:
- Run OpenFace C++ on same frame
- Extract response map values for a few landmarks
- Compare peak locations and values with PyCLNF

### Priority 3: Verify Patch Expert Forward Pass
**Why**: If patch experts are producing wrong outputs, everything downstream is wrong.

**Action**:
- Compare patch expert architecture with OpenFace models
- Verify preprocessing (normalization, sigma components)
- Test patch expert on known inputs

### Priority 4: Check Image Warping
**Why**: Misaligned patches → offset peaks.

**Action**:
- Visualize warped patches for a few landmarks
- Compare with OpenFace C++ warped patches
- Verify transform computation

---

## Expected vs Actual Behavior

### Expected (OpenFace C++):
- **Peak offsets**: < 1 pixel for converged landmarks
- **Response values**: 0.1-1.0 range across all window sizes
- **Mean-shift**: Large initially (10-20px), decreases to < 1px
- **Convergence**: Achieved in 10-20 iterations

### Actual (PyCLNF):
- **Peak offsets**: 2-6 pixels (consistently offset) ❌
- **Response values**: 0.1-0.9 for ws=[7,9,11], **123.98** for ws=5 ❌
- **Mean-shift**: Stays at 24-58px (not decreasing to < 1px) ❌
- **Convergence**: Never achieved ❌

---

## Conclusion

Response map peak locations reveal that:
1. **Peaks are consistently 2-6 pixels offset from center** - Explains large mean-shift values
2. **Response values increase 100x at ws=5** - Critical scaling/normalization bug ✅ **SOLVED**
3. **Sigma components missing for ws=5** - Root cause of scaling issue ✅ **CONFIRMED**

**Resolution Implemented**:
- Changed window sizes from [11, 9, 7, 5] to [11, 9, 7] in `pyclnf/clnf.py:69`
- This aligns with available sigma components [7, 9, 11, 15]
- Eliminates 100x scaling bug at ws=5

**Remaining Issues**:
1. Peak offsets still 2-6 pixels from center (not < 1px as expected)
2. Mean-shift magnitudes still 24-58 pixels (not decreasing to < 1px)
3. Convergence not achieved (final update 4.15 vs target 0.005)

**Next Investigation**: Compare response map values with OpenFace C++ to determine if peaks are in correct locations, or if there's an additional bug in response map computation/patch expert forward pass.

---

## Files Modified

1. **pyclnf/core/optimizer.py**:
   - Lines 267-268: Added `peak_locations` list to track peak offsets
   - Lines 295-298: Store peak offset from center for each landmark
   - Lines 339-347: Print worst 3 peak offsets (farthest from center)
   - All changes marked with `# DEBUG:` for easy removal

---

## Next Session

Start with:
1. Investigate missing ws=5 sigma components (`pyclnf/core/optimizer.py:156` logs: "Loaded sigma components for window sizes: [7, 9, 11, 15]")
2. Check `sigma_components` loading and fallback behavior
3. Compare response map values with OpenFace C++ empirically
