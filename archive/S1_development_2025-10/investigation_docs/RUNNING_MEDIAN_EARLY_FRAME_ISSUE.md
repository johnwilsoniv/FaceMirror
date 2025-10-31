# Running Median Early Frame Issue

## Current Status
**Overall Correlation: r = 0.947** (71% of AUs at r > 0.90)

## The Problem

OpenFace 2.2 outputs **exactly 0.00 for frames 0-12** for dynamic AUs, then starts producing non-zero predictions from frame 13 onwards. Our Python implementation predicts non-zero values (0.17-0.44) for these same early frames.

### Evidence

From `analyze_early_frames.py`:

```
Frame  Update?  Median Mean    Median Std     Raw Pred     OF2.2        Error
-----------------------------------------------------------------------------------------------
0      False    0.107989       0.862132       0.1734       0.0000            +0.1734
1      True     0.109298       0.859392       0.1734       0.0000            +0.1734
2      False    0.109861       0.862037       0.1734       0.0000            +0.1734
3      True     0.095731       0.869473       0.1582       0.0000            +0.1582
4      False    0.095731       0.869473       0.3957       0.0000            +0.3957
5      True     0.109763       0.860332       0.3908       0.0000            +0.3908
6      False    0.109763       0.860332       0.4396       0.0000            +0.4396
7      True     0.104497       0.862127       0.3120       0.0000            +0.3120
8      False    0.104497       0.862127       0.3044       0.0000            +0.3044
9      True     0.110041       0.859771       0.2915       0.0000            +0.2915
10     False    0.110041       0.859771       0.2180       0.0000            +0.2180
11     True     0.106241       0.859423       0.2836       0.0000            +0.2836
12     False    0.106241       0.859423       0.2884       0.0000            +0.2884
13     True     0.110112       0.859366       0.1821       0.0200            +0.1621  ← FIRST NON-ZERO!
14     False    0.110112       0.859366       0.3266       0.0200            +0.3066
```

**Key Observations:**
- Frames 0-12: OpenFace = 0.00, Python = 0.17-0.44
- Frame 13+: Both produce non-zero predictions
- By frame 12: 6 histogram updates have occurred (frames 1,3,5,7,9,11)
- Static AUs work perfectly (r > 0.999) - issue ONLY affects dynamic models

## Hypotheses Tested

### ❌ Hypothesis 1: 1-Frame Lag
**Test:** Use `running_medians_per_frame[i-1]` instead of `running_medians_per_frame[i]`
**Result:** Made things WORSE (r = 0.939 vs 0.942)
**Conclusion:** Not a timing/lag issue

### ✅ Hypothesis 2: HOG Median Clamping
**Test:** Clamp HOG median to >= 0 after update (line 405: `hog_desc_median.setTo(0, hog_desc_median < 0)`)
**Result:** MAJOR improvement (r = 0.942 → 0.947), AU15 improved +0.076!
**Conclusion:** Critical fix, now implemented

### 🔍 Hypothesis 3: Temporal Smoothing Edge Effects
**Analysis:** 3-frame moving average only affects frames [1, size-2]
**Conclusion:** Cannot explain frames 0-12 being zero (smoothing only affects 2 edge frames)

### 🔍 Hypothesis 4: Early Frame Initialization Bug
**C++ Code Analysis (FaceAnalyser.cpp:764-800):**

Frame 0 execution flow when `update=False`:
1. Line 772-776: `if(histogram.empty())` → Initialize histogram, set `median = descriptor.clone()`
2. Line 778-795: `if(update)` → FALSE (skip, hist_count stays at 0)
3. Line 797: `if(hist_count == 1)` → FALSE (it's 0, not 1!)
4. Line 802+: Goes to `else` block → calls `_compute_median()` on **EMPTY histogram**

**Potential Bug:** C++ might compute median from empty histogram on frame 0, producing zeros!

**Our Python Implementation (histogram_median_tracker.py:96-104):**
```python
if self.hist_count == 0:
    # Frame 0: histogram not updated yet, use descriptor directly
    self.current_median = features.copy()
elif self.hist_count == 1:
    # Frame 1: histogram updated once, still use descriptor directly
    self.current_median = features.copy()
else:
    # Frame 2+: compute from histogram
    self._compute_median()
```

We handle `hist_count==0` as special case, but C++ might not!

## Root Cause Analysis

**Why frames 0-12 are zero in OpenFace:**

1. **Frame 0:** `update=False`, hist_count=0
   - C++ might compute median from empty histogram → zeros?
   - Python: Uses descriptor directly (non-zero)

2. **Frames 1-12:** Histogram updates every 2nd frame
   - By frame 12: 6 updates total
   - Possibly insufficient for stable median
   - OpenFace may have minimum hist_count threshold (≥7?) before activating dynamic models

3. **Frame 13+:** Both systems converge
   - Running median stabilized
   - Dynamic models fully active
   - Excellent correlation (r > 0.94)

## Impact Assessment

### Remaining Problematic AUs (r < 0.90): 5/17

| AU | Correlation | Type | Notes |
|----|-------------|------|-------|
| AU05 | 0.853 | Dynamic | Upper lid raiser |
| AU09 | 0.894 | Dynamic | Nose wrinkler (very close!) |
| AU15 | 0.868 | Dynamic | Lip corner depressor |
| AU20 | 0.823 | Dynamic | Lip stretcher |
| AU23 | 0.868 | Dynamic | Lip tightener |

**All problematic AUs are dynamic models** - confirms issue is with running median.

### Production Ready AUs: 12/17 (71%)

- **Static models:** 6/6 at r > 0.999 (perfect!)
- **Dynamic models:** 6/11 at r > 0.95

## Potential Solutions

### Option 1: Match C++ Empty Histogram Behavior
Force median to zeros when `hist_count < threshold`:

```python
if self.hist_count < 7:  # Or some threshold
    self.current_median = np.zeros(self.feature_dim)
elif self.hist_count == 1:
    self.current_median = features.copy()
else:
    self._compute_median()
```

**Risk:** May not be the actual C++ logic

### Option 2: Skip Early Frames in Validation
Exclude frames 0-12 from correlation calculation:

```python
# Only compare frames 13+ where both systems are active
python_preds = python_preds[13:]
of22_preds = of22_preds[13:]
r, p = pearsonr(python_preds, of22_preds)
```

**Expected Result:** r = 0.95+ for all AUs

### Option 3: Deep Dive C++ _compute_median()
Trace exactly what C++ does when:
- `hist_count == 0`
- Histogram is empty
- What median value is computed?

Need to check lines 803+ in FaceAnalyser.cpp

### Option 4: Accept r=0.947 as "Good Enough"
- 71% of AUs at production quality (r > 0.90)
- Early frame discrepancy may be inherent limitation
- Proceed with FHOG extraction

## Next Steps

### Immediate (Current Debug)
1. ✅ Found HOG median clamping issue (FIXED)
2. ✅ Identified early frame zeroing pattern
3. 🔄 Determine C++ behavior for empty histogram median
4. ⏳ Test minimum hist_count threshold hypothesis
5. ⏳ Implement fix and re-validate

### After Resolution
1. **Implement Python FHOG Extraction**
   - Port dlib/OpenCV FHOG algorithm
   - Validate against OpenFace .hog files
   - Eliminate dependency on C++ FeatureExtraction binary

2. **Create Complete OF2.2 AU Predictor Class**
   - Integrate all components
   - End-to-end Python pipeline
   - Production-ready API

## Files Modified in This Session

1. `histogram_median_tracker.py`
   - Added HOG median clamping (line 209)
   - Fixed frame 0/1 initialization

2. `validate_svr_predictions.py`
   - Added prediction clamping [0, 5]
   - Uses per-frame running medians

3. `analyze_early_frames.py` (NEW)
   - Diagnostic tool for early frame analysis
   - Reveals frame-by-frame running median evolution

4. `RUNNING_MEDIAN_EARLY_FRAME_ISSUE.md` (THIS FILE)
   - Comprehensive problem documentation

## Key Code References

### C++ Frame 0 Logic (FaceAnalyser.cpp:764-821)
```cpp
void FaceAnalyser::UpdateRunningMedian(..., bool update, ...) {
    // Line 772-776: Initialize on first call
    if(histogram.empty()) {
        histogram = cv::Mat_<int>(descriptor.cols, num_bins, (int)0);
        median = descriptor.clone();  // ← Sets median to descriptor
    }

    // Line 778-795: Update histogram (only if update=true)
    if(update) {
        // ... bin features and increment hist_count
        hist_count++;
    }

    // Line 797-821: Recompute median
    if(hist_count == 1) {
        median = descriptor.clone();
    }
    else {  // ← FRAME 0 GOES HERE (hist_count==0!)
        // Recompute median from histogram...
        // What happens with empty histogram?
    }
}
```

### Python Implementation (histogram_median_tracker.py:92-104)
```python
# Compute median (matches C++: always sets median, even on frame 0)
if self.hist_count == 0:
    # Frame 0: histogram not updated yet, use descriptor directly
    self.current_median = features.copy()
elif self.hist_count == 1:
    # Frame 1: histogram updated once, still use descriptor directly
    self.current_median = features.copy()
else:
    # Frame 2+: compute from histogram
    self._compute_median()
```

**Key Difference:** We handle `hist_count==0` explicitly, C++ might not!

## Diagnostic Commands

```bash
# Run early frame analysis
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"
python3 analyze_early_frames.py

# Full validation
python3 validate_svr_predictions.py

# Check specific AU
python3 debug_au15_running_median.py

# Extract first 20 frames from CSV
head -22 of22_validation/IMG_0942_left_mirrored.csv | cut -d',' -f1,690
```

## Summary

We've achieved **r = 0.947** through systematic debugging:
- ✅ Prediction clamping [0, 5]
- ✅ Frame 0/1 initialization
- ✅ HOG median clamping to >= 0
- ✅ Temporal smoothing (3-frame window)
- ✅ PDM-based geometric features

**Remaining issue:** OpenFace zeros frames 0-12, we don't. This is likely an initialization period or minimum hist_count threshold. Once resolved, expect **r > 0.95 for all 17 AUs**.
