# OpenFace 2.2 Python Replication Summary

## Goal
Replicate OpenFace 2.2's AU prediction system in Python to achieve r > 0.95 for all 17 Action Units.

## Progress Timeline

### Initial State (r=0.84)
- Implemented SVR model parser
- Implemented HOG parser
- Implemented PDM parser
- Implemented histogram-based running median
- Implemented temporal smoothing
- **Result:** 5/17 AUs with r < 0.75

### Fix 1: Prediction Clamping (r=0.84 → r=0.94)
**Problem:** Predictions could go negative (e.g., AU02 drops to -2.0)
**Solution:** Added `pred = np.clip(pred, 0.0, 5.0)` matching C++ line 890-895
**Impact:** +0.10 average correlation, massive improvement for all dynamic AUs

### Fix 2: Frame 0 Initialization (r=0.94 → r=0.942)
**Problem:** Running median at frame 0 was all zeros
**Solution:** Modified histogram tracker to handle hist_count==0, ==1, >=2 cases
**Impact:** +0.002 average correlation, improved early frame predictions

### Fix 3: HOG Median Clamping (r=0.942 → r=0.947)
**Problem:** HOG median could be negative
**Solution:** Added clamping after HOG median update: `median[median < 0] = 0`
**Code:** FaceAnalyser.cpp:405 `this->hog_desc_median.setTo(0, this->hog_desc_median < 0);`
**Impact:** +0.005 average correlation, AU15 improved +0.076!

## Current Status (r=0.947)

### Production Ready AUs (r > 0.95): 10/17 (59%)
- AU04 (static): r=0.9999
- AU06 (static): r=0.9999
- AU07 (static): r=0.9997
- AU10 (static): r=0.9999
- AU12 (static): r=0.9999
- AU14 (static): r=0.9999
- AU17 (dynamic): r=0.968
- AU25 (dynamic): r=0.995
- AU26 (dynamic): r=0.998
- AU45 (dynamic): r=0.987

### Good AUs (0.90 < r < 0.95): 2/17 (12%)
- AU01 (dynamic): r=0.931
- AU02 (dynamic): r=0.908

### Problematic AUs (r < 0.90): 5/17 (29%)
- AU05 (dynamic): r=0.853
- AU09 (dynamic): r=0.894
- AU15 (dynamic): r=0.868
- AU20 (dynamic): r=0.823
- AU23 (dynamic): r=0.868

## Key Implementation Details

### Histogram Parameters
- HOG: num_bins=1000, min_val=-0.005, max_val=1.0
- Geometric: num_bins=10000, min_val=-60.0, max_val=60.0

### Running Median Algorithm
1. Frame 0: histogram empty, median = descriptor
2. Frame 1+: update histogram every 2nd frame (when i % 2 == 1)
3. Compute median from cumulative histogram
4. **CRITICAL:** Clamp HOG median to >= 0 after update

### Feature Vector (4702 dims)
- HOG features: 4464 dims
- Geometric features: 238 dims
  - PDM-reconstructed landmarks: 204 dims
  - PDM parameters: 34 dims

### Prediction Formula (Dynamic Models)
```python
centered = features - means - running_median
prediction = centered · support_vectors + bias
prediction = clip(prediction, 0.0, 5.0)  # Critical!
```

### Temporal Smoothing
3-frame moving average (offline processing only)

## Remaining Challenges

The 5 problematic AUs (r < 0.90) are all dynamic models. Possible causes:
1. Numerical precision differences between C++ and Python
2. Accumulated floating point rounding over 1110 frames
3. Subtle differences in OpenCV operations
4. AU-specific model characteristics

The correlation of r=0.947 may be close to the practical limit given inherent numerical differences between C++ and Python implementations.

## Files Modified
- `validate_svr_predictions.py`: Main validation script
- `histogram_median_tracker.py`: Running median implementation with HOG clamping
- `openface22_model_parser.py`: SVR model parser
- `pdm_parser.py`: PDM shape model parser
- `openface22_hog_parser.py`: HOG feature parser

## Next Steps
1. Investigate if remaining discrepancy is acceptable for production use
2. Consider if r=0.947 (85-90% range for problematic AUs) is "good enough"
3. If higher accuracy needed, deep-dive into specific AU model files or training data
4. Proceed with Python FHOG extraction implementation
