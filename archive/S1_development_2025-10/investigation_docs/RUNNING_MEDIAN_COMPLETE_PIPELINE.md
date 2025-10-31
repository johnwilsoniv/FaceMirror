# Running Median Complete Pipeline

**Reference:** OpenFace 2.2 FaceAnalyser.cpp (lines 764-821, 504-554)

## Overview

OpenFace uses a **histogram-based running median** to perform person-specific normalization for dynamic AU models. This allows the system to calibrate AU predictions to each individual's neutral expression baseline.

## Complete Pipeline

### Phase 1: Online Processing (First Pass)

Process video frame-by-frame, building the running median incrementally.

#### Frame-by-Frame Processing

For each frame `i` (0-indexed):

**Step 1: Extract Features**
- HOG features: 4464 dimensions
- Geometric features: 238 dimensions (204 PDM-reconstructed landmarks + 34 PDM parameters)

**Step 2: Update Running Median**

Call `UpdateRunningMedian(histogram, hist_count, median, descriptor, update, num_bins, min_val, max_val)`

Where:
- `update = (i % 2 == 1)` (update histogram every 2nd frame)
- Separate histograms for HOG and geometric features

**Frame 0 Execution (update=False):**
```cpp
if(histogram.empty()) {
    histogram = Mat_<int>(descriptor.cols, num_bins, (int)0);
    median = descriptor.clone();  // Line 775
}

if(update) {
    // SKIPPED - update=False
}

// Recompute median
if(hist_count == 1) {
    median = descriptor.clone();  // NOT executed (hist_count=0)
}
else {
    // Compute median from histogram
    // With hist_count=0 and empty histogram, this produces unreliable results
    // BUT the initial median = descriptor.clone() from line 775 is used
}
```

**Frame 1+ Execution (update alternates):**
```cpp
if(histogram.empty()) {
    // SKIPPED - already initialized
}

if(update) {
    // Bin each feature value into histogram
    converted = (descriptor - min_val) * num_bins / (max_val - min_val)
    converted = clip(converted, 0, num_bins-1)

    // Increment histogram bins
    for each dimension i:
        bin_idx = (int)converted[i]
        histogram[i, bin_idx]++

    hist_count++  // Increment count
}

// Recompute median
if(hist_count == 1) {
    median = descriptor.clone()  // Frame 1 only
}
else {
    // Find median from cumulative histogram
    cutoff_point = (hist_count + 1) / 2

    for each dimension i:
        cumsum = 0
        for each bin j:
            cumsum += histogram[i, j]
            if cumsum >= cutoff_point:
                median[i] = min_val + j * bin_width + 0.5 * bin_width
                break
}
```

**Step 3: HOG Median Clamping (CRITICAL!)**
```cpp
// Line 405 in FaceAnalyser.cpp
this->hog_desc_median.setTo(0, this->hog_desc_median < 0);
```

Clamp all HOG median values to >= 0 after updating.

**Step 4: Make Prediction**

Static models:
```python
pred = (features - means) · support_vectors + bias
```

Dynamic models:
```python
pred = (features - means - running_median) · support_vectors + bias
```

**Step 5: Clamp Prediction**
```python
pred = clip(pred, 0.0, 5.0)
```

**Step 6: Store for Postprocessing**

If frame index < max_init_frames (3000):
- Store HOG features
- Store geometric features
- Store frame index

#### Histogram Parameters

**HOG Histogram:**
- Dimensions: 4464
- Number of bins: 1000
- Range: [-0.005, 1.0]
- Bin width: 1.005 / 1000 = 0.001005

**Geometric Histogram:**
- Dimensions: 238
- Number of bins: 10000
- Range: [-60.0, 60.0]
- Bin width: 120.0 / 10000 = 0.012

### Phase 2: Offline Postprocessing (Second Pass)

After processing all frames, **re-process the first `max_init_frames` (3000) frames** using the **final** running median.

**Reference:** FaceAnalyser.cpp lines 504-554

```cpp
void FaceAnalyser::PostprocessPredictions() {
    if(!postprocessed) {
        int success_ind = 0;
        int all_ind = 0;

        // Re-process first max_init_frames successful frames
        while(all_ind < all_frames_size && success_ind < max_init_frames) {
            if(valid_preds[all_ind]) {
                // Use stored HOG and geometric features
                this->hog_desc_frame = hog_desc_frames_init[success_ind];
                this->geom_descriptor_frame = geom_descriptor_frames_init[success_ind];

                // Re-predict using FINAL running median
                auto AU_predictions_reg = PredictCurrentAUs(views[success_ind]);

                // Update stored predictions
                for(each AU) {
                    AU_predictions_reg_all_hist[AU][all_ind] = AU_predictions_reg[AU];
                }

                success_ind++;
            }
            all_ind++;
        }

        postprocessed = true;
    }
}
```

**Key Points:**
1. Uses the **final running median** computed from entire video
2. Only re-processes first 3000 frames (early frames benefit most)
3. Overwrites original predictions with re-computed values
4. Uses stored features (no re-extraction needed)

### Phase 3: Temporal Smoothing

After postprocessing, apply 3-frame moving average:

```python
window_size = 3
for frame i in range(1, len(predictions)-1):
    smoothed[i] = mean([predictions[i-1], predictions[i], predictions[i+1]])

# Edge frames (0 and last) are NOT smoothed
smoothed[0] = predictions[0]
smoothed[-1] = predictions[-1]
```

**Reference:** FaceAnalyser.cpp lines 651-666

## Why Two-Pass Processing Matters

### Early Frame Problem

**Without Postprocessing:**
- Frame 0: Uses descriptor as median (hist_count=0)
- Frames 1-12: Uses immature running median (hist_count=1-6)
- Running median converges slowly over first ~300 frames

**With Postprocessing:**
- All early frames re-predicted using final, stable running median
- Neutral baseline properly calibrated
- Significantly improves AU predictions for low-intensity AUs

### Impact on Correlation

Problematic AUs (r<0.90) are typically:
- Low intensity (mean < 0.15)
- Sparse activation (<40% non-zero frames)
- More sensitive to early frame errors

Two-pass processing should improve correlation by:
- Fixing early frame predictions (frames 0-300)
- Using stable neutral baseline throughout
- Reducing numerical errors from immature median

## Implementation Checklist

**Phase 1 (Online - DONE):**
- [x] Histogram initialization
- [x] Bin feature values
- [x] Update histogram counts
- [x] Compute median from cumulative histogram
- [x] HOG median clamping (>= 0)
- [x] Separate HOG and geometric histograms
- [x] Store features for postprocessing

**Phase 2 (Postprocessing - TODO):**
- [ ] Detect when all frames processed
- [ ] Store final running median
- [ ] Re-predict first 3000 frames using final median
- [ ] Overwrite original predictions

**Phase 3 (Smoothing - DONE):**
- [x] 3-frame moving average
- [x] Skip edge frames (0 and last)

## Python Implementation Plan

```python
class TwoPassAUPredictor:
    def __init__(self):
        self.median_tracker = DualHistogramMedianTracker(...)
        self.stored_features = []  # Store (hog, geom) for first 3000 frames

    def predict_video(self, video_frames):
        # PASS 1: Process all frames
        all_predictions = []
        for i, (hog, geom) in enumerate(video_frames):
            # Update running median
            update = (i % 2 == 1)
            self.median_tracker.update(hog, geom, update)
            running_median = self.median_tracker.get_combined_median()

            # Predict
            pred = self._predict_frame(hog, geom, running_median)
            all_predictions.append(pred)

            # Store features for postprocessing
            if i < 3000:
                self.stored_features.append((hog, geom))

        # PASS 2: Reprocess early frames with final median
        final_median = self.median_tracker.get_combined_median()

        for i in range(min(3000, len(all_predictions))):
            hog, geom = self.stored_features[i]
            # Re-predict with final median
            all_predictions[i] = self._predict_frame(hog, geom, final_median)

        # PASS 3: Temporal smoothing
        smoothed = self._apply_smoothing(all_predictions)

        return smoothed
```

## Expected Improvement

**Current (single-pass):**
- Average correlation: r = 0.947
- Problematic AUs: 5/11 at r < 0.90

**Expected (two-pass):**
- Average correlation: r > 0.97 (estimated)
- Problematic AUs: 0-2/11 at r < 0.90

Improvement primarily from:
- Better calibration for low-intensity AUs
- Stable neutral baseline from frame 0
- Matching OpenFace's offline processing mode

## References

- FaceAnalyser.cpp:764-821 (UpdateRunningMedian)
- FaceAnalyser.cpp:504-554 (PostprocessPredictions)
- FaceAnalyser.cpp:651-666 (Temporal smoothing)
- FaceAnalyser.cpp:405 (HOG median clamping)
