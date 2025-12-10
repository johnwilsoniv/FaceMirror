# Detection Validator Implementation - Handoff Document

## Overview

This document describes the implementation of the CNN-based detection validator from C++ OpenFace, ported to Python for use with pyCLNF. The validator determines if detected landmarks are valid, triggering re-detection when tracking fails.

## Files Created/Modified

### New Files
- `pyclnf/pyclnf/core/detection_validator.py` - Full CNN validator implementation

### Modified Files
- `pymtcnn/README.md` - Added notes about supported input formats and interpolation differences
- C++ debug code added to `LandmarkDetectionValidator.cpp` (temporary, for comparison)

## Implementation Status

### Completed
1. **PAW (Piecewise Affine Warp)** - `detection_validator.py:PAW`
   - Reads OpenFace binary format for triangulation, alpha/beta coefficients
   - `calc_coeff()` - Computes affine transform coefficients per triangle
   - `warp_region()` - Maps destination template coords to source image coords
   - `warp()` - Full face warping using cv2.remap

2. **CNN Architecture** - `detection_validator.py:DetectionValidator`
   - Loads 7 view models (frontal + various angles)
   - Layer types: Conv, MaxPool, FC, ReLU, Sigmoid
   - Convolution uses im2col approach matching C++
   - Properly handles weight matrix format (bias in last row)

3. **Normalization Pipeline**
   - Two-stage normalization matching C++:
     1. Local: zero-mean, unit-variance on current warped image
     2. Global: apply pre-computed mean/std images
   - Transposed iteration order matching C++

### Issue Found: pyMTCNN Bbox Mismatch on PNG Files

**Symptom**: On static PNG images, the bbox returned by pyMTCNN doesn't contain the detected keypoints.

**Example** (test_face_clean.png):
```
Bbox y-range: 559 to 641
Keypoints y-range: 831 to 1082
Gap: ~270 pixels!
```

**Root Cause**: Interpolation differences between C++ and Python image pyramid generation cause different detection results on static images. Video frames work correctly.

**Workaround**: Use video files for testing, or compute bbox from keypoints.

### Bugs Fixed (2024-12-09)

**Bug 1: Missing ROI extraction in `check()`**
- C++ extracts a cropped ROI around landmarks (bbox ± 3 pixels), adjusts landmarks to ROI coordinates, and converts image to float32 before warping
- Python was passing full image with absolute landmarks
- Fixed by adding ROI extraction and landmark adjustment in `check()`

**Bug 2: Wrong flatten order in `fully_connected()`**
- C++ transposes each feature map before flattening (`add.t().reshape(0,1)`) - column-major order
- Python was using row-major flatten (`inp.flatten()`)
- Fixed by using `inp.flatten('F')` (Fortran/column-major order)

**Result after fixes:**
| Metric | C++ | Python |
|--------|-----|--------|
| Validation Confidence | 0.9750 | 0.9750 |
| Threshold | 0.725 | 0.725 |
| Result | VALID | VALID |

Layer-by-layer CNN outputs now match within floating point precision.

## Architecture

```
DetectionValidator
├── orientations[7]          # View angles (pitch, yaw, roll)
├── paws[7]                  # PAW for each view
│   ├── triangulation        # Triangle vertex indices
│   ├── alpha, beta          # Barycentric coordinates per triangle
│   ├── destination_landmarks # Template landmark positions
│   └── pixel_mask           # Valid face region mask
├── mean_images[7]           # Per-view normalization means
├── standard_deviations[7]   # Per-view normalization stds
├── cnn_layer_types[7]       # Layer type sequence per view
├── cnn_convolutional_layers_weights[7]
├── cnn_fully_connected_layers_weights[7]
└── cnn_fully_connected_layers_biases[7]

Flow:
1. get_view_id(orientation) → select closest view model
2. paw.warp(image, landmarks) → 60x60 warped face
3. normalise_warped_to_vector() → local + global normalization
4. check_cnn() → run CNN layers → quantized output → confidence
5. confidence > 0.725 → valid
```

## Key Code Sections

### PAW Warp (lines 100-180)
```python
def warp(self, image, landmarks):
    # Convert landmarks to (2*n, 1) format
    # calc_coeff() computes affine transforms per triangle
    # warp_region() builds map_x, map_y
    # cv2.remap() applies the warp
```

### CNN Forward Pass (lines 450-495)
```python
for layer_type in self.cnn_layer_types[view_id]:
    if layer_type == 0:  # Convolution
        input_maps = self.convolution_direct(input_maps, weights, kernel_h, kernel_w)
    elif layer_type == 1:  # MaxPool
        input_maps = self.max_pooling(input_maps)
    elif layer_type == 2:  # FC
        input_maps = self.fully_connected(input_maps, weights, biases)
    elif layer_type == 3:  # ReLU
        input_maps = [np.maximum(0, inp) for inp in input_maps]
```

### Output Conversion (lines 485-495)
```python
# CNN outputs 20 bins representing confidence levels
output = input_maps[0].flatten()
max_idx = np.argmax(output)
bins = len(output)
step_size = 2.0 / bins  # range [-1, 1]
unquantized = -1.0 + step_size / 2.0 + max_idx * step_size
confidence = 0.5 * (1.0 - unquantized)
```

## Debug Code Added to C++

Location: `LandmarkDetectionValidator.cpp`

### CheckCNN() (lines 367-522)
- Dumps feature_vec size, min, max, first 10 values
- Dumps layer-by-layer stats (min, max, shape)
- Dumps final output array and max_idx

### NormaliseWarpedToVector() (lines 527-624)
- Dumps raw pixel stats before normalization
- Dumps local mean/std
- Dumps mean_images and std_dev values

**To remove debug code**: Search for `do_debug` and `debug_dump_count` in the file.

## Next Steps

1. **Integrate validator into CLNF**: Add validation check after landmark fitting:
   ```python
   # In CLNF.detect_and_fit() or similar
   if not validator.validate(orientation, image, landmarks):
       tracking_failures += 1
       if tracking_failures >= 2:  # reinit_video_every = 2
           # Trigger re-detection
   ```

2. **Test with pyCLNF landmarks**: Verify the validator works correctly with actual pyCLNF-detected landmarks (not just C++ landmarks).

## Testing

```python
from pyclnf.core.detection_validator import DetectionValidator

model_path = ".../model/detection_validation/validator_cnn_68.txt"
validator = DetectionValidator(model_path)

# Check validation
confidence = validator.check(orientation, image, landmarks)
is_valid = confidence > validator.validation_boundary  # 0.725
```

## References

- C++ source: `OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectionValidator.cpp`
- C++ header: `OpenFace/lib/local/LandmarkDetector/include/LandmarkDetectionValidator.h`
- Model file: `model/detection_validation/validator_cnn_68.txt`
