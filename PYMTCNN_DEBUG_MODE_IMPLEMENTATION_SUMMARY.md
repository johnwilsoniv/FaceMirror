# PyMTCNN & PyFaceAU Debug Mode Implementation Summary

## Current Status

✅ **PyMTCNN Debug Mode: COMPLETE**
- ONNX backend debug mode implemented
- CoreML backend debug mode implemented
- Debug mode tested and validated
- Captures PNet, RNet, ONet, and final stage outputs with timing

✅ **PyFaceAU Debug Mode: COMPLETE**
- Added `debug_mode` parameter to `__init__`
- Modified `_process_frame` signature to support `return_debug`
- Implemented debug tracking for all 8 pipeline steps:
  1. Face Detection (bbox, cached status, timing)
  2. Landmark Detection (68-point landmarks, CLNF refinement status, timing)
  3. Pose Estimation (scale, rotation, translation, params shape, timing)
  4. Face Alignment (aligned face shape, timing)
  5. HOG Feature Extraction (feature shape, timing)
  6. Geometric Feature Extraction (feature shape, timing)
  7. Running Median Update (median shape, update status, timing)
  8. AU Prediction (AU count, timing)
- TODO: Test debug mode locally

We're implementing a comprehensive validation system that requires debug modes in both PyMTCNN and PyFaceAU to capture stage-by-stage outputs for comparison against C++ OpenFace.

## Implementation Plan

### PyMTCNN Debug Mode

**Goal:** Capture box counts and outputs at each MTCNN stage (PNet, RNet, ONet, Final)

**Required Changes:**

1. **detector.py (MTCNN class)**:
   ```python
   def __init__(self, backend=None, model_dir=None, verbose=False, debug_mode=False, **kwargs):
       self.debug_mode = debug_mode
       # Pass debug_mode to backend

   def detect(self, img, return_debug=False):
       if return_debug or self.debug_mode:
           bboxes, landmarks, debug_info = self._detector.detect_with_debug(img)
           return bboxes, landmarks, debug_info
       else:
           return self._detector.detect(img)
   ```

2. **Backend implementations** (CoreML, ONNX, base.py):
   - Add `detect_with_debug()` method
   - Capture intermediate outputs:
     ```python
     debug_info = {
         'pnet': {
             'num_boxes': len(pnet_boxes),
             'boxes': pnet_boxes.copy(),
             'time_ms': pnet_time
         },
         'rnet': {...},
         'onet': {...},
         'final': {...}
     }
     ```

### PyFaceAU Debug Mode

**Goal:** Capture component outputs and timing throughout the pipeline

**Required Changes:**

1. **pipeline.py (FullPythonAUPipeline)**:
   ```python
   def __init__(self, ..., debug_mode=False):
       self.debug_mode = debug_mode

   def _process_frame(self, frame, frame_idx, timestamp, return_debug=False):
       if return_debug or self.debug_mode:
           # Capture all intermediate outputs
           debug_info = {
               'face_detection': {...},
               'landmarks_initial': {...},  # 5-point from MTCNN
               'landmarks_final': {...},    # 68-point from PFLD
               'pose_estimation': {...},
               'alignment': {...},
               'hog_extraction': {...},
               'au_prediction': {...}
           }
           return result, debug_info
       else:
           return result
   ```

## Next Steps

Given our 17-task comprehensive validation plan, we should:

1. ✅ Add debug modes (in progress)
2. Test debug output locally
3. Build validation script
4. Run on patient dataset
5. Generate report

**Current blocker:** Need to implement debug mode in PyMTCNN backends

**Estimated time:**
- PyMTCNN debug: 30min
- PyFaceAU debug: 20min
- Testing: 10min
- Total: ~1 hour for Phase 1

After Phase 1, we can proceed with extracting test frames and running the full validation suite.
