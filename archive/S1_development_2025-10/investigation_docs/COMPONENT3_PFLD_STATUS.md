# Component 3: 68-Point Landmark Detection - PFLD Status

## Summary

Successfully downloaded and integrated PFLD (Practical Facial Landmark Detector) ONNX model for 68-point landmark detection. The detector works and produces landmarks in the right approximate locations, but accuracy is lower than target.

## What Was Done

### 1. Model Selection & Download ✅
- **Rejected dlib**: Too large for PyInstaller (~180MB with dependencies)
- **Rejected STAR 98→68 mapping**: No official mapping exists between WFLW 98-point and iBUG 68-point formats
- **Selected PFLD ONNX**: Lightweight (2.8MB), fast (~100 FPS CPU), PyInstaller-friendly

**Model Details:**
- Source: pytorch_face_landmark repository
- Download: https://drive.google.com/file/d/1me3-AC6rVcvVyyxNP7FxqdAN5SoDTj95/view?usp=sharing
- Location: `weights/pfld_68_landmarks.onnx`
- Input: [batch, 3, 112, 112] RGB normalized [0-1]
- Output: [batch, 136] (68 × 2 coordinates, normalized)

### 2. Implementation ✅
Created complete PFLD landmark detector in `pfld_landmark_detector.py`:

- **PFLDLandmarkDetector class**: ONNX inference wrapper
- **Preprocessing**: BGR→RGB, resize to 112×112, normalize, CHW format
- **Postprocessing**: Denormalize coordinates from model output to absolute pixel positions
- **Batch support**: Can process multiple faces
- **Visualization utilities**: For debugging and verification

### 3. Validation ⚠️

**Quick Test (`quick_pfld_test.py`)**: ✅ PASSED
- Successfully detects 68 landmarks on test frame
- Model loads and runs correctly
- Output format is correct (68, 2)

**Accuracy Test (`simple_pfld_validation.py`)**: ⚠️ ISSUES
- **Detection Rate**: 100% (50/50 frames) ✅
- **Mean RMSE**: 13.26 pixels ⚠️ (target: < 3 pixels)
- **Median RMSE**: 13.52 pixels ⚠️
- **Range**: 10.37 - 14.76 pixels

**Detailed Debug (`debug_pfld_coordinates.py`)**: ⚠️ SYSTEMATIC OFFSET
- Landmarks in approximately correct location
- **Mean error**: 9.55 pixels per landmark
- **Median error**: 7.71 pixels
- **Error distribution**: Jaw/chin points highest (16-20px), eye region lowest (3-5px)
- Coordinates systematically offset, not random

## Current Status: ⚠️ NEEDS IMPROVEMENT

### What Works ✅
1. Model downloads and loads successfully
2. Inference runs correctly (100% detection rate)
3. Output format is correct (68 landmarks, 2D coordinates)
4. Landmarks are in approximately the right spatial region
5. No crashes or errors during processing

### What Doesn't Work ⚠️
1. **Accuracy is below target**: 9-13 pixels error vs <3 pixel target
2. **Systematic offset**: Especially on jaw/chin points (16-20 pixels)
3. **Jaw landmarks**: Points 0-16 have highest error
4. **Not production-ready**: Current accuracy insufficient for high-quality AU extraction

## Root Cause Analysis

### Hypothesis 1: Landmark Definition Mismatch 🔴 LIKELY
**Problem**: PFLD model outputs landmarks in a slightly different convention than iBUG 68-point format used by OpenFace.

**Evidence**:
- Error is systematic, not random
- Jaw points (boundary landmarks) have highest error
- Internal facial features (eyes, nose) have lower error
- PFLD trained on custom datasets, not standard iBUG annotations

**Impact**: HIGH - requires model retraining or finding iBUG-trained model

### Hypothesis 2: Coordinate Transformation Issues 🟡 POSSIBLE
**Problem**: Postprocessing might incorrectly transform normalized model outputs to absolute coordinates.

**Evidence**:
- Coordinates in right general region but offset
- Could be scaling or offset error in `postprocess_landmarks()`

**Impact**: MEDIUM - fixable with code adjustments

**Next Steps to Test**:
1. Check if landmarks need additional alignment/registration step
2. Verify the model outputs are truly in [0, 1] normalized range
3. Test with different padding strategies (current: 10% padding)

### Hypothesis 3: Model Quality 🟡 POSSIBLE
**Problem**: This specific PFLD model may not be highly accurate.

**Evidence**:
- No published accuracy metrics found for this specific model
- Error of 10 pixels on 1080p video might be expected for lightweight model
- Other PFLD implementations report ~2-3 pixel error on 256×256 input

**Impact**: HIGH - might need different model

## Options Moving Forward

### Option A: Find Better PFLD Model 🟢 RECOMMENDED
**Approach**: Search for PFLD models specifically trained on iBUG 68-point format

**Pros**:
- Maintains lightweight ONNX approach (PyInstaller-friendly)
- Many PFLD variants exist with better accuracy
- Could find model trained on higher resolution input (256×256 vs 112×112)

**Cons**:
- Requires research and testing
- May not exist in ONNX format

**Time**: 2-4 hours research + testing

### Option B: Use dlib (Original Plan) 🟡 FALLBACK
**Approach**: Return to dlib shape_predictor_68_face_landmarks.dat

**Pros**:
- Industry standard iBUG 68-point detector
- Sub-pixel accuracy (<1 pixel typical)
- Known to work with OpenFace pipeline
- Guaranteed compatibility

**Cons**:
- Large size: ~180MB total with dependencies
- PyInstaller packaging complexity
- Slower inference (~30 FPS vs 100 FPS)

**Time**: 4-6 hours for proper PyInstaller integration

### Option C: Calibrate/Retrain PFLD 🔴 NOT RECOMMENDED
**Approach**: Fine-tune current PFLD model on iBUG annotations or add calibration layer

**Pros**:
- Could achieve both size and accuracy goals

**Cons**:
- Requires iBUG training dataset
- Requires PyTorch training setup
- 1-2 days minimum work
- May not solve fundamental architecture limitations

**Time**: 1-2 days minimum

### Option D: Accept Current Accuracy 🟡 RISKY
**Approach**: Test if 10-pixel landmark error is acceptable for AU extraction

**Pros**:
- No additional work
- Already implemented

**Cons**:
- May degrade AU prediction accuracy
- Violates gold standard principle
- No guarantee it will work

**Time**: 2-3 hours to test AU extraction with PFLD landmarks

**Test Plan**:
1. Integrate PFLD into full pipeline (replace CSV landmarks)
2. Run AU prediction on validation video
3. Compare AU correlation: target r > 0.80
4. If r < 0.75, accuracy is insufficient

## Recommendation

**Go with Option A (Find Better PFLD Model) first, fallback to Option B (dlib) if needed.**

**Rationale**:
1. Research effort is low (2-4 hours)
2. Lightweight ONNX approach is highly desirable for PyInstaller
3. Better PFLD models likely exist
4. Can pivot to dlib quickly if search fails

**Search Strategy**:
1. Look for PFLD models trained on 300W/AFW/HELEN datasets (standard iBUG benchmarks)
2. Check for higher resolution variants (256×256 input)
3. Search face_alignment, face-alignment, or face_recognition repos for ONNX models
4. Consider MediaPipe Face Mesh (468 points, can subset to 68 iBUG landmarks)

## Files Created

### Implementation
- `pfld_landmark_detector.py` - Main PFLD detector class
- `weights/pfld_68_landmarks.onnx` - Model file (2.8MB)

### Testing
- `quick_pfld_test.py` - Simple one-frame test ✅
- `simple_pfld_validation.py` - Accuracy test on 50 frames ⚠️
- `debug_pfld_coordinates.py` - Coordinate system debugging ⚠️
- `validate_pfld_landmarks.py` - Full validation (had RetinaFace import issues)

### Results
- `pfld_test_output.jpg` - Single frame visualization
- `pfld_debug_frame0.jpg` - Debug visualization (GREEN=GT, RED=PFLD, BLUE=bbox)
- `simple_pfld_results.csv` - Per-frame RMSE results

## Next Session

1. **Decision Point**: Choose Option A or Option B
2. **If Option A**: Research better PFLD/landmark ONNX models
3. **If Option B**: Begin dlib PyInstaller integration
4. **Goal**: Achieve <3 pixel landmark RMSE to proceed with AU integration

---

**Session Date**: 2025-10-29
**Component**: 3 (68-Point Landmark Detection)
**Status**: Implemented but needs accuracy improvement
**Blocker**: 10+ pixel error vs <3 pixel target
