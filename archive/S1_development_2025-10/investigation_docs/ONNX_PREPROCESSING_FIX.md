# ONNX Preprocessing Fix - January 2025

## Problem Identified

ONNX models were producing significantly different AU outputs compared to PyTorch models, despite being "direct conversions." Comparison showed:

- **Average correlation**: 0.695 (69.5%) - classified as POOR equivalence
- **Average absolute difference**: 0.072
- **Worst AUs**: AU01 (0.31 difference), AU12 (0.28 difference)
- **Best AU**: AU45 (0.91-0.99 correlation)

## Root Cause

**cv2 interpolation on uint8 vs float32 produces different results!**

The ONNX detectors were doing:
```python
# WRONG ORDER
img_uint8 = cv2.resize(img_uint8, ...)  # Resize on uint8
img_float = img_uint8.astype(np.float32) / 255.0  # Then convert
```

But PyTorch does:
```python
# CORRECT ORDER
img_float = img.float() / 255.0  # Convert first
img_resized = F.interpolate(img_float, ...)  # Then resize
```

### Test Results

When tested on identical input:
- **cv2.resize on uint8**: Mean difference = 0.000790 (small but cumulative)
- **cv2.resize on float32**: Mean difference = 0.000000 ✓ PERFECT MATCH

## Fixes Applied

### 1. MTL Detector (`onnx_mtl_detector.py`)

**Before:**
```python
face_resized = cv2.resize(face_rgb, (224, 224))
face_float = face_resized.astype(np.float32) / 255.0
```

**After:**
```python
face_float = face_rgb.astype(np.float32) / 255.0  # Convert FIRST
face_resized = cv2.resize(face_float, (224, 224))  # Then resize
```

### 2. STAR Detector (`onnx_star_detector.py`)

**Before:**
```python
input_crop = cv2.warpPerspective(image, matrix, ...)
input_tensor = input_crop.astype(np.float32) / 255.0
```

**After:**
```python
image_float = image.astype(np.float32) / 255.0  # Convert FIRST
input_crop = cv2.warpPerspective(image_float, matrix, ...)  # Then transform
```

### 3. RetinaFace Detector (`onnx_retinaface_detector.py`)

**Already correct!** Already converts to float32 before resize.

## Expected Impact

After this fix, ONNX outputs should match PyTorch outputs nearly perfectly:
- Correlation should increase from 0.695 to >0.99
- Absolute differences should drop from 0.072 to <0.001
- Clinical AU measurements should be consistent across backends

## Testing

To verify the fix worked, run:
```bash
cd "S1 Face Mirror"
python3 compare_onnx_pytorch.py
```

Look for:
- ✓ Correlation > 0.99
- ✓ Mean absolute difference < 0.001
- ✓ "EXCELLENT equivalence" assessment

## Lesson Learned

**Always convert to float BEFORE any geometric transformations (resize, warp, etc.)**

This is because:
1. Interpolation on integers has limited precision (rounds to nearest int)
2. Interpolation on floats preserves sub-pixel accuracy
3. PyTorch always operates on float tensors
4. Small interpolation differences compound through the neural network
