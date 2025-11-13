# MTCNN Weight Extraction Root Cause - FOUND!

## Executive Summary

**Root Cause**: Weight extraction does NOT transpose convolution kernels, but C++ does.  
**Location**: extract_cpp_mtcnn_weights.py:122  
**Fix**: Add `.T` to transpose each kernel during extraction

## Problem Statement

Python ONNX MTCNN outputs were scaled ~2.08x compared to C++ .dat outputs:
- Python: logit[0]=-1.637, logit[1]=+1.638
- C++: logit[0]=-3.414, logit[1]=+3.413

## Investigation Method - Rigorous Debugging with Actual Data

### Step 1: Generated Fresh Debug Files
- Modified C++ code to save ONet input and layer 0 output (32 channels)
- Rebuilt C++ and ran detection to generate verified debug files:
  - `/tmp/cpp_onet_input.bin` (48x48x3)
  - `/tmp/cpp_layer0_after_conv_output.bin` (32x46x46)
  - `/tmp/cpp_conv6_weight.bin` (28x32 weight matrix)

### Step 2: Verified Debug File Consistency
Manual computation using C++ weight matrix:
- Result: 0.063729
- C++ saved output [0,0,0]: 0.063729
- **Difference: 2.2e-08 (essentially zero!)** ✅

This proves all debug files are from the same run and internally consistent.

### Step 3: Identified First Point of Divergence
Compared C++ vs Python layer 0 outputs using same input:
- **C++ layer 0 [0,0,0]: 0.063729**
- **Python layer 0 [0,0,0]: 0.066004**
- Max difference across all pixels: **4.81 (huge!)**
- Mean difference: **0.248**

**Conclusion: Divergence occurs at the FIRST convolution layer.**

### Step 4: Isolated Root Cause
Compared extracted Python weights vs C++ runtime weight matrix:
- **Max difference: 1.12 (significant!)**
- Weights contain same values but in **different order** (shuffled)

Sample comparison:
```
C++ weights:    [0.481, 0.302, -0.062, 0.453, -0.090]
Python weights: [0.481, 0.453, 0.212, 0.302, -0.090]
                  ^^^^   ^^^^   ^^^^   ^^^^   ^^^^
                  match  swap   diff   swap   match
```

### Step 5: Found the Bug
**C++ code (FaceDetectorMTCNN.cpp:437)**:
```cpp
cv::Mat_<float> k_flat = kernels_rearr[k][i].t();  // Transpose kernel!
k_flat = k_flat.reshape(0, 1).t();
```

**Python extraction code (extract_cpp_mtcnn_weights.py:122)**:
```python
weights[out_ch, in_ch, :, :] = kernels[idx]  # No transpose!
```

C++ transposes each kernel for column-major im2col flattening, but Python extraction doesn't!

### Step 6: Verified the Fix
Tested transposing kernels during reconstruction:
```python
kernel_transposed = kernel.T  # Add transpose like C++ does
```

**Result**:
- Max difference: **0.000000e+00** (EXACT MATCH!)
- Mean difference: **0.000000e+00**

```
C++ weights:           [0.481, 0.302, -0.062, 0.453, -0.090]
Python (w/ transpose): [0.481, 0.302, -0.062, 0.453, -0.090]
                        ✅    ✅     ✅      ✅     ✅
```

## The Fix

**File**: `extract_cpp_mtcnn_weights.py`  
**Line**: 122  
**Change**:
```python
# Before:
weights[out_ch, in_ch, :, :] = kernels[idx]

# After:
weights[out_ch, in_ch, :, :] = kernels[idx].T
```

## Why This Happens

1. MATLAB training saves kernels in row-major order
2. C++ reads kernels and transposes them (line 437) for column-major im2col
3. Python extraction reads kernels but doesn't transpose
4. Result: Python weights are in wrong order, causing incorrect convolution outputs
5. Error propagates through all layers, causing final 2.08x scaling discrepancy

## Technical Details

### C++ Weight Matrix Construction (FaceDetectorMTCNN.cpp:395-454)
1. Read kernels from .dat file (line 405)
2. Rearrange kernels (lines 411-421)
3. **Transpose each kernel** (line 437): `k_flat = kernels_rearr[k][i].t()`
4. Flatten to column vector for im2col format (line 438)
5. Build weight matrix (28x32 for ONet first layer)
6. Transpose entire matrix (line 444)
7. Add bias column (lines 447-452)
8. Store W.t() for runtime (line 454)

### Python Extraction (extract_cpp_mtcnn_weights.py:110-123)
1. Read kernels from .dat file using `read_matrix()`
2. Store in (out_ch, in_ch, H, W) format for PyTorch
3. **BUG: No transpose applied** (line 122)
4. Result: Kernels in wrong orientation

## Impact

- All convolutional layers have incorrect weights
- Error propagates through entire network
- Final output scaled by ~2.08x (accumulated error)
- Detection still works because relative ordering preserved
- But absolute values wrong, breaking compatibility with C++

## Next Steps

1. Apply fix to extract_cpp_mtcnn_weights.py
2. Re-extract all MTCNN weights (PNet, RNet, ONet)
3. Rebuild ONNX models
4. Verify outputs match C++ exactly
5. Test on full detection pipeline
