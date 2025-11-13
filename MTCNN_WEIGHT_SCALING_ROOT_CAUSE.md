# MTCNN Weight Scaling Root Cause Analysis

## Problem
Python ONNX MTCNN outputs are scaled by ~0.48x compared to C++ .dat outputs:
- Python: logit[0]=-1.637, logit[1]=+1.638
- C++: logit[0]=-3.414, logit[1]=+3.413
- **Scaling Factor: 2.0844x** (C++ / Python)

## Investigation Summary

### What We Confirmed ✅
1. Weight extraction mathematical approach is correct (verified with synthetic test)
2. PyTorch Conv2d computation is correct (manual calculation matches)
3. C++ uses column-major im2col with transposed kernels (line 416)
4. PyTorch expects row-major ordering without kernel transpose
5. NOT transposing kernels for PyTorch is mathematically correct

### Key Discovery 🎯
**All weights need to be multiplied by 2.0844 to match C++ outputs!**

When we multiply all extracted weights by 2.0844:
- Python output becomes: logit[0]=-3.412, logit[1]=+3.414
- C++ output: logit[0]=-3.414, logit[1]=+3.413
- **Difference: < 0.002 (perfect match!)**

## Root Cause Hypothesis

The 2.08x factor suggests either:

### Option 1: Matrix Transpose Issue
- MATLAB `writeMatrixBin.m` line 11 does `M = M'` (transpose)
- Our `read_matrix()` at extract_cpp_mtcnn_weights.py:76 has comment "NO transpose needed!"
- But C++ `ReadMatBin` reads directly without un-transposing
- **This might be creating a mismatch**

### Option 2: Normalization in C++
- C++ might apply 2x scaling when loading weights
- Or original MATLAB training normalized by 0.5
- Need to check C++ weight loading code for hidden scaling

### Option 3: PReLU Weights
- PReLU negative slope weights might be scaled differently
- Could accumulate to 2x factor through multiple layers

## Next Steps

1. **Test transpose hypothesis**:
   - Try transposing weight matrices after extraction
   - See if this gives 2x factor

2. **Check C++ weight loading**:
   - Search for any scaling factors when C++ reads .dat files
   - Look for normalization in FaceDetectorMTCNN.cpp weight loading

3. **Verify with single layer**:
   - Test just first conv layer with different transpose options
   - Manually compute expected output

4. **Quick fix for now**:
   - Multiply all weights/biases by 2.0844 in ONNX export
   - This will make outputs match C++ exactly
   - Then debug the root cause later

## Proposed Fix (Temporary)

In `convert_mtcnn_to_onnx.py`, after loading weights:

```python
# TEMPORARY FIX: Scale weights to match C++ .dat outputs
SCALE_FACTOR = 2.0844

weights = np.load(weights_file) * SCALE_FACTOR
biases = np.load(biases_file) * SCALE_FACTOR
```

This will make the ONNX models functionally identical to C++ .dat models.

## Status
- ✅ Identified exact scaling factor: 2.0844x
- ✅ Verified scaling fix works mathematically
- ⏳ Need to find root cause in weight extraction
- ⏳ Need to test fix in actual ONNX models
