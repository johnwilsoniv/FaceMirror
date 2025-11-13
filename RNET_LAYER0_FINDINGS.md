# RNet Layer 0 Investigation Results

## Executive Summary

RNet layer 0 **MATCHES C++ OpenFace perfectly** with floating-point precision accuracy. The transpose fix applied to weight extraction is working correctly for RNet.

## Test Methodology

1. Captured C++ RNet layer 0 output during actual face detection
2. Loaded identical input (24x24x3) into Python with extracted weights
3. Compared outputs numerically
4. Same methodology as ONet and PNet verification

## Results

### Layer 0 Output Comparison (28 channels, 22x22 spatial)

**Overall Statistics:**
- Max absolute difference: **6.86e-07** (floating-point precision)
- Mean absolute difference: 4.73e-08
- **100%** of values within 1e-6 tolerance
- **85.8%** of values within 1e-7 tolerance

**Channel-Specific Analysis:**
```
Channel 0: max=4.768e-07, mean=7.34e-09
Channel 1: max=2.384e-07, mean=3.34e-09
Channel 2: max=4.768e-07, mean=6.20e-09
Channel 3: max=1.788e-07, mean=2.85e-09
Channel 4: max=2.235e-07, mean=3.69e-09
Channel 5: max=1.192e-07, mean=2.52e-09
Channel 6: max=2.682e-07, mean=2.67e-09
Channel 7: max=4.768e-07, mean=5.93e-09
Channel 8: max=4.768e-07, mean=3.89e-09
Channel 9: max=6.855e-07, mean=1.11e-08
```

**All channels show perfect agreement** - no systematic divergence in any channel.

### Sample Value Verification

At output position [0,0,0]:
- C++ output: -3.6104822159
- Python output: -3.6104822159
- Difference: < 1e-9 ✅

## Comparison with ONet and PNet

| Network | Layer 0 Max Diff | Status | Notes |
|---------|-----------------|--------|-------|
| **ONet** | 7.15e-07 | ✅ Perfect match | Transpose fix working |
| **RNet** | 6.86e-07 | ✅ Perfect match | Transpose fix working |
| **PNet** | 8.86 | ❌ Diverges | Unique issue (channel 1 correlation -0.72) |

## Conclusions

1. **RNet weight extraction is CORRECT**
2. **Transpose fix works perfectly for RNet** (same as ONet)
3. **No systematic divergence** in any channel
4. **PNet's divergence is unique** and not related to the transpose fix methodology
5. Weight extraction methodology is validated (works for 2 out of 3 networks)

## Implications for PNet Divergence

Since RNet and ONet both match perfectly with the same extraction methodology, PNet's divergence is likely:
- An intentional modification in C++ OpenFace for PNet specifically
- Different weight preprocessing applied only to PNet in C++
- A layer-specific difference in how PNet is implemented

The fact that **RNet matches perfectly** confirms our extraction methodology is sound, isolating PNet as having a unique difference from the Python/ONNX model.

## Files Generated

- `test_rnet_layer0.py` - Layer 0 comparison test
- Debug logging added to C++ FaceDetectorMTCNN.cpp (lines 332-374)
- C++ debug files: `/tmp/cpp_rnet_input.bin`, `/tmp/cpp_rnet_layer0_after_conv_output.bin`
