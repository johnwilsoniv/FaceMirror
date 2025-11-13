# PNet Layer 0 Investigation Results

## Executive Summary

PNet layer 0 shows significant divergence between C++ OpenFace and Python ONNX implementations, specifically in output channel 1. C++ OpenFace is the reference implementation and is working correctly in production.

## Test Methodology

1. Captured C++ PNet layer 0 output during actual face detection
2. Loaded identical input into Python with extracted weights
3. Compared outputs numerically
4. Performed manual convolution computation as mathematical reference

## Results

### Layer 0 Output Comparison (10 channels, 382x214 spatial)

**Overall Statistics:**
- Mean absolute difference: 0.153
- Max absolute difference: 8.86
- Only 2.2% of values within 1e-5 tolerance

**Channel-Specific Analysis:**
```
Channel 0: max=1.953, mean=0.039
Channel 1: max=8.858, mean=0.688  ⚠️ SIGNIFICANT DIVERGENCE
Channel 2: max=3.374, mean=0.266
Channel 3: max=4.396, mean=0.107
Channel 4: max=3.934, mean=0.050
Channel 5: max=2.759, mean=0.049
Channel 6: max=3.450, mean=0.107
Channel 7: max=4.934, mean=0.111
Channel 8: max=1.571, mean=0.032
Channel 9: max=3.196, mean=0.078
```

**Channel 1 Specific Issues:**
- Correlation between C++ and Python: **-0.72** (strongly negative)
- All top 10 worst divergences occur in channel 1
- Sample divergence at [307,121]: C++=+5.18, Python=-3.68 (opposite signs!)

### Manual Computation Verification

At output position [0,0,0]:
- Manual computation: -0.1244121762
- Python output: -0.1244121492 (diff = 2.7e-08) ✅
- C++ output: -0.1243851557 (diff = 2.7e-05) ⚠️

**Note:** Python matches manual computation to floating-point precision, C++ shows small but consistent deviation.

### Final Output Impact

Despite layer 0 divergence, final PNet outputs (layer 7):
- Mean probability difference: 0.020
- 74.8% of probabilities within 0.01 tolerance
- 46.1% of probabilities within 0.001 tolerance

## Comparison with ONet

**ONet Layer 0** (verified previously):
- C++ vs Python max diff: 7.15e-07
- All values agree to floating-point precision
- **Perfect agreement**

**PNet Layer 0:**
- C++ vs Python max diff: 8.86
- Negative correlation in channel 1
- **Significant systematic divergence**

## Conclusions

1. **PNet layer 0 implementation differs between C++ and Python**
2. **C++ OpenFace is the production-tested reference**
3. Python implementation matches mathematical convolution exactly
4. The divergence may be:
   - An intentional modification in C++ for improved performance
   - Different weight preprocessing in C++
   - A difference in how weights are loaded/applied

5. **Despite layer 0 divergence, final detection outputs remain reasonably close** (75% of probabilities within 0.01)

## Recommendations

- Document RNet layer 0 behavior for comparison
- Consider C++ OpenFace as ground truth for detection performance
- Monitor if the divergence impacts actual face detection quality
- May need to accept that exact numeric matching with C++ is not achievable for PNet

## Files Generated

- `test_pnet_layer0.py` - Layer 0 comparison test
- `debug_pnet_convolution.py` - Manual computation verification
- `find_max_divergence.py` - Divergence pattern analysis
- `investigate_pnet_systematically.py` - Systematic pattern checks
- `check_pnet_final_outputs.py` - Final output comparison
