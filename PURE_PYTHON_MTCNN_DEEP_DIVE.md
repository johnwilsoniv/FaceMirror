# Pure Python MTCNN - Deep Dive Investigation

## Summary of Work

We've conducted extensive debugging to understand why Pure Python MTCNN doesn't match C++ MTCNN performance.

## What We've Proven Works ✅

1. **RNet Implementation**: Standalone testing proved RNet achieves 0.94 score on real face crops
2. **ONet Implementation**: Layer-by-layer analysis shows all layers function correctly
3. **Max Pool Fix**: Changed from `floor()` to `round()` to match C++ behavior
4. **Face Cropping**: Simplified logic removes distortion
5. **Bbox Regression**: ONet regression is being applied correctly
6. **Image Pyramid**: Covers the right range (40px to 2945px detectable faces)

## The Core Problem ❌

**PNet fails to detect the actual face at the appropriate pyramid scale.**

### Evidence:

**Test Image**: `calibration_frames/patient1_frame1.jpg` (1920×1080)
**C++ Gold Standard**: x=331, y=753, w=368, h=423 (area: 155,548 px²)
**Pure Python Best Detection**: x=553, y=841, w=30, h=30 (area: 901 px²)

**Size Discrepancy**: Our detection is **99.4% smaller** than expected!
**IoU Overlap**: Only **0.6%** with C++ gold standard

### Detailed Findings:

#### 1. Pyramid Scales Are Correct
```
Scale 0: m=0.3000, detects faces ≥133px
Scale 1: m=0.2127, detects faces ≥188px
Scale 2: m=0.1508, detects faces ≥265px
Scale 3: m=0.1069, detects faces ≥374px  ← Should detect 368×423 face!
...
Scale 9: m=0.0136, detects faces ≥2945px
```

The pyramid SHOULD detect a 368×423 face at Scale 3, but it doesn't.

#### 2. Lowering Thresholds Doesn't Help
- Official PNet threshold: 0.6
- Lowered to: 0.3 (50% reduction)
- **Result**: Still only 1 tiny detection, no improvement

This proves the issue ISN'T just being too strict with thresholds.

#### 3. PNet Detects Many Small Features
- With threshold 0.6: 161 PNet boxes detected
- Size range: 40-80 pixels
- **None in the 300-400 pixel range for the actual face!**

PNet is finding lots of small features (eyes, shadows, edges) but completely missing the full face detection at larger scales.

#### 4. ONet Correctly Rejects Partial Crops
- Face crop 3: Just shows an eye (42×42 pixel region)
- ONet score: 0.55 (uncertain)
- **This is correct behavior!** ONet should be uncertain about a partial face

The "low ONet scores" aren't a bug - they're the correct response to bad inputs.

## Root Cause Hypothesis

**PNet weights or implementation have a subtle bug that prevents large-scale face detection.**

### Why We Think This:

1. ✅ RNet proven to work (0.94 score)
2. ✅ ONet proven to work (all layers functional)
3. ✅ Pyramid construction correct
4. ✅ Bbox regression applied correctly
5. ❌ PNet output doesn't include appropriate-scale face detections

### Possible Causes:

#### A. Image Preprocessing Difference
```python
# Our preprocessing:
img_norm = (img.astype(np.float32) - 127.5) * 0.0078125
img_chw = np.transpose(img_norm, (2, 0, 1))
```

Questions:
- Is the C++ using RGB vs our BGR?
- Is there a gamma correction we're missing?
- Are we handling the image dtype correctly?

#### B. PNet Conv Layer Issue
- RNet has 3 conv layers (works)
- PNet has 3 conv layers (might have issue?)
- ONet has 4 conv layers (works)

Maybe PNet's specific layer configuration has a bug we haven't caught?

#### C. PNet Output Interpretation
```python
# Our interpretation:
logit_not_face = output[:, :, 0]
logit_face = output[:, :, 1]
prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))
```

Maybe the output format is different for PNet vs RNet/ONet?

#### D. Fully Convolutional Output
- PNet is fully convolutional (outputs a 2D map)
- RNet/ONet output single vectors
- Maybe the reshape/transpose of PNet output is wrong?

#### E. Regression Map Handling
PNet outputs:
- Channels 0-1: Face/not-face logits
- Channels 2-5: Bbox regression

Maybe we're mixing up the channels?

## Next Steps (Prioritized)

### 1. Compare PNet Implementation to C++ Source
Read the C++ PNet implementation line-by-line and verify:
- Input preprocessing
- Layer execution order
- Output interpretation
- Channel ordering

### 2. Test PNet on Synthetic Input
Create a known test pattern and verify PNet output matches expectations.

### 3. Compare Against Reference Implementation
Find a known-working Python MTCNN implementation and compare:
- How they preprocess images
- How they interpret PNet output
- What their detection results are on our test image

### 4. Debug PNet Output Maps Directly
Visualize PNet's output confidence maps at each scale:
- Do they show high confidence in the face region?
- Or is PNet actually producing low scores there?

## Conclusion

We've narrowed the problem to **PNet not detecting faces at appropriate scales**. All other components (RNet, ONet, bbox transforms, pyramid construction) are proven to work correctly. The next phase must focus specifically on PNet's implementation and comparing it against the C++ source.

## Files Involved

- `cpp_cnn_loader.py`: CNN implementation (MaxPool fix applied)
- `pure_python_mtcnn_v2.py`: Full pipeline (bbox regression verified)
- `debug_rnet_standalone.py`: Proved RNet works (0.94 score)
- `debug_onet_standalone.py`: Proved ONet works (all layers functional)
- `debug_bbox_transformations.py`: Showed PNet only outputs 40-80px boxes
- `debug_onet_layer_by_layer.py`: Verified ONet layer-by-layer execution

## Test Data

- Image: `calibration_frames/patient1_frame1.jpg` (1920×1080)
- C++ Reference: x=331.6, y=753.5, w=367.9, h=422.8
- Model Files: `~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/face_detection/mtcnn/convert_to_cpp/*.dat`
