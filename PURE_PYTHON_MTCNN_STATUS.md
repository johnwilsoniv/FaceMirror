# Pure Python CNN MTCNN - Status Report

## Goal
Achieve parity with **C++ MTCNN** from OpenFace (NOT the ONNX version).

## C++ MTCNN Gold Standard
Using `.dat` files from:
```
~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/face_detection/mtcnn/convert_to_cpp/
```

Test image: `calibration_frames/patient1_frame1.jpg`
Expected detection:
- x=331.6, y=753.5, w=367.9, h=422.8

## Current Status

### ✅ Completed
1. **Pure Python CNN Loader** (`cpp_cnn_loader.py`)
   - Loads `.dat` files (MATLAB `writeMatrixBin` format)
   - Fixed dimension calculation: MaxPool uses `round()` not `floor()` to match C++
   - RNet dimension test: SUCCESS (576 inputs matched)
   - All three networks (PNet, RNet, ONet) run end-to-end

### ❌ Current Issue
**Pure Python CNN MTCNN V2 detects 0 faces**

Debug findings:
- PNet Stage 1: ✓ Working (detects 138 boxes)
- RNet Stage 2: ❌ **Rejecting ALL faces** (all scores < 0.7 threshold)
  - Highest RNet score: 0.6943 (below 0.7 threshold)
  - RNet outputs appear correct structurally but scores are too low

### Question
Why is Pure Python RNet (from `.dat` files) producing low scores when C++ RNet (same `.dat` files) should work?

Possible causes:
1. Weight loading issue from `.dat` format?
2. PReLU implementation difference?
3. Convolution implementation difference?
4. FC layer implementation difference?

## Next Steps
1. Compare Pure Python RNet vs C++ RNet on identical input
2. Check if weights are being loaded correctly from `.dat`
3. Verify layer-by-layer outputs match C++ implementation
