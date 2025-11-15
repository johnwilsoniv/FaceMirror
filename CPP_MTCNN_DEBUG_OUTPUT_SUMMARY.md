# C++ OpenFace MTCNN Debug Output Summary

## Status: ✅ ALREADY IMPLEMENTED

The C++ OpenFace binary (`/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction`) **already contains comprehensive debug output** for all MTCNN stages that is comparable to the PyMTCNN debug mode we just implemented.

## Debug Output Locations

### File: `FaceDetectorMTCNN.cpp`
Location: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/FaceDetectorMTCNN.cpp`

### PNet Debug Output (Lines 919-1101)

**Captured Information:**
- Scale-by-scale processing
- Boxes generated per scale
- Before/after within-scale NMS
- Before/after cross-scale NMS (line 1056, 1080)
- Top scoring proposals
- Bbox regression details
- All boxes dumped to `/tmp/cpp_pnet_all_boxes.txt`

**Example Output:**
```
[C++ PNet] Scale 1/8 BEFORE within-scale NMS: 45 boxes
[C++ PNet] Scale 1/8 AFTER within-scale NMS: 32 boxes
[C++ PNET DEBUG] Before cross-scale NMS: 245 boxes
[C++ PNET DEBUG] After cross-scale NMS: 104 boxes
```

### RNet Debug Output (Lines 1158-1279)

**Captured Information:**
- Input bbox for each detection
- Preprocessed pixel values (lines 1158-1178)
- RNet CNN output (logits, probabilities)
- ALL RNet scores dumped to `/tmp/cpp_rnet_scores.txt` (line 1211-1225)
- Boxes before/after regression
- Boxes before/after NMS
- Final RNet output dumped to `/tmp/cpp_rnet_output.txt` (line 1265)

**Example Output:**
```
C++ RNet Debug - Detection 0
Input bbox: x=123.4 y=456.7 w=89.2 h=92.1
RNet output: logit[0]=-2.34 logit[1]=1.56 prob=0.923
```

### ONet Debug Output (Lines 1284-1469)

**Captured Information:**
- Boxes going into ONet (line 1284)
- ONet input tensor saved to `/tmp/cpp_onet_input.bin` (line 1356)
- ONet raw CNN output (line 1377-1393)
- Landmark predictions
- Boxes before/after final NMS
- Final ONet boxes dumped to file (line 1466)

**Example Output:**
```
C++ Boxes BEFORE ONet: 6 boxes
C++ ONet: 4 boxes BEFORE NMS
C++ ONet: 2 boxes after final NMS
```

## Output Files Generated

The C++ binary writes debug information to these files:

1. **`/tmp/cpp_pnet_all_boxes.txt`** - All PNet proposals after cross-scale NMS
2. **`/tmp/cpp_rnet_scores.txt`** - All RNet scores before threshold filter
3. **`/tmp/cpp_before_rnet_regression.txt`** - Boxes before RNet regression
4. **`/tmp/cpp_rnet_output.txt`** - Final RNet output boxes
5. **`/tmp/cpp_before_onet.txt`** - Boxes going into ONet
6. **`/tmp/cpp_onet_input.bin`** - Binary dump of ONet input tensor (first detection)
7. **`/tmp/cpp_original_float.bin`** - Original preprocessed image
8. **`/tmp/cpp_pnet_input_scale0.bin`** - PNet input for first scale
9. **`/tmp/cpp_scale7_probs.bin`** - All probabilities for scale 7

## Comparison with PyMTCNN Debug Mode

| Component | PyMTCNN Debug | C++ OpenFace Debug | Status |
|-----------|---------------|-------------------|--------|
| PNet box count | ✅ Captured | ✅ Captured | ✅ Comparable |
| PNet timing | ✅ Captured | ❌ Not captured | ⚠️ Minor difference |
| RNet box count | ✅ Captured | ✅ Captured | ✅ Comparable |
| RNet timing | ✅ Captured | ❌ Not captured | ⚠️ Minor difference |
| ONet box count | ✅ Captured | ✅ Captured | ✅ Comparable |
| ONet timing | ✅ Captured | ❌ Not captured | ⚠️ Minor difference |
| Landmarks | ✅ Captured | ✅ Captured | ✅ Comparable |
| Bboxes | ✅ Captured | ✅ Captured | ✅ Comparable |

## Recommendations

### For Stage-by-Stage Comparison:

The C++ code provides **sufficient debug output** for stage-by-stage MTCNN comparison:

1. **Box Count Comparison:** Parse stdout for "BEFORE/AFTER NMS" messages
2. **Box Coordinate Comparison:** Read `/tmp/cpp_*_output.txt` files
3. **Landmark Comparison:** Already captured in ONet debug output

### Optional Enhancements (NOT REQUIRED):

If needed for timing comparisons, could add:
- High-resolution timestamps for each stage
- JSON/CSV output format for easier parsing

## Conclusion

✅ **The C++ OpenFace binary already has comprehensive MTCNN debug output that is directly comparable to the PyMTCNN debug mode we implemented.**

No modifications to the C++ binary are required for the comprehensive pipeline validation.
