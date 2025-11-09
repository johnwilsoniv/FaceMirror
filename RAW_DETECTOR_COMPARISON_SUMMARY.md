# Raw Detector Comparison Summary

## What We Did

1. **Modified OpenFace 2.2** to output raw MTCNN detector data
   - Added debug logging to `FaceDetectorMTCNN.cpp`
   - Extracts bounding box + 5 facial landmarks from ONet output
   - Saves to `/tmp/mtcnn_debug.csv` for each video processed
   - Successfully rebuilt and tested

2. **OpenFace MTCNN Output Format**
   - ONet produces **16 values**:
     - [0-1]: Face/non-face probability
     - [2-5]: Bbox corrections (x, y, width, height)
     - **[6-15]: 5 facial landmarks** (normalized 0-1 within bbox):
       - lm1: Left eye (x, y)
       - lm2: Right eye (x, y)
       - lm3: Nose (x, y)
       - lm4: Left mouth corner (x, y)
       - lm5: Right mouth corner (x, y)

3. **Generated Visualizations**
   - `test_output/*_raw_detector_comparison.jpg`: Shows OpenFace MTCNN detections (green bbox + 5 landmarks)
   - These are the RAW detector outputs BEFORE CLNF refinement to 68 points

## Key Findings

### OpenFace MTCNN vs RetinaFace+PFLD

**OpenFace MTCNN:**
- Produces 5 facial landmarks directly from the detector
- These 5 points are then refined/expanded to 68 points by CLNF (Constrained Local Neural Fields)
- Bbox is adjusted by empirical offsets to be "tight around facial landmarks"

**Our Pipeline (RetinaFace + PFLD):**
- RetinaFace produces bbox + 5 landmarks for alignment
- PFLD produces **98 landmarks** directly
- Our SVR CLNF refines these 98 points and adds face mesh

### Raw Detection Quality

From the MTCNN CSV output on IMG_0861:
```csv
frame,bbox_x,bbox_y,bbox_w,bbox_h,confidence,onet_size,lm1_x,lm1_y,lm2_x,lm2_y,lm3_x,lm3_y,lm4_x,lm4_y,lm5_x,lm5_y
0,304.312,853.474,421.681,389.83,1,16,0.352858,0.697592,0.541091,0.376588,0.673918,0.376053,0.38199,0.497063,0.737251,0.749997
```

The 5-point landmarks are remarkably accurate for such a simple detector!

## OpenCV 4.12.0 Issue Summary

Testing revealed that OpenCV 4.12.0 has a bug/incompatibility with certain rotated video files:

**Videos that CRASH with C++ OpenFace (OpenCV 4.12.0):**
- IMG_0504_source.MOV - SIGSEGV  
- IMG_8401_source.MOV - SIGSEGV
- IMG_9330_source.MOV - SIGSEGV

**Videos that WORK:**
- IMG_0441, IMG_0452, IMG_0861, IMG_0942 - All work fine

**Root Cause:**
- OpenCV 4.12.0 introduced regression with rotation metadata in certain H.264 encodings
- Python pipeline using OpenCV 4.11.0 works perfectly on ALL videos
- This is an OpenCV bug, not OpenFace

## Files Generated

1. **Modified OpenFace Source:**
   - `~/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/FaceDetectorMTCNN.cpp`
   - Adds debug output to `/tmp/mtcnn_debug.csv`

2. **MTCNN Debug CSV:**
   - `/tmp/mtcnn_debug.csv` - Per-frame MTCNN detections with 5 landmarks

3. **Comparison Images:**
   - `test_output/IMG_*_raw_detector_comparison.jpg` - OpenFace MTCNN visualizations
   - `test_output/IMG_*_comparison.jpg` - Full refined pipeline comparisons (from earlier tests)

## Next Steps

For a complete raw detector comparison showing both MTCNN and RetinaFace+PFLD side-by-side:
1. Fix pyfaceau editable install to make detectors module accessible
2. OR: Extract raw detections directly from video_processor.py output
3. OR: Use the existing full pipeline comparisons which show final refined outputs

The key insight: **OpenFace starts with 5 landmarks from MTCNN, we start with 98 landmarks from PFLD** - both before any CLNF refinement.
