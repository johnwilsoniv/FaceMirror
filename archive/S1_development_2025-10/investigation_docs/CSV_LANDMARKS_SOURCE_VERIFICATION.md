# CSV Landmarks Source Verification

## Finding: CSV Contains PDM-Reconstructed Landmarks

**Date:** 2025-10-29

### Code Trace

**FeatureExtraction.cpp line 227:**
```cpp
open_face_rec.SetObservationLandmarks(face_model.detected_landmarks,
    face_model.GetShape(...), ...)
```

**RecorderOpenFace.cpp line 466:**
```cpp
void SetObservationLandmarks(const cv::Mat_<float>& landmarks_2D, ...) {
    this->landmarks_2D = landmarks_2D;
}
```

**RecorderOpenFace.cpp line 339:**
```cpp
this->csv_recorder.WriteLine(face_id, frame_number, timestamp,
    landmark_detection_success, landmark_detection_confidence,
    landmarks_2D, landmarks_3D, pdm_params_local, pdm_params_global, ...)
```

**RecorderCSV.cpp line 256:**
```cpp
// Output the detected 2D facial landmarks
if (output_2D_landmarks) {
    for (auto lmk : landmarks_2D) {
        output_file << "," << lmk;
    }
}
```

### What is face_model.detected_landmarks?

From LandmarkDetectorModel.cpp line 635:
```cpp
// Store the landmarks converged on in detected_landmarks
pdm.CalcShape2D(detected_landmarks, params_local, params_global);
```

**This proves:**
1. `detected_landmarks` is the OUTPUT of `PDM.CalcShape2D()`
2. `CalcShape2D()` takes params_global + params_local and reconstructs the face shape
3. This happens AFTER PDM fitting (CalcParams)
4. The landmarks are expression-factored (expression is in params_local)

### Implication

**The CSV landmarks (x_0...x_67, y_0...y_67) are PDM-reconstructed, not raw detections.**

This means:
- ✅ Expression has already been factored into params_local
- ✅ Rigid pose has already been factored into params_global
- ✅ The landmarks represent the "best fit" PDM model to the detected patches

### Why This Matters

**If landmarks are already expression-factored, why is our Python alignment expression-sensitive?**

The PDM reconstruction applies:
```cpp
out_shape.at<float>(i,0) = s * (R[0,0]*x + R[0,1]*y + R[0,2]*z) + tx;
out_shape.at<float>(i+n,0) = s * (R[1,0]*x + R[1,1]*y + R[1,2]*z) + ty;
```

Where R is the 3D rotation matrix from params_global (p_rx, p_ry, p_rz).

**The landmarks already have the 3D → 2D projection applied, including rotation.**

This means the landmarks in the CSV are NOT in the same coordinate frame as the PDM mean_shape! They've been rotated by params_global rotation matrix.

## Next Investigation

If the CSV landmarks are rotated by params_global, then we need to either:
1. **UN-rotate them back** to PDM canonical frame before aligning
2. **Rotate the PDM mean_shape** by params_global before aligning
3. Use params_global rotation directly instead of computing via Kabsch

This explains why C++ AlignFace works but our Python doesn't - C++ must be handling this coordinate transform that we're missing.
