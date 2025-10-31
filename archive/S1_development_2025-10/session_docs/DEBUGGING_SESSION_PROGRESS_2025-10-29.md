# Debugging Session Progress Report
**Date:** 2025-10-29
**Session Duration:** ~4 hours
**Status:** Significant progress - Core mystery identified

---

## Executive Summary

We have **successfully replicated OpenFace C++ PDM reconstruction** with perfect accuracy (RMSE < 0.1 pixels). We now fully understand how C++ processes landmarks from detection through PDM fitting to reconstruction.

**The remaining mystery:** Why does Python Kabsch alignment produce -8° to +2° rotation while C++ produces ~0° rotation when both use identical CSV landmarks and identical reference shapes?

---

## Major Achievements

### ✅ 1. Verified CSV Landmarks Source

**Finding:** CSV landmarks ARE PDM-reconstructed (not raw detections)

**Evidence:**
- Code trace: `FeatureExtraction.cpp:227` → `face_model.detected_landmarks` → `PDM.CalcShape2D()`
- Full pipeline: `CalcParams()` → `CalcShape2D()` → CSV output
- Confirmed by perfect reconstruction (see Achievement #2)

**Files:**
- `CSV_LANDMARKS_SOURCE_VERIFICATION.md` - Code trace documentation

### ✅ 2. Perfect PDM Reconstruction

**Achievement:** Replicated `CalcShape3D` + `CalcShape2D` with RMSE < 0.1 pixels

**Results:**
```
Frame    1: RMSE X=0.035 px, Y=0.047 px  ✓ Perfect match
Frame  493: RMSE X=0.045 px, Y=0.037 px  ✓ Perfect match
Frame  617: RMSE X=0.039 px, Y=0.038 px  ✓ Perfect match (eyes closed!)
Frame  863: RMSE X=0.060 px, Y=0.066 px  ✓ Perfect match
```

**Implementation:**
```python
# CalcShape3D: Apply expression deformation
shape_3d = mean_shape + principal_components @ params_local

# CalcShape2D: Apply pose transformation
R = euler_to_rotation_matrix(rx, ry, rz)  # 3D rotation
rotated = scale * (R @ shape_3d_points.T).T
landmarks_2d = rotated[:, :2] + translation
```

**Implications:**
- We fully understand C++ PDM pipeline
- Expression is encoded in `params_local` (p_0 through p_33)
- Rotation convention confirmed: XYZ Euler angles in radians
- PDM format confirmed: `[x0...x67, y0...y67, z0...z67]`

**Files:**
- `full_pdm_reconstruction.py` - Complete implementation
- `PDM_RECONSTRUCTION_SUCCESS.md` - Detailed documentation

### ✅ 3. Rotation Convention Verification

**Confirmed:** XYZ Euler angle convention, all parameters in radians

**Tests performed:**
- Tested radians vs degrees: ✓ Radians confirmed
- Tested XYZ vs ZYX order: ✓ XYZ confirmed (matches C++ line 72 comment)
- Tested rotation matrix construction: ✓ Matches C++ exactly

**Files:**
- `test_3d_to_2d_rotation_effect.py` - Rotation testing
- `compare_csv_vs_simulated_landmarks.py` - Validation

### ✅ 4. Kabsch Implementation Verification

**Finding:** Our Kabsch implementation is mathematically correct

**Evidence:**
- Identity test: landmarks = reference → rotation = 0.000000° ✓
- Matrix output: `[[1.0, 0.0], [0.0, 1.0]]` (perfect identity) ✓
- Follows standard Kabsch algorithm with SVD correction for reflection ✓

**Files:**
- `debug_python_vs_cpp_rotation.py` - Kabsch verification

---

## The Core Mystery

### Problem Statement

**Given:** Identical inputs (CSV landmarks, PDM reference shape, rigid indices)
**Observed:** Different rotation outputs

| Frame | Python Rotation | C++ Rotation (Visual) | Difference |
|-------|----------------|----------------------|------------|
| 1     | -8.79°        | ~0°                  | -8.79°     |
| 493   | -4.27°        | ~0°                  | -4.27°     |
| 617   | +2.17°        | ~0° (eyes closed!)   | +2.17°     |
| 863   | -2.89°        | ~0°                  | -2.89°     |

**Key observation:** Python shows 10.96° range and 6.44° expression sensitivity
**C++ shows:** ~0° across all frames, expression-invariant

### What We've Ruled Out

❌ **CSV landmarks are different** → No, we perfectly reconstruct them (RMSE < 0.1 px)
❌ **Rotation convention differs** → No, XYZ Euler confirmed via perfect reconstruction
❌ **Kabsch implementation bug** → No, produces perfect identity for identical inputs
❌ **Reference shape differs** → No, both use `pdm.mean_shape * 0.7` projected to 2D
❌ **Rigid indices differ** → No, hardcoded array matches C++ exactly

### Remaining Hypotheses

**Hypothesis A: C++ applies post-Kabsch correction**
- Maybe C++ computes rotation via Kabsch but then applies a correction
- Need to check if there's code after `AlignShapesWithScale()` we're missing
- Confidence: 40%

**Hypothesis B: C++ uses different reference shape at runtime**
- Maybe `pdm.mean_shape` gets transformed before alignment
- Could be related to params_global somehow
- Confidence: 30%

**Hypothesis C: Coordinate system difference**
- Maybe there's a flip/transform in how C++ interprets landmarks
- OpenCV vs NumPy coordinate conventions
- Confidence: 20%

**Hypothesis D: Something we haven't thought of**
- Unknown unknown
- Confidence: 10%

---

## Key Insights Gained

### 1. PDM Pipeline Understanding

**Complete flow:**
```
Raw Image
  ↓
Patch Experts (CE-CLM) → Raw landmark detections
  ↓
PDM.CalcParams() → Separates:
  - params_global [scale, rx, ry, rz, tx, ty]
  - params_local [p_0...p_33]
  ↓
PDM.CalcShape2D() → Reconstructs landmarks:
  - CalcShape3D: mean_shape + PCA × params_local
  - Apply 3D rotation (params_global)
  - Project to 2D
  - Add translation
  ↓
CSV Output (detected_landmarks)
  ↓
AlignFace() → Kabsch alignment
  ↓
Aligned face image
```

### 2. Expression Factorization Works

When we simulated the PDM pathway (3D rotation → 2D projection → Kabsch):
- Frame 493 vs 617 (eyes closed): Only 1.49° change
- This proves PDM-based approach IS expression-invariant
- But our actual Python alignment on CSV shows 6.44° change

**This suggests:** The CSV landmarks have expression factored in (via params_local), but something about how we're using them causes expression sensitivity to reappear.

### 3. The Rotation Sign Puzzle

When simulating PDM reconstruction → Kabsch:
- Simulated angles: +2.52°, +5.76°, +7.25°, +7.38° (POSITIVE)
- Actual Python: -8.79°, -4.27°, +2.17°, -2.89° (MOSTLY NEGATIVE)

**This suggests:** There might be a sign flip or coordinate system difference we're not accounting for.

---

## Investigation Tools Created

### Analysis Scripts
1. `check_landmark_coordinate_system.py` - Pose parameter analysis
2. `test_reference_rotation_correction.py` - Reference shape rotation testing
3. `test_pose_rotation_correction.py` - Pose-based correction attempts
4. `test_3d_to_2d_rotation_effect.py` - 3D rotation projection analysis
5. `compare_csv_vs_simulated_landmarks.py` - PDM reconstruction validation
6. `full_pdm_reconstruction.py` - Complete PDM replication
7. `debug_python_vs_cpp_rotation.py` - Kabsch verification

### Documentation
1. `CSV_LANDMARKS_SOURCE_VERIFICATION.md` - Code trace evidence
2. `PDM_RECONSTRUCTION_SUCCESS.md` - Achievement summary
3. `STRATEGIC_DECISION_PYTHON_OPENFACE_ARCHITECTURE.md` - Architecture analysis

---

## External LLM Analysis Summary

We received strategic guidance on Python OpenFace implementation:

**Key Recommendations:**
1. **Option B (Hybrid architecture) preferred** - Use C++ for PDM fitting, Python for alignment/HOG/AU
2. **PDM implementation complexity:** 2-4 weeks, moderate difficulty, high numerical precision risk
3. **PyInstaller feasibility:** Confirmed viable with 70-140 MB overhead
4. **Validation strategies:** Comprehensive tolerance-based testing approach

**Alternative approaches suggested:**
- `eos-py` library (actively maintained, Dec 2024 release)
- `img2pose` for 6-DOF alignment without PDM
- pybind11 wrapper for selective C++ integration

**Licensing:** All dependencies (OpenBLAS, OpenCV, Boost, dlib) use permissive licenses - no restrictions

---

## Next Steps (Priority Order)

### Immediate (Next 1-2 days)

**Option 1: Find the C++ difference**
- [ ] Instrument C++ AlignFace to print scale_rot matrix
- [ ] Compare C++ matrix values to Python matrix values
- [ ] Check if there's any post-processing after Kabsch we're missing
- [ ] Look for coordinate transforms between landmark detection and alignment

**Option 2: Test alternative hypothesis**
- [ ] Check if C++ uses raw detections instead of PDM-reconstructed landmarks for alignment
- [ ] Verify face_analyser.AddNextFrame() passes correct landmarks
- [ ] Compare GetLatestAlignedFace() output to our Python alignment

### Short-term (This week)

If debugging succeeds:
- [ ] Document the fix
- [ ] Validate AU model compatibility
- [ ] Test on multiple videos

If debugging fails after 2 more days:
- [ ] Implement pybind11 wrapper for CalcParams/CalcShape2D (2-3 days)
- [ ] Keep Python alignment/HOG/AU inference
- [ ] Package with PyInstaller

### Medium-term (Next 2 weeks)

- [ ] Cross-platform testing (Windows, macOS, Linux)
- [ ] Performance optimization
- [ ] Comprehensive validation suite
- [ ] Distribution packaging

---

## Resources and References

### Code Locations

**OpenFace C++ (critical files):**
- `/OpenFace/lib/local/LandmarkDetector/src/PDM.cpp` - PDM implementation
  - Line 153: CalcShape3D
  - Line 159-188: CalcShape2D
- `/OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp` - Alignment
  - Line 110-146: AlignFace
- `/OpenFace/lib/local/Utilities/include/RotationHelpers.h`
  - Line 47-70: Euler2RotationMatrix
  - Line 195-242: AlignShapesWithScale
- `/OpenFace/exe/FeatureExtraction/FeatureExtraction.cpp`
  - Line 193: face_analyser.AddNextFrame
  - Line 227: SetObservationLandmarks

**Our Python Implementation:**
- `pdm_parser.py` - PDM file parser ✅
- `openface22_face_aligner.py` - Face alignment (needs fix)
- `full_pdm_reconstruction.py` - PDM reconstruction ✅

### Test Data

- `of22_validation/IMG_0942_left_mirrored.csv` - C++ output with params
- `of22_validation/IMG_0942_left_mirrored/` - Video frames
- `pyfhog_validation_output/IMG_0942_left_mirrored_aligned/` - C++ aligned faces (ground truth)
- `In-the-wild_aligned_PDM_68.txt` - PDM model file

---

## Metrics

**Code Quality:**
- PDM reconstruction: RMSE < 0.1 pixels (perfect)
- Kabsch identity test: 0.000000° (perfect)
- Rotation convention: Confirmed via reconstruction

**Remaining Gap:**
- Rotation difference: 3° to 11° (needs fix)
- Expression sensitivity: 6.44° (vs C++ ~0°)

**Timeline:**
- Investigation time so far: ~4 hours
- Estimated time to solution:
  - If debugging succeeds: 1-2 days
  - If hybrid approach needed: 3-5 days

---

## Decision Point

**We are at a critical juncture:**

### Path A: Continue Debugging (1-2 more days)
**Pros:**
- We're very close - perfect PDM reconstruction achieved
- Pure Python solution if successful
- Deep understanding of the system

**Cons:**
- Might hit a wall (unknown unknown)
- Time investment with uncertain outcome

### Path B: Hybrid Approach Now (3-5 days)
**Pros:**
- Guaranteed to work (use C++ PDM fitting)
- Matches industry standard (every major CV project uses hybrid)
- Leverages our successful HOG/AU replication

**Cons:**
- Small C++ dependency (~70-140 MB)
- Feels like "giving up" on pure Python

**Recommendation:** Give Path A one more focused day. If we don't find the rotation difference cause by tomorrow evening, switch to Path B.

---

## Questions for User

1. **Timeline pressure:** Is 1-2 more days of debugging acceptable, or should we switch to hybrid approach now?

2. **Purity vs pragmatism:** How important is "100% Python" vs "Python-first with minimal C++ for complex PDM fitting"?

3. **Distribution target:** Are we packaging for researchers who need exact OpenFace replication, or is close-enough acceptable?

---

## Appendix: Technical Details

### PDM Format

```
Storage: [x0, x1, ..., x67, y0, y1, ..., y67, z0, z1, ..., z67]
Shape: (204, 1)
Principal components: (204, 34)
```

### Euler Angle Convention

```python
# XYZ order (confirmed)
R = Rx(pitch) @ Ry(yaw) @ Rz(roll)

# Matrix:
R[0,0] = c2*c3
R[0,1] = -c2*s3
R[0,2] = s2
R[1,0] = c1*s3 + c3*s1*s2
R[1,1] = c1*c3 - s1*s2*s3
R[1,2] = -c2*s1
R[2,0] = s1*s3 - c1*c3*s2
R[2,1] = c3*s1 + c1*s2*s3
R[2,2] = c1*c2
```

### Rigid Indices

```python
[1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35,
 36, 39, 40, 41, 42, 45, 46, 47]
# 24 landmarks from forehead, nose bridge, eye corners
```

---

**End of Progress Report**
