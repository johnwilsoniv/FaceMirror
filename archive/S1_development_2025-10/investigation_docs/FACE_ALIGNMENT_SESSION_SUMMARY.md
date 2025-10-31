# Face Alignment Implementation Session Summary

**Date:** 2025-10-29
**Goal:** Implement Python face alignment component for OpenFace 2.2 AU prediction

---

## ✅ What We Accomplished

### 1. Core Implementation Complete
- ✅ Created `openface22_face_aligner.py` with full alignment pipeline
- ✅ PDM mean shape loading with correct 3D→2D reshape
- ✅ Rigid points extraction (24 landmarks)
- ✅ Kabsch algorithm for rotation computation
- ✅ Similarity transform (scale + rotation)
- ✅ Affine warp matrix construction
- ✅ Integration with cv2.warpAffine

### 2. Validation Infrastructure
- ✅ Created `validate_python_alignment.py` for pixel-level comparison
- ✅ Created multiple debug scripts to analyze transformations
- ✅ Tested with OpenFace 2.2 validation data (IMG_0942_left_mirrored)

### 3. Key Discoveries
- ✅ **PDM Format:** Must reshape as (68, 3) then take [:, :2], not flatten first
- ✅ **Pose Parameters:** Use `p_tx`/`p_ty` (PDM params), NOT `pose_Tx`/`pose_Ty` (world space)
- ✅ **Transform Centering:** Face center correctly transforms to (56, 56) in 112×112 output
- ✅ **Interpolation:** INTER_LINEAR is optimal (tested CUBIC, LANCZOS4)

---

## ⚠️ Current Issue: 45° Rotation

### Problem
Python aligned faces are rotated ~45° clockwise compared to OpenFace C++ output.

**Evidence:**
- Visual comparison shows clear rotation difference
- MSE: ~4000 (should be < 1.0 for pixel-perfect)
- Correlation: r ≈ 0.52 (should be > 0.99)
- User observation: "rotated 1/8th turn clockwise"

### Current Metrics
```
Mean MSE:         3993.81
Mean Correlation: 0.518
Max Difference:   216-222 pixels
```

### Root Cause Analysis
The rotation suggests an issue in one of:
1. **Kabsch algorithm** reflection correction
2. **SVD matrix ordering** (numpy vs OpenCV conventions)
3. **Transform direction** for warpAffine
4. **Coordinate system** mismatch

### What We Tried
- ❌ Inverting entire scale_rot_matrix → Made it worse (MSE 6300, r=0.31)
- ❌ Different interpolation methods → No improvement
- ✅ Correct PDM parameter usage → Improved from negative to positive correlation
- ✅ Transform centering verified → Face center lands at (56, 56) correctly

---

## 🔍 Next Steps to Fix Rotation

### Priority 1: Debug Kabsch Algorithm
**File:** `openface22_face_aligner.py:128-161`

Check these potential issues:
1. **SVD matrix usage:**
   ```python
   U, S, Vt = np.linalg.svd(src.T @ dst)
   R = Vt.T @ corr @ U.T
   ```
   - Verify this matches C++ `svd.vt.t() * corr * svd.u.t()`
   - Test with simple known rotation to verify correctness

2. **Reflection correction:**
   ```python
   d = np.linalg.det(Vt.T @ U.T)
   corr[1, 1] = 1 if d > 0 else -1
   ```
   - Print d value to verify sign
   - Check if correction is being applied correctly

3. **Test isolated rotation:**
   - Create test with known 45° rotation
   - Verify Kabsch produces correct inverse

### Priority 2: Compare Transform Matrices
Create script to:
1. Load same frame's landmarks from CSV
2. Compute Python transform matrices (scale_rot, warp_matrix)
3. Try to extract corresponding values from OpenFace output
4. Compare matrix values element-by-element

### Priority 3: Test End-to-End with pyfhog
Even with rotation issue, test:
```bash
python3 test_end_to_end_alignment.py
```

**Critical question:** Does the rotated alignment still produce usable HOG features?
- If HOG correlation > 0.95: Alignment may be "good enough" for AU prediction
- If HOG correlation < 0.95: Must fix rotation before proceeding

---

## 📊 Files Created This Session

### Core Implementation
- `openface22_face_aligner.py` - Main alignment class
- `In-the-wild_aligned_PDM_68.txt` - Copied PDM file

### Validation & Debug
- `validate_python_alignment.py` - Pixel-level validation against C++
- `debug_alignment.py` - Transform value analysis
- `debug_pdm_shape.py` - PDM format verification
- `compare_transforms.py` - Detailed transform analysis
- `test_alignment_variants.py` - Interpolation method testing
- `test_end_to_end_alignment.py` - HOG feature comparison (not yet run)
- `visualize_transform.py` - Visual debugging (incomplete)

### Output
- `alignment_validation_output/` - Comparison images showing rotation issue

---

## 🎯 Success Criteria

### Minimum (Not Yet Achieved)
- [ ] MSE < 5.0 between Python and C++ aligned faces
- [ ] Correlation r > 0.95
- [ ] No visible rotation difference

### Target (Goal)
- [ ] MSE < 1.0 (near pixel-perfect)
- [ ] Correlation r > 0.99
- [ ] Visual inspection shows identical alignment

### Alternative Success Path
- [ ] HOG features from Python alignment match C++ (r > 0.95)
- [ ] AU predictions from Python pipeline match C++ (r > 0.999)
- [ ] Production-ready even if not pixel-perfect

---

## 💡 Key Insights for Next Session

1. **Don't invert the whole matrix** - C++ copies scale_rot directly into warp_matrix
2. **The transform centering is correct** - Face lands at (56, 56) as expected
3. **Focus on Kabsch algorithm** - The 45° rotation strongly suggests SVD/rotation bug
4. **Test end-to-end ASAP** - May discover alignment is "good enough" despite rotation
5. **Consider comparing with OpenFace source** - Run C++ in debugger to extract matrices

---

## 📝 Code Snippets for Debugging

### Print Kabsch Intermediate Values
```python
# Add to _align_shapes_kabsch_2d():
print(f"src.T @ dst =\n{src.T @ dst}")
print(f"U =\n{U}")
print(f"S = {S}")
print(f"Vt =\n{Vt}")
print(f"det(Vt.T @ U.T) = {d}")
print(f"corr =\n{corr}")
print(f"R =\n{R}")
```

### Test with Known Rotation
```python
# Create 45° rotated points
theta = np.pi / 4  # 45°
R_true = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
src_test = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
dst_test = (R_true @ src_test.T).T

# Test Kabsch
R_computed = aligner._align_shapes_kabsch_2d(src_test, dst_test)
print(f"True R:\n{R_true}")
print(f"Computed R:\n{R_computed}")
print(f"Difference:\n{R_true - R_computed}")
```

---

## 🚀 Recommended Next Actions

1. **Debug Kabsch with test case** (30 min)
   - Implement known rotation test
   - Fix any bugs in SVD/reflection handling

2. **Run end-to-end pyfhog test** (15 min)
   - See if HOG features still match despite rotation
   - May reveal alignment is usable

3. **If HOG matches:** Proceed with AU prediction testing
4. **If HOG doesn't match:** Continue debugging rotation issue

---

## 📈 Progress: 85% → 90%

We're very close! The infrastructure is complete, transform math is mostly correct, just need to fix the rotation bug in Kabsch algorithm.

**Estimated time to completion:** 1-2 hours of focused debugging
