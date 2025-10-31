# Face Alignment Investigation - Current Status

**Date:** 2025-10-29
**Session Focus:** Achieving C++-equivalent face alignment for AU prediction

## Summary of Progress

### What We Fixed ✅

**1. Face Alignment Rotation**
- **Problem:** Python produced tilted faces (preserving video head tilt)
- **Solution:** Apply inverse of CSV `p_rz` rotation: `angle = -p_rz`
- **Result:** Faces now corrected for head tilt, mean pixel diff ~17-18 from C++
- **File:** `openface22_face_aligner.py:107-118`

**2. Masking Support**
- **Status:** Already implemented in code
- **Features:** Masks out neck, ears, background using triangulation
- **File:** `openface22_face_aligner.py:117-139`
- **Triangulation file:** `tris_68_full.txt` (copied to working directory)

### Current Alignment Quality

**Visual Comparison (test_final_alignment_frame_*.png):**
- Python faces are upright (correcting video head tilt)
- Still appears slightly tilted compared to C++ (~1-2° remaining)
- Mean pixel difference: 17-18 (best result achieved)

**User Feedback:**
> "C++ output looks more like they used some sort of 3D effect to fully reorient the face"

This suggests C++ CalcParams does full 3D pose estimation (rx, ry, rz), not just 2D rotation (rz).

## Key Discovery: CalcParams Complexity

**From agent search:**
- No Python CalcParams implementation exists in codebase
- C++ CalcParams is ~200 lines of iterative optimization
- Estimates 40 parameters: 6 global (scale, rx, ry, rz, tx, ty) + 34 local (PCA)
- Uses 3D→2D projection with full 3D rotation matrix
- **Estimated effort to implement:** 3-4 weeks

## Blocking Issues

### 1. OpenFace Build Incomplete ❌

**Problem:**
- `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/` has no binaries
- FeatureExtraction not available for testing
- Persistent build errors: missing `limits` header, libraries not linking

**Impact:**
- Cannot run C++ CalcParams
- Cannot test AU predictions with our alignment
- Cannot validate if current alignment is "good enough"

### 2. AU Testing Blocked 🚫

**Test Strategy (created but cannot run):**
- File: `test_python_alignment_with_cpp_au.py`
- Plan: Create video of Python-aligned faces → Run C++ OpenFace → Compare AUs
- **Blocked by:** No FeatureExtraction binary

## Current Dependencies

### For Face Alignment (Working)
- ✅ PDM file: `In-the-wild_aligned_PDM_68.txt`
- ✅ CSV with landmarks + pose: `of22_validation/IMG_0942_left_mirrored.csv`
- ✅ Triangulation: `tris_68_full.txt`
- ✅ Python implementation: `openface22_face_aligner.py`

### For AU Prediction (Blocked)
- ❌ HOG extraction: Need either C++ binary or Python implementation
- ❌ AU validation: Need working OpenFace binary

## Three Paths Forward

### Option A: Fix OpenFace Build
**Goal:** Get FeatureExtraction binary working

**Steps:**
1. Investigate `limits` header issue (macOS/Xcode problem)
2. Try different compiler flags or CMake configuration
3. Build minimal version (just FeatureExtraction, not full OpenFace)

**Estimated Time:** 1-2 days
**Risk:** High (persistent build issues)
**Benefit:** Enables full testing + CalcParams access

### Option B: Implement Python HOG
**Goal:** Remove C++ dependency for feature extraction

**Steps:**
1. Implement OpenFace-compatible HOG in Python
2. Match C++ parameters exactly (cell size 8×8, block 2×2, 9 bins)
3. Validate features match C++ (correlation > 0.999)
4. Test AU predictions with Python HOG + current alignment

**Estimated Time:** 1-2 days
**Risk:** Medium (HOG is well-documented)
**Benefit:** Near-complete Python solution, only CSV preprocessing needs C++

### Option C: Accept Current Quality + Empirical Tuning
**Goal:** Work with what we have

**Steps:**
1. Assume current alignment is "close enough"
2. Apply small empirical corrections if needed
3. Test AU predictions qualitatively (visual inspection)
4. Document limitations

**Estimated Time:** < 1 day
**Risk:** Low
**Benefit:** Quick solution, may be sufficient

## Recommended Next Steps

**Priority 1: Locate Working OpenFace Binary**
- Check if CSV files came from different OpenFace installation
- Search system for any FeatureExtraction binary
- If found, run AU comparison test immediately

**Priority 2: Implement Python HOG (if no binary found)**
- Most practical path to completion
- Removes C++ dependency except CSV preprocessing
- Enables full AU testing

**Priority 3: CalcParams (only if AU predictions fail)**
- Complex 3-4 week effort
- Only pursue if simpler solutions don't work

## Questions for Decision

1. **Where did the CSV files come from originally?**
   - There may be a working OpenFace installation elsewhere

2. **What's the acceptable AU prediction error?**
   - If correlation > 0.90 is acceptable, current alignment may be sufficient
   - If need r > 0.95, may need CalcParams

3. **Can we visually validate AU predictions?**
   - Even without quantitative testing, qualitative checks may show if alignment works

4. **Is pure Python solution worth the effort?**
   - Python HOG: 1-2 days
   - Or accept C++ preprocessing dependency?

## Technical Documentation Created

- `CPP_DEPENDENCY_ANALYSIS.md` - Analysis of remaining C++ dependencies
- `ALIGNMENT_FIX_COMPLETE.md` - Documentation of alignment fix
- `CRITICAL_FINDING_DOUBLE_PDM_FITTING.md` - CalcParams discovery
- `test_python_alignment_with_cpp_au.py` - AU testing script (ready to run when binary found)

## Summary

**Alignment Quality:** Good (~17-18 pixel diff, ~1-2° tilt remaining)
**Main Blocker:** Cannot test AU predictions (no OpenFace binary)
**Best Path:** Find working OpenFace binary OR implement Python HOG
**CalcParams:** Complex, only if absolutely necessary
