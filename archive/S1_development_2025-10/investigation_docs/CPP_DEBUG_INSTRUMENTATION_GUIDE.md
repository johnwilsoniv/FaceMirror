# C++ Debug Instrumentation Guide

**Goal:** Find exact point where Python and C++ alignment diverge

**Status:** Ready to apply patch and compare

---

## What We're Debugging

**Known:** Both Python and C++ use:
- Same CSV landmarks (PDM-reconstructed)
- Same PDM reference shape
- Same rigid point indices
- Same Kabsch algorithm

**Mystery:** Different rotation outputs:
- Python: -8.79° to +2.17° (10.96° range, 6.44° expression sensitivity)
- C++: ~0° (upright, expression-invariant)

**Hypothesis:** There's a subtle difference in either:
1. The input values to Kabsch
2. The Kabsch computation itself
3. Post-Kabsch processing

---

## Step 1: Apply C++ Debug Patch

**Run:**
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"
./apply_cpp_debug_patch.sh
```

**What this does:**
1. Backs up original `Face_utils.cpp`
2. Adds debug output after `AlignShapesWithScale()` call
3. Rebuilds OpenFace (takes ~2-5 minutes)
4. Creates instrumented binary

**Debug output includes:**
- Frame number (for frames 1, 493, 617, 863)
- First 3 source landmarks (to verify they match CSV)
- First 3 destination landmarks (to verify they match PDM)
- Scale-rot matrix (2×2)
- Rotation angle computed from matrix
- params_global values

---

## Step 2: Run Instrumented C++ FeatureExtraction

**Find the video file:**
```bash
# The video should be wherever you have it
# Example path (adjust if needed):
VIDEO="/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
```

**Run with debug output:**
```bash
cd /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin

./FeatureExtraction -f "$VIDEO" 2>&1 | tee /tmp/cpp_debug_output.txt
```

**Extract just the debug lines:**
```bash
grep -A 30 "=== DEBUG Frame" /tmp/cpp_debug_output.txt > /tmp/cpp_debug_clean.txt
cat /tmp/cpp_debug_clean.txt
```

---

## Step 3: Compare C++ vs Python Values

**Python values (already captured):**

```
=== DEBUG Frame 1 ===
Source landmarks (first 3):
  [0]: (316.799988, 906.900024)
  [1]: (320.700012, 956.099976)
  [2]: (328.299988, 1005.400024)
Dest landmarks (first 3):
  [0]: (-49.373547, -46.795041)
  [1]: (-41.853131, -33.858281)
  [2]: (-23.884771, -12.512788)
Scale-rot matrix:
  [0.246873, 0.038175]
  [-0.038175, 0.246873]
Rotation angle: -8.790155°

=== DEBUG Frame 493 ===
Source landmarks (first 3):
  [0]: (262.200012, 953.099976)
  [1]: (269.799988, 1002.599976)
  [2]: (281.399994, 1051.800049)
Dest landmarks (first 3):
  [0]: (-49.373547, -46.795041)
  [1]: (-41.853131, -33.858281)
  [2]: (-23.884771, -12.512788)
Scale-rot matrix:
  [0.242406, 0.018090]
  [-0.018090, 0.242406]
Rotation angle: -4.267885°

=== DEBUG Frame 617 ===
Source landmarks (first 3):
  [0]: (269.500000, 951.200012)
  [1]: (275.799988, 1001.000000)
  [2]: (287.700012, 1050.000000)
Dest landmarks (first 3):
  [0]: (-49.373547, -46.795041)
  [1]: (-41.853131, -33.858281)
  [2]: (-23.884771, -12.512788)
Scale-rot matrix:
  [0.249250, -0.009450]
  [0.009450, 0.249250]
Rotation angle: 2.171295°

=== DEBUG Frame 863 ===
Source landmarks (first 3):
  [0]: (269.399994, 944.000000)
  [1]: (277.399994, 993.500000)
  [2]: (288.799988, 1042.500000)
Dest landmarks (first 3):
  [0]: (-49.373547, -46.795041)
  [1]: (-41.853131, -33.858281)
  [2]: (-23.884771, -12.512788)
Scale-rot matrix:
  [0.243468, 0.012291]
  [-0.012291, 0.243468]
Rotation angle: -2.890081°
```

**What to compare:**

### Check 1: Source Landmarks
**Expected:** Should be IDENTICAL (both from same CSV)
**If different:** Something is wrong with how C++ loads landmarks from CSV
**Action:** Need to trace back further in C++ pipeline

### Check 2: Destination Landmarks
**Expected:** Should be IDENTICAL (both from same PDM × 0.7)
**If different:** PDM loading or scaling differs
**Action:** Check PDM file loading code

### Check 3: Scale-Rot Matrix
**Expected:** THIS is where we expect the difference
**C++ likely produces:** Matrix with near-zero rotation (~0°)
**Python produces:** Matrix with -8.79° to +2.17° rotation

**Analysis:**
- If C++ matrix = `[[0.25, 0.00], [0.00, 0.25]]` (pure scale, no rotation)
  → C++ is applying some correction we're missing
- If C++ matrix has rotation but different from Python
  → Kabsch algorithm implementation differs
- If C++ matrix matches Python
  → Problem is after Kabsch (unlikely but possible)

### Check 4: Rotation Angle
**Expected:** C++ ~0°, Python -8.79° to +2.17°
**This confirms the problem we're trying to fix**

---

## Step 4: Analyze the Deviation

### Scenario A: Source or Dest landmarks differ

**Problem:** Input to Kabsch is different
**Solution:** Need to trace why landmarks differ
**Action:**
- Check if C++ uses raw detections vs PDM-reconstructed
- Verify rigid point extraction logic
- Check coordinate system transforms

### Scenario B: Landmarks identical, matrix differs

**Problem:** Kabsch computation differs
**Solution:** Compare Kabsch implementation line-by-line
**Action:**
- Check AlignShapesWithScale() in RotationHelpers.h
- Compare centering, normalization, SVD, correction matrix
- Look for subtle numerical differences

### Scenario C: Matrix identical, but output still differs

**Problem:** Post-Kabsch processing differs
**Solution:** Check what happens after scale_rot_matrix computation
**Action:**
- Look at warp_matrix construction
- Check translation computation
- Verify cv::warpAffine parameters

---

## Step 5: Apply Fix

Once we identify the deviation:

### If C++ applies a correction:
```python
# In openface22_face_aligner.py, after Kabsch:
scale_rot = self._align_shapes_with_scale(source_rigid, dest_rigid)
# ADD CORRECTION HERE based on what C++ does
```

### If Kabsch differs:
```python
# Fix _align_shapes_with_scale() method to match C++ exactly
```

### If inputs differ:
```python
# Fix landmark extraction or reference shape loading
```

---

## Quick Reference Commands

**Apply patch:**
```bash
./apply_cpp_debug_patch.sh
```

**Run instrumented C++:**
```bash
cd /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin
./FeatureExtraction -f /path/to/IMG_0942_left_mirrored.mp4 2>&1 | grep -A 30 "=== DEBUG Frame"
```

**Restore original C++:**
```bash
cp /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp.backup \
   /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp
cd /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build
make
```

**Regenerate Python debug values:**
```bash
python3 print_python_debug_values.py
```

---

## Expected Timeline

- **Step 1 (Apply patch):** 5 minutes (includes rebuild)
- **Step 2 (Run C++):** 2-3 minutes (video processing)
- **Step 3 (Compare):** 5 minutes (visual inspection)
- **Step 4 (Analyze):** 15-30 minutes (debugging)
- **Step 5 (Apply fix):** 10-30 minutes (depends on fix complexity)

**Total:** 1-2 hours to solution

---

## Success Criteria

After applying fix:
- [ ] Python rotation angle matches C++ (~0° ± 1°)
- [ ] Rotation stability across frames (std < 2°)
- [ ] Expression invariance (< 2° change eyes open vs closed)
- [ ] Visual inspection: Aligned faces look upright
- [ ] AU model compatibility: Predictions match C++ output

---

## Files Created

1. `apply_cpp_debug_patch.sh` - Automated patch application
2. `print_python_debug_values.py` - Python debug output
3. `cpp_debug_patch.txt` - Manual patch instructions
4. `CPP_DEBUG_INSTRUMENTATION_GUIDE.md` - This file

---

## Next Steps

1. Run `./apply_cpp_debug_patch.sh`
2. Capture C++ debug output
3. Report findings in this conversation
4. We'll analyze together and apply the fix
