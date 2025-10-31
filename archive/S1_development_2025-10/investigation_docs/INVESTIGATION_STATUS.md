# Face Alignment Investigation - Current Status

## Latest Findings (2025-10-29)

### Problem Summary
Python implementation produces faces with consistent ~5° counter-clockwise tilt, while C++ OpenFace produces upright faces.

### Key Discovery: PDM Mean Shape is Rotated!

The PDM mean shape in `In-the-wild_aligned_PDM_68.txt` is itself rotated approximately **45° counter-clockwise**.

**Evidence:**
- Visualization shows face tilted diagonally (see `pdm_mean_shape_visualization.png`)
- Jaw line (landmarks 0-16) runs from upper-left to lower-right
- Nose (landmarks 27-35) points diagonally up-right
- Measured angles:
  - Nose: 64° from vertical
  - Eyes: -130° from horizontal

### Rotation Consistency Analysis

After implementing C++-matching algorithm (no transpose, transform pose through rotation):

| Metric | Value | Status |
|--------|-------|--------|
| Mean rotation | -4.78° | Consistent CCW tilt |
| Std deviation | 2.76° | ✓ Good consistency |
| Range | 10.96° | Frame 1: -8.79°, Frame 617: +2.17° |

**Conclusion:** Rotation is now **consistent** across frames (no progressive drift), but **offset** by ~5° CCW.

### Current Implementation vs C++

| Aspect | Python | C++ | Match? |
|--------|--------|-----|--------|
| PDM file | In-the-wild_aligned_PDM_68.txt | Same | ✓ |
| PDM loading | First 136 values, reshape(68,2) | First 136, reshape(1,2).t() | ✓ (equivalent) |
| Rigid points | 24 indices | 24 indices | ✓ |
| Kabsch algorithm | Correct | Same | ✓ |
| Matrix transpose | NO (removed) | NO | ✓ |
| Pose transform | Through scale-rotation | Through scale-rotation | ✓ |
| Empirical shifts | NO (removed) | NO | ✓ |
| Rigid parameter | true | true | ✓ |

**All steps match C++, yet output differs!**

### Hypothesis

The PDM mean shape rotation is intentional and represents the average orientation in the training data. However:

1. **C++ may apply additional correction** we haven't found yet
2. **The landmarks in the CSV may be pre-rotated** to account for this
3. **There may be a coordinate system difference** between PDM space and image space

### Questions to Investigate

1. ✓ Does C++ use rigid points? **YES** (rigid=true)
2. ✓ Does C++ transpose the matrix? **NO**
3. ✓ Does C++ transform pose through rotation? **YES**
4. ❓ **Is the PDM mean shape supposed to be rotated?**
5. ❓ **Does C++ apply an additional rotation correction step we're missing?**
6. ❓ **Are the CSV landmarks already adjusted for PDM orientation?**

### Next Steps

**Option A: Search for rotation correction in C++**
- Look for any post-processing after AlignFace
- Check if params_global[1:4] (rx, ry, rz) are used anywhere
- Search for additional rotation matrices

**Option B: Test with identity rotation**
- Try forcing rotation matrix to identity (scale only)
- See if that produces upright faces

**Option C: Examine actual C++ transform matrices**
- Add debug output to C++ and compare exact matrix values
- See if C++ matrices also have ~5° rotation

### Files for Reference

- `CPP_ALIGNMENT_ALGORITHM_ANALYSIS.md` - Complete C++ breakdown
- `CPP_VS_PYTHON_COMPARISON.md` - Side-by-side comparison
- `pdm_mean_shape_visualization.png` - Visual proof PDM is rotated
- `measure_rotation_consistency.py` - Tool to measure rotation angles

### Current Metrics

- Correlation: 0.749 (mean)
- MSE: 2213 (mean)
- Rotation std: 2.76° (good consistency)
- Visual quality: Faces look good but tilted ~5° CCW compared to C++
