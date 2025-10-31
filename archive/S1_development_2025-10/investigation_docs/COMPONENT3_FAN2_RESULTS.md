# Component 3: FAN2 Results Summary

## Final Results

After fixing the coordinate system bug, FAN2 shows **significant improvement** over PFLD:

| Model | Mean RMSE | Median RMSE | Min RMSE | Max RMSE | Size | Improvement |
|-------|-----------|-------------|----------|----------|------|-------------|
| **PFLD** | 13.26 px | 13.52 px | 10.37 px | 14.76 px | 2.8MB | Baseline |
| **FAN2** | **6.95 px** | 7.03 px | 6.22 px | 7.37 px | 50.9MB | **47.6% better** |
| **Target** | < 3 px | - | - | - | - | - |

## Model Details

**FAN2 (Face Alignment Network 2)**
- **Source**: HuggingFace bluefoxcreation/FaceAlignment
- **Training**: 300W-LP + iBUG datasets (official face alignment benchmarks)
- **Architecture**: Hourglass network (coordinate-based variant)
- **Input**: 256×256 RGB, normalized [0-1]
- **Output**: 68 landmarks in 64×64 heatmap space + confidence scores
- **Size**: 50.9MB (vs 2.8MB PFLD, vs ~95MB dlib)

## Key Finding: Coordinate System

**Critical Bug Discovery**: FAN2 outputs coordinates in **64×64 space** (heatmap resolution), not 256×256 (input resolution).

**Incorrect scaling** (initial implementation):
```python
scale = crop_size / 256.0  # WRONG - caused 318px error!
```

**Correct scaling** (fixed):
```python
scale = crop_size / 64.0  # CORRECT - achieves 6.95px accuracy
```

This aligns with FAN architecture which uses 64×64 heatmaps for landmark localization.

## Accuracy Analysis

### Strengths ✅
- **47.6% more accurate** than PFLD
- **100% detection rate** (50/50 frames)
- **Consistent error**: 6.22-7.37px range (low variance)
- **High confidence**: Mean score 0.754
- **Trained on iBUG format**: Should match OpenFace expectations

### Limitations ⚠️
- **Above 3-pixel target**: 6.95px vs <3px goal
- **2.3× target error**: May impact AU accuracy
- **Larger size**: 50.9MB vs 2.8MB PFLD
- **Slower inference**: ~0.03s vs ~0.01s (still acceptable for CPU)

## Comparison to Published Benchmarks

**FAN (original paper)** on 300W:
- Common subset: 3.28 NME
- Challenging subset: 5.87 NME
- Full set: 3.78 NME

**Our FAN2 ONNX variant**:
- ~6.95 pixel error on 1920×1080 video
- Inter-ocular distance ~120px → 6.95/120 = **5.79% NME**
- Slightly worse than paper but in reasonable range

**Possible reasons for difference**:
1. ONNX conversion may introduce small accuracy loss
2. Different test video (in-the-wild vs controlled 300W)
3. 64×64 heatmap resolution limit
4. Coordinate-based variant vs original heatmap-based

## Decision Matrix

### Option A: Use FAN2 (Test with AU Extraction) ✓ RECOMMENDED
**Pros**:
- 47.6% better than PFLD
- Reasonable size (50.9MB)
- Already integrated and working
- Worth testing AU impact before trying dlib

**Cons**:
- Above 3-pixel target
- May degrade AU accuracy

**Next Step**: Integrate FAN2 into AU pipeline and measure AU correlation (target r > 0.80)

**Time**: 2-3 hours to test

---

### Option B: Use dlib (Guaranteed Accuracy) ✓ FALLBACK
**Pros**:
- Sub-pixel accuracy (<1 pixel typical)
- Proven with OpenFace pipeline
- Industry standard

**Cons**:
- Large size (~180MB total)
- PyInstaller packaging complexity
- Slower inference (~30 FPS vs 100 FPS)

**Next Step**: Download dlib model and integrate with PyInstaller handling

**Time**: 4-6 hours for full integration

---

### Option C: Search for Better ONNX Model ⚠️ NOT RECOMMENDED
**Pros**:
- Might find higher accuracy lightweight model

**Cons**:
- Time consuming (already spent 3+ hours)
- No guarantee of finding better option
- FAN2 is already state-of-the-art for ONNX

**Time**: 2-4 hours, uncertain outcome

---

### Option D: Accept Current Accuracy △ RISKY
**Pros**:
- No additional work
- FAN2 significantly better than PFLD

**Cons**:
- Unknown AU impact
- Below gold standard principle

**Time**: 0 hours, but may need rollback later

## Recommendation

**Go with Option A: Test FAN2 with AU Extraction**

**Rationale**:
1. FAN2 is **47.6% better** than PFLD - significant improvement
2. Quick to test (~2-3 hours) before committing to dlib
3. If AU correlation stays r > 0.80, FAN2 is production-ready
4. If AU correlation drops below r = 0.75, fall back to dlib (Option B)

**Test Plan**:
1. Integrate FAN2 into AU prediction pipeline
2. Run on validation video (1110 frames)
3. Compare AU correlations vs CSV baseline
4. **Success criteria**: r > 0.80 for majority of AUs

**Expected Outcome**:
- 6.95px landmark error → estimated 5-10% AU accuracy drop
- Likely r = 0.75-0.80 (vs current r = 0.83)
- If r ≥ 0.80: FAN2 validated ✅
- If r < 0.75: Switch to dlib

## Files Created

### Implementation
- `fan2_landmark_detector.py` - FAN2 detector class with corrected scaling
- `weights/fan2_68_landmark.onnx` - Model file (50.9MB)

### Testing & Validation
- `test_fan2_accuracy.py` - Accuracy validation vs CSV baseline
- `debug_fan2_output.py` - Coordinate system debugging
- `fan2_accuracy_results.csv` - Per-frame results

### Documentation
- `COMPONENT3_PFLD_STATUS.md` - Initial PFLD findings
- `COMPONENT3_FAN2_RESULTS.md` - This document

## Next Session Actions

1. **Immediate**: Decide Option A (test FAN2 with AUs) or Option B (dlib)
2. **If Option A**: Integrate FAN2 → Run AU validation → Measure correlation
3. **If Option B**: Download dlib → Integrate with PyInstaller → Validate

---

**Session Date**: 2025-10-29 (evening)
**Component**: 3 (68-Point Landmark Detection)
**Status**: FAN2 integrated, 47.6% better than PFLD, ready for AU testing
**Best RMSE**: 6.95 pixels (vs 13.26 PFLD, vs <3 target)
