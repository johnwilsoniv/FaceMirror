# Parallel Path Status - Running Median + CalcParams

**Date:** 2025-10-29 Late Evening
**Status:** Pursuing two parallel approaches to fix AU predictions

---

## Current Problem

**AU Prediction Correlation:** r = 0.0000 (complete failure)

This zero correlation suggests a **fundamental feature issue**, not just alignment:
- Partial correlation (r = 0.3-0.7) would indicate alignment problems
- Zero correlation (r = 0.0) indicates wrong features or missing normalization

---

## PATH A: Running Median Normalization ⚡ **HIGHEST PRIORITY**

### Status: ✅ Implementation Already Exists!

**File:** `histogram_median_tracker.py` (complete implementation)

### Why This is Likely the Issue

From `PHASE2_COMPLETE_SUCCESS.md`:
- Dynamic AU models require **running median normalization**
- This is person-specific normalization applied to features
- Without it, SVR predictions will be completely wrong

### Evidence
1. C++ baseline AU values are reasonable (0.0 to 4.5 range)
2. Dynamic models loaded correctly (AU01, AU02, AU05, AU09, AU15, AU17, AU20, AU23, AU25, AU26, AU45)
3. Feature dimensions are correct (4702 = 4464 HOG + 238 geometric)
4. **But predictions have r=0.0000** → Missing normalization!

### Implementation Plan

**Step 1:** Integrate `HistogramBasedMedianTracker` into AU prediction pipeline
```python
from histogram_median_tracker import HistogramBasedMedianTracker

# Create separate trackers for HOG and geometric
hog_tracker = HistogramBasedMedianTracker(
    feature_dim=4464,
    num_bins=1000,
    min_val=-0.005,
    max_val=1.0
)

geom_tracker = HistogramBasedMedianTracker(
    feature_dim=238,
    num_bins=10000,
    min_val=-60.0,
    max_val=60.0
)
```

**Step 2:** Apply running median for dynamic models
```python
# For each frame:
hog_tracker.update(hog_features)
geom_tracker.update(geom_features)

# Get normalized features
hog_median = hog_tracker.get_median()
geom_median = geom_tracker.get_median()

# Normalize
hog_normalized = hog_features - hog_median
geom_normalized = geom_features - geom_median

# Combine for complete normalized features
normalized_features = np.concatenate([hog_normalized, geom_normalized])

# Predict AUs using normalized features (for dynamic models)
```

**Step 3:** Test AU predictions with normalization
- Expected result: **Significant improvement in correlation**
- Target: r > 0.90 for dynamic AUs

**Estimated Time:** 2-3 hours
**Probability of Success:** **HIGH** (this is the most likely cause)

---

## PATH B: CalcParams for Better Alignment

### Status: ⚠️ Multiple Blockers

### Current Alignment Status
- Using **inverse CSV p_rz** (2D rotation only)
- Mean pixel diff from C++: ~17-18 pixels
- Visual: Slight tilt remaining (~1-2°)

### CalcParams Would Provide
- Full **3D pose estimation** (rx, ry, rz) not just 2D (rz)
- Re-fitting PDM to CSV landmarks → params_global₂
- "3D effect" user noticed in C++ output

### Options for CalcParams

#### Option B1: Fix OpenFace Build (HIGH RISK)
**Blockers:**
- Missing `limits` header (macOS compiler issue)
- Libraries not linking (LandmarkDetector.a doesn't exist)
- Persistent build failures

**Estimated Time:** 1-2 days
**Probability of Success:** LOW-MEDIUM (compiler environment issues)

#### Option B2: Implement Simplified CalcParams in Python
**Requirements:**
- Implement 3D→2D projection with full rotation matrix
- Optimize 6 global params (scale, rx, ry, rz, tx, ty)
- Skip local params (34 PCA coefficients) - use CSV values

**Estimated Time:** 2-3 days
**Probability of Success:** MEDIUM (complex but doable)

#### Option B3: Use Kabsch on 3D PDM Shape
**Idea:** Align 3D PDM shape to 2D landmarks
- Might capture 3D pose better than 2D-only alignment
- Simpler than full CalcParams

**Estimated Time:** 3-4 hours
**Probability of Success:** MEDIUM

### Why PATH B Might Not Matter Yet

**If running median (Path A) gives r > 0.90:**
- Alignment is "good enough"
- CalcParams not needed
- Focus on other improvements

**If running median gives r < 0.90:**
- Alignment may be limiting factor
- Pursue CalcParams
- But first check HOG extraction differences

---

## Recommended Strategy

### Phase 1: Running Median (IMMEDIATE)
1. ✅ Found existing implementation (`histogram_median_tracker.py`)
2. 🔄 Integrate into `test_python_au_predictions.py`
3. 🔄 Test AU correlation with normalization
4. ✅ If r > 0.90: **SUCCESS! PATH A solved it**

### Phase 2: CalcParams (IF NEEDED)
**Only if Phase 1 gives r < 0.90:**

5. Try Option B3 (Kabsch on 3D shape) - quickest
6. If still poor, try Option B2 (simplified CalcParams)
7. Last resort: Option B1 (fix OpenFace build)

---

## Key Metrics to Watch

### Success Criteria

**Running Median Test:**
- Dynamic AUs (AU01, AU02, AU05, AU09, AU15, AU17, AU20, AU23, AU25, AU26, AU45): r > 0.90
- Static AUs (AU04, AU06, AU07, AU10, AU12, AU14): r > 0.95

**If CalcParams Needed:**
- All AUs: r > 0.95
- Visual: No tilt in aligned faces
- Mean pixel diff < 10 from C++

### Failure Modes

**If running median doesn't help (r still ~0.0):**
- HOG extraction parameters wrong
- Geometric feature construction wrong
- Model loading issue

**If running median helps but r < 0.80:**
- Alignment is likely the issue
- Pursue CalcParams

---

## Files to Use

### PATH A Files
- ✅ `histogram_median_tracker.py` - Running median implementation
- 🔄 `test_python_au_predictions.py` - Test script (needs normalization added)
- 📖 `PHASE2_COMPLETE_SUCCESS.md` - Documentation on running median

### PATH B Files
- 📖 `CRITICAL_FINDING_DOUBLE_PDM_FITTING.md` - CalcParams discovery
- 📖 `calc_params_tool.cpp` - Attempted C++ wrapper (failed to compile)
- 🔄 `openface22_face_aligner.py` - Current alignment (could be improved)

---

## Next Actions

**IMMEDIATE (tonight/tomorrow):**
1. Add running median to AU prediction pipeline
2. Run test on 100+ frames
3. Compute correlations

**Expected outcome:** Significant jump in correlation (r: 0.0 → 0.85-0.95)

**IF successful:** PATH A complete, CalcParams not needed! 🎉

**IF unsuccessful:** Deep dive into PATH B options

---

## Confidence Assessment

**Running Median Fix (PATH A):** 🟢 **85% confident** this is the main issue
- Dynamic models REQUIRE normalization
- Zero correlation strongly suggests missing normalization
- Implementation already exists and validated

**CalcParams Fix (PATH B):** 🟡 **50% confident** this is needed
- May help but probably not the primary blocker
- More likely a refinement after running median works
- Worth pursuing if PATH A isn't sufficient

---

## Summary

**What we learned today:**
1. PyFHOG produces correct 4464 features ✅
2. Geometric features correctly constructed (238 dims) ✅
3. Complete feature vector (4702 dims) ✅
4. Masking fixed and working ✅
5. **Missing: Running median normalization** ❌

**Most likely solution:** Add running median → instant improvement

**Backup solution:** Improve alignment with CalcParams

**Timeline:**
- Running median: 2-3 hours
- Test & validate: 1 hour
- CalcParams (if needed): 2-3 days

We're very close! 🚀
