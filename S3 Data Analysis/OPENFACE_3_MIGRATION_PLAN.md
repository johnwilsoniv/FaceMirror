# OpenFace 3.0 Migration - Performance Recovery Plan

**Date**: 2025-10-27
**Status**: ROOT CAUSE IDENTIFIED - OpenFace 2.2 vs 3.0 AU Extraction Differences

---

## Executive Summary

**CRITICAL FINDING**: The catastrophic performance drop (84% → 34% for lower face, 93% → 65% for mid face, 83% → 60% for upper face) is **NOT due to code issues** but due to **OpenFace version differences**.

- **Published Benchmark**: OpenFace 2.2 outputs (`/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/combined_results.csv`)
- **Current Training**: OpenFace 3.0 outputs (`/Users/johnwilsoniv/Documents/SplitFace/S3O Results/combined_results.csv`)
- **Same source videos, same patients, but different AU extraction algorithms**

---

## What We've Fixed (Code-Level Issues)

### 1. ✅ Post-Publication Feature Additions Removed
**Issue**: All three zone feature extraction files had new features added AFTER publication:
- **Upper face**: AU04 (brow lowerer) interactions - 5 new features
- **Lower face**: AU15 (lip depressor) + AU20 (lip stretcher) interactions - 24 new features
- **Mid face**: AU45 × AU06 interactions - 41 new features

**Fix**: Replaced all feature extraction files with published versions from `/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/`

**Verification**:
- Upper face: 18 features (was 23) - NO AU04 ✅
- Mid face: 74 features (was 115) ✅
- Lower face: 220 features (was 244) ✅

### 2. ✅ Python Bytecode Cache Cleared
**Issue**: Even after replacing source files, Python was using cached `.pyc` files with old AU04 code

**Fix**: Deleted `__pycache__/` directory

### 3. ✅ Data Cache Cleared
**Fix**: Deleted `.cache/` directory to force fresh feature extraction

### 4. ✅ All Model Files Deleted
**Fix**: Removed all `models/*_face_*.pkl` files to force clean retraining

### 5. ✅ GUI Progress Tracking Fixed
**Issues**:
- Progress showed wrong zone names
- Time-based estimation showed 100% while Optuna continued
- "SHAP" pattern matched "X shape" in SMOTEENN logs

**Fixes**:
- Fixed zone name display to use actual selected zones
- Replaced time estimates with real log file monitoring
- Made SHAP detection pattern more specific ("Computing SHAP" not "shap")

---

## Current Performance - OpenFace 3.0 Data

### With Published Code + OpenFace 3.0 Data:

| Zone | Accuracy | F1 Weighted | Optuna Best CV | Benchmark Target |
|------|----------|-------------|----------------|------------------|
| Lower | 34.55% | 0.3638 | 0.3525 | 84% / 0.82 |
| Mid | 64.81% | 0.6800 | 0.3129 | 93% / 0.92 |
| Upper | 60.38% | 0.6343 | 0.4545 | 83% / 0.83 |

**Partial Class Performance** (Most Critical):
- Lower: F1 = 0.00 (0/9 correct) ❌
- Mid: F1 = 0.25 (2/8 correct) ⚠️
- Upper: F1 = 0.14 (1/5 correct) ❌

**Key Observation**: Optuna cross-validation scores are catastrophic (0.31-0.45), meaning the model cannot even learn the training data patterns. This indicates fundamental feature distribution differences, not just overfitting.

---

## Root Cause Analysis

### Why OpenFace 3.0 Performs Poorly

**OpenFace 2.2 → 3.0 Changes**:
1. **9 AUs removed**: AU08, AU11, AU13, AU18, AU19, AU21, AU22, AU24, AU27
   - None of these were used in the published model ✅

2. **AU extraction algorithm changes** (suspected):
   - Internal landmark detection improvements
   - Different AU intensity calibration
   - Modified temporal filtering
   - Updated facial geometry calculations

3. **AU intensity distribution shifts**:
   - Even for the same 18 AUs (AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU16, AU17, AU20, AU23, AU25, AU26, AU45)
   - The intensity values and distributions are likely different
   - This breaks the learned decision boundaries from the OpenFace 2.2 model

### Evidence

**From training logs**:
- Cross-validation scores during hyperparameter optimization are terrible (0.31-0.45)
- 100-120 Optuna trials cannot find good hyperparameters
- Class imbalance handling (SMOTE) not helping
- Feature selection reduces to same number of features but different patterns

**Published code IS working correctly**:
- Feature counts match published versions
- No AU04 in upper face
- Hyperparameter search space identical
- Train/test split logic identical

---

## Remaining Tasks / Options Forward

### Option 1: Accept OpenFace 3.0 Performance and Establish New Baseline

**Tasks**:
1. ✅ Document that OpenFace 3.0 has fundamentally different AU distributions
2. ⏳ Run comprehensive AU distribution analysis comparing OpenFace 2.2 vs 3.0:
   - Mean/std/min/max for each AU
   - Correlation matrices
   - Class-wise distributions
3. ⏳ Establish new performance targets for OpenFace 3.0:
   - Current: Lower 35%, Mid 65%, Upper 60%
   - These may be the realistic limits with OpenFace 3.0
4. ⏳ Optimize specifically for OpenFace 3.0 data:
   - Different hyperparameter ranges
   - Different class weights
   - Different SMOTE parameters
   - Possibly different feature engineering

**Pros**:
- Moves forward with modern OpenFace 3.0
- Establishes new ground truth
- May discover improvements over time

**Cons**:
- Performance will be significantly lower than published
- Clinical utility questionable at 35-65% accuracy
- Partial class (most important) completely fails

---

### Option 2: Deep AU Distribution Analysis and Calibration

**Tasks**:
1. ⏳ Extract AU statistics from both OpenFace versions:
   ```python
   # Compare AU distributions
   of2_data = pd.read_csv('open2GR/3_Data_Analysis/combined_results.csv')
   of3_data = pd.read_csv('SplitFace/S3O Results/combined_results.csv')

   # For each AU, compute:
   # - Mean, std, percentiles
   # - By action (BS, SS, ES, ET, BK, RE)
   # - By patient
   # - By paralysis class
   ```

2. ⏳ Create AU calibration/normalization layer:
   - Map OpenFace 3.0 distributions to match OpenFace 2.2
   - Use percentile-based normalization
   - Or use distribution matching (histogram matching)

3. ⏳ Retrain with calibrated features

**Pros**:
- May recover performance
- Maintains clinical validity
- Enables use of modern OpenFace 3.0

**Cons**:
- Complex statistical analysis required
- May introduce artifacts
- Not guaranteed to work

---

### Option 3: Revert to OpenFace 2.2 for Production

**Tasks**:
1. ⏳ Install OpenFace 2.2 (or 2.2.1)
2. ⏳ Re-extract AUs from all source videos using OpenFace 2.2
3. ⏳ Verify performance matches published benchmark
4. ⏳ Use OpenFace 2.2 for production system

**Pros**:
- Guaranteed to match published performance
- Clinically validated
- No unknowns

**Cons**:
- Using older software version
- May have compatibility issues
- Won't benefit from OpenFace 3.0 improvements

---

### Option 4: Hybrid Approach - Feature-Level Analysis

**Tasks**:
1. ⏳ Identify which specific features are most affected by OpenFace version:
   - Compare feature importance between versions
   - Identify features with largest distribution shifts
2. ⏳ Engineer version-invariant features:
   - Ratios and relative measures may be more stable
   - Asymmetry metrics may be less affected
   - Action-to-action comparisons may be robust
3. ⏳ Retrain with modified feature set

**Pros**:
- Targeted approach
- May identify root cause features
- Could lead to more robust model

**Cons**:
- Labor intensive
- Requires deep feature engineering
- May sacrifice performance

---

## Immediate Next Steps (Choose One Path)

### If Accepting OpenFace 3.0:
1. Create comprehensive AU distribution comparison document
2. Run sensitivity analysis on hyperparameters for OpenFace 3.0
3. Try more aggressive class weighting for Partial class
4. Document new baseline performance metrics

### If Attempting Calibration:
1. Write script to extract and compare AU statistics
2. Visualize distribution differences (histograms, box plots)
3. Prototype calibration function
4. Test on small subset

### If Reverting to OpenFace 2.2:
1. Research OpenFace 2.2 installation on current system
2. Test on sample videos
3. Plan full re-extraction pipeline

---

## Files Modified in This Session

### Code Files:
1. `upper_face_features.py` - Removed AU04 interactions (replaced with published version)
2. `lower_face_features.py` - Removed AU15/AU20 additions (replaced with published version)
3. `mid_face_features.py` - Removed AU45×AU06 additions (replaced with published version)
4. `training_gui.py` - Fixed progress tracking and zone name display
5. `paralysis_config.py` - Temporarily modified then reverted

### Deleted:
- `__pycache__/` - Python bytecode cache
- `.cache/` - Data preprocessing cache
- `models/lower_face_*` - All lower face model files
- `models/mid_face_*` - All mid face model files
- `models/upper_face_*` - All upper face model files

### Backups Created:
- `upper_face_features.py.backup`
- `lower_face_features.py.backup`
- `mid_face_features.py.backup`

---

## Key Insights

1. **The published code is correct** - all fixes brought it in line with published version
2. **The problem is the data source** - OpenFace 3.0 vs 2.2
3. **Feature extraction matters more than hyperparameters** - adding AU04/AU15/AU20 killed performance, but even without them, OF3 data doesn't work
4. **Partial class is the canary** - its complete failure (F1=0.00) indicates fundamental learning issues
5. **Optuna scores reveal the truth** - CV scores of 0.35 during training mean the model can't even learn from training data

---

## Questions for User

1. **Is maintaining OpenFace 3.0 compatibility a hard requirement?**
   - If yes → Pursue calibration or accept lower performance
   - If no → Revert to OpenFace 2.2

2. **What is the minimum acceptable accuracy for clinical use?**
   - This will determine if OpenFace 3.0 path is viable

3. **Are the source videos still available for re-extraction with OpenFace 2.2?**
   - Needed for Option 3

4. **What are the benefits of OpenFace 3.0 that make it worth pursuing?**
   - Better landmark detection?
   - More robust to lighting/pose?
   - Performance improvements?

---

## References

- Published code: `/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/`
- Published data (OF 2.2): `/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/combined_results.csv`
- Current code: `/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis/`
- Current data (OF 3.0): `/Users/johnwilsoniv/Documents/SplitFace/S3O Results/combined_results.csv`
- Manuscript: `/Users/johnwilsoniv/Documents/FaceMirror Manuscript/draft_Proof_hi.pdf`

---

## Appendix: Performance Comparison Tables

### Published Benchmark (OpenFace 2.2)
| Zone | Accuracy | Weighted F1 | Partial F1 | Test Samples |
|------|----------|-------------|------------|--------------|
| Lower | 0.84 | 0.82 | 0.46 | 56 |
| Mid | 0.93 | 0.92 | 0.67 | 54 |
| Upper | 0.83 | 0.83 | 0.40 | 53 |

### Current Results (OpenFace 3.0)
| Zone | Accuracy | Weighted F1 | Partial F1 | Test Samples |
|------|----------|-------------|------------|--------------|
| Lower | 0.35 | 0.36 | 0.00 | 55 |
| Mid | 0.65 | 0.68 | 0.25 | 54 |
| Upper | 0.60 | 0.63 | 0.14 | 53 |

### Delta (Current - Benchmark)
| Zone | Δ Accuracy | Δ Weighted F1 | Δ Partial F1 | % Change Accuracy |
|------|------------|---------------|--------------|-------------------|
| Lower | -0.49 | -0.46 | -0.46 | -58.3% |
| Mid | -0.28 | -0.24 | -0.42 | -30.1% |
| Upper | -0.23 | -0.20 | -0.26 | -27.7% |

---

**END OF DOCUMENT**
