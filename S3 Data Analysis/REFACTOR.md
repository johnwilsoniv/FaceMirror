# S3 Data Analysis Refactor Plan: PyFaceAU Integration

**Date**: November 2, 2025
**Status**: Planning Phase - NO CODE CHANGES YET

## Executive Summary

S3 Data Analysis needs to be updated to work with PyFaceAU (pure Python OpenFace 2.2 implementation) instead of the failed OpenFace 3.0 migration. This requires reverting AU mappings to OpenFace 2.2 standard and retraining paralysis models.

**Goal**: Restore OpenFace 2.2 AU mappings, retrain paralysis models with updated training pipeline, and update branding to PyFaceAU.

---

## Background

### Historical Timeline

1. **Original Version** (open2GR/3_Data_Analysis)
   - Built for OpenFace 2.2 C++
   - Used AU07 (Lid Tightener) for eye closure detection
   - 18 AUs available
   - Paralysis detection working correctly

2. **OpenFace 3.0 Migration Attempt** (Failed)
   - Attempted migration to OpenFace 3.0
   - **Critical Discovery**: 9 AUs had 100% reduction in values
     - AU05, AU07, AU09, AU10, AU14, AU17, AU23, AU26: completely disappeared
     - AU06: 93% reduction
   - 5 AUs had 59-85% loss (AU04, AU02, AU25, AU12, AU01)
   - Models trained on OpenFace 2.2 became incompatible
   - Switched AU07 → AU06 in ACTION_TO_AUS as workaround
   - **Result**: System failed, models unusable

3. **Current Version** (S3 Data Analysis)
   - Uses OpenFace 3.0 AU mappings (AU06 instead of AU07)
   - Improved training pipeline and GUI
   - Still references "OpenFace" in user-facing contexts
   - **Problem**: Configured for non-functional OpenFace 3.0

4. **Solution** (PyFaceAU Integration)
   - Pure Python OpenFace 2.2 implementation
   - r=0.8640 correlation with original OpenFace 2.2
   - All 17 AUs functional and validated
   - 6x faster than OpenFace 2.2 C++
   - Enables reversion to correct AU mappings

---

## Critical AU Mapping Differences

### Action Definitions That Need Reverting

| Action | Current (OpenFace 3.0) | Correct (OpenFace 2.2) | Reason |
|--------|----------------------|----------------------|--------|
| **ET** (Close Eyes Tightly) | `['AU06_r', 'AU45_r']` | `['AU07_r', 'AU45_r']` | AU07 is Lid Tightener (correct for eye closure)<br>AU06 is Cheek Raiser (smile-related, incorrect) |
| **BS** (Big Smile) | `['AU12_r', 'AU25_r', 'AU06_r']` | `['AU12_r', 'AU25_r', 'AU07_r']` | Same correction |

**Why This Matters**:
- AU07 (Lid Tightener) is anatomically correct for forceful eye closure
- AU06 (Cheek Raiser) was a desperate workaround for OpenFace 3.0's failure
- PyFaceAU properly detects AU07 with r=0.9413 correlation

### All 18 AUs Available in PyFaceAU

```python
ALL_AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
    'AU25_r', 'AU26_r', 'AU45_r'
]
```

---

## What to Keep from Current Version

### Improvements to Preserve

1. **GUI Enhancements** (`facial_au_gui.py`)
   - Better user interface
   - Improved progress reporting
   - More intuitive workflow

2. **Training Pipeline Improvements**
   - `paralysis_training_pipeline.py`
   - `paralysis_model_trainer.py`
   - `paralysis_training_helpers.py`
   - More efficient training process
   - Better cross-validation
   - Improved feature selection

3. **Shared Detector Loading** (`facial_au_analyzer.py` line 31-44)
   - Detectors can be pre-loaded and shared across analyses
   - Significant performance improvement for batch processing

4. **Hardware Detection** (`hardware_detection.py`)
   - Automatic CPU/GPU detection
   - Optimized resource usage

5. **Config Paths System** (`config_paths.py`)
   - Centralized path management
   - Better for deployment

6. **Batch Processing Improvements** (`facial_au_batch_processor.py`)
   - More robust error handling
   - Better logging

---

## Refactor Strategy

### Phase 1: AU Mapping Corrections

**Objective**: Restore anatomically correct AU mappings for OpenFace 2.2/PyFaceAU

**Files to Modify**:

1. **`facial_au_constants.py`**
   ```python
   # Change from (INCORRECT - OpenFace 3.0 workaround):
   ACTION_TO_AUS = {
       'ET': ['AU06_r', 'AU45_r'],  # Wrong: AU06 is Cheek Raiser
       'BS': ['AU12_r', 'AU25_r', 'AU06_r'],  # Wrong
       # ... rest unchanged
   }

   # Change to (CORRECT - OpenFace 2.2/PyFaceAU):
   ACTION_TO_AUS = {
       'ET': ['AU07_r', 'AU45_r'],  # Correct: AU07 is Lid Tightener
       'BS': ['AU12_r', 'AU25_r', 'AU07_r'],  # Correct
       # ... rest unchanged
   }
   ```

2. **`paralysis_config.py`** (if needed)
   - Verify mid-face AU configuration
   - Ensure AU07 is included, AU06 may need adjustment
   - Check all AU lists match OpenFace 2.2 standard

**Testing**:
- Load PyFaceAU-generated CSV files
- Verify peak frame detection works correctly
- Check that all expected AU columns are present
- Validate data loading and preprocessing

**Estimated Effort**: 1-2 hours
**Risk**: Low (simple mapping changes)

---

### Phase 2: Model Retraining with Correct AUs

**Objective**: Retrain all three paralysis models using OpenFace 2.2 AU scheme with improved training pipeline

**Models to Retrain**:
1. **Lower Face Paralysis Model**
   - Old AUs used: AU12, AU15, AU20, AU25
   - Config AUs: AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26
   - Review which AUs are actually needed based on old feature engineering

2. **Mid Face Paralysis Model** ⚠️ CRITICAL
   - Old AUs used: AU45, AU06
   - Should use: AU45, **AU07** (not AU06)
   - This is the primary correction needed

3. **Upper Face Paralysis Model**
   - Old AUs used: AU01, AU02, AU04
   - No change needed (these AUs were stable across versions)

**Process**:
1. **Review Old Feature Engineering**
   - Check `/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/lower_face_features.py`
   - Check `/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/mid_face_features.py`
   - Check `/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/upper_face_features.py`
   - Identify which AUs were actually used in feature extraction
   - Ensure current feature extraction matches old logic

2. **Update Feature Extraction** (if needed)
   - Modify `paralysis_utils.py` if feature engineering differs
   - Ensure asymmetry calculations use correct AUs
   - Verify baseline normalization logic

3. **Retrain Models**
   - Use current improved training pipeline
   - Process training data with PyFaceAU (if not already done)
   - Train with correct AU features
   - Validate with held-out test set
   - Compare performance to original OpenFace 2.2 models

**Estimated Effort**: 8-16 hours (including training time)
**Risk**: Medium (depends on training data availability and model convergence)

---

### Phase 3: PyFaceAU Branding Updates

**Objective**: Replace all user-facing "OpenFace" references with "PyFaceAU"

**Files to Update**:

1. **`facial_au_gui.py`**
   - Window titles
   - Status messages
   - Help text
   - About dialog

2. **`facial_au_batch_processor.py`**
   - Log messages shown to user
   - Progress updates
   - Error messages

3. **`main.py`**
   - Startup messages
   - Command line help text

4. **`README.md`**
   - Documentation
   - Installation instructions
   - Usage examples

5. **`config_paths.py`** (if it has user-facing messages)

**Search Pattern**:
```bash
grep -r "OpenFace" --include="*.py" .
grep -r "openface" --include="*.py" .
```

**Changes**:
- "OpenFace" → "PyFaceAU"
- "OpenFace 2.2" → "PyFaceAU"
- "OpenFace 3.0" → remove or replace with "PyFaceAU"

**Keep Internal References**:
- Variable names can stay (e.g., `openface_processor`)
- Internal comments can reference OpenFace for clarity
- Only change USER-FACING text

**Estimated Effort**: 2-3 hours
**Risk**: Low (cosmetic changes)

---

## Implementation Plan

### Step-by-Step Checklist

#### Preparation
- [x] Create REFACTOR.md plan
- [ ] Back up current S3 Data Analysis directory
- [ ] Create Git branch: `pyfaceau-integration`
- [ ] Verify PyFaceAU is working in S1 Face Mirror

#### Phase 1: AU Corrections (Day 1)
- [ ] Modify `facial_au_constants.py`:
  - [ ] Change ET: `AU06_r` → `AU07_r`
  - [ ] Change BS: `AU06_r` → `AU07_r`
  - [ ] Verify all other ACTION_TO_AUS entries
  - [ ] Verify ALL_AU_COLUMNS list is complete
- [ ] Review `paralysis_config.py`:
  - [ ] Check mid-face AU configuration
  - [ ] Ensure AU07 is included
  - [ ] Verify lower/upper face configs
- [ ] Test with PyFaceAU CSV:
  - [ ] Load test data
  - [ ] Verify peak frame detection
  - [ ] Check AU column availability
  - [ ] Test feature extraction
- [ ] Commit Phase 1 changes

#### Phase 2: Review Old Feature Engineering (Day 2)
- [ ] Compare old vs current feature extraction:
  - [ ] `lower_face_features.py` - old vs current
  - [ ] `mid_face_features.py` - old vs current (CRITICAL)
  - [ ] `upper_face_features.py` - old vs current
- [ ] Document differences in feature engineering
- [ ] Identify if current `paralysis_utils.py` needs updates
- [ ] Update feature extraction if needed
- [ ] Test feature extraction with PyFaceAU data

#### Phase 3: Model Retraining (Day 3-4)
- [ ] Prepare training data:
  - [ ] Ensure training videos processed with PyFaceAU
  - [ ] Verify ground truth labels available
  - [ ] Split train/validation/test sets
- [ ] Retrain mid-face model (PRIORITY):
  - [ ] Train with AU07 instead of AU06
  - [ ] Validate on test set
  - [ ] Compare to baseline performance
- [ ] Retrain lower-face model:
  - [ ] Use correct AU features
  - [ ] Validate performance
- [ ] Retrain upper-face model:
  - [ ] Should need minimal changes
  - [ ] Validate performance
- [ ] Document model performance metrics
- [ ] Save trained models

#### Phase 4: PyFaceAU Branding (Day 5)
- [ ] Search for "OpenFace" in user-facing files
- [ ] Update `facial_au_gui.py`:
  - [ ] Window title
  - [ ] Status messages
  - [ ] Help text
- [ ] Update `facial_au_batch_processor.py`:
  - [ ] Log messages
  - [ ] Progress updates
- [ ] Update `main.py`:
  - [ ] Startup banner
  - [ ] Help text
- [ ] Update `README.md`:
  - [ ] Replace all OpenFace references
  - [ ] Update installation instructions
- [ ] Test GUI to verify all changes

#### Phase 5: Integration Testing (Day 6)
- [ ] Full pipeline test:
  - [ ] Load PyFaceAU CSV files
  - [ ] Run analysis on test patients
  - [ ] Verify predictions match expected
  - [ ] Check GUI displays results correctly
- [ ] Batch processing test:
  - [ ] Process multiple patients
  - [ ] Verify consistent results
  - [ ] Check error handling
- [ ] Performance test:
  - [ ] Measure processing time
  - [ ] Check memory usage
  - [ ] Verify no regression

#### Phase 6: Documentation and Cleanup (Day 7)
- [ ] Update documentation:
  - [ ] README.md
  - [ ] User guide (if exists)
  - [ ] Installation instructions
- [ ] Clean up temp files:
  - [ ] Remove old OpenFace 3.0 analysis docs
  - [ ] Archive migration planning docs
- [ ] Final code review
- [ ] Commit all changes
- [ ] Merge to main branch

---

## File-by-File Changes

### Critical Files - MUST CHANGE

1. **`facial_au_constants.py`**
   - Line ~46: `'ET': ['AU07_r', 'AU45_r']` (currently AU06)
   - Line ~48: `'BS': ['AU12_r', 'AU25_r', 'AU07_r']` (currently AU06)
   - Verify: ALL_AU_COLUMNS is complete

2. **`paralysis_config.py`** (if needed)
   - Check mid-face config (around line 170)
   - May need to adjust AU06/AU07

3. **`paralysis_utils.py`** (if feature engineering differs)
   - Compare with old version
   - Update if needed for AU07 handling

### Important Files - REVIEW & UPDATE

1. **`facial_au_gui.py`**
   - Search & replace "OpenFace" → "PyFaceAU" in user-facing strings

2. **`facial_au_batch_processor.py`**
   - Search & replace "OpenFace" → "PyFaceAU" in logs/messages

3. **`main.py`**
   - Update startup banner
   - Update help text

4. **`README.md`**
   - Complete rewrite of OpenFace references

### Files to Keep - NO CHANGES

1. **`facial_au_analyzer.py`** - Core logic stays the same
2. **`paralysis_detector.py`** - Detector logic unchanged
3. **`facial_paralysis_detection.py`** - Dispatcher logic unchanged
4. **`paralysis_training_pipeline.py`** - Keep improved training
5. **`paralysis_model_trainer.py`** - Keep improved training
6. **`hardware_detection.py`** - Keep as is
7. **`config_paths.py`** - Keep as is

---

## Old vs Current Feature Engineering Comparison

### Need to Verify

1. **Lower Face Features** (`lower_face_features.py`)
   - Old: Lines 1-250 (need to review)
   - Current: May be in `paralysis_utils.py`
   - Check: AU12, AU15, AU20, AU25 feature extraction

2. **Mid Face Features** (`mid_face_features.py`) ⚠️ CRITICAL
   - Old: Used AU07 (Lid Tightener)
   - Current: Uses AU06 (Cheek Raiser) - WRONG
   - **Must verify**: How features are extracted and if logic needs updating

3. **Upper Face Features** (`upper_face_features.py`)
   - Old: AU01, AU02, AU04
   - Current: Should be same
   - Check: Verify feature extraction matches

### Questions to Answer

1. Does current `paralysis_utils.py` replicate old feature engineering?
2. Are asymmetry calculations the same?
3. Is baseline normalization the same?
4. Are any new features added that should be kept?

---

## Model Performance Expectations

### Expected Outcomes

**After AU Corrections**:
- Peak frame detection should be more accurate (correct AU for eye closure)
- Feature extraction should align with original OpenFace 2.2 logic

**After Model Retraining**:
- Performance should match or exceed original OpenFace 2.2 models
- PyFaceAU correlation (r=0.8640) means slight variation is expected
- Mid-face model should improve (was using wrong AU)

### Validation Metrics

Track these for each model:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Confusion matrix

Compare to baseline (if available) from original OpenFace 2.2 models.

---

## Risk Mitigation

### Backup Strategy
- Full backup before starting
- Git branch for all changes
- Keep old models for comparison

### Rollback Plan
- If Phase 2 fails: Can revert to current models
- If Phase 1 breaks something: Git reset to pre-refactor
- Keep both old and new models until validation complete

### Testing Strategy
- Test each phase before proceeding
- Validate with known ground truth
- Compare to original model outputs (if available)

---

## Success Criteria

### Phase 1 Complete When:
- [ ] AU mappings match OpenFace 2.2 standard
- [ ] No user-facing references to incorrect AUs
- [ ] Peak frame detection works with PyFaceAU data
- [ ] All tests pass

### Phase 2 Complete When:
- [ ] All three models retrained with correct AUs
- [ ] Model performance validated
- [ ] Performance meets or exceeds baseline
- [ ] Models work correctly with PyFaceAU data

### Phase 3 Complete When:
- [ ] All user-facing "OpenFace" references changed to "PyFaceAU"
- [ ] GUI shows PyFaceAU branding
- [ ] Documentation updated
- [ ] No internal functionality broken

### Final Success:
- [ ] Full analysis pipeline works with PyFaceAU
- [ ] Models accurate and validated
- [ ] GUI polished and branded correctly
- [ ] Batch processing functional
- [ ] Performance at least as good as original
- [ ] Ready for production use

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: AU Corrections | 1-2 hours | None |
| Phase 2: Feature Review | 4-6 hours | Phase 1 complete |
| Phase 2: Model Retraining | 8-16 hours | Training data ready |
| Phase 3: Branding | 2-3 hours | Phase 1 complete |
| Testing & Validation | 4-8 hours | Phases 1-3 complete |
| Documentation | 2-4 hours | All phases complete |
| **Total** | **21-39 hours** | **~1 week** |

---

## Next Steps

1. ✅ User approval of refactor plan
2. Create backup of S3 Data Analysis
3. Create Git branch `pyfaceau-integration`
4. Begin Phase 1: AU corrections
5. Review old feature engineering code
6. Proceed with model retraining
7. Update branding
8. Validate and test
9. Merge to main

---

## Notes

- This refactor corrects the failed OpenFace 3.0 migration
- PyFaceAU enables return to anatomically correct AU mappings
- Keep all current training pipeline improvements
- Focus exclusively on paralysis detection
- User-facing branding should reflect PyFaceAU
