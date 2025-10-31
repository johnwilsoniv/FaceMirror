# Feature Set Optimization Analysis for OpenFace 3.0

**Date:** 2025-10-22
**Status:** Pre-Retraining Analysis

---

## Current Feature Extraction Architecture

### Base Features (Per AU × Action)
The `_extract_base_au_features()` function creates **6 features per AU×Action combination**:

1. `val_side` - AU value on current side (normalized if configured)
2. `val_opp` - AU value on opposite side
3. `Asym_Diff` - Difference (current - opposite)
4. `Asym_Ratio` - Ratio (current / opposite)
5. `Asym_PercDiff` - Percent difference (capped at 200%)
6. `Is_Weaker_Side` - Binary indicator (1 if current side weaker)

### Zone-Specific Custom Features

#### Upper Face (AU01, AU02, AU04 - **UPDATED**)
**Action:** RE (Raise Eyebrows)

**Base Features:** 3 AUs × 1 action × 6 features = **18 base features**

**Custom Features (lines 31-48 in upper_face_features.py):**
1. `RE_avg_Asym_Ratio` - Average of AU01 and AU02 asymmetry ratios
2. `RE_avg_Asym_PercDiff` - Average of AU01 and AU02 percent differences
3. `RE_max_Asym_PercDiff` - Max of AU01 and AU02 percent differences
4. `RE_AU01_AU02_product_side` - AU01 × AU02 product
5. `RE_AU01_AU02_sum_side` - AU01 + AU02 sum

**Total Features:** 18 + 5 + 1 (side_indicator) = **~24 features**

**Issues with Current Implementation:**
- ❌ **AU04 not included in custom features** - Only AU01 and AU02 are used in averages/products
- ❌ **Missing AU04 interactions** - No AU01×AU04, AU02×AU04, or AU01×AU02×AU04 features
- ⚠️ **No temporal features** - Brow movements may have diagnostic temporal patterns

---

#### Mid Face (AU45, AU07, AU06)
**Actions:** ES (Eyes Shut), ET (Eyes Tight), BK (Blink)

**Base Features:** 3 AUs × 3 actions × 6 features = **54 base features**

**Custom Features (lines 31-60 in mid_face_features.py):**
For each AU (AU45, AU07, AU06):
1. `{AU}_ETES_Ratio_Side` - ET/ES ratio on current side
2. `{AU}_ETES_Ratio_Opp` - ET/ES ratio on opposite side
3. `{AU}_ETES_Asym_Diff` - Difference of ET/ES ratios
4. `{AU}_ETES_Asym_Ratio` - Ratio of ET/ES ratios
5. `{AU}_ETES_Asym_PercDiff` - Percent difference of ET/ES ratios
6. `{AU}_ETES_Is_Weaker_Side` - Binary indicator

**Custom Features:** 3 AUs × 6 ETES features = **18 custom features**

**Total Features:** 54 + 18 + 1 (side_indicator) = **~73 features**

**Critical Issues with OpenFace 3.0:**
- 🔴 **AU07 is NON-FUNCTIONAL** (all zeros) - 18 base features + 6 ETES features = **24 dead features**
- 🔴 **AU06 values dropped 93%** - Features exist but have very low signal
- ✓ **AU45 improved +137%** - These features should be more informative now
- ❌ **No cross-AU interactions** - AU45 × AU06 could be valuable
- ❌ **No BK-specific features** - Blink asymmetry features missing

**Expected Impact:**
- ~33% of features are dead or near-dead (AU07 completely dead, AU06 very weak)
- Model may struggle without additional feature engineering

---

#### Lower Face (AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26)
**Actions:** BS (Big Smile), SS (Soft Smile), SO (Snarl), SE (Snarl)

**Base Features:** 9 AUs × 4 actions × 6 features = **216 base features**

**Custom Features (lines 34-56 in lower_face_features.py):**
1. `avg_AU12_Asym_Ratio` - Average AU12 asymmetry ratio across all actions
2. `max_AU12_Asym_PercDiff` - Max AU12 asymmetry percent diff across all actions
3. `BS_Asym_Ratio_Product_12_25` - Product of BS_AU12 and BS_AU25 asymmetry ratios

**Total Features:** 216 + 3 + 1 (side_indicator) = **~220 features**

**Critical Issues with OpenFace 3.0:**
- 🔴 **5 AUs completely non-functional** (AU10, AU14, AU17, AU23, AU26) - **120 dead features (56% of base features!)**
- ✓ **4 AUs functional** (AU12, AU15, AU20, AU25) - **96 features remaining (44%)**
- ✓ **AU20 improved +372%** - Should be very informative
- ✓ **AU15 stable** - Good reliability
- ⚠️ **AU12 dropped 80%** - Still functional but weaker
- ⚠️ **AU25 dropped 87%** - Still functional but weaker
- ❌ **Custom features only use AU12 and AU25** - Need to add AU15 and AU20 features
- ❌ **No AU20 × AU25 features** - AU20 improvement should be leveraged

**Expected Impact:**
- ~55% of features are dead
- Heavy reliance on AU12 (which dropped 80%) and AU25 (which dropped 87%)
- AU20's massive improvement is underutilized

---

## Feature Set Optimization Recommendations

### Priority 1: Remove Dead Features (Pre-Training)

**Objective:** Eliminate features based on non-functional AUs to reduce noise and improve model efficiency.

#### Upper Face
- ✓ No dead features (all AUs functional)

#### Mid Face
**Remove all AU07-based features:**
- Base: `ES_AU07_r_*`, `ET_AU07_r_*`, `BK_AU07_r_*` (18 features)
- ETES: `AU07_r_ETES_*` (6 features)
- **Total to remove: 24 features**

**Result:** 73 → 49 features (~33% reduction)

#### Lower Face
**Remove all non-functional AU features:**
- AU10: 24 features (4 actions × 6)
- AU14: 24 features
- AU17: 24 features
- AU23: 24 features
- AU26: 24 features
- **Total to remove: 120 features**

**Result:** 220 → 100 features (~55% reduction)

---

### Priority 2: Add New Interaction Features (High Value)

Given the reduced AU set, **interaction features become critical** for maintaining F1 scores.

#### Upper Face - ADD THESE FEATURES

**AU04 Integration (AU04 is new, not in custom features):**
```python
# Add to upper_face_features.py after line 48

# AU04 interactions with AU01 and AU02
au4_val_side = feature_data.get('RE_AU04_r_val_side', pd.Series(0.0, index=df.index))
au4_ratio = feature_data.get('RE_AU04_r_Asym_Ratio', pd.Series(1.0, index=df.index))
au4_pd = feature_data.get('RE_AU04_r_Asym_PercDiff', pd.Series(0.0, index=df.index))

# AU01 × AU04 (antagonistic brow movements)
feature_data['RE_AU01_AU04_product_side'] = au1_val_side * au4_val_side
feature_data['RE_AU01_AU04_diff_side'] = au1_val_side - au4_val_side

# AU02 × AU04
feature_data['RE_AU02_AU04_product_side'] = au2_val_side * au4_val_side
feature_data['RE_AU02_AU04_diff_side'] = au2_val_side - au4_val_side

# 3-way interaction
feature_data['RE_AU01_AU02_AU04_sum_side'] = au1_val_side + au2_val_side + au4_val_side

# Updated averages to include AU04
feature_data['RE_avg_Asym_Ratio'] = (au1_ratio + au2_ratio + au4_ratio) / 3.0
feature_data['RE_avg_Asym_PercDiff'] = (au1_pd + au2_pd + au4_pd) / 3.0
feature_data['RE_max_Asym_PercDiff'] = pd.concat([au1_pd, au2_pd, au4_pd], axis=1).max(axis=1)
```

**New Features Added:** 8 (5 interactions + 3 updated averages)
**New Total:** 24 + 8 = **32 features**

---

#### Mid Face - ADD THESE FEATURES

**AU45 × AU06 Interactions (leverage AU45's improvement):**
```python
# Add to mid_face_features.py after ETES features

# AU45 × AU06 interactions for each action
for action in ['ES', 'ET', 'BK']:
    au45_val_side = feature_data.get(f'{action}_AU45_r_val_side', pd.Series(0.0, index=df.index))
    au06_val_side = feature_data.get(f'{action}_AU06_r_val_side', pd.Series(0.0, index=df.index))

    # Product and ratio
    feature_data[f'{action}_AU45_AU06_product_side'] = au45_val_side * au06_val_side
    feature_data[f'{action}_AU45_AU06_ratio_side'] = calculate_ratio(au45_val_side, au06_val_side, min_value=min_val_cfg)

    # Asymmetry of product
    au45_val_opp = feature_data.get(f'{action}_AU45_r_val_opp', pd.Series(0.0, index=df.index))
    au06_val_opp = feature_data.get(f'{action}_AU06_r_val_opp', pd.Series(0.0, index=df.index))
    product_opp = au45_val_opp * au06_val_opp
    product_side = au45_val_side * au06_val_side
    feature_data[f'{action}_AU45_AU06_product_asym'] = product_side - product_opp

# BK-specific asymmetry (blink asymmetry is critical for lagophthalmos)
bk_au45_asym_diff = feature_data.get('BK_AU45_r_Asym_Diff', pd.Series(0.0, index=df.index))
bk_au45_asym_ratio = feature_data.get('BK_AU45_r_Asym_Ratio', pd.Series(1.0, index=df.index))
feature_data['BK_AU45_strong_asymmetry'] = (bk_au45_asym_diff.abs() > 0.5).astype(int)  # Binary flag

# AU06 dominance (when AU06 >> AU45, different pattern)
es_au06_val = feature_data.get('ES_AU06_r_val_side', pd.Series(0.0, index=df.index))
es_au45_val = feature_data.get('ES_AU45_r_val_side', pd.Series(0.0, index=df.index))
feature_data['ES_AU06_dominant'] = (es_au06_val > es_au45_val).astype(int)
```

**New Features Added:** 11 (3 actions × 3 + 2 special features)
**New Total (after removing AU07):** 49 + 11 = **60 features**

---

#### Lower Face - ADD THESE FEATURES

**AU15, AU20 Integration (currently only AU12 and AU25 have custom features):**
```python
# Add to lower_face_features.py after AU12/AU25 features

# AU15 summary features (similar to AU12)
avg_au15_ratio_vals = []
max_au15_pd_vals = []
for act in actions:
    ratio_key = f"{act}_AU15_r_Asym_Ratio"
    pd_key = f"{act}_AU15_r_Asym_PercDiff"
    if ratio_key in feature_data:
        avg_au15_ratio_vals.append(feature_data[ratio_key])
    if pd_key in feature_data:
        max_au15_pd_vals.append(feature_data[pd_key])

feature_data['avg_AU15_Asym_Ratio'] = pd.concat(avg_au15_ratio_vals, axis=1).mean(axis=1) if avg_au15_ratio_vals else pd.Series(1.0, index=df.index)
feature_data['max_AU15_Asym_PercDiff'] = pd.concat(max_au15_pd_vals, axis=1).max(axis=1) if max_au15_pd_vals else pd.Series(0.0, index=df.index)

# AU20 summary features (leverage +372% improvement!)
avg_au20_ratio_vals = []
max_au20_pd_vals = []
for act in actions:
    ratio_key = f"{act}_AU20_r_Asym_Ratio"
    pd_key = f"{act}_AU20_r_Asym_PercDiff"
    if ratio_key in feature_data:
        avg_au20_ratio_vals.append(feature_data[ratio_key])
    if pd_key in feature_data:
        max_au20_pd_vals.append(feature_data[pd_key])

feature_data['avg_AU20_Asym_Ratio'] = pd.concat(avg_au20_ratio_vals, axis=1).mean(axis=1) if avg_au20_ratio_vals else pd.Series(1.0, index=df.index)
feature_data['max_AU20_Asym_PercDiff'] = pd.concat(max_au20_pd_vals, axis=1).max(axis=1) if max_au20_pd_vals else pd.Series(0.0, index=df.index)

# AU12/AU15 ratio (smile vs frown balance - antagonistic movements)
bs_au12_val = feature_data.get('BS_AU12_r_val_side', pd.Series(0.0, index=df.index))
bs_au15_val = feature_data.get('BS_AU15_r_val_side', pd.Series(0.0, index=df.index))
feature_data['BS_AU12_AU15_ratio'] = calculate_ratio(bs_au12_val, bs_au15_val, min_value=0.0001)

# AU20/AU25 ratio (lip stretch vs part balance)
bs_au20_val = feature_data.get('BS_AU20_r_val_side', pd.Series(0.0, index=df.index))
bs_au25_val = feature_data.get('BS_AU25_r_val_side', pd.Series(0.0, index=df.index))
feature_data['BS_AU20_AU25_ratio'] = calculate_ratio(bs_au20_val, bs_au25_val, min_value=0.0001)

# AU12 × AU20 product (both smile-related)
feature_data['BS_AU12_AU20_product'] = bs_au12_val * bs_au20_val

# AU15 × AU20 product (frown + stretch = snarl?)
so_au15_val = feature_data.get('SO_AU15_r_val_side', pd.Series(0.0, index=df.index))
so_au20_val = feature_data.get('SO_AU20_r_val_side', pd.Series(0.0, index=df.index))
feature_data['SO_AU15_AU20_product'] = so_au15_val * so_au20_val

# Overall lower face activity (sum of all functional AUs)
feature_data['BS_total_activity'] = bs_au12_val + bs_au15_val + bs_au20_val + bs_au25_val
```

**New Features Added:** 11
**New Total (after removing dead AUs):** 100 + 11 = **111 features**

---

### Priority 3: Consider Temporal Features (Medium Value)

**Rationale:** With fewer AUs, temporal patterns may help discriminate paralysis severity.

**Potential Features (would require frame-level data, not just max frames):**
- AU velocity (rate of change)
- AU smoothed values (moving average)
- AU standard deviation (irregular movement)
- Time to peak AU activation
- AU activation duration

**Recommendation:**
- ⏸ **Skip for Phase 1 retraining** (requires data pipeline changes)
- ✅ **Consider for Phase 3** if F1 scores remain below target

---

## Implementation Plan

### Phase 1A: Feature Set Updates (Before Retraining)

**1. Update paralysis_config.py**
- ✅ Already done: Added AU04 to upper face

**2. Update upper_face_features.py**
- Add AU04 interactions (8 new features)
- Update averages to include AU04

**3. Update mid_face_features.py**
- Remove AU07 references (already handled by config change - just verify)
- Add AU45 × AU06 interactions (11 new features)
- Add BK-specific blink asymmetry features

**4. Update lower_face_features.py**
- Remove AU10, AU14, AU17, AU23, AU26 references (already handled by config)
- Add AU15 summary features (2 new)
- Add AU20 summary features (2 new)
- Add AU12/AU15, AU20/AU25 ratio features (2 new)
- Add AU interaction products (3 new)
- Add total activity feature (1 new)

**Estimated Effort:** 2-3 hours of coding + testing

---

### Phase 1B: Feature Count Verification

**Expected Feature Counts After Updates:**

| Zone | Before | Dead Features | After Removal | New Features | Final Count |
|------|--------|---------------|---------------|--------------|-------------|
| Upper | 24 | 0 | 24 | +8 | **32** |
| Mid | 73 | 24 | 49 | +11 | **60** |
| Lower | 220 | 120 | 100 | +11 | **111** |

**Total Features:** 203 (down from 317, but more informative)

---

### Phase 1C: Retrain and Evaluate

**After implementing feature changes:**

1. Clear feature cache: `rm -rf .cache/`
2. Run training pipeline for each zone
3. Compare F1 scores to targets:
   - Upper: Target ≥ 0.80 (was 0.83)
   - Mid: Target ≥ 0.70 (was 0.92)
   - Lower: Target ≥ 0.65 (was 0.82)

4. Analyze feature importance:
   - Verify new features are being used
   - Check if dead AUs were properly excluded
   - Identify which features are most predictive

---

## Risk Assessment

### High Risk
1. **Mid face AU06 weakness** - Even with new features, AU06's 93% drop may limit performance
2. **Lower face AU12/AU25 degradation** - Both dropped 80-87%, yet are heavily relied upon

### Medium Risk
1. **Feature engineering complexity** - More features = more hyperparameter tuning needed
2. **Overfitting risk** - With fewer samples and more features, overfitting is possible
3. **SMOTE sensitivity** - Class balancing may behave differently with new feature distributions

### Low Risk
1. **Upper face** - Should perform well with 100% AU retention
2. **AU20 and AU45** - Both improved significantly and should boost performance

---

## Success Criteria

### Phase 1 Success
After feature updates and retraining:
- Upper Face: F1 ≥ 0.78
- Mid Face: F1 ≥ 0.68
- Lower Face: F1 ≥ 0.63

### Phase 2 Success (if Phase 1 insufficient)
After hyperparameter re-optimization:
- Upper Face: F1 ≥ 0.80
- Mid Face: F1 ≥ 0.72
- Lower Face: F1 ≥ 0.68

### Phase 3 Success (if Phase 2 insufficient)
After additional data collection:
- Upper Face: F1 ≥ 0.83 (maintain original)
- Mid Face: F1 ≥ 0.75
- Lower Face: F1 ≥ 0.72

---

## Next Steps

1. ✅ Add AU04 to upper face config (DONE)
2. ⏭ Implement new interaction features in all three zone files
3. ⏭ Verify dead AU features are excluded
4. ⏭ Run Phase 1 retraining
5. ⏭ Analyze results and determine if Phase 2 needed

**Estimated Timeline:** 1-2 days for feature implementation + testing, 1 day for retraining and analysis

---

**Report prepared by:** Claude (AI Assistant)
**Last Updated:** 2025-10-22
