# OpenFace 2.2 vs 3.0 Transition Analysis

**Analysis Date:** 2025-10-26  
**Dataset:** 110 overlapping patients

## Executive Summary

OpenFace 3.0 produces **drastically different AU intensity values** compared to OpenFace 2.2, making models trained on OpenFace 2.2 incompatible with OpenFace 3.0 data.

### Critical Findings

**9 AUs have effectively DISAPPEARED** (100% reduction):
- AU05, AU07, AU09, AU10, AU14, AU17, AU23, AU26, AU06 (93%)

**5 AUs have MAJOR LOSS** (59-85% reduction):
- AU04, AU02, AU25, AU12, AU01

**This explains the poor model performance** - the features the models were trained on are essentially zero in OpenFace 3.0.

---

## Detailed AU Analysis

### Critical Loss AUs (>90% reduction)

| AU | Description | OF2.2 Mean | OF3.0 Mean | Change | Impact Area |
|----|-------------|------------|------------|--------|-------------|
| **AU07** | Lid Tightener | 1.460 | 0.000 | -100% | **MID-FACE (CRITICAL)** |
| **AU06** | Cheek Raiser | 0.823 | 0.057 | -93% | **MID-FACE (CRITICAL)** |
| **AU05** | Upper Lid Raiser | 0.153 | 0.000 | -100% | **UPPER-FACE** |
| **AU09** | Nose Wrinkler | 0.258 | 0.000 | -100% | **MID-FACE** |
| **AU10** | Upper Lip Raiser | 0.888 | 0.000 | -100% | **LOWER-FACE** |
| **AU14** | Dimpler | 0.786 | 0.000 | -100% | **LOWER-FACE** |
| **AU17** | Chin Raiser | 0.712 | 0.000 | -100% | **LOWER-FACE** |
| **AU23** | Lip Tightener | 0.205 | 0.000 | -100% | **LOWER-FACE** |
| **AU26** | Jaw Drop | 0.646 | 0.000 | -100% | **LOWER-FACE** |

### Major Loss AUs (50-90% reduction)

| AU | Description | OF2.2 Mean | OF3.0 Mean | Change |
|----|-------------|------------|------------|--------|
| AU04 | Brow Lowerer | 0.833 | 0.121 | -85% |
| AU02 | Outer Brow Raiser | 0.216 | 0.018 | -83% |
| AU25 | Lips Part | 0.840 | 0.110 | -80% |
| AU12 | Lip Corner Puller | 0.922 | 0.185 | -77% |
| AU01 | Inner Brow Raiser | 0.384 | 0.051 | -59% |

### Anomalous Gains

| AU | Description | OF2.2 Mean | OF3.0 Mean | Change | Likely Cause |
|----|-------------|------------|------------|--------|--------------|
| AU45 | Blink | 0.836 | 1.947 | +556% | Different measurement method (EAR-based?) |
| AU20 | Lip Stretcher | 0.135 | 0.649 | +1553% | Detection algorithm change |

---

## Impact by Face Region

### Mid-Face Detection (MOST CRITICAL)

Your mid-face models rely heavily on:
- **AU07 (lid tightener)**: 100% loss ⚠️
- **AU06 (cheek raiser)**: 93% loss ⚠️
- **AU09 (nose wrinkler)**: 100% loss ⚠️

**Result:** Mid-face models cannot function with OpenFace 3.0 data as-is.

### Upper-Face Detection

Your upper-face models rely on:
- **AU01 (inner brow raiser)**: 59% loss
- **AU02 (outer brow raiser)**: 83% loss
- **AU04 (brow lowerer)**: 85% loss

**Result:** Upper-face models will perform poorly without recalibration.

### Lower-Face Detection

Your lower-face models rely on:
- **AU10, AU12, AU14, AU17, AU23, AU25, AU26**: 77-100% loss

**Result:** Lower-face models will fail without retraining.

---

## Why This Happened

OpenFace 3.0 likely uses:
1. **Different detection algorithms** for certain AUs
2. **Different intensity scaling** (more conservative thresholds)
3. **Different landmark models** affecting AU calculations
4. **Improved specificity** (fewer false positives, but lower sensitivity)

The drastically reduced values suggest OpenFace 3.0 is **much more conservative** in detecting AU activation, possibly to reduce false positives.

---

## Recommended Solutions

### Option 1: Retrain Models on OpenFace 3.0 (RECOMMENDED)

**Pros:**
- Most accurate long-term solution
- Models will be optimized for OpenFace 3.0 features
- No scaling/transformation artifacts

**Cons:**
- Requires retraining all three models (upper, mid, lower)
- Need to re-collect or re-process all training data with OpenFace 3.0
- May require feature engineering if some AUs are truly missing

**Implementation:**
1. Re-process all training videos with OpenFace 3.0
2. Analyze which AUs are still functional in OpenFace 3.0
3. Retrain models using only functional AUs
4. May need to derive missing AUs from available features

---

### Option 2: Value Calibration/Scaling

**Approach:** Create transformation functions to map OpenFace 3.0 values to OpenFace 2.2 equivalents.

**Pros:**
- Faster implementation
- Can use existing models
- No need to re-label training data

**Cons:**
- Won't work for AUs with 100% loss (AU07, AU09, AU10, etc.)
- May introduce artifacts or errors
- Unreliable for edge cases

**Feasibility:** ❌ **NOT VIABLE** due to the 100% loss of critical AUs like AU07.

---

### Option 3: Hybrid Approach - Feature Engineering

**Approach:** Derive missing AUs from available features using correlations.

**For example:**
- **AU07** might be correlated with other eye-region AUs
- **AU10** might be derivable from AU12 + AU25 patterns
- Use landmark positions directly as supplementary features

**Pros:**
- May recover some lost information
- Allows use of existing model architecture

**Cons:**
- Derived features will be approximations
- May not capture true AU07 behavior
- Still requires significant retraining

**Feasibility:** ⚠️ **POSSIBLE BUT RISKY** - would need thorough validation.

---

### Option 4: Stay on OpenFace 2.2

**Pros:**
- Models continue to work
- No retraining needed

**Cons:**
- ❌ You already stated OpenFace 2.2 is too hard to distribute
- Technical debt accumulates
- Missing out on OpenFace 3.0 improvements

**Recommendation:** ❌ **NOT VIABLE** given your distribution constraints.

---

## Recommended Action Plan

### Phase 1: Verify Functional AUs (1-2 days)

1. Identify which AUs in OpenFace 3.0 are actually functional and have meaningful variance
2. Check if AU07 is truly absent or just measured differently
3. Analyze raw landmark data to see if we can compute AU07 manually

**Action:** Run detailed analysis on raw OpenFace 3.0 output files.

---

### Phase 2: Feature Engineering Exploration (2-3 days)

1. Investigate if missing AUs can be derived from:
   - Landmark distances/ratios
   - Correlations with other AUs
   - Combinations of available AUs
   
2. For AU07 specifically:
   - Check eye landmark distances
   - Compare with AU45 (blink) behavior
   - See if we can compute lid tightness from landmarks

**Action:** Create custom AU derivation functions.

---

### Phase 3: Model Retraining (1-2 weeks)

1. Re-process all training videos with OpenFace 3.0
2. Retrain models using:
   - Functional OpenFace 3.0 AUs
   - Engineered features (if validated)
   - Raw landmark features as supplementary input
   
3. Validate performance against OpenFace 2.2 baseline

**Action:** Full model retraining pipeline.

---

### Phase 4: Validation & Deployment (3-5 days)

1. Compare OpenFace 3.0 models against OpenFace 2.2 models on held-out test set
2. Ensure clinical accuracy is maintained
3. Deploy updated models

---

## Immediate Next Steps

**I recommend starting with Phase 1:**

1. **Analyze raw OpenFace 3.0 output** to see which AUs are truly functional
2. **Check if AU07 can be manually computed** from landmark positions
3. **Identify AU correlations** in OpenFace 2.2 data to inform feature engineering

**Would you like me to:**
- A) Investigate if we can manually derive AU07 from OpenFace 3.0 landmarks?
- B) Start analyzing feature correlations to engineer missing AUs?
- C) Begin setting up the retraining pipeline for all models?
- D) Something else?

---

## Technical Notes

### Data Quality Observations

- **110 overlapping patients** between datasets
- **Same facial expressions** performed in both datasets
- **Column structure is identical** (1008 AU columns each)
- **No all-NaN columns** in OpenFace 3.0
- **576 columns have >50% NaN** (mostly FR condition - "Frown"?)

### Condition-Specific Impact

Most affected conditions (tasks where AUs differ most):
1. **ET (Eyes Tight)**: -61% average AU intensity
2. **BK (Blink)**: -61% average AU intensity
3. **ES (Eyes Shut)**: -46% average AU intensity

Least affected conditions:
1. **SO (Soft Smile)**: +836% (anomaly)
2. **SE (Snarl)**: +20%
3. **LT (?)**: +18%

---

## Conclusion

**The core issue is clear:** OpenFace 3.0 produces fundamentally different AU intensity values, with 9 critical AUs essentially absent.

**Recommended path forward:**
1. Investigate if missing AUs can be derived from landmarks
2. If not, proceed with full model retraining on OpenFace 3.0

**Bottom line:** You cannot use OpenFace 2.2-trained models with OpenFace 3.0 data without significant modifications. Retraining is the most reliable solution.
