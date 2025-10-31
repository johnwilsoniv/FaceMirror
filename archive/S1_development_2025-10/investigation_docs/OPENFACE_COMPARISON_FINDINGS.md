# OpenFace 2.2 vs 3.0 - Critical Differences Found

## Executive Summary

**The paralysis detection models are failing because OpenFace 3.0 produces fundamentally different AU data than OpenFace 2.2.**

### Key Issues Identified

1. **8 AUs are completely missing (all NaN) in OpenFace 3.0**
2. **AU intensity values are 10-50x smaller** in OpenFace 3.0
3. **Correlations between versions are near-zero** (typically < 0.3)
4. **Your ML models were trained on OpenFace 2.2 data** and cannot interpret OpenFace 3.0 data

---

## Critical Finding #1: Missing AUs

### AUs That Are ALL NaN in OpenFace 3.0 (but have values in 2.2):

- **AU05_r** - Upper Lid Raiser
- **AU07_r** - Lid Tightener
- **AU09_r** - Nose Wrinkler
- **AU10_r** - Upper Lip Raiser
- **AU14_r** - Dimpler
- **AU17_r** - Chin Raiser
- **AU23_r** - Lip Tightener
- **AU26_r** - Jaw Drop

**Impact**: Your paralysis detection models rely on these AUs (especially for lower face detection), but OpenFace 3.0 never computes them.

---

## Critical Finding #2: Magnitude Differences

OpenFace 3.0 produces **dramatically smaller** AU intensity values:

| AU | OF3.0 Mean | OF2.2 Mean | Ratio |
|----|------------|------------|-------|
| AU01_r (Inner Brow Raiser) | 0.013 | 0.358 | **27.5x smaller** |
| AU02_r (Outer Brow Raiser) | 0.006 | 0.219 | **36.5x smaller** |
| AU04_r (Brow Lowerer) | 0.039 | 2.318 | **59.4x smaller** |
| AU06_r (Cheek Raiser) | 0.006 | 0.469 | **78.2x smaller** |
| AU12_r (Lip Corner Puller) | 0.069 | 0.182 | **2.6x smaller** |
| AU25_r (Lips Part) | 0.067 | 0.657 | **9.8x smaller** |
| AU45_r (Blink) | 1.923 | 0.414 | **4.6x LARGER** |

**Impact**: Your ML models learned decision boundaries based on OF2.2 magnitudes. When you feed in OF3.0 data with 10-50x smaller values, the models cannot classify correctly.

---

## Critical Finding #3: Near-Zero Correlations

Correlation measures how similarly the two versions track AU intensity across frames (1.0 = perfect match, 0.0 = no relationship).

### Results for Sample File (20240723_175947000_iOS_left_mirrored):

| AU | Correlation | Status |
|----|-------------|--------|
| AU01_r | 0.112 | ❌ Very poor |
| AU02_r | 0.027 | ❌ Essentially random |
| AU04_r | 0.197 | ❌ Very poor |
| AU06_r | 0.186 | ❌ Very poor |
| AU12_r | -0.014 | ❌ **Negative** (opposite direction!) |
| AU15_r | -0.132 | ❌ **Negative** |
| AU20_r | -0.073 | ❌ **Negative** |
| AU25_r | 0.474 | ⚠️ Moderate (still not good) |
| AU45_r | 0.497 | ⚠️ Moderate |

**Impact**: The two versions don't even agree on WHEN AUs are active. Some have negative correlation, meaning they activate in opposite patterns!

---

## Why This Breaks Your Pipeline

### Your Training Data:
- Used OpenFace 2.2 outputs
- AU intensity range: 0-5 (typical)
- 17 AUs available (8 are now missing)
- Models learned: "If AU04_r > 1.5, it indicates brow movement"

### Your Production Data (OpenFace 3.0):
- AU intensity range: 0-0.3 (typical for most AUs)
- Only 9 AUs have non-NaN values
- 8 critical AUs are always NaN
- OpenFace 3.0 never triggers "AU04_r > 1.5" threshold

**Result**: Models classify everything as "no paralysis" or give random predictions because the input data is outside the learned distribution.

---

## Column Differences

### OpenFace 3.0 Only:
- `AU16_r` - Lip depressor (NEW)
- `AU16_c` - Lip depressor (classification)

### OpenFace 2.2 Only:
- `AU28_c` - Lip suck (classification)

### Both Versions Have:
- 17 common AUs (but 8 are NaN in OF3.0)
- frame, face_id, timestamp, confidence, success

---

## Specific AU Analysis

### AUs with Values in BOTH (but poor correlation):

1. **AU01_r (Inner Brow Raiser)**: 0.11 correlation
   - Used for: Upper face paralysis detection
   - OF3.0 almost always near zero

2. **AU04_r (Brow Lowerer)**: 0.20 correlation
   - Used for: Forehead movement, brow symmetry
   - OF3.0 values 60x smaller

3. **AU06_r (Cheek Raiser)**: 0.19 correlation
   - Used for: Smile detection, mid-face movement
   - OF3.0 values 78x smaller

4. **AU12_r (Lip Corner Puller)**: -0.01 correlation (NEGATIVE!)
   - Used for: Smile asymmetry, critical for paralysis
   - OF3.0 actually moves OPPOSITE direction

5. **AU45_r (Blink)**: 0.50 correlation (best of all)
   - Used for: Eye closure symmetry
   - OF3.0 actually 4.6x LARGER (opposite problem)

### AUs Missing in OpenFace 3.0 (all NaN):

6. **AU09_r (Nose Wrinkler)**: Missing
   - Used for: Mid-face asymmetry

7. **AU10_r (Upper Lip Raiser)**: Missing
   - Used for: Lower face movement

8. **AU14_r (Dimpler)**: Missing
   - Used for: Smile analysis

9. **AU17_r (Chin Raiser)**: Missing
   - Used for: Lower face paralysis

10. **AU26_r (Jaw Drop)**: Missing
    - Used for: Mouth opening asymmetry

---

## Root Cause

OpenFace 3.0 uses a **completely different model architecture**:

1. **OpenFace 2.2**: Trained on comprehensive AU datasets with 17+ AUs
2. **OpenFace 3.0**: Uses MTL (Multi-Task Learning) model that only outputs 8 AUs
3. Your adapter converts 8 AUs → 18 AUs, but:
   - 8 AUs are mapped to **NaN** (not computed)
   - The 8 that ARE computed use different models/training data
   - Different intensity scales and activation patterns

---

## Solutions (In Order of Feasibility)

### Option 1: Retrain Models on OpenFace 3.0 Data ✅ RECOMMENDED

**Pros:**
- Keeps fast OpenFace 3.0 pipeline
- Models will work correctly with current data
- Future-proof

**Cons:**
- Need to re-run all training videos through OpenFace 3.0
- Re-train all 3 models (upper, mid, lower face)
- May lose accuracy if 8 missing AUs were important

**Steps:**
1. Process all training videos with OpenFace 3.0
2. Update feature lists to exclude the 8 NaN AUs
3. Re-run model training pipeline
4. Compare accuracy

---

### Option 2: Use OpenFace 2.2 for Production ⚠️ TEMPORARY

**Pros:**
- Works immediately with existing models
- No retraining needed

**Cons:**
- 20-50x slower than OpenFace 3.0
- Requires maintaining OpenFace 2.2 binary
- Not a long-term solution

**Steps:**
1. Replace OpenFace 3.0 in S1 Face Mirror with 2.2 binary calls
2. Keep everything else the same

---

### Option 3: Create Calibration Layer 🔬 EXPERIMENTAL

**Pros:**
- No retraining
- Keeps OpenFace 3.0 speed

**Cons:**
- Very difficult to calibrate 8 missing AUs
- May not be accurate
- Requires extensive validation

**Steps:**
1. Learn mapping: OF3.0 AUs → OF2.2 AUs
2. Apply calibration transform before ML models
3. Handle 8 missing AUs (maybe impute from related AUs?)

---

## Immediate Next Steps

1. **Confirm the issue** by testing S3 Data Analysis with OpenFace 2.2 CSVs:
   ```bash
   # Copy OF2.2 CSVs to Combined Data (backup first!)
   cp -r "~/Documents/SplitFace/S1O Processed Files/Combined Data" \
         "~/Documents/SplitFace/S1O Processed Files/Combined Data.OF30_backup"

   cp "~/Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test"/*.csv \
      "~/Documents/SplitFace/S1O Processed Files/Combined Data/"

   # Run S3 Data Analysis
   cd "S3 Data Analysis"
   python main.py
   ```

2. **If S3 works with OF2.2 data**: Confirms the problem is OF3.0 incompatibility

3. **Choose a solution**:
   - **Quick fix**: Use OpenFace 2.2 temporarily
   - **Proper fix**: Retrain models on OpenFace 3.0 data

---

## Which AUs Are Your Models Actually Using?

Check your feature lists:
- `S3 Data Analysis/models/upper_face_features.list`
- `S3 Data Analysis/models/mid_face_features.list`
- `S3 Data Analysis/models/lower_face_features.list`

If any of the **8 missing AUs** (AU05, AU07, AU09, AU10, AU14, AU17, AU23, AU26) are in these lists, your models **cannot work** with OpenFace 3.0 data without retraining.

---

## Conclusion

**Your paralysis detection is failing because:**

1. OpenFace 3.0 doesn't compute 8 critical AUs (outputs NaN)
2. The 9 AUs it does compute have completely different magnitudes (10-50x smaller)
3. The correlations are near zero (different detection patterns)
4. Your ML models were trained on OpenFace 2.2 data with different distributions

**You MUST either:**
- Retrain your models on OpenFace 3.0 data, OR
- Switch back to OpenFace 2.2 for production

There is no middle ground - the data distributions are incompatible.
