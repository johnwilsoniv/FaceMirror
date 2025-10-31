# Comprehensive Performance Recovery Plan
## OpenFace 2.2 (Published) vs OpenFace 3.0 (Current)

**Date**: 2025-10-27
**Status**: CRITICAL - Same cohort, ~50-60% performance degradation
**Root Cause**: Methodology changes between published and current code

---

## Executive Summary

Since you've confirmed this is the **SAME PATIENT COHORT** as the published manuscript, the performance drop is **NOT** due to different training data. This narrows the root cause to one or more of the following:

1. **OpenFace 2.2 → 3.0 AU extraction differences**
2. **Code/methodology changes made after publication**
3. **Configuration differences between published and current**
4. **Data caching or preprocessing artifacts**

---

## Code Comparison Analysis

### Critical Differences Identified

#### 1. **Upper Face AU Configuration** ⚠️ MAJOR CHANGE

**Published Code** (`open2GR/3_Data_Analysis/paralysis_config.py` line 194):
```python
'aus': ['AU01_r', 'AU02_r']  # Only 2 AUs
```

**Current Code** (`SplitFace Open3/S3 Data Analysis/paralysis_config.py` line 194):
```python
'aus': ['AU01_r', 'AU02_r', 'AU04_r']  # 3 AUs - AU04 ADDED!
```

**Impact**:
- Published upper face used only AU01 + AU02 (brow raisers)
- Current adds AU04 (brow lowerer - antagonistic action)
- This **fundamentally changes** the feature space
- AU04 was likely in OpenFace 2.2 output but **not used** in the published model
- May cause training issues since brow lowering conflicts with brow raising

**Analysis**: This is likely a **deliberate enhancement** you made post-publication, but it may be causing model confusion. The published model achieved 0.83 accuracy WITHOUT AU04.

#### 2. **Data Caching System** ⚠️ ADDED POST-PUBLICATION

**Published Code**: No caching mechanism

**Current Code** (lines 114-155 in current `paralysis_training_pipeline.py`):
```python
def get_data_cache_key(zone_key, results_csv, expert_csv):
    """Generate cache key based on zone and input file modification times"""

def load_cached_data(cache_path):
    """Load preprocessed data from cache"""

def save_data_cache(cache_path, features_df, targets_arr, metadata_df):
    """Save preprocessed data to cache"""
```

**Impact**:
- Could be loading stale/corrupted cached data
- Cache key uses file modification times - may not detect all changes
- **CRITICAL**: If you modified OpenFace outputs or processing but cache wasn't invalidated, you're training on old data!

**Test**: Delete `.cache` directory and retrain to rule this out.

#### 3. **Optuna Trial Counts**

**Published**:
- Lower: 120 trials
- Mid: 80 trials (DIFFERENT from current!)
- Upper: 70 trials (DIFFERENT from current!)

**Current**:
- Lower: 120 trials ✓ (matches)
- Mid: 100 trials (was 80)
- Upper: 100 trials (was 70)

**Impact**: More trials should improve performance, not degrade it. This is not the cause.

#### 4. **Class Weights**

**Published**:
- Lower: {0: 1.0, 1: 3.5, 2: 2.5}
- Mid: {0: 1.0, 1: 4.0, 2: 3.0}
- Upper: {0: 1.0, 1: 3.0, 2: 2.0}

**Current**:
- Lower: {0: 1.0, 1: 3.5, 2: 2.5} ✓ (matches)
- Mid: {0: 1.0, 1: 4.0, 2: 3.0} ✓ (matches)
- Upper: {0: 1.0, 1: 3.0, 2: 2.0} ✓ (matches)

**Impact**: These match! Not the issue.

#### 5. **Feature Selection Top N**

**Published**:
- Lower: 60 features
- Mid: 40 features
- Upper: 25 features

**Current**:
- Lower: 60 features ✓ (matches)
- Mid: 40 features ✓ (matches)
- Upper: 25 features ✓ (matches)

**Impact**: These match! Not the issue.

#### 6. **SMOTE + ENN Configuration**

**Published Code** (line 105):
```python
'use_smoteenn_after': True,  # SMOTE then ENN cleaning
```

**Current Code**:
```python
'use_smoteenn_after': True,  # Same ✓
```

**Impact**: Matches, but worth verifying SMOTEENN is working correctly.

---

## Root Cause Hypothesis (Prioritized)

### #1 MOST LIKELY: Data Caching Issue 🔴🔴🔴

**Evidence**:
- New caching system not in published code
- Cache invalidation based only on file modification time
- If OpenFace outputs were regenerated but cache wasn't cleared, using stale data
- `.cache/` directory may contain corrupted data

**Test**:
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
rm -rf .cache/
# Then retrain all zones
```

**Expected Outcome**: If this is the issue, performance should improve immediately.

### #2 HIGHLY LIKELY: AU04 Addition to Upper Face 🔴🔴

**Evidence**:
- Upper face published used AU01, AU02 only
- Current adds AU04 (brow lowerer)
- Upper face performance: 0.83 → 0.58 (-29.5%)
- AU04 is antagonistic to AU01/AU02, may confuse model

**Test**:
```python
# In paralysis_config.py, change line 194:
# FROM:
'aus': ['AU01_r', 'AU02_r', 'AU04_r']
# TO:
'aus': ['AU01_r', 'AU02_r']  # Match published
```

**Expected Outcome**: Upper face performance should improve to ~0.83.

### #3 LIKELY: OpenFace 3.0 AU Intensity Scaling Differences 🔴

**Evidence**:
- Same AUs present, but extraction algorithms may differ
- Manuscript used OpenFace 2.0
- You're using OpenFace 3.0
- Even subtle scaling changes could affect feature engineering

**Test**:
```python
# Compare AU intensity distributions
import pandas as pd
import numpy as np

# Load OpenFace 2.0 outputs (if you still have them)
df_v2 = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S2O Coded Files/..._coded.csv')  # OpenFace 2.0
df_v3 = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S3O Results/combined_results.csv')  # OpenFace 3.0

au_cols = [col for col in df_v2.columns if col.startswith('AU') and col.endswith('_r')]

print("AU Intensity Comparison (OpenFace 2.0 vs 3.0):")
for au in au_cols:
    v2_mean = df_v2[au].dropna().mean()
    v3_mean = df_v3[au].dropna().mean()
    delta = v3_mean - v2_mean
    pct_change = (delta / v2_mean * 100) if v2_mean != 0 else 0
    print(f"{au:10s}: v2={v2_mean:.3f}, v3={v3_mean:.3f}, delta={delta:+.3f} ({pct_change:+.1f}%)")
```

**Expected Outcome**: If significant scaling differences exist, may need to retrain with normalized AUs or adjust feature extraction.

### #4 POSSIBLE: Training Data Preprocessing Differences 🔶

**Evidence**:
- Your current `paralysis_utils.py` may differ from published version
- Feature extraction code may have changed
- Asymmetry calculations may have subtle differences

**Test**: Compare `prepare_data_generalized` implementation between published and current.

### #5 POSSIBLE: Random Seed Differences 🔶

**Evidence**:
- Published and current both use `random_state: 42`
- But if data ordering changed due to caching, train/test split could differ

**Test**: Verify train/test split consistency.

---

## Action Plan

### PHASE 1: Quick Diagnostics (30 minutes)

#### Step 1.1: Clear Data Cache
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
rm -rf .cache/
echo "Cache cleared"
```

#### Step 1.2: Revert Upper Face to Published Config
```python
# Edit paralysis_config.py line 194
# Change FROM:
'aus': ['AU01_r', 'AU02_r', 'AU04_r']
# TO:
'aus': ['AU01_r', 'AU02_r']
```

#### Step 1.3: Retrain Upper Face ONLY
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
python3 paralysis_training_pipeline.py upper
```

**Expected Results**:
- If cache was the issue: ALL zones should improve when fully retrained
- If AU04 was the issue: Upper face should improve to ~0.83 accuracy

---

### PHASE 2: Verify Data Consistency (1 hour)

#### Step 2.1: Compare AU Distributions
```python
# Create: verify_openface_outputs.py
import pandas as pd
import numpy as np
import glob

# Find all OpenFace 3.0 coded files
coded_files = glob.glob('/Users/johnwilsoniv/Documents/SplitFace/S2O Coded Files/*_coded.csv')

print(f"Found {len(coded_files)} coded files")

# Sample a few files and check AU distributions
for i, file_path in enumerate(coded_files[:5]):
    df = pd.read_csv(file_path)
    print(f"\n{'='*60}")
    print(f"File {i+1}: {os.path.basename(file_path)}")
    print('='*60)

    au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

    for au in au_cols:
        data = df[au].dropna()
        if len(data) > 0:
            print(f"{au:10s}: mean={data.mean():6.3f}, std={data.std():6.3f}, "
                  f"count={len(data):4d}, nan%={df[au].isna().sum()/len(df)*100:5.1f}%")
```

#### Step 2.2: Verify Expert Labels Unchanged
```python
# verify_expert_labels.py
import pandas as pd

df = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/FPRS FP Key.csv', dtype=str, keep_default_na=False)

zones = {
    'Lower': ['Paralysis - Left Lower Face', 'Paralysis - Right Lower Face'],
    'Mid': ['Paralysis - Left Mid Face', 'Paralysis - Right Mid Face'],
    'Upper': ['Paralysis - Left Upper Face', 'Paralysis - Right Upper Face']
}

for zone_name, cols in zones.items():
    print(f"\n{zone_name} Face Distribution:")
    all_labels = pd.concat([df[col] for col in cols if col in df.columns])
    print(all_labels.value_counts())

    # Calculate percentages
    total = len(all_labels[all_labels != ''])
    for label, count in all_labels.value_counts().items():
        if label != '':
            pct = count / total * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
```

#### Step 2.3: Check Train/Test Split Consistency
```python
# verify_train_test_split.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Simulate the split process
# This should match what paralysis_training_pipeline.py does

# Load data (simplified - just checking split)
# After prepare_data_generalized runs, check if split is deterministic

# Create dummy data with same characteristics
n_samples = 222  # Example from your test sets
random_state = 42

# Generate indices
indices = np.arange(n_samples)

# Stratified split
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.25,
    random_state=random_state,
    stratify=None  # Simplified
)

print(f"Train size: {len(train_idx)}")
print(f"Test size: {len(test_idx)}")
print(f"First 10 test indices: {sorted(test_idx)[:10]}")
```

---

### PHASE 3: Full Comparison (2 hours)

#### Step 3.1: Copy Published Code Feature Extraction
```bash
# Backup current feature extraction
cp "S3 Data Analysis/lower_face_features.py" "S3 Data Analysis/lower_face_features.py.backup"
cp "S3 Data Analysis/mid_face_features.py" "S3 Data Analysis/mid_face_features.py.backup"
cp "S3 Data Analysis/upper_face_features.py" "S3 Data Analysis/upper_face_features.py.backup"

# Copy published versions
cp "/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/lower_face_features.py" "S3 Data Analysis/"
cp "/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/mid_face_features.py" "S3 Data Analysis/"
cp "/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/upper_face_features.py" "S3 Data Analysis/"
```

#### Step 3.2: Retrain with Published Feature Extraction
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
python3 paralysis_training_pipeline.py
```

**Expected**: If feature extraction differences are the cause, performance should match published.

#### Step 3.3: Compare Optuna Hyperparameter Search Results

Check if Optuna is finding similar hyperparameters:

**Published Lower Face Best Params** (from paper):
```
learning_rate: 0.1181
max_depth: 5
n_estimators: 418
min_child_weight: 6
gamma: 0.07956
subsample: 0.953
colsample_bytree: 0.9174
reg_alpha: 0.02496
reg_lambda: 0.004963
```

**Your Current Lower Face Best Params**:
```
learning_rate: 0.1181  (EXACT MATCH!)
max_depth: 5  (EXACT MATCH!)
n_estimators: 418  (EXACT MATCH!)
min_child_weight: 6  (EXACT MATCH!)
gamma: 0.07956  (EXACT MATCH!)
subsample: 0.953  (EXACT MATCH!)
colsample_bytree: 0.9174  (EXACT MATCH!)
reg_alpha: 0.02496  (EXACT MATCH!)
reg_lambda: 0.004963  (EXACT MATCH!)
```

**CRITICAL FINDING**: Your hyperparameters are **IDENTICAL** to the published ones. This strongly suggests:
1. You loaded a saved Optuna study from the published work, OR
2. You're using the exact same random seed and getting identical results, OR
3. These are the optimal parameters for this problem and Optuna consistently finds them

This RULES OUT hyperparameter differences as the cause!

---

### PHASE 4: OpenFace Version Testing (3 hours)

#### Step 4.1: Check if OpenFace 2.2 Outputs Still Exist
```bash
# Search for original OpenFace 2.2 outputs
find /Users/johnwilsoniv/Documents -name "*_coded.csv" -type f | head -10

# Check modification dates
ls -lt /Users/johnwilsoniv/Documents/SplitFace/S2O\ Coded\ Files/ | head
```

#### Step 4.2: If OpenFace 2.2 Outputs Exist, Retrain on Them
```python
# Temporarily modify INPUT_FILES in paralysis_config.py
INPUT_FILES = {
    'results_csv': '/path/to/openface2.2/combined_results.csv',  # If it exists
    'expert_key_csv': 'FPRS FP Key.csv'
}
```

Then retrain and compare.

#### Step 4.3: Document OpenFace Version Differences
Create test comparing same video through OpenFace 2.2 vs 3.0:

```bash
# If you have both versions installed:
openface2.2 -f test_video.mp4 -out test_v2.csv
openface3.0 -f test_video.mp4 -out test_v3.csv

# Compare outputs
python3 compare_openface_versions.py test_v2.csv test_v3.csv
```

---

## Decision Tree

```
START
 |
 ├─> Clear cache and retrain
 |   ├─> Performance improves? → CACHE WAS THE ISSUE ✓
 |   └─> No improvement → Continue
 |
 ├─> Revert AU04 from upper face
 |   ├─> Upper performance improves to ~0.83? → AU04 WAS THE ISSUE ✓
 |   └─> No improvement → Continue
 |
 ├─> Compare AU distributions v2 vs v3
 |   ├─> Significant differences? → OPENFACE VERSION IS THE ISSUE
 |   |   └─> Retrain on OpenFace 2.2 outputs if available
 |   └─> No significant differences → Continue
 |
 ├─> Use published feature extraction code
 |   ├─> Performance improves? → FEATURE EXTRACTION CHANGED ✓
 |   └─> No improvement → Continue
 |
 └─> Deep dive into prepare_data_generalized
     └─> Line-by-line comparison with published version
```

---

## Immediate Actions (Start Here)

### Action 1: Clear Cache (5 minutes)
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
rm -rf .cache/
```

### Action 2: Revert Upper Face Config (2 minutes)
```python
# Edit S3 Data Analysis/paralysis_config.py line 194
'aus': ['AU01_r', 'AU02_r']  # Remove AU04_r
```

### Action 3: Quick Retrain Upper Face (10 minutes)
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
python3 paralysis_training_pipeline.py upper
```

### Action 4: Check Result
- If upper face accuracy improves to ~0.80-0.83 → AU04 was causing issues
- If still poor (~0.58) → Problem is deeper

### Action 5: Full Retrain All Zones (30 minutes)
```bash
python3 paralysis_training_pipeline.py
```

Compare all results to published benchmarks.

---

## Expected Outcomes

### Scenario A: Cache + AU04 Fix (MOST LIKELY)
- **After cache clear + AU04 removal + full retrain**:
  - Lower: 0.82-0.84 accuracy ✓
  - Mid: 0.90-0.93 accuracy ✓
  - Upper: 0.81-0.83 accuracy ✓
  - Partial F1 scores all > 0.35 ✓

### Scenario B: OpenFace Version Issue
- **After all fixes, still poor performance**:
  - Need to retrain on OpenFace 2.2 outputs
  - Or adjust feature extraction to normalize OpenFace 3.0 outputs
  - May need to collect new training data with OpenFace 3.0

### Scenario C: Code Regression
- **After reverting to published code, performance matches**:
  - Systematic code review to find regressions
  - Unit tests to prevent future regressions
  - Version control to track changes

---

## Success Metrics

After completing actions, success is defined as:

| Zone | Target Accuracy | Target Weighted F1 | Target Partial F1 |
|------|----------------|--------------------|--------------------|
| Lower | ≥ 0.82 | ≥ 0.80 | ≥ 0.40 |
| Mid | ≥ 0.90 | ≥ 0.90 | ≥ 0.60 |
| Upper | ≥ 0.81 | ≥ 0.81 | ≥ 0.35 |

---

## Monitoring & Validation

After fixes are applied:

1. **Re-run full training 3 times** with different random seeds to verify stability
2. **Save baseline metrics** for future comparison
3. **Document all changes** made to recover performance
4. **Create regression tests** to prevent future degradation
5. **Version lock** all dependencies (scikit-learn, xgboost, imbalanced-learn, etc.)

---

## Next Steps

Please execute **Actions 1-5** in order and report back the results. Based on which action resolves the issue, we can:
1. Document the root cause
2. Prevent recurrence
3. Update your code/process accordingly

**Start with Action 1 (clear cache) - this takes 5 minutes and could immediately solve the problem.**
