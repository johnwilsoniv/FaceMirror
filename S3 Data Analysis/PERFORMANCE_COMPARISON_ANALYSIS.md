# Performance Comparison: Current vs. Benchmark

## Executive Summary

**CRITICAL ISSUE**: Your current model performance is dramatically below the published benchmark across all facial zones. This represents a **50-60% drop** in accuracy and weighted F1-scores.

---

## Performance Metrics Comparison

### Lower Face (Mouth)

| Metric | Benchmark (Paper) | Current | Delta | % Change |
|--------|------------------|---------|-------|----------|
| **Accuracy** | **0.84** | **0.3455** | **-0.4945** | **-58.9%** |
| **Weighted F1** | **0.82** | **0.3808** | **-0.4392** | **-53.6%** |
| F1 (None) | 0.92 | 0.5091 | -0.4109 | -44.7% |
| F1 (Partial) | 0.46 | **0.0000** | -0.46 | **-100%** |
| F1 (Complete) | 0.81 | 0.3030 | -0.5070 | -62.6% |
| Test Samples | 56 | 55 | -1 | -1.8% |

**Status**: ❌ **SEVERE DEGRADATION**

---

### Mid Face (Eye)

| Metric | Benchmark (Paper) | Current | Delta | % Change |
|--------|------------------|---------|-------|----------|
| **Accuracy** | **0.93** | **0.6667** | **-0.2633** | **-28.3%** |
| **Weighted F1** | **0.92** | **0.7019** | **-0.2181** | **-23.7%** |
| F1 (None) | 0.98 | 0.8169 | -0.1631 | -16.6% |
| F1 (Partial) | 0.67 | 0.2222 | -0.4478 | -66.8% |
| F1 (Complete) | 0.83 | 0.5263 | -0.3037 | -36.6% |
| Test Samples | 54 | 54 | 0 | 0% |

**Status**: ⚠️ **SIGNIFICANT DEGRADATION**

---

### Upper Face (Eyebrows)

| Metric | Benchmark (Paper) | Current | Delta | % Change |
|--------|------------------|---------|-------|----------|
| **Accuracy** | **0.83** | **0.5849** | **-0.2451** | **-29.5%** |
| **Weighted F1** | **0.83** | **0.5840** | **-0.2460** | **-29.6%** |
| F1 (None) | 0.88 | 0.7606 | -0.1194 | -13.6% |
| F1 (Partial) | 0.40 | **0.0000** | -0.40 | **-100%** |
| F1 (Complete) | 0.86 | 0.3333 | -0.5267 | -61.2% |
| Test Samples | 53 | 53 | 0 | 0% |

**Status**: ❌ **SEVERE DEGRADATION**

---

## Critical Issues Identified

### 1. **Partial Class Catastrophic Failure** ⚠️⚠️⚠️

The "Partial" class (Incomplete paralysis) shows complete or near-complete failure:

| Zone | Benchmark Partial F1 | Current Partial F1 | Correct Predictions |
|------|---------------------|-------------------|---------------------|
| Lower | 0.46 | **0.0000** | **0/9 (0%)** |
| Mid | 0.67 | 0.2222 | 2/8 (25%) |
| Upper | 0.40 | **0.0000** | **0/5 (0%)** |

**Analysis**:
- Lower face: Model predicted 0 partial cases correctly - classified 2 as None, 7 as Complete
- Upper face: Model predicted 0 partial cases correctly - classified 3 as None, 2 as Complete
- Mid face: Only 25% correct - classified 1 as None, 5 as Complete

**This is the most clinically significant problem** as partial paralysis is the most important class to detect for treatment decisions.

---

### 2. Confusion Matrix Analysis

#### Lower Face - Current Performance
```
True \ Pred     None   Partial  Complete
None              14       11         9      (Only 41% correct)
Partial            2        0         7      (0% correct!)
Complete           5        2         5      (42% correct)
```

#### Mid Face - Current Performance
```
True \ Pred     None   Partial  Complete
None              29        8         4      (71% correct)
Partial            1        2         5      (25% correct)
Complete           0        0         5      (100% correct)
```

#### Upper Face - Current Performance
```
True \ Pred     None   Partial  Complete
None              27        3         5      (77% correct)
Partial            3        0         2      (0% correct!)
Complete           6        3         4      (31% correct)
```

---

## Root Cause Analysis

### Likely Primary Causes:

#### 1. **Training Data Issues** (Most Likely)
- **Different patient cohort**: Your test set may have different characteristics than the original training data
- **Label quality differences**: The expert labels may differ from the original manuscript labels
- **Class distribution mismatch**: The current data shows severe class imbalance
  - Lower: 34 None, 9 Partial, 12 Complete
  - Mid: 41 None, 8 Partial, 5 Complete
  - Upper: 35 None, 5 Partial, 13 Complete
- **Partial class severely underrepresented** (only 9-16% of test samples)

#### 2. **OpenFace Version Differences**
From your earlier analysis:
- Manuscript used OpenFace 2.0
- You're using OpenFace 3.0
- Even though the same 18 AUs are present, the **AU extraction algorithms may have changed**
- Different calibration or intensity scaling in OpenFace 3.0

#### 3. **SMOTE Not Working Effectively**
Your config shows SMOTE is enabled with:
```python
'variant': 'borderline'
'sampling_strategy': 'adaptive'
'target_ratio_partial_to_majority': 0.8
```

But the Partial class is still failing completely. Possible issues:
- SMOTE may be creating synthetic samples that don't generalize
- Adaptive strategy parameters may need tuning for this dataset
- Feature space may not be conducive to SMOTE interpolation

#### 4. **Feature Engineering Differences**
Your current feature extraction may differ from the manuscript's approach:
- The manuscript mentions "baseline-normalized AU intensities"
- Check if your normalization is consistent
- Asymmetry metric calculations may differ

#### 5. **Hyperparameter Optimization Converging to Poor Solution**
Your Optuna results show:
- **Lower face best score: 0.3380** (terrible for 120 trials!)
- **Mid face best score: 0.3550** (still poor for 100 trials)
- **Upper face best score: 0.4238** (mediocre for 100 trials)

This suggests the optimization itself is struggling, not just overfitting.

---

## Key Differences: Manuscript vs. Current

| Aspect | Manuscript (Benchmark) | Current | Impact |
|--------|----------------------|---------|--------|
| OpenFace Version | 2.0 | 3.0 | ⚠️ **High** - Different AU extraction |
| Dataset | 112 videos (100 patients, 12 controls) | Unknown | ⚠️ **Critical** - May be entirely different |
| Class Labels | "None, Incomplete, Complete" | "None, Partial, Complete" | ✅ Same (just naming) |
| Feature Approach | Baseline-normalized + raw + asymmetry | Same approach | ✅ Should be same |
| Model | XGBoost + SMOTE + isotonic calibration | XGBoost + VotingClassifier + SMOTE + isotonic | ⚠️ **Medium** - More complex |
| Cross-validation | 5-fold for HPT | 5-fold for HPT | ✅ Same |

---

## Recommendations (Prioritized)

### IMMEDIATE ACTIONS (Critical - Do First):

#### 1. **Verify Training Data Quality** 🔴
```bash
# Check class distribution
python -c "
import pandas as pd
df = pd.read_csv('FPRS FP Key.csv')
print('Lower Face Distribution:')
print(df['Paralysis - Left Lower Face'].value_counts())
print(df['Paralysis - Right Lower Face'].value_counts())
print('\nMid Face Distribution:')
print(df['Paralysis - Left Mid Face'].value_counts())
print(df['Paralysis - Right Mid Face'].value_counts())
print('\nUpper Face Distribution:')
print(df['Paralysis - Left Upper Face'].value_counts())
print(df['Paralysis - Right Upper Face'].value_counts())
"
```

**Action**: If Partial class has < 30 samples per zone, you need more data or different class weighting.

#### 2. **Test with OpenFace 2.2 Data** 🔴
- Find the original OpenFace 2.2 outputs from the manuscript study
- Re-run training on that exact data
- If performance matches the paper → OpenFace 3.0 is the problem
- If performance still poor → problem is in your training code

#### 3. **Verify Feature Extraction** 🔴
Compare your features to what the manuscript describes:

```python
# Check feature statistics for known samples
import pandas as pd
df = pd.read_csv('combined_results.csv')
print("AU12_r statistics by action:")
for action in ['BS', 'SS']:
    action_data = df[df['action'] == action]
    print(f"\n{action}:")
    print(f"  Mean: {action_data['AU12_r'].mean():.3f}")
    print(f"  Std: {action_data['AU12_r'].std():.3f}")
    print(f"  NaN%: {action_data['AU12_r'].isna().sum() / len(action_data) * 100:.1f}%")
```

### SHORT-TERM ACTIONS (High Priority):

#### 4. **Increase Class Weights for Partial Class**
```python
# In paralysis_config.py, try more aggressive weights:
ZONE_CONFIG['lower']['training']['class_weights'] = {0: 1.0, 1: 10.0, 2: 3.0}
ZONE_CONFIG['mid']['training']['class_weights'] = {0: 1.0, 1: 12.0, 2: 4.0}
ZONE_CONFIG['upper']['training']['class_weights'] = {0: 1.0, 1: 10.0, 2: 3.0}
```

#### 5. **Try Regular SMOTE Instead of Borderline**
```python
# Change SMOTE variant
'smote': {
    'enabled': True,
    'variant': 'regular',  # Changed from 'borderline'
    'k_neighbors': 3,  # Reduced from 5 (since Partial class is small)
    'sampling_strategy': 'not majority',  # Simpler strategy
}
```

#### 6. **Simplify Model Architecture**
```python
# Disable ensemble temporarily to isolate issues
ZONE_CONFIG['lower']['training']['use_ensemble'] = False
ZONE_CONFIG['mid']['training']['use_ensemble'] = False
ZONE_CONFIG['upper']['training']['use_ensemble'] = False
```

#### 7. **Increase Optuna Trials with Better Scoring**
```python
# Focus optimization on Partial class
'optuna': {
    'n_trials': 200,  # More trials
    'scoring': 'f1_partial',  # Optimize for partial class directly!
    # Or use a custom scorer that heavily weights partial class
}
```

### MEDIUM-TERM ACTIONS:

#### 8. **Collect More Training Data**
- Minimum 30-50 examples of Partial paralysis per zone
- Ensure expert labels are consistent
- Consider data augmentation techniques specific to paralysis assessment

#### 9. **Feature Selection Review**
Your current selected features:
- Lower: 60 features (from 244) - 75.4% reduction
- Mid: 40 features (from 115) - 65.2% reduction
- Upper: 25 features (from 29) - 13.8% reduction

**Action**: Check if Partial-class-discriminative features are being kept:
```python
# Add this to your training pipeline
from sklearn.feature_selection import mutual_info_classif
# Calculate mutual information for Partial vs. others
partial_mask = (y_train == 1)
other_mask = ~partial_mask
# Select features with high MI for Partial class
```

#### 10. **Threshold Optimization Check**
Your thresholds are all at ~0.50, which suggests threshold optimization isn't helping:
```
Lower: 0: 0.50, 1: 0.51, 2: 0.50
```

This is essentially unchanged from default 0.5, meaning the model's probability outputs are already well-calibrated OR threshold optimization isn't working.

---

## Diagnostic Commands to Run

### 1. Check Training Data Stats
```python
# Run this in your S3 Data Analysis directory
python3 << 'EOF'
import pandas as pd
import numpy as np

# Load expert key
expert_df = pd.read_csv('FPRS FP Key.csv', dtype=str, keep_default_na=False)

zones = {
    'Lower': ['Paralysis - Left Lower Face', 'Paralysis - Right Lower Face'],
    'Mid': ['Paralysis - Left Mid Face', 'Paralysis - Right Mid Face'],
    'Upper': ['Paralysis - Left Upper Face', 'Paralysis - Right Upper Face']
}

for zone_name, cols in zones.items():
    print(f"\n{'='*60}")
    print(f"{zone_name} Face - Expert Label Distribution")
    print('='*60)
    all_labels = pd.concat([expert_df[col] for col in cols if col in expert_df.columns])
    print(all_labels.value_counts())
    print(f"\nTotal assessments: {len(all_labels)}")
    print(f"Not Assessed / NA: {all_labels.isin(['', 'NA', 'N/A', 'Not Assessed']).sum()}")
EOF
```

### 2. Compare OpenFace AU Distributions
```python
# Compare AU intensity distributions between OpenFace 2.2 (if available) and 3.0
python3 << 'EOF'
import pandas as pd
import numpy as np

# Load your current OpenFace 3.0 data
df = pd.read_csv('combined_results.csv')

au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

print("AU Intensity Summary Statistics (OpenFace 3.0):")
print("="*70)
for au in au_cols:
    data = df[au].dropna()
    if len(data) > 0:
        print(f"{au:15s}: mean={data.mean():6.3f}, std={data.std():6.3f}, "
              f"nan={df[au].isna().sum():4d} ({df[au].isna().sum()/len(df)*100:5.1f}%)")
EOF
```

### 3. Analyze Current Model Predictions
```python
# Check if model is biased toward certain classes
python3 << 'EOF'
import pandas as pd

# These values from your performance summary
zones = {
    'Lower': {
        'confusion': [[14, 11, 9], [2, 0, 7], [5, 2, 5]],
        'classes': ['None', 'Partial', 'Complete']
    },
    'Mid': {
        'confusion': [[29, 8, 4], [1, 2, 5], [0, 0, 5]],
        'classes': ['None', 'Partial', 'Complete']
    },
    'Upper': {
        'confusion': [[27, 3, 5], [3, 0, 2], [6, 3, 4]],
        'classes': ['None', 'Partial', 'Complete']
    }
}

for zone, data in zones.items():
    cm = data['confusion']
    print(f"\n{zone} Face Prediction Bias:")
    print("="*50)

    # Total predictions per class
    total_pred_per_class = [sum(row[i] for row in cm) for i in range(3)]
    total = sum(total_pred_per_class)

    for i, cls in enumerate(data['classes']):
        pct = total_pred_per_class[i] / total * 100
        print(f"  Predicted as {cls:10s}: {total_pred_per_class[i]:3d} ({pct:5.1f}%)")
EOF
```

---

## Success Criteria for Next Training Run

To consider the model "fixed", you should achieve:

| Zone | Minimum Accuracy | Minimum Weighted F1 | Minimum Partial F1 |
|------|-----------------|--------------------|--------------------|
| Lower | 0.75 | 0.75 | 0.35 |
| Mid | 0.85 | 0.85 | 0.50 |
| Upper | 0.75 | 0.75 | 0.30 |

**Target** (to match paper):
- Lower: Acc 0.84, F1 0.82, Partial F1 0.46
- Mid: Acc 0.93, F1 0.92, Partial F1 0.67
- Upper: Acc 0.83, F1 0.83, Partial F1 0.40

---

## Summary

Your current performance is **critically below benchmark** with the most severe issue being **complete failure on the Partial paralysis class**. This is likely due to:

1. **Different training data** (most likely root cause)
2. **OpenFace 3.0 AU extraction differences**
3. **Class imbalance handling not working**
4. **Possible feature engineering differences**

**Next Steps**:
1. ✅ Verify training data distribution and labels
2. ✅ Test with OpenFace 2.2 data if available
3. ✅ Increase Partial class weights significantly
4. ✅ Simplify SMOTE to regular variant
5. ✅ Optimize hyperparameters specifically for Partial class F1

**Most Critical**: The Partial class is the most clinically important (represents incomplete paralysis that needs monitoring/treatment), and it's completely failing. This must be fixed before the model can be considered usable.
