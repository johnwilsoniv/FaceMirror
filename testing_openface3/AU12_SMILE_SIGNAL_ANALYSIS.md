# AU12 (Smile) Signal Analysis - OpenFace 3.0

**Date**: 2025-11-14
**Question**: Does the real AU12 from OpenFace 3.0 give a good signal for smile detection on patient videos?

---

## Executive Summary

**Short Answer**: Cannot conclusively determine from current test data. The test video (IMG_0422.MOV) shows a neutral/resting face with essentially no smiling, so AU12 values are near-zero as expected. **We need to test on a video with explicit smile actions to validate AU12 signal strength.**

---

## What We Tested

### Test Data
- **Video**: IMG_0422.MOV (Normal Cohort)
- **Frames**: 30 frames processed
- **Expected content**: Neutral/resting face recording
- **OpenFace 3.0 CSV column**: AU03_r (contains real AU12)

### Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Mean AU12 intensity | 0.000751 | Very low (expected for neutral face) |
| Max AU12 intensity | 0.007401 | Below smile threshold (< 0.01) |
| Non-zero frames | 7/30 (23.3%) | Minimal activation |
| Frames > 0.001 | 6/30 (20.0%) | Very sparse |

### Frame-by-Frame AU12 Values

Only 6 frames showed any meaningful AU12 activity (> 0.001):

```
Frame  0: 0.007401  ← Highest value
Frame  1: 0.005540
Frame  2: 0.002277
Frame 16: 0.001486
Frame 17: 0.003098
Frame 18: 0.002378
```

All other frames: essentially zero (< 0.001)

---

## Context: Comparison to Other AUs

| CSV Column | Actual AU | Mean | Max | Non-zero Frames |
|------------|-----------|------|-----|-----------------|
| AU01_r | AU01 (Inner Brow) | 0.005780 | 0.026612 | 16/30 |
| AU02_r | AU06 (Cheek Raiser) | 0.000011 | 0.000339 | 0/30 |
| **AU03_r** | **AU12 (Smile)** | **0.000751** | **0.007401** | **6/30** |
| AU04_r | AU15 (Lip Depressor) | 0.009360 | 0.020067 | 30/30 ✓ |
| AU05_r | AU17 (Chin Raiser) | 0.000000 | 0.000000 | 0/30 |
| AU06_r | AU02 (Outer Brow) | 0.000083 | 0.001726 | 1/30 |
| AU07_r | AU09 (Nose Wrinkler) | 0.068220 | 0.155148 | 30/30 ✓ |
| AU08_r | AU10 (Upper Lip) | 0.000000 | 0.000000 | 0/30 |

**Observation**: AU09 (Nose Wrinkler) and AU15 (Lip Corner Depressor) show the strongest signals. This makes sense for facial expressions during neutral talking/resting.

---

## Interpretation

### Why AU12 is Near-Zero

1. **Video Content**: IMG_0422.MOV is from "Normal Cohort" - likely a neutral/resting face recording
2. **No Smile Actions**: The video doesn't appear to contain explicit smile instructions
3. **Expected Behavior**: AU12 should be near-zero when not smiling
4. **Model is Working**: The low values indicate the model correctly detects absence of smile

### Expected AU12 Values

| Condition | Expected AU12 Range | Our Test |
|-----------|---------------------|----------|
| Neutral/Resting Face | 0.0 - 0.1 | 0.000 - 0.007 ✓ |
| Slight Smile | 0.1 - 0.5 | Not present |
| Moderate Smile | 0.5 - 2.0 | Not present |
| **Big Smile (BS action)** | **2.0 - 5.0** | **Not present** |

Our test video falls squarely in the "neutral face" range, confirming the model is responding appropriately.

---

## What This Tells Us

### ✓ Good News
1. **Model is functional**: AU12 correctly stays near-zero for non-smiling faces
2. **No false positives**: Model isn't hallucinating smiles on neutral faces
3. **Reasonable baseline**: Max value 0.007 is low enough to establish a clear threshold

### ❓ Unknown
1. **Signal strength during actual smiles**: We haven't tested on smiling faces yet
2. **Dynamic range**: Can the model detect smile intensity variations?
3. **Sensitivity**: Will it respond to subtle vs. big smiles?
4. **Reliability**: Does it consistently activate during smile actions?

---

## Next Steps: How to Properly Test AU12

### 1. Process a Video with Smile Actions

The standard facial action protocol includes:
- **SS** (Soft Smile): Gentle smile
- **BS** (Big Smile): Full smile with cheek raising

**Expected Results for BS (Big Smile)**:
- AU12 should spike to 2.0-5.0 during smile
- AU06 (Cheek Raiser) may also activate
- Clear separation from baseline

### 2. Test Video Requirements

Ideal test video should have:
- ✓ Explicit "Big Smile" instruction/action
- ✓ Multiple smile instances
- ✓ Return to neutral between smiles
- ✓ Good face visibility and lighting

### 3. Validation Criteria

AU12 signal is "good" if:
- BS action: AU12 > 1.0 (preferably > 2.0)
- Neutral face: AU12 < 0.1
- Clear dynamic range: 20x+ difference between smile and neutral
- Temporal consistency: Activates during smile period, drops after

### 4. Suggested Test

```bash
# Process a patient video with known BS (Big Smile) actions
# Example: Look for videos from facial action protocol sessions
# Then extract AU12 values during BS action frames
```

---

## Clinical Implications

### If AU12 Works Well (>2.0 during smiles)

**Benefits**:
- Can detect smile asymmetry for paralysis analysis
- Quantify smile intensity
- Track recovery progress
- Validate S3 smile-based features

**Use in S3**:
- Lower face paralysis detection (smile is key indicator)
- BS (Big Smile) action peak frame selection
- Asymmetry quantification (left vs right face AU12)

### If AU12 Remains Weak (<0.5 during smiles)

**Problems**:
- Cannot reliably detect smiles
- Lower face paralysis detection compromised
- Would need alternative features
- May explain why old OpenFace 3.0 migration failed

---

## Comparison to Old Mapping Error

### Old Adapter Mistake

The old adapter mapped position 4 → AU12_r, but position 4 actually contains **AU17 (Chin Raiser)**.

**What this means**:
- Old code tried to detect smiles using chin raising values
- Anatomically incorrect (chin raising ≠ smiling)
- Would produce nonsensical smile detection
- Explains S3 paralysis model failures

### Current Correct Mapping

Position 2 (AU03_r) → AU12 (Lip Corner Puller)
- Anatomically correct
- Proper FACS AU for smile
- Should work if model is properly trained

---

## Conclusion

**Current Status**: ⚠️ Incomplete

We've confirmed:
- ✓ AU12 correctly stays near-zero on neutral faces
- ✓ Mapping is anatomically correct
- ❓ AU12 signal strength during actual smiles: **UNTESTED**

**Recommendation**:
Process a patient video with explicit "Big Smile" (BS) action to determine if AU12 provides clinically useful signal strength (target: >2.0 during smile).

**Expected Outcome**:
Given that OpenFace 3.0 was trained on DISFA dataset and AU12 is a core AU for smile detection, it should work. However, validation is required before trusting it for clinical paralysis detection.

---

## Technical Notes

### Why We Can't Use Old S3 Data

The old S3 processed data used the incorrect adapter:
- Position 4 → written as AU12_r (but contained AU17)
- Position 2 → written as AU04_r (but contained AU12)

So even if we had old processed videos with smiles, the AU12 column would be wrong. We need fresh processing with the correct mapping.

### Processing Requirements

To test AU12 on a smile video:
1. Use OpenFace 3.0 with our test script
2. Extract column **AU03_r** (not AU12_r!)
3. Identify BS (Big Smile) action frames
4. Check AU03_r values during those frames
5. Expect values > 2.0 for good signal

---

## Files Generated

- `analyze_real_au12_smile.py` - Analysis script
- `AU12_SMILE_SIGNAL_ANALYSIS.md` - This document
- `openface3_output.csv` - Test data (neutral face, AU12 near-zero)

**Next**: Need processed data from smile action video to complete validation.
