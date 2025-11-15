# OpenFace 3.0 AU Output Mapping - COMPLETE EXPLANATION

## Your Original Question

**"How did the code map OpenFace 3.0 AU outputs to specific AUs to know which ones were which?"**

## The Answer

OpenFace 3.0 CSV files have **misleading column names**. The columns are named `AU01_r`, `AU02_r`, `AU03_r`, etc., but these **DO NOT** directly correspond to FACS AU numbers!

### Actual Mapping

| CSV Column | FACS AU Number | FACS AU Name | Status in Test Data |
|------------|----------------|--------------|---------------------|
| `AU01_r` | **AU01** | Inner Brow Raiser | ✓ ACTIVE (Mean: 0.0058) |
| `AU02_r` | **AU06** | Cheek Raiser | ✗ INACTIVE (Mean: 0.0000) |
| `AU03_r` | **AU12** | Lip Corner Puller (Smile) | ✓ ACTIVE (Mean: 0.0008) |
| `AU04_r` | **AU15** | Lip Corner Depressor | ✓ ACTIVE (Mean: 0.0094) |
| `AU05_r` | **AU17** | Chin Raiser | ✗ INACTIVE (Mean: 0.0000) |
| `AU06_r` | **AU02** | Outer Brow Raiser | ✗ INACTIVE (Mean: 0.0001) |
| `AU07_r` | **AU09** | Nose Wrinkler | ✓ ACTIVE (Mean: 0.0682) |
| `AU08_r` | **AU10** | Upper Lip Raiser | ✗ INACTIVE (Mean: 0.0000) |

### Where This Mapping Comes From

In the OpenFace 3.0 training code (`train_mix_au_dev.py`), line 42:

```python
eval_au = [0, 4, 8, 10, 11, 1, 6, 7]
label = [label[i] for i in eval_au]
```

This selects indices from the DISFA dataset's AU ordering:
- Index 0 → AU01
- Index 1 → AU02
- Index 4 → AU06
- Index 6 → AU09
- Index 7 → AU10
- Index 8 → AU12
- Index 10 → AU15
- Index 11 → AU17

The model outputs these 8 AUs in the order: `[0, 4, 8, 10, 11, 1, 6, 7]`

Which becomes the CSV columns: `AU01_r, AU02_r, AU03_r, AU04_r, AU05_r, AU06_r, AU07_r, AU08_r`

## Why Your S3 Code Had Problems

### What Documentation Said (WRONG)

Your S3 Data Analysis README (commit c6ca4701) claimed OpenFace 3.0 had **9 available AUs**:
- AU01, AU02, AU04, **AU06**, AU12, AU15, AU20, AU25, AU45

### What Actually Exists

OpenFace 3.0 model only outputs **8 AUs**:
- AU01, AU02, **AU06**, **AU09**, **AU10**, AU12, AU15, **AU17**

### The Mismatch

| What Docs Said | What Model Has | Impact |
|----------------|----------------|--------|
| AU04 (Brow Lowerer) | Missing | Lost eye-related detection |
| AU20 (Lip Stretcher) | Missing | Lost mouth movement |
| AU25 (Lips Part) | Missing | Lost mouth opening |
| AU45 (Blink) | Missing | Lost blink detection |
| - | AU09 (Nose Wrinkler) | Extra (not documented) |
| - | AU10 (Upper Lip Raiser) | Extra (not documented) |
| - | AU17 (Chin Raiser) | Extra (not documented) |

## Even Worse: Most AUs Don't Work

Out of the 8 AUs the model outputs, only **4 actually produce meaningful values**:

### Working AUs (from our test)
1. **AU01** (Inner Brow Raiser) - Mean: 0.0058, Max: 0.0266
2. **AU09** (Nose Wrinkler) - Mean: 0.0682, Max: 0.1551 ← Most active
3. **AU12** (Lip Corner Puller/Smile) - Mean: 0.0008, Max: 0.0074
4. **AU15** (Lip Corner Depressor) - Mean: 0.0094, Max: 0.0201

### Non-Working AUs
- **AU02** (via AU06_r) - Completely zero
- **AU06** (via AU02_r) - Nearly zero (max 0.0003)
- **AU10** (via AU08_r) - Completely zero
- **AU17** (via AU05_r) - Completely zero

## How Your Code Adapted

Your `facial_au_constants.py` (commit 6bd590ab) changed:

```python
# Before (OpenFace 2.2 - Correct):
ACTION_TO_AUS = {
    'ET': ['AU07_r', 'AU45_r'],  # Close Eyes Tightly
    'BS': ['AU12_r', 'AU25_r', 'AU07_r'],  # Big Smile
}

# After (OpenFace 3.0 - Workaround):
ACTION_TO_AUS = {
    'ET': ['AU06_r', 'AU45_r'],  # AU07→AU06 because AU07 not available
    'BS': ['AU12_r', 'AU25_r', 'AU06_r'],  # Same substitution
}
```

**But this was wrong!** Because:
1. CSV column `AU06_r` actually maps to FACS **AU02** (Outer Brow Raiser), not AU06
2. The actual AU06 (Cheek Raiser) is in column `AU02_r`
3. AU07 (Lid Tightener) doesn't exist in OpenFace 3.0 at all
4. AU45 (Blink) doesn't exist either

## Why PyFaceAU Was Created

PyFaceAU (pure Python OpenFace 2.2 implementation) outputs **17 functional AUs**:
- AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45

With proper column naming that **actually matches FACS AU numbers**.

## Summary

**OpenFace 3.0 AU mapping is a mess:**

1. **Misleading column names**: `AU01_r` doesn't mean FACS AU01 for all columns
2. **Only 8 AUs**: Not 9 as documented, not 17 like OpenFace 2.2
3. **Only 4 work**: Half the AUs output zeros
4. **Wrong documentation**: The README didn't match the actual model
5. **No AU45**: Critical blink detection missing
6. **No AU07**: Lid tightener missing (needed for eye closure)

This is why your S1 code eventually migrated to PyFaceAU, which properly implements OpenFace 2.2 with accurate AU column naming and all 17 functional AUs.
