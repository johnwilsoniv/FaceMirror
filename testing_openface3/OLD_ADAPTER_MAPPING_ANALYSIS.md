# Analysis: Old OpenFace 3.0 Adapter Mapping Errors

**Date**: 2025-11-14
**Finding**: The original OpenFace 3.0 adapter had incorrect AU mappings. Only AU01 was correct.

---

## What the Old Adapter THOUGHT It Was Doing

From commit `6bd590ab`, file `S1 Face Mirror/openface3_to_18au_adapter.py`:

```python
self.of3_au_mapping = {
    0: 'AU01_r',  # Inner Brow Raiser
    1: 'AU02_r',  # Outer Brow Raiser
    2: 'AU04_r',  # Brow Lowerer
    3: 'AU06_r',  # Cheek Raiser
    4: 'AU12_r',  # Lip Corner Puller (Smile) ← WRONG!
    5: 'AU15_r',  # Lip Corner Depressor
    6: 'AU20_r',  # Lip Stretcher
    7: 'AU25_r',  # Lips Part
}
```

**Assumption**: The adapter assumed OpenFace 3.0 output positions corresponded to sequential FACS AU numbers.

---

## What OpenFace 3.0 ACTUALLY Does

From `OpenFace-3.0/train_mix_au_dev.py` line 121:
```python
eval_au = [0, 4, 8, 10, 11, 1, 6, 7]
label = [label[i] for i in eval_au]
```

This selects 8 AUs from DISFA dataset (which uses standard FACS ordering) in this scrambled order:

**ACTUAL OpenFace 3.0 Mapping**:
```python
{
    0: 'AU01_r',  # Position 0 → DISFA index 0  → AU01 (Inner Brow Raiser)
    1: 'AU06_r',  # Position 1 → DISFA index 4  → AU06 (Cheek Raiser)
    2: 'AU12_r',  # Position 2 → DISFA index 8  → AU12 (Lip Corner Puller)
    3: 'AU15_r',  # Position 3 → DISFA index 10 → AU15 (Lip Corner Depressor)
    4: 'AU17_r',  # Position 4 → DISFA index 11 → AU17 (Chin Raiser)
    5: 'AU02_r',  # Position 5 → DISFA index 1  → AU02 (Outer Brow Raiser)
    6: 'AU09_r',  # Position 6 → DISFA index 6  → AU09 (Nose Wrinkler)
    7: 'AU10_r',  # Position 7 → DISFA index 7  → AU10 (Upper Lip Raiser)
}
```

---

## The Mapping Errors

| Position | Old Adapter Wrote | Actual FACS AU | Error Description |
|----------|-------------------|----------------|-------------------|
| 0 | AU01_r | AU01 | ✓ **CORRECT** (only one!) |
| 1 | AU02_r | **AU06** | ✗ Wrote Cheek Raiser as Outer Brow Raiser |
| 2 | AU04_r | **AU12** | ✗ Wrote Smile as Brow Lowerer |
| 3 | AU06_r | **AU15** | ✗ Wrote Lip Depressor as Cheek Raiser |
| 4 | **AU12_r** | **AU17** | ✗ **Wrote Chin Raiser as Smile!** |
| 5 | AU15_r | **AU02** | ✗ Wrote Outer Brow Raiser as Lip Depressor |
| 6 | AU20_r | **AU09** | ✗ Wrote Nose Wrinkler as Lip Stretcher |
| 7 | AU25_r | **AU10** | ✗ Wrote Upper Lip Raiser as Lips Part |

---

## Critical Problems

### 1. AU12 (Smile) Was Completely Wrong

**What users expected**: Column `AU12_r` contains smile intensity
**What it actually contained**: AU17 (Chin Raiser) values
**Where the real smile was**: In column `AU04_r` (labeled as "Brow Lowerer")

This means:
- Smile detection in S3 Data Analysis was reading chin raising values
- The actual smile data was ignored (labeled as AU04)
- Paralysis models trained on this data learned incorrect patterns

### 2. AU06 (Cheek Raiser) Was Swapped with AU02 (Outer Brow)

**What the "AU06 workaround" actually did**:
- Old code used `AU06_r` for eye closure and nose wrinkle
- But `AU06_r` (position 3) actually contained **AU15 (Lip Corner Depressor)**
- The real AU06 was at position 1, labeled as `AU02_r`

### 3. AU09 (Nose Wrinkler) Was Labeled as AU20

- Real AU09 was at position 6, written as `AU20_r`
- When code looked for nose wrinkle, it read the wrong column entirely

---

## Impact on S3 Data Analysis

From the commit message for `6bd590ab`:
> - ET (Close Eyes Tightly): AU07→AU06
> - BS (Big Smile): AU07→AU06
> - WN (Wrinkle Nose): AU09→AU06

**What actually happened**:
- ET used column `AU06_r` which contained **AU15 (Lip Corner Depressor)** - completely wrong for eye closure
- BS used column `AU06_r` for smile enhancement, but got **AU15** instead of AU06
- WN used column `AU06_r` for nose wrinkle, but got **AU15** instead of AU09

**The system was reading lip depression values for eye closure and nose wrinkling.**

---

## Why This Explains the OpenFace 3.0 Failure

From the REFACTOR.md at commit `8edc9242`:
> AU05, AU07, AU09, AU10, AU14, AU17, AU23, AU26: completely disappeared

**They didn't disappear - they were mislabeled!**

- AU07 doesn't exist in OpenFace 3.0 (correct assessment)
- AU09 was in the output, but labeled as `AU20_r`
- AU10 was in the output, but labeled as `AU25_r`
- AU17 was in the output, but labeled as `AU12_r`

The "disappearance" was a mapping error, not a model failure.

---

## Conclusion

**Only 1 out of 8 AUs was mapped correctly** (AU01).

The old adapter assumed OpenFace 3.0 used sequential AU numbering, but it actually used a scrambled selection order from the DISFA dataset. This caused:

1. Smile detection to read chin raising values
2. Eye closure detection to read lip depression values
3. Nose wrinkle detection to read lip depression values
4. Complete corruption of the paralysis detection models

The system didn't fail because OpenFace 3.0 was broken - it failed because the AU mapping adapter was based on incorrect assumptions about how OpenFace 3.0 ordered its outputs.

---

## Correct Mapping Reference

For future reference, the correct OpenFace 3.0 CSV mapping is:

```python
OPENFACE3_CORRECT_MAPPING = {
    'AU01_r': 'AU01',  # Inner Brow Raiser
    'AU02_r': 'AU06',  # Cheek Raiser (NOT AU02!)
    'AU03_r': 'AU12',  # Lip Corner Puller (NOT AU04!)
    'AU04_r': 'AU15',  # Lip Corner Depressor (NOT AU06!)
    'AU05_r': 'AU17',  # Chin Raiser (NOT AU12!)
    'AU06_r': 'AU02',  # Outer Brow Raiser (NOT AU15!)
    'AU07_r': 'AU09',  # Nose Wrinkler (NOT AU20!)
    'AU08_r': 'AU10',  # Upper Lip Raiser (NOT AU25!)
}
```

**Source**: `OpenFace-3.0/train_mix_au_dev.py` line 121, traced through DISFA dataset standard FACS ordering.
