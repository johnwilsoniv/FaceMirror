# OpenFace 3.0 AU Mapping - Quick Reference

## CSV Column → FACS AU Mapping Table

| CSV Column | FACS AU | AU Name | Status |
|------------|---------|---------|--------|
| `AU01_r` | **AU01** | Inner Brow Raiser | ✓ Works |
| `AU02_r` | **AU06** | Cheek Raiser | ✗ Zeros |
| `AU03_r` | **AU12** | Lip Corner Puller (Smile) | ✓ Works |
| `AU04_r` | **AU15** | Lip Corner Depressor | ✓ Works |
| `AU05_r` | **AU17** | Chin Raiser | ✗ Zeros |
| `AU06_r` | **AU02** | Outer Brow Raiser | ✗ Zeros |
| `AU07_r` | **AU09** | Nose Wrinkler | ✓ Works |
| `AU08_r` | **AU10** | Upper Lip Raiser | ✗ Zeros |

## Reverse Lookup (FACS AU → CSV Column)

| FACS AU | CSV Column | AU Name |
|---------|------------|---------|
| AU01 | `AU01_r` | Inner Brow Raiser |
| AU02 | `AU06_r` | Outer Brow Raiser ⚠️ |
| AU06 | `AU02_r` | Cheek Raiser ⚠️ |
| AU09 | `AU07_r` | Nose Wrinkler |
| AU10 | `AU08_r` | Upper Lip Raiser |
| AU12 | `AU03_r` | Lip Corner Puller (Smile) |
| AU15 | `AU04_r` | Lip Corner Depressor |
| AU17 | `AU05_r` | Chin Raiser |

⚠️ = **Warning**: Counter-intuitive mapping!

## Python Dictionary

```python
OPENFACE3_CSV_TO_FACS_AU = {
    'AU01_r': 'AU01',
    'AU02_r': 'AU06',  # ← Misleading!
    'AU03_r': 'AU12',
    'AU04_r': 'AU15',
    'AU05_r': 'AU17',
    'AU06_r': 'AU02',  # ← Misleading!
    'AU07_r': 'AU09',
    'AU08_r': 'AU10',
}
```

## Available AUs

OpenFace 3.0 detects **8 AUs total**:
- AU01, AU02, AU06, AU09, AU10, AU12, AU15, AU17

## Working AUs (Non-Zero Values in Testing)

Only **4 AUs** produced meaningful values:
- **AU01** (Inner Brow Raiser)
- **AU09** (Nose Wrinkler) ← Most active
- **AU12** (Lip Corner Puller/Smile)
- **AU15** (Lip Corner Depressor)

## Missing AUs (Not in OpenFace 3.0)

These AUs are **NOT available** in OpenFace 3.0:
- AU04 (Brow Lowerer)
- AU05 (Upper Lid Raiser)
- AU07 (Lid Tightener) ← Critical for eye closure!
- AU14 (Dimpler)
- AU20 (Lip Stretcher)
- AU23 (Lip Tightener)
- AU25 (Lips Part)
- AU26 (Jaw Drop)
- AU45 (Blink) ← Critical for blink detection!

## Usage Example

### Reading OpenFace 3.0 CSV with Correct Labels

```python
import pandas as pd

# Read CSV
df = pd.read_csv('openface3_output.csv')

# Rename columns to actual FACS AU numbers
mapping = {
    'AU01_r': 'AU01_r',
    'AU02_r': 'AU06_r',  # Rename AU02_r to AU06_r
    'AU03_r': 'AU12_r',
    'AU04_r': 'AU15_r',
    'AU05_r': 'AU17_r',
    'AU06_r': 'AU02_r',  # Rename AU06_r to AU02_r
    'AU07_r': 'AU09_r',
    'AU08_r': 'AU10_r',
}

df.rename(columns=mapping, inplace=True)

# Now AU columns are correctly named!
# df['AU06_r'] contains Cheek Raiser intensity
# df['AU02_r'] contains Outer Brow Raiser intensity
```

## Files

- **Full Python module**: `openface3_au_mapping.py`
- **Complete explanation**: `OPENFACE_3_AU_MAPPING_EXPLAINED.md`
- **Test output CSV**: `openface3_output.csv`
- **Test video**: Sample 30 frames from IMG_0422.MOV
