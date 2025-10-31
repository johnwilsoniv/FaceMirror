# Three-Way OpenFace Comparison Guide

## Purpose

This script answers two critical questions:

1. **Is ONNX conversion broken?**
   - Compares ONNX vs PyTorch outputs
   - If they differ → ONNX has bugs

2. **Is MTL model incompatible with OpenFace 2.2?**
   - Compares PyTorch vs OpenFace 2.2 outputs
   - If they differ → MTL fundamentally incompatible

## Quick Start

### Step 1: Pick a short test video (30-60 seconds)

Use an already-mirrored video from your Face Mirror output:

```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"

# Example video (pick a short one!)
python3 three_way_comparison.py "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/20240723_175947000_iOS_left_mirrored.mp4"
```

### Step 2: Wait 5-15 minutes

The script will:
1. Process with OpenFace 3.0 ONNX (fast - ~2 min)
2. Process with OpenFace 3.0 PyTorch (slower - ~5 min)
3. Process with OpenFace 2.2 binary (slowest - ~5 min)

### Step 3: Read the report

Open the generated report:
```
~/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/<video_name>_COMPARISON_REPORT.txt
```

## Understanding The Results

### Scenario 1: ONNX is Broken ✅ FIXABLE

```
ONNX vs PyTorch Correlation: 0.45
PyTorch vs OF2.2 Correlation: 0.85

VERDICT: ONNX conversion is BROKEN
         MTL model is COMPATIBLE
```

**What this means:**
- PyTorch works well (matches OF2.2)
- ONNX conversion has bugs
- **FIX**: Debug your ONNX conversion scripts
- After fixing ONNX, you can use OpenFace 3.0 successfully

**Action Plan:**
1. Review ONNX conversion scripts for bugs
2. Check input normalization
3. Check output scaling
4. Retest after fixes

---

### Scenario 2: MTL Model is Incompatible ❌ FUNDAMENTAL

```
ONNX vs PyTorch Correlation: 0.98
PyTorch vs OF2.2 Correlation: 0.25

VERDICT: ONNX conversion is CORRECT
         MTL model is INCOMPATIBLE
```

**What this means:**
- ONNX is fine (matches PyTorch)
- MTL model produces fundamentally different AUs than OF2.2
- **NO QUICK FIX**: Must retrain models OR use OF2.2

**Action Plan:**
1. Switch to OpenFace 2.2 for production (temporary)
2. Re-process training data with OpenFace 3.0
3. Retrain all models on OpenFace 3.0 data

---

### Scenario 3: Both Are Broken ❌❌ DOUBLE PROBLEM

```
ONNX vs PyTorch Correlation: 0.50
PyTorch vs OF2.2 Correlation: 0.30

VERDICT: ONNX is BROKEN
         MTL is INCOMPATIBLE
```

**What this means:**
- Two separate problems
- ONNX has bugs AND MTL is incompatible
- **RECOMMEND**: Switch to OF2.2 while you fix both

**Action Plan:**
1. Switch to OpenFace 2.2 immediately
2. Fix ONNX conversion issues
3. Still need to retrain models on OF3.0

---

### Scenario 4: Everything Works ✓✓ MYSTERY

```
ONNX vs PyTorch Correlation: 0.97
PyTorch vs OF2.2 Correlation: 0.89

VERDICT: ONNX is CORRECT
         MTL is COMPATIBLE
```

**What this means:**
- ONNX works
- MTL works
- Problem is NOT in AU extraction!
- Issue must be in data analysis pipeline

**Action Plan:**
1. Check feature engineering code
2. Check model loading
3. Check data preprocessing
4. Verify no data corruption

---

## What The Script Does

### Pipeline 1: OpenFace 3.0 with ONNX (Current Production)
- Uses your current optimized ONNX models
- Uses CoreML Neural Engine on M-series chips
- **Fast** (~2 minutes for 1000 frames)
- Outputs: `<video>_OF30_ONNX.csv`

### Pipeline 2: OpenFace 3.0 with Pure PyTorch
- Bypasses ONNX completely
- Uses original PyTorch models
- **Slower** (~5 minutes for 1000 frames)
- More accurate reference for testing
- Outputs: `<video>_OF30_PyTorch.csv`

### Pipeline 3: OpenFace 2.2 Binary (Baseline)
- Your proven working baseline
- What your models were trained on
- **Slow** (~5 minutes for 1000 frames)
- Outputs: `<video>_OF22_Baseline.csv`

### Comparison Report
- Calculates correlations between all three
- Diagnoses root cause
- Provides action plan
- Outputs: `<video>_COMPARISON_REPORT.txt`

---

## Output Location

All files go to:
```
~/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/
```

Contents:
- `<video>_OF30_ONNX.csv` - ONNX results
- `<video>_OF30_PyTorch.csv` - PyTorch results
- `<video>_OF22_Baseline.csv` - OpenFace 2.2 results
- `<video>_COMPARISON_REPORT.txt` - Diagnostic report

---

## Interpreting Correlations

| Correlation | Meaning |
|-------------|---------|
| > 0.95 | Excellent - nearly identical |
| 0.80 - 0.95 | Good - minor differences |
| 0.50 - 0.80 | Moderate - significant differences |
| 0.30 - 0.50 | Poor - very different |
| < 0.30 | Very poor - fundamentally different |
| < 0.0 | Negative - opposite patterns |

### Key Thresholds:

**ONNX vs PyTorch:**
- Should be > 0.95 (nearly identical if ONNX correct)
- If < 0.80, ONNX has bugs

**PyTorch vs OF2.2:**
- Should be > 0.80 for models to work
- If < 0.50, must retrain models

---

## Tips

### Use a Short Video
- Pick 30-60 seconds (500-1000 frames)
- Shorter = faster testing
- Still representative of the problem

### Pick a Mirrored Video
- Use output from Face Mirror (pre-aligned)
- Skips face detection step
- More consistent comparison

### Check Progress
- Script prints progress every 100 frames
- Total time: 5-15 minutes depending on video length
- PyTorch and OF2.2 are slow but necessary

---

## Troubleshooting

**Error: "OpenFace 2.2 binary not found"**
- Update `OPENFACE2_BINARY` path in script
- Or install OpenFace 2.2 at expected location

**Error: "Failed to import openface_integration"**
- Run from S1 Face Mirror directory
- Make sure all dependencies installed

**PyTorch processing is very slow**
- Normal - PyTorch CPU is 5-10x slower than ONNX
- Consider using shorter test video
- OR enable CUDA if you have NVIDIA GPU

**Script hangs during processing**
- Check terminal for error messages
- May need to Ctrl+C and retry
- Check if enough disk space

---

## What To Do After Testing

Based on the report verdict:

### If ONNX is broken but MTL works:
1. Debug ONNX conversion scripts
2. Check normalization, scaling, operators
3. Retest with this script
4. Once fixed, use OpenFace 3.0 in production

### If MTL is incompatible:
1. Switch to OpenFace 2.2 for production NOW
2. Start retraining models on OF3.0 data
3. This is a weeks-long effort
4. Or stay on OF2.2 permanently (slower but works)

### If both have issues:
1. Fix ONNX first (easier)
2. Then retrain models (harder)
3. Use OF2.2 in meantime

### If everything works:
1. Problem is NOT in AU extraction
2. Check S3 Data Analysis code
3. Check feature engineering
4. Check model loading/preprocessing
