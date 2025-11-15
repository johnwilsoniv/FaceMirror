# MTCNN Stage Box Count Regression - Investigation Plan

**Date**: November 14, 2025
**Issue**: CoreML and ONNX backends showing different box counts vs C++ baseline
**Previous State**: IMG_0434 showed identical box counts across all stages
**Current State**: Divergence in RNet/ONet stages

---

## Executive Summary

PyMTCNN previously achieved **bit-for-bit accuracy** with C++ OpenFace (documented in `MTCNN_COMPLETE_VERIFICATION.md`). The image IMG_0434 shows a working state where all stage box counts matched perfectly. Current testing reveals:

**Current Box Counts (patient1_frame1.jpg)**:
| Stage | C++ | CoreML | ONNX |
|---|---|---|---|
| PNet | 95 | 99 (✗ +4) | 71 (✗ -24) |
| RNet | 95 | 9 (✗ -86) | 3 (✗ -92) |
| ONet | 1 | 1 (✓) | 3 (✗ +2) |
| Final | 1 | 1 (✓) | 3 (✗ +2) |

**Bounding Box IoU**:
- CoreML: 96.83% (✓ **PASS**)
- ONNX: 42.86% (✗ **FAIL**)

---

## Historical Context

### Working State (from documentation)

**Pure Python MTCNN** (`MTCNN_COMPLETE_VERIFICATION.md`):
- Bit-for-bit accuracy with C++ (0.0 max diff on all layers)
- All 120,872 weight values matched perfectly
- All layer outputs identical

**ONNX Conversion** (`PHASE2_ONNX_SUCCESS_SUMMARY.md`):
- PNet: 0.0000006 max diff (0.6 ppm)
- RNet: 0.0000014 max diff
- ONet: 0.0000019 max diff
- Status: "bit-for-bit accuracy"

**CoreML Conversion** (`PHASE3_COREML_CONVERSION_RESULTS.md`):
- PNet: 0.0181 max diff (1.81%)
- RNet: 0.0215 max diff (2.15%)
- ONet: 0.0177 max diff (1.77%)
- Status: "acceptable for ML" (<2.5% tolerance)

### Key Differences Between Backends

**CoreML FP32**:
- Uses Apple Accelerate framework
- May use mixed FP32/FP16 internally for ANE
- ~2% numerical differences accepted

**ONNX Runtime**:
- CPU backend using standard BLAS
- Should have <2 ppm differences (nearly bit-for-bit)
- Currently showing catastrophic failure (42.86% IoU)

---

## Root Cause Hypotheses

### Hypothesis 1: ONNX Runtime Configuration Change ⭐⭐⭐⭐⭐

**Likelihood**: VERY HIGH

**Evidence**:
- ONNX previously had <2e-6 differences (bit-for-bit)
- Now shows 42.86% IoU (catastrophic failure)
- This suggests model inference itself is broken

**Possible Causes**:
1. **ONNX Runtime version change** - Different ONNX Runtime version with bugs
2. **Model file corruption** - ONNX files may be corrupted/outdated
3. **Input preprocessing mismatch** - BGR vs RGB, normalization, etc.
4. **Batch dimension handling** - ONNX may expect different input shape
5. **Output parsing error** - May be reading wrong outputs

**Investigation Steps**:
1. Check ONNX Runtime version: `python -c "import onnxruntime; print(onnxruntime.__version__)"`
2. Verify ONNX model files exist and have correct MD5 hashes
3. Compare input preprocessing between CoreML and ONNX backends
4. Add debug logging to ONNX inference to check input/output shapes
5. Test ONNX models with same input as CoreML (ensure identical preprocessing)

### Hypothesis 2: Threshold Differences Between Backends ⭐⭐⭐⭐

**Likelihood**: HIGH

**Evidence**:
- CoreML shows different box counts at PNet/RNet but same final result
- This suggests different NMS or confidence thresholds
- RNet: C++ 95 → CoreML 9 (aggressive filtering!)

**Possible Causes**:
1. **Hardcoded thresholds** - Different threshold values in CoreML vs ONNX backend code
2. **Floating point precision** - 2% numerical diff changes which boxes pass threshold
3. **NMS IoU threshold** - Different NMS settings between backends
4. **Score calibration** - CoreML/ONNX may apply different score transformations

**Investigation Steps**:
1. Check threshold values in `coreml_backend.py` vs `onnx_backend.py`
2. Log actual confidence scores at each stage for same input
3. Compare NMS IoU thresholds between backends
4. Check if CoreML backend has more aggressive box filtering

### Hypothesis 3: CoreML ~2% Precision Accumulates ⭐⭐⭐

**Likelihood**: MODERATE

**Evidence**:
- CoreML has ~2% numerical differences vs Pure Python
- These differences compound through pipeline
- Small logit difference → different confidence → box pass/fail

**Explanation**:
- PNet logit diff of 0.018 can flip box from 0.598 → 0.602 (threshold 0.6)
- RNet shows 95→9 boxes (90% filtered!) - extreme threshold sensitivity
- Final result still good (96.83% IoU) because dominant box survives

**Investigation Steps**:
1. Log PNet confidence scores near threshold (0.55-0.65 range)
2. Count how many boxes are "near-threshold" (within ±0.05 of cutoff)
3. Test with slightly relaxed thresholds to see if box counts match
4. Compare score distributions between CoreML and C++

### Hypothesis 4: Model Files Out of Sync ⭐⭐

**Likelihood**: LOW

**Evidence**:
- CoreML works well (96.83% IoU)
- Suggests CoreML models are correct
- But ONNX catastrophically fails

**Possible Causes**:
1. ONNX models rebuilt without critical fixes
2. CoreML updated but ONNX not regenerated
3. Different source .dat files used for conversions

**Investigation Steps**:
1. Check file timestamps on CoreML vs ONNX models
2. Verify both were generated from same C++ .dat files
3. Re-convert ONNX models from C++ .dat files with latest code
4. Compare model architectures (layer counts, shapes)

### Hypothesis 5: Batching Implementation Bug ⭐⭐

**Likelihood**: LOW-MODERATE

**Evidence**:
- Current implementation uses batching for performance
- Batching code may handle CoreML/ONNX differently
- ONNX showing multiple false detections (3 faces vs 1)

**Possible Causes**:
1. Batch result regrouping logic differs between backends
2. ONNX batch inference has different semantics
3. Index tracking bug causes boxes to be assigned to wrong frames

**Investigation Steps**:
1. Test both backends with batch_size=1 (disable batching)
2. Compare box assignment logic in CoreML vs ONNX batch code
3. Add assertions to verify batch indices are correct
4. Log which frame each box belongs to

### Hypothesis 6: Pipeline Code Changes ⭐

**Likelihood**: VERY LOW

**Evidence**:
- CoreML works correctly
- Both backends share same pipeline code
- Unlikely to be shared code issue

**Possible Causes**:
1. Recent refactoring broke ONNX-specific paths
2. Conditional logic branches differently for ONNX vs CoreML

**Investigation Steps**:
1. Git diff the pipeline code since last known working state
2. Check for ONNX-specific code paths
3. Review recent commits for unintended changes

---

## Investigation Priority

### Phase 1: Quick Diagnostics (15 minutes)

**Goal**: Identify if ONNX models or configuration are broken

```bash
# Test 1: Check ONNX Runtime version
python3 -c "import onnxruntime; print(onnxruntime.__version__)"

# Test 2: Verify model files exist
ls -lh pymtcnn/models/onnx/*.onnx
md5 pymtcnn/models/onnx/*.onnx

# Test 3: Test both backends with debug logging
PYTHONPATH=pymtcnn python3 -c "
from pymtcnn import MTCNN
import cv2

# Test CoreML
detector_coreml = MTCNN(backend='coreml', debug_mode=True)
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
bbox_coreml, lm_coreml, debug_coreml = detector_coreml.detect(img)
print(f'CoreML PNet: {debug_coreml[\"pnet\"][\"num_boxes\"]}')
print(f'CoreML RNet: {debug_coreml[\"rnet\"][\"num_boxes\"]}')
print(f'CoreML ONet: {debug_coreml[\"onet\"][\"num_boxes\"]}')

# Test ONNX
detector_onnx = MTCNN(backend='onnx', debug_mode=True)
bbox_onnx, lm_onnx, debug_onnx = detector_onnx.detect(img)
print(f'ONNX PNet: {debug_onnx[\"pnet\"][\"num_boxes\"]}')
print(f'ONNX RNet: {debug_onnx[\"rnet\"][\"num_boxes\"]}')
print(f'ONNX ONet: {debug_onnx[\"onet\"][\"num_boxes\"]}')
"
```

### Phase 2: Threshold Investigation (30 minutes)

**Goal**: Check if threshold differences cause box count mismatch

```python
# Compare thresholds
import pymtcnn.backends.coreml_backend as coreml
import pymtcnn.backends.onnx_backend as onnx

print("CoreML thresholds:", coreml.DEFAULT_THRESHOLDS)
print("ONNX thresholds:", onnx.DEFAULT_THRESHOLDS)

# Test with relaxed thresholds
detector = MTCNN(backend='coreml', thresholds=[0.5, 0.5, 0.5])  # Lower from 0.6
```

### Phase 3: Model Regeneration (1 hour)

**Goal**: Rebuild ONNX models from C++ .dat files

```bash
# Re-convert all ONNX models
cd archive/conversion_scripts
python3 convert_mtcnn_to_onnx_v2.py

# Validate new models
python3 validate_onnx_models.py
```

### Phase 4: Deep Debugging (2-4 hours)

**Goal**: Layer-by-layer comparison between backends

1. Add logging to capture PNet/RNet/ONet outputs for same input
2. Compare raw logits before softmax
3. Compare confidence scores after softmax
4. Compare bbox regression outputs
5. Compare NMS results

---

## Success Criteria

**Validation PASS Requirements**:
1. ✅ PNet box count within ±5% of C++ baseline
2. ✅ RNet box count within ±10% of C++ baseline
3. ✅ ONet/Final box count exact match with C++
4. ✅ Bounding box IoU > 95% for both CoreML and ONNX
5. ✅ Cross-platform consistency: CoreML and ONNX within ±2% box counts

---

## Fallback Plans

### If ONNX Cannot Be Fixed:
1. **Option A**: Use CoreML-only backend (96.83% IoU is production-ready)
2. **Option B**: Regenerate ONNX models from scratch with current toolchain
3. **Option C**: Use Pure Python MTCNN as ONNX replacement (slow but accurate)

### If Both Backends Fail:
1. Revert to last known working git commit (find IMG_0434 timestamp)
2. Rebuild from Pure Python MTCNN with verified bit-for-bit accuracy
3. Re-apply conversion fixes one at a time with validation at each step

---

## Reference Documentation

- `MTCNN_COMPLETE_VERIFICATION.md` - Pure Python bit-for-bit accuracy baseline
- `PHASE2_ONNX_SUCCESS_SUMMARY.md` - ONNX conversion with <2 ppm accuracy
- `PHASE3_COREML_CONVERSION_RESULTS.md` - CoreML conversion with ~2% tolerance
- `FINAL_OPTIMIZATION_SUMMARY.md` - Performance benchmarks (95% IoU target)
- `IMG_0434` - Known working state screenshot showing identical box counts

---

## Next Steps

1. **Run Phase 1 diagnostics** to identify quick-win fixes
2. **Compare thresholds** between CoreML and ONNX backends
3. **Test ONNX model integrity** by validating against Pure Python CNN
4. **If needed, regenerate ONNX models** from C++ .dat files
5. **Document findings** and update this plan

---

**Status**: INVESTIGATION PENDING - Awaiting Phase 1 diagnostics
