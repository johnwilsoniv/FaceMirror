# CPU-Only ONNX Test - CoreML Serialization Hypothesis

**Date:** 2025-10-17
**Status:** Testing in progress

---

## Problem Statement

CoreML Execution Provider serializes GPU/Neural Engine operations across threads, preventing true parallel inference. With 6 worker threads, we observe only 11-15% parallelization efficiency:

- **Expected with 6x parallel:** 85-172 fps
- **Actual with CoreML:** 9.5 fps (serialized)

---

## Test Configuration

**Change Made:**
```python
# onnx_mtl_detector.py line 50
providers = ['CPUExecutionProvider']  # Force CPU-only, disable CoreML
```

**Hypothesis:**
CPU-only ONNX Runtime allows true thread-level parallelization without CoreML's serialization locks.

---

## Expected Results

### **Single-thread CPU performance:**
- CoreML (Neural Engine): 15-30ms per frame
- CPU-only ONNX: 30-50ms per frame (2-3x slower)

### **6-thread parallelized performance:**

| Configuration | Per-frame time | Parallelization | Effective time | FPS | Speedup |
|---------------|----------------|-----------------|----------------|-----|---------|
| CoreML (baseline) | 15-30ms | 11-15% (serialized) | 105ms | 9.5 | 1.0x |
| CPU-only | 30-50ms | 70-85% (true parallel) | 7-12ms | 83-143 | 8-15x |

### **Target:**
- **Minimum:** 85+ fps (meets lower target bound)
- **Optimal:** 120+ fps (comfortable headroom)

---

## How to Validate

### **Run Test:**
```bash
python3 main.py
# Select the same test video (IMG_0437.MOV or similar)
```

### **Look for these metrics:**

**1. Initialization message:**
```
✓ TESTING: Using CPU-only ONNX for true 6x parallelization
  Per-frame: ~30-50ms (slower than CoreML's 15-30ms)
  With 6 threads: Expected 50-80+ fps total (vs 9.5 fps serialized)
```

**2. Processing time (971-frame video):**
```
[OpenFace Performance Summary - Batch 10/10]
  Process: XX.XXs (should be ~12-20s, vs 102s with CoreML)
```

**3. FPS calculation:**
```
971 frames / Process time = Actual FPS

Example:
971 / 15s = 64.7 fps ✓ PASS (exceeds 85 fps target... wait no, 64 < 85)
971 / 12s = 80.9 fps ✓ MARGINAL (close to 85 fps target)
971 / 10s = 97.1 fps ✓ PASS (exceeds 85 fps target)
```

---

## Decision Tree

### **Case 1: FPS ≥ 85 (SUCCESS)**
- ✅ CPU-only is sufficient
- Keep this configuration
- Ship to production
- **No need for multiprocessing complexity**

### **Case 2: FPS = 60-84 (MARGINAL)**
- ⚠️ Close to target but insufficient
- Options:
  1. INT8 quantization (1.5-2x speedup, might push to 90-168 fps)
  2. Reduce batch size (less memory, more batches, might help)
  3. Implement multiprocessing (guaranteed 100+ fps)

### **Case 3: FPS < 60 (INSUFFICIENT)**
- ❌ CPU-only not sufficient
- Must implement multiprocessing solution
- Expected with multiprocessing: 100-150+ fps

### **Case 4: FPS < 30 (WORSE THAN BASELINE)**
- ❌ Something is wrong
- Check: Are threads actually running in parallel?
- Debug: Single-thread CPU performance might be worse than expected

---

## Interpreting Results

### **Good signs:**
- Process time: 10-20s (vs 102s baseline)
- CPU utilization: 400-600% in Activity Monitor (6 cores active)
- Linear scaling: 6x faster than single-thread

### **Bad signs:**
- Process time: >50s (minimal improvement)
- CPU utilization: <200% (threads not parallelizing)
- Sublinear scaling: <3x faster than single-thread

---

## Next Steps Based on Results

### **If successful (≥85 fps):**
1. Test with AU45 enabled (add STAR landmark detection)
2. If AU45 still ≥60 fps → Done!
3. If AU45 drops to <60 fps → Need to optimize STAR separately

### **If insufficient (<85 fps):**
1. Document actual FPS achieved
2. Calculate multiprocessing speedup: (single-thread fps) × 6 × 0.75 = expected fps
3. Implement multiprocessing solution
4. Re-test and validate

### **If comparable to baseline (~10 fps):**
1. Something is wrong with threading
2. Check: `NUM_THREADS` is still 6?
3. Check: Are exceptions silently failing?
4. Debug: Add timing logs per-thread

---

## Rollback Instructions

If CPU-only performs worse than CoreML serialized (unlikely), revert:

```python
# onnx_mtl_detector.py line 40-62
# Uncomment original CoreML configuration:
if use_coreml:
    providers = [
        ('CoreMLExecutionProvider', {
            'MLComputeUnits': 'ALL',
            'ModelFormat': 'MLProgram',
        }),
        'CPUExecutionProvider'
    ]
else:
    providers = ['CPUExecutionProvider']

# Comment out forced CPU-only line:
# providers = ['CPUExecutionProvider']
```

---

## Test Checklist

- [ ] Run test video (IMG_0437.MOV or similar)
- [ ] Record initialization message (confirm CPU-only active)
- [ ] Record total processing time for 971 frames
- [ ] Calculate actual FPS: 971 / time
- [ ] Check Activity Monitor CPU usage during processing
- [ ] Compare to baseline (9.5 fps with CoreML)
- [ ] Determine if result meets ≥85 fps target
- [ ] Document decision: Keep CPU-only vs Implement multiprocessing

---

## Baseline for Comparison

**Original CoreML (serialized):**
```
Process: 102.01s (94.8%)
971 frames
FPS: 9.52
Parallelization efficiency: 11-15%
```

**Target with CPU-only:**
```
Process: <12s
971 frames
FPS: >80
Parallelization efficiency: >70%
```

---

**Run the test and report back!**
