# CoreML/ONNX Performance Analysis - Face Mirror v2.0.0

## Neural Engine Utilization by Model

### 1. MTL EfficientNet (AU Extraction)
**File**: `mtl_efficientnet_b0_coreml.onnx`
```
Total operators:           520
Neural Engine operators:   359
CPU fallback operators:    161
Neural Engine coverage:    69.0%
CoreML partitions:         28
```

**Analysis**: Only 69% of operators run on Neural Engine. The 28 partitions indicate significant fragmentation - the model is split into many small chunks. This suggests:
- High overhead from CPU ↔ Neural Engine transfers
- Some operator types not supported by CoreML
- Potential optimization opportunity

---

### 2. RetinaFace (Face Detection)
**File**: `retinaface_mobilenet025_coreml.onnx`
```
Total operators:           144
Neural Engine operators:   117
CPU fallback operators:    27
Neural Engine coverage:    81.3%
CoreML partitions:         3
```

**Analysis**: Better coverage at 81.3%, with only 3 partitions. This means:
- More efficient execution (fewer CPU-GPU transitions)
- Most detection operators well-supported by CoreML
- ~5-10x speedup achieved (as noted in logs: ~20-40ms per detection)

---

### 3. STAR Landmark Detector
**File**: `star_landmark_98_coreml.onnx`
```
Total operators:           855
Neural Engine operators:   706
CPU fallback operators:    149
Neural Engine coverage:    82.6%
CoreML partitions:         18
```

**Analysis**: Highest operator count, good 82.6% coverage, but 18 partitions:
- Moderate fragmentation impacts performance
- ~10-20x speedup achieved (logs show ~90-180ms per frame)
- Most expensive model in pipeline

---

## CoreML Warnings Detected

### Warning Type: Partition Fragmentation
All three models show the warning:
```
[W:onnxruntime:, coreml_execution_provider.cc:107 GetCapability]
CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: [N]
```

**Implications**:
- Each partition boundary requires data transfer between CPU and Neural Engine
- Memory copies add latency overhead
- Partitions indicate unsupported operators that break the graph

**Common unsupported operators**:
- Dynamic shapes/reshapes
- Certain activation functions
- Loop/control flow operators
- Custom normalization layers

---

## Performance Characteristics

### AU Extraction (per frame)
```
Processing time:    ~113ms per frame (8.6-8.8 FPS)
Model inference:    ~95% of processing time
Read/Cleanup:       ~5% overhead
```

**Breakdown**:
- MTL model dominates (~100-110ms)
- 69% Neural Engine + 31% CPU fallback
- 28 partitions = ~28 CPU↔GPU transitions per frame

### Face Detection + Landmarks (during mirroring)
```
Detection time:     248.6ms per detection
Landmark time:      183.0ms per frame
Effective FPS:      4.3 (when detecting)
```

**Notes**:
- Detection runs every ~5 frames (adaptive)
- RetinaFace (81.3% NE) faster than expected
- STAR landmarks (82.6% NE) still bottleneck

---

## Memory Patterns (Inferred)

**From logs**:
- Models frozen from garbage collection (reduces GC overhead)
- Isolated CoreML sessions per process (multiprocessing)
- Batch processing: 100 frames at a time
- 6 parallel threads used

**Expected memory footprint**:
```
MTL model:           ~50-100 MB
RetinaFace model:    ~10-20 MB
STAR model:          ~30-50 MB
Frame buffers:       ~100 frames × 1080×1920×3 bytes = ~600 MB
Total per process:   ~700-800 MB
```

**Multiprocessing implications**:
- Each worker process has isolated CoreML session
- Memory NOT shared between processes
- 6 threads = potentially 6× memory for CoreML sessions

---

## Optimization Recommendations

### 1. Reduce MTL Partitioning (Priority: HIGH)
Current: 28 partitions at 69% coverage
```
Options:
- Re-export with newer CoreML version
- Simplify operator types during conversion
- Use CoreML-native operators where possible
```
**Expected gain**: 15-25% speedup

### 2. Pre-compiled CoreML Models
Current: ONNX → CoreML at runtime
```
Convert to .mlmodel/.mlpackage format:
- Pre-compile during build
- Eliminate runtime compilation overhead
- Better operator optimization
```
**Expected gain**: 10-15% speedup

### 3. Batch Inference
Current: Single-frame inference
```
Process multiple frames per inference:
- Reduce partition overhead per frame
- Better utilize Neural Engine parallelism
```
**Expected gain**: 20-30% speedup (but adds latency)

---

## Summary

| Model | Neural Engine | CPU | Partitions | Performance Impact |
|-------|--------------|-----|------------|-------------------|
| MTL (AU) | 69.0% | 31.0% | 28 | HIGH overhead |
| RetinaFace | 81.3% | 18.7% | 3 | LOW overhead |
| STAR | 82.6% | 17.4% | 18 | MEDIUM overhead |

**Overall Assessment**:
- Neural Engine is utilized but heavily fragmented
- MTL model is the primary bottleneck (69% coverage, 28 partitions)
- ~30% of compute still runs on CPU across all models
- Optimization potential: 30-50% overall speedup possible
