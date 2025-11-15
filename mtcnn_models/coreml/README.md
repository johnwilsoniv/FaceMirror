# Phase 3 CoreML MTCNN Models

Native CoreML models for direct Apple Neural Engine (ANE) execution on Apple Silicon.

## Files (Planned)

```
coreml/
├── pnet_fp32.mlpackage/      # Full precision baseline
├── rnet_fp32.mlpackage/
├── onet_fp32.mlpackage/
├── pnet_fp16.mlpackage/      # FP16 quantized (production)
├── rnet_fp16.mlpackage/
├── onet_fp16.mlpackage/
├── validation/
│   ├── fp32_vs_onnx.json         # FP32 CoreML vs ONNX baseline
│   ├── fp16_vs_fp32.json         # FP16 accuracy impact
│   └── cpp_vs_coreml.json        # End-to-end vs C++ gold standard
└── README.md                      # This file
```

## Model Specifications (Target)

### PNet CoreML
- **Input**: Variable size RGB image
- **Output**: Face probability map + bbox regression
- **Format**: .mlpackage (macOS 13+)
- **Compute Units**: ALL (ANE + GPU + CPU)
- **Versions**:
  - FP32: ~29 KB (validation baseline)
  - FP16: ~15 KB (production)

### RNet CoreML
- **Input**: 24×24×3 RGB patches
- **Output**: Face probability + bbox regression
- **Format**: .mlpackage
- **Versions**:
  - FP32: ~396 KB
  - FP16: ~200 KB

### ONet CoreML
- **Input**: 48×48×3 RGB patches
- **Output**: Face probability + bbox regression + landmarks
- **Format**: .mlpackage
- **Versions**:
  - FP32: ~1.5 MB
  - FP16: ~750 KB

## Creation Details (Planned)

**Date**: TBD (Phase 3)
**Converter**: `convert_onnx_to_coreml.py` (to be created)
**Source**: Phase 2 ONNX models (validated < 2e-6 vs Pure Python)
**CoreML Tools Version**: 7.0+
**Deployment Target**: macOS 13.0+
**Git Commit**: [TBD]

## Conversion Process (Planned)

### Step 1: ONNX → CoreML FP32

```python
import coremltools as ct

model = ct.converters.onnx.convert(
    model='../onnx/pnet.onnx',
    minimum_deployment_target=ct.target.macOS13,
    compute_units=ct.ComputeUnit.ALL  # Enable ANE + GPU + CPU
)

model.save('pnet_fp32.mlpackage')
```

### Step 2: FP16 Quantization

```python
model_fp16 = ct.optimize.coreml.linear_quantize_weights(
    model,
    mode='linear_symmetric',
    dtype=np.float16
)

model_fp16.save('pnet_fp16.mlpackage')
```

### Step 3: ANE Optimization

CoreML Tools will automatically optimize for Neural Engine:
- Layer fusion (combine Conv+PReLU)
- Memory layout optimization
- FP16 operations on ANE
- GPU/CPU fallback for unsupported ops

## Validation Requirements

### FP32 Validation (vs ONNX Baseline)

| Metric | Threshold | Status |
|--------|-----------|--------|
| Max Abs Diff | < 1e-5 | ⏳ Not yet tested |
| Mean Abs Diff | < 1e-6 | ⏳ Not yet tested |
| Layer-by-Layer | All pass | ⏳ Not yet tested |

### FP16 Validation (vs FP32)

| Metric | Threshold | Status |
|--------|-----------|--------|
| Max Abs Diff | < 1e-3 | ⏳ Not yet tested |
| Mean IoU | > 0.999 | ⏳ Not yet tested |
| Detection Agreement | 100% | ⏳ Not yet tested |

### End-to-End Accuracy (vs C++ Gold Standard)

| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Mean IoU | > 0.90 | > 0.75 | ⏳ Not yet tested |
| Detection Agreement | 100% | 95% | ⏳ Not yet tested |
| False Positives | 0 | < 2 | ⏳ Not yet tested |

## Performance Targets

**Test Platform**: Apple Silicon M-series
**Test Dataset**: 30-frame Patient Data videos

| Metric | Phase 2 (ONNX) | Phase 3 Target | Stretch Goal |
|--------|---------------|----------------|--------------|
| Mean FPS | 5.87 | **30+** | 50+ |
| Mean Latency | 170 ms | **< 33 ms** | < 20 ms |
| P95 Latency | 195 ms | **< 50 ms** | < 30 ms |
| Memory | ~200 MB | < 200 MB | < 100 MB |
| Speedup vs Phase 2 | 1.0x | **5.1x** | 8.5x |

### Expected Performance by Platform

| Device | FP32 FPS | FP16 FPS | Notes |
|--------|----------|----------|-------|
| M1 | 20-25 | **30-40** | Base model |
| M1 Pro | 25-30 | **35-45** | More GPU cores |
| M1 Max | 30-40 | **40-50** | Max GPU + ANE |
| M2 | 25-30 | **35-45** | Better ANE |
| M2 Pro | 30-35 | **40-50** | - |
| M2 Max/Ultra | 40-50 | **50-70** | Highest performance |
| M3/M4 | 35-45 | **45-60** | Latest ANE optimizations |

## Usage (Planned)

### Python (coremltools)

```python
import coremltools as ct
import numpy as np

# Load model
model = ct.models.MLModel('mtcnn_models/coreml/pnet_fp16.mlpackage')

# Prepare input
preprocessed = (image_rgb - 127.5) * 0.0078125

# Run inference
result = model.predict({'input': preprocessed})
```

### Full MTCNN Pipeline (Planned)

```python
from pure_python_mtcnn_coreml import CoreMLMTCNNDetector

detector = CoreMLMTCNNDetector()  # Auto-detects .mlpackage files
bboxes = detector.detect(image)

# Expected: 30+ FPS on Apple Silicon
```

### Automatic Fallback

```python
from mtcnn_detector_auto import MTCNNDetector

detector = MTCNNDetector(model_dir='mtcnn_models/')

# Fallback chain:
# 1. Try CoreML FP16 (fastest)
# 2. Try CoreML FP32 (more accurate)
# 3. Try ONNX + CoreMLExecutionProvider (5.87 FPS)
# 4. Try ONNX + CPUExecutionProvider
# 5. Try Pure Python (slowest)

bboxes = detector.detect(image)
```

## Platform Requirements

### Minimum Requirements
- **Hardware**: Apple Silicon (M1 or newer)
- **OS**: macOS 13.0 (Ventura) or newer
- **Python**: 3.8+
- **coremltools**: 7.0+

### Recommended
- **Hardware**: M2 Pro/Max or M3/M4
- **OS**: macOS 14.0 (Sonoma) or newer
- **RAM**: 16 GB+

### Not Supported
- Intel Macs (no Neural Engine)
- Linux/Windows (CoreML is macOS-only)
- iOS/iPadOS (different CoreML API, could be ported)

## Known Risks & Mitigations

### Risk 1: CoreML Conversion Failure
**Likelihood**: Medium
**Impact**: High (blocks Phase 3)
**Mitigation**:
- Use well-tested ONNX → CoreML path
- Validate ONNX models are CoreML-compatible
- Have ONNX fallback ready
- Start with PNet (smallest) to validate process

### Risk 2: ANE Incompatibility
**Likelihood**: Low-Medium
**Impact**: Medium (reduced performance)
**Mitigation**:
- Some ops may fall back to GPU/CPU
- Use coremltools analysis to check ANE usage:
  ```python
  model.compute_units = ct.ComputeUnit.CPU_AND_NE  # Force ANE
  ```
- Acceptable if 80%+ ops run on ANE

### Risk 3: FP16 Accuracy Loss
**Likelihood**: Low
**Impact**: Medium (may need FP32)
**Mitigation**:
- Extensive validation on 30-frame test set
- If IoU < 0.90, use FP32 instead
- FP16 typically < 0.5% accuracy loss
- Document trade-offs clearly

### Risk 4: Platform Lock-in
**Likelihood**: High (by design)
**Impact**: Low (expected)
**Mitigation**:
- Keep ONNX models as cross-platform baseline
- Clear documentation of platform requirements
- Automatic fallback to ONNX on non-Apple platforms

## Development Checklist

### Conversion
- [ ] Install coremltools 7.0+
- [ ] Verify ONNX models are CoreML-compatible
- [ ] Convert PNet ONNX → CoreML FP32
- [ ] Convert RNet ONNX → CoreML FP32
- [ ] Convert ONet ONNX → CoreML FP32
- [ ] Analyze ANE utilization for each network
- [ ] Create FP16 quantized versions

### Validation
- [ ] Validate PNet FP32 vs ONNX (< 1e-5 max diff)
- [ ] Validate RNet FP32 vs ONNX (< 1e-5 max diff)
- [ ] Validate ONet FP32 vs ONNX (< 1e-5 max diff)
- [ ] Validate PNet FP16 vs FP32 (> 0.999 IoU)
- [ ] Validate RNet FP16 vs FP32 (> 0.999 IoU)
- [ ] Validate ONet FP16 vs FP32 (> 0.999 IoU)
- [ ] End-to-end FP16 vs C++ OpenFace (> 0.90 IoU target)

### Integration
- [ ] Create CoreMLMTCNNDetector class
- [ ] Implement full pipeline with CoreML models
- [ ] Add platform detection
- [ ] Implement fallback chain
- [ ] Write integration tests

### Performance
- [ ] Benchmark FP32 on M-series hardware
- [ ] Benchmark FP16 on M-series hardware
- [ ] Verify 30+ FPS target achieved
- [ ] Profile ANE vs GPU vs CPU usage
- [ ] Measure memory consumption

### Documentation
- [ ] Update this README with results
- [ ] Create validation reports
- [ ] Document platform requirements
- [ ] Write usage examples
- [ ] Update main README

## Validation Scripts (To Be Created)

```bash
# Convert models
python convert_onnx_to_coreml.py

# Validate FP32
python validate_coreml_fp32_vs_onnx.py

# Validate FP16
python validate_coreml_fp16_vs_fp32.py

# End-to-end accuracy
python compare_coreml_vs_cpp_openface.py

# Performance benchmark
python benchmark_coreml_performance.py

# ANE utilization analysis
python analyze_coreml_ane_usage.py
```

## Critical Pitfalls to Avoid

See `PHASE3_COREML_CONVERSION_STRATEGY.md` for complete list. Most critical for CoreML:

1. **Input Format**: CoreML expects RGB (not BGR)
2. **Memory Layout**: Verify C-contiguous vs Fortran-contiguous
3. **Operation Support**: Not all ONNX ops map to ANE
4. **Quantization**: FP16 may introduce accuracy drift
5. **Batch Size**: CoreML typically runs batch=1

## Success Criteria

Phase 3 complete when:
1. ✅ All 3 networks converted to CoreML FP32 + FP16
2. ✅ FP32 validated < 1e-5 diff vs ONNX
3. ✅ FP16 validated > 0.999 IoU vs FP32
4. ✅ End-to-end IoU ≥ 0.90 vs C++ gold standard
5. ✅ Performance ≥ 30 FPS on Apple Silicon
6. ✅ Fallback chain working on all platforms
7. ✅ Documentation complete

---

**Model Status**: ⏳ Not Yet Created (Phase 3 In Progress)
**Target Completion**: TBD
**Next Steps**: Install coremltools, convert ONNX → CoreML FP32
