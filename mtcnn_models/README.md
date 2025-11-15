# MTCNN Model Repository

This directory contains all MTCNN face detection models across different optimization phases and formats.

## Directory Structure

```
mtcnn_models/
├── source/         # C++ binary weights (gold standard)
├── python/         # Pure Python CNN implementation
├── onnx/           # Phase 2 ONNX models (cross-platform)
└── coreml/         # Phase 3 CoreML models (Apple Silicon)
```

## Performance Timeline

| Phase | Format | Platform | FPS | Speedup | Status |
|-------|--------|----------|-----|---------|--------|
| Baseline | Pure Python | All | 0.195 | 1.0x | ✅ Complete |
| Phase 1 | Optimized Python | All | 0.910 | 4.3x | ✅ Complete |
| Phase 2 | ONNX + CoreMLEP | Apple Silicon | 5.87 | 30.1x | ✅ Complete |
| **Phase 3** | **CoreML Native** | **Apple Silicon** | **30+ (target)** | **153x (target)** | 🔄 **In Progress** |
| Phase 4 | TensorRT | NVIDIA | 100+ (est.) | 500x (est.) | ⏳ Future |

## Model Usage

### Cross-Platform (ONNX)
```python
from pure_python_mtcnn_onnx import ONNXMTCNNDetector

detector = ONNXMTCNNDetector(
    pnet_path='mtcnn_models/onnx/pnet.onnx',
    rnet_path='mtcnn_models/onnx/rnet.onnx',
    onet_path='mtcnn_models/onnx/onet.onnx'
)

bboxes = detector.detect(image)  # 5.87 FPS on Apple Silicon
```

### Apple Silicon (CoreML) - Coming in Phase 3
```python
from pure_python_mtcnn_coreml import CoreMLMTCNNDetector

detector = CoreMLMTCNNDetector(
    pnet_path='mtcnn_models/coreml/pnet_fp16.mlpackage',
    rnet_path='mtcnn_models/coreml/rnet_fp16.mlpackage',
    onet_path='mtcnn_models/coreml/onet_fp16.mlpackage'
)

bboxes = detector.detect(image)  # Target: 30+ FPS
```

### Automatic Platform Detection
```python
from mtcnn_detector_auto import MTCNNDetector

# Automatically selects best available backend
detector = MTCNNDetector(model_dir='mtcnn_models/')
# Apple Silicon M-series → CoreML FP16
# Apple Silicon (no CoreML) → ONNX + CoreMLExecutionProvider
# NVIDIA GPU → ONNX + TensorRT (Phase 4)
# CPU → ONNX + CPU

bboxes = detector.detect(image)
```

## Model Provenance

### Phase 2 ONNX Models
- **Created**: 2025-01-14
- **Source**: C++ OpenFace MTCNN binary weights (det1.dat, det2.dat, det3.dat)
- **Converter**: `convert_mtcnn_to_onnx_v2.py`
- **Git Commit**: [hash]
- **Validation**: < 2e-6 max diff vs Pure Python CNN
- **Accuracy**: 75.3% mean IoU vs C++ OpenFace (30/30 detection agreement)
- **Performance**: 5.87 FPS, 170ms latency, 30.1x speedup vs baseline

### Phase 3 CoreML Models
- **Status**: Not yet created
- **Planned Source**: Phase 2 ONNX models
- **Converter**: TBD (`convert_onnx_to_coreml.py`)
- **Target Validation**: FP32 < 1e-5 vs ONNX, FP16 IoU > 0.999 vs FP32
- **Target Accuracy**: > 90% mean IoU vs C++ OpenFace
- **Target Performance**: 30+ FPS (5.1x improvement over Phase 2)

## Accuracy Validation

### Pure Python CNN (Baseline)
- **vs C++ OpenFace**: 86.4% mean IoU, 100% detection agreement
- **CNN Layer Outputs**: < 1.4e-6 max diff (floating-point precision)
- **Status**: Gold standard for validation

### Phase 2 ONNX
- **vs Pure Python CNN**: < 2e-6 max diff (perfect match)
- **vs C++ OpenFace**: 75.3% mean IoU, 100% detection agreement
- **IoU Gap Analysis**: Pipeline logic differences (precision, rounding), not CNN accuracy
- **Status**: Production-ready for cross-platform

### Phase 3 CoreML (Target)
- **FP32 vs ONNX**: < 1e-5 max diff required
- **FP16 vs FP32**: > 0.999 IoU required
- **vs C++ OpenFace**: > 0.90 IoU target
- **Status**: To be measured

## Known Issues

### Phase 2 ONNX
1. **IoU Gap** (75.3% vs 86.4%): Acceptable for production given 100% detection agreement and 30x speedup
2. **Platform Limitations**: CoreMLExecutionProvider on Apple Silicon only
3. **Debugging**: Limited layer-by-layer debugging compared to Pure Python

### Phase 3 CoreML (Anticipated)
1. **Platform Lock-in**: Apple Silicon only (no cross-platform fallback)
2. **FP16 Accuracy**: May have small accuracy loss vs FP32
3. **ANE Compatibility**: Not all ops may run on Neural Engine
4. **Debugging**: Even more limited than ONNX

## Fallback Strategy

The detector should automatically fall back through this chain:

1. **Try**: CoreML FP16 (Apple Silicon, fastest)
2. **Fallback**: CoreML FP32 (Apple Silicon, more accurate)
3. **Fallback**: ONNX + CoreMLExecutionProvider (Apple Silicon, cross-platform code)
4. **Fallback**: ONNX + CUDAExecutionProvider (NVIDIA)
5. **Fallback**: ONNX + CPUExecutionProvider (any platform)
6. **Fallback**: Pure Python (slowest but guaranteed to work)

## Critical Conversion Pitfalls

See `PHASE3_COREML_CONVERSION_STRATEGY.md` for comprehensive list of 12 pitfalls to avoid during model conversion.

Most critical:
- **Pitfall #1**: Kernel transpose bug (5,400,000x accuracy loss)
- **Pitfall #2**: MaxPool rounding (dimension mismatch)
- **Pitfall #6**: Kernel spatial transpose (14.31 → 0.000002 max diff)

## Development Notes

### Adding New Model Formats

When adding a new format (e.g., TensorRT in Phase 4):

1. Create subdirectory: `mtcnn_models/tensorrt/`
2. Add validation subdirectory: `mtcnn_models/tensorrt/validation/`
3. Document conversion process in README
4. Run full validation suite
5. Update automatic platform detection
6. Add to fallback chain
7. Update this README

### Model Versioning

Each model directory should contain a `PROVENANCE.md` file with:
- Creation date
- Source models (with hashes)
- Conversion script (with git commit)
- Validation results
- Known issues
- Performance benchmarks

---

**Last Updated**: 2025-01-14
**Current Phase**: Phase 3 (CoreML Conversion)
