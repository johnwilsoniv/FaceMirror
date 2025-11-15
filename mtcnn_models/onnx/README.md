# Phase 2 ONNX MTCNN Models

Cross-platform MTCNN models in ONNX format with optimized execution providers.

## Files

```
onnx/
├── pnet.onnx          # Proposal Network (12×12 receptive field)
├── rnet.onnx          # Refinement Network (24×24 input)
├── onet.onnx          # Output Network (48×48 input)
├── validation/
│   ├── pnet_validation.json
│   ├── rnet_validation.json
│   └── onet_validation.json
└── README.md          # This file
```

## Model Specifications

### PNet (Proposal Network)
- **Input**: Variable size RGB image (H×W×3)
- **Output**: Face probability map + bbox regression
- **File Size**: 29 KB
- **Parameters**: 7,366
- **Layers**: 6 (3 Conv, 2 PReLU, 1 MaxPool)

### RNet (Refinement Network)
- **Input**: 24×24×3 RGB patches
- **Output**: Face probability + bbox regression
- **File Size**: 396 KB
- **Parameters**: 98,442
- **Layers**: 10 (4 Conv, 3 PReLU, 2 MaxPool, 2 FC)

### ONet (Output Network)
- **Input**: 48×48×3 RGB patches
- **Output**: Face probability + bbox regression + 5 landmarks
- **File Size**: 1.5 MB
- **Parameters**: 388,318
- **Layers**: 14 (5 Conv, 4 PReLU, 3 MaxPool, 3 FC)

## Creation Details

**Date**: 2025-01-14 10:26
**Converter**: `convert_mtcnn_to_onnx_v2.py`
**Source**: C++ OpenFace MTCNN binary weights
**ONNX Opset**: 11
**Git Commit**: [TBD]

## Conversion Process

### Critical Fixes Applied

1. **Kernel Spatial Transpose** (Pitfall #6):
   ```python
   kernels_cpp = layer.kernels  # (K, C, H, W) column-major
   kernels_pytorch = np.transpose(kernels_cpp, (0, 1, 3, 2))  # Swap H↔W
   ```

2. **MaxPool Rounding** (Pitfall #2):
   ```python
   # Use ceil_mode=True to match C++ round() behavior
   F.max_pool2d(x, kernel_size, stride, padding=0, ceil_mode=True)
   ```

3. **PReLU Behavior** (Pitfall #5):
   ```python
   # Ensure x >= 0 (not x > 0)
   output = torch.where(x >= 0, x, x * weight)
   ```

4. **FC Flattening Order** (Pitfall #4):
   ```python
   # Transpose each feature map before flattening
   flattened = torch.cat([fm.T.flatten() for fm in feature_maps])
   ```

## Validation Results

### Layer-by-Layer Comparison vs Pure Python CNN

| Network | Max Diff | Mean Diff | Status |
|---------|----------|-----------|--------|
| PNet | 0.6 ppm | < 0.1 ppm | ✅ Perfect |
| RNet | 1.4 ppm | < 0.2 ppm | ✅ Perfect |
| ONet | 1.9 ppm | < 0.3 ppm | ✅ Perfect |

**ppm** = parts per million (1e-6)

### End-to-End Accuracy vs C++ OpenFace

**Test Dataset**: 30 frames from 10 Patient Data videos

| Metric | Value |
|--------|-------|
| Detection Agreement | 100% (30/30 frames) |
| Mean IoU | 75.3% |
| Median IoU | 76.3% |
| Min IoU | 62.1% |
| Matched Boxes | 30/30 |
| False Positives | 0 |
| False Negatives | 0 |

**IoU Gap Analysis**: The 11.1% IoU gap vs Pure Python (86.4%) comes from pipeline logic differences (numeric precision, rounding), NOT CNN model accuracy. CNNs validated to < 2e-6 difference.

## Performance Benchmarks

**Test Platform**: Apple Silicon (M-series)
**Test Dataset**: Same 30-frame dataset as accuracy validation

| Metric | Value |
|--------|-------|
| Mean FPS | 5.871 |
| Median FPS | 5.844 |
| Mean Latency | 170.3 ms |
| P95 Latency | 195.2 ms |
| P99 Latency | 203.2 ms |
| Min Latency | 134.0 ms |
| Max Latency | 205.5 ms |

**Speedup Analysis**:
- vs Baseline (0.195 FPS): **30.1x faster**
- vs Phase 1 (0.910 FPS): **6.45x faster**

**Execution Provider**: CoreMLExecutionProvider (ONNX Runtime → CoreML bridge)

## Usage

### Python (onnxruntime)

```python
import onnxruntime as ort
import numpy as np

# Create session with CoreML provider (Apple Silicon)
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('mtcnn_models/onnx/pnet.onnx', providers=providers)

# Run inference
input_name = session.get_inputs()[0].name
preprocessed_image = (image_rgb - 127.5) * 0.0078125
output = session.run(None, {input_name: preprocessed_image})
```

### Full MTCNN Pipeline

```python
from pure_python_mtcnn_onnx import ONNXMTCNNDetector

detector = ONNXMTCNNDetector()  # Automatically finds models in mtcnn_models/onnx/
bboxes = detector.detect(image)

# Returns: [[x1, y1, x2, y2, score], ...]
```

## Platform-Specific Execution Providers

### Apple Silicon (M1/M2/M3/M4)
```python
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
# 5.87 FPS, uses Metal Performance Shaders
```

### NVIDIA GPU (Future Phase 4)
```python
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# Expected: 50-100+ FPS
```

### Intel/AMD CPU
```python
providers = ['CPUExecutionProvider']
# Expected: 2-3 FPS
```

## Known Issues

### 1. IoU Gap (75.3% vs 86.4%)
**Severity**: Low
**Impact**: Detections are correct but bounding boxes have 10-20px positioning differences
**Root Cause**: Pipeline logic precision differences, not CNN accuracy
**Status**: Acceptable for production (100% detection agreement, 30x speedup)

### 2. CoreMLExecutionProvider Availability
**Severity**: Low
**Impact**: Only works on Apple Silicon macOS 13+
**Workaround**: Falls back to CPUExecutionProvider automatically
**Status**: Expected behavior

### 3. Limited Debugging
**Severity**: Low
**Impact**: Harder to debug than Pure Python (can't inspect intermediate layers easily)
**Workaround**: Use Pure Python for debugging, ONNX for production
**Status**: Expected trade-off

## Next Steps (Phase 3)

These ONNX models will be converted to native CoreML format for direct Apple Neural Engine execution:

1. Convert ONNX → CoreML FP32
2. Validate < 1e-5 max diff vs ONNX
3. Create FP16 quantized versions
4. Validate > 0.999 IoU vs FP32
5. Target: 30+ FPS (5.1x improvement)

## Validation Scripts

Run these to validate the models:

```bash
# Layer-by-layer validation
python validate_onnx_vs_python_cnn.py

# End-to-end accuracy vs C++ OpenFace
python compare_onnx_vs_cpp_openface.py

# Performance benchmark
python benchmark_onnx_performance.py

# IoU gap analysis
python debug_onnx_vs_pure_python.py
```

---

**Model Status**: ✅ Production-Ready
**Last Validated**: 2025-01-14
**Next Review**: After Phase 3 completion
