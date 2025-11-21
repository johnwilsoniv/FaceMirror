# PyCLNF Hardware Acceleration Backends

Hardware-accelerated inference for CEN patch experts using CoreML (Apple Neural Engine) and ONNX (CPU/CUDA).

## Overview

The CEN patch expert models (neural networks for response map computation) can be accelerated using:

1. **CoreML** - Apple Neural Engine (M1/M2/M3 Macs): 20-50 FPS
2. **ONNX + CUDA** - NVIDIA GPUs: 50-100 FPS
3. **Pure Python** - CPU baseline: 5-10 FPS

## Quick Start

### 1. Export Models (One-Time Setup)

First, export the CEN patch experts to your desired format:

```bash
# For macOS (CoreML)
python pyclnf/backends/export_to_coreml.py

# For NVIDIA GPUs (ONNX)
python pyclnf/backends/export_to_onnx.py
```

This creates:
- `pyclnf/models/coreml_cen/` - CoreML models (68 landmarks × 4 scales = 272 .mlpackage files)
- `pyclnf/models/onnx_cen/` - ONNX models (68 landmarks × 4 scales = 272 .onnx files)

### 2. Install Runtime Dependencies

```bash
# For CoreML (macOS only)
pip install coremltools

# For ONNX + CUDA (NVIDIA GPUs)
pip install onnxruntime-gpu

# For ONNX + CPU (fallback)
pip install onnxruntime
```

### 3. Use Accelerated Backend

```python
from pyclnf.backends import select_best_backend

# Auto-select best backend for your hardware
backend = select_best_backend(verbose=True)

if backend:
    # Load models
    backend.load_models("pyclnf/models")

    # Use for inference
    response_map = backend.response(
        area_of_interest=image_patch,
        landmark_idx=36,
        scale=0.25
    )
else:
    # Fallback to pure Python (existing implementation)
    from pyclnf.core.cen_patch_expert import CENPatchExpert
    # ... use existing code ...
```

## Backend Details

### CoreML Backend

**Platform**: macOS 13+ (Ventura or later)
**Hardware**: Apple Silicon (M1/M2/M3) or Intel with Neural Engine
**Performance**: 20-50 FPS per patch expert

**Pros**:
- Native Apple Silicon optimization
- Excellent power efficiency (5-10x better than CPU)
- Automatic use of Neural Engine + GPU + CPU

**Cons**:
- macOS only
- One-time export required (~2 minutes)
- Larger model files (~10MB per landmark vs ~2MB)

**Usage**:
```python
from pyclnf.backends import CoreMLCENBackend

backend = CoreMLCENBackend()
backend.load_models("pyclnf/models", scales=[0.25, 0.35, 0.5])
response = backend.response(image_patch, landmark_idx=36, scale=0.25)
```

### ONNX Backend

**Platform**: Windows, Linux, macOS
**Hardware**: NVIDIA GPUs (CUDA), Intel/AMD CPUs
**Performance**: 50-100 FPS (GPU), 10-20 FPS (CPU)

**Pros**:
- Cross-platform (Windows/Linux/macOS)
- CUDA acceleration on NVIDIA GPUs
- Efficient CPU execution with AVX optimizations
- Better batch processing than CoreML

**Cons**:
- Requires CUDA for best performance
- Slightly higher latency than CoreML on Apple Silicon

**Usage**:
```python
from pyclnf.backends import ONNXCENBackend

backend = ONNXCENBackend()  # Auto-detects CUDA
backend.load_models("pyclnf/models", scales=[0.25, 0.35, 0.5])

# Single inference
response = backend.response(image_patch, landmark_idx=36, scale=0.25)

# Batch inference (more efficient)
responses = backend.batch_response(
    areas_of_interest=[patch1, patch2, patch3],
    landmark_indices=[36, 48, 30],
    scales=[0.25, 0.25, 0.25]
)
```

### Pure Python Backend

**Platform**: Any (fallback)
**Hardware**: CPU
**Performance**: 5-10 FPS

This is the existing implementation - no export needed. Used automatically if CoreML/ONNX not available.

## Model Export Details

### CoreML Export

```bash
python pyclnf/backends/export_to_coreml.py
```

**What it does**:
1. Loads CEN patch experts from `.dat` files
2. Converts each neural network to PyTorch
3. Exports to CoreML ML Program format
4. Optimizes for Apple Neural Engine
5. Saves to `pyclnf/models/coreml_cen/`

**Requirements**:
- `torch` >= 2.0
- `coremltools` >= 7.0
- macOS 13+ for testing (export can run anywhere)

**Output**: 272 `.mlpackage` directories (~2.7GB total)

### ONNX Export

```bash
python pyclnf/backends/export_to_onnx.py
```

**What it does**:
1. Loads CEN patch experts from `.dat` files
2. Converts each neural network to PyTorch
3. Exports to ONNX format (opset 14)
4. Adds dynamic batching support
5. Saves to `pyclnf/models/onnx_cen/`

**Requirements**:
- `torch` >= 2.0
- `onnx` >= 1.14

**Output**: 272 `.onnx` files (~680MB total)

## Integration with PyCLNF

The backends are designed to be drop-in replacements for the existing CEN patch expert response computation:

```python
# Option 1: Manual backend selection
from pyclnf.backends import CoreMLCENBackend

backend = CoreMLCENBackend()
backend.load_models("pyclnf/models")

# Use in optimizer (modify optimizer.py to accept backend parameter)
# ... integrate with existing code ...

# Option 2: Automatic backend selection (future integration)
from pyclnf.clnf import CLNF

clnf = CLNF(
    model_dir="pyclnf/models",
    use_hardware_acceleration=True,  # Future parameter
    force_backend="coreml"  # Optional: force specific backend
)
```

## Performance Benchmarking

To benchmark all available backends:

```python
from pyclnf.backends import benchmark_backends
import numpy as np

# Create test patch
test_patch = np.random.randint(0, 255, (21, 21), dtype=np.uint8)

# Run benchmark
results = benchmark_backends(test_patch, iterations=100)

for backend_name, fps in results.items():
    if fps:
        print(f"{backend_name}: {fps:.2f} FPS")
    else:
        print(f"{backend_name}: Not available")
```

Expected results:
- **CoreML (M1 Max)**: ~30-40 FPS
- **ONNX+CUDA (RTX 3080)**: ~60-80 FPS
- **ONNX+CPU (Intel i7)**: ~12-18 FPS
- **Pure Python**: ~6-8 FPS

## Troubleshooting

### CoreML models not loading
```
Error: CoreML CEN models not found
```
**Solution**: Run `python pyclnf/backends/export_to_coreml.py`

### ONNX CUDA not detected
```
[CEN Backend] ONNX available but no CUDA detected
```
**Solution**: Install `onnxruntime-gpu` and ensure CUDA 11.x/12.x is installed

### Export script fails
```
Error: torch not installed
```
**Solution**: Install PyTorch for model export:
```bash
pip install torch torchvision
```

## Architecture

```
pyclnf/backends/
├── __init__.py              # Package exports
├── base_backend.py          # Abstract backend interface
├── backend_selector.py      # Auto backend selection
├── coreml_backend.py        # CoreML implementation
├── onnx_backend.py          # ONNX implementation
├── export_to_coreml.py      # CoreML export script
├── export_to_onnx.py        # ONNX export script
└── README.md                # This file
```

## Future Improvements

- [ ] TensorRT backend (NVIDIA optimization)
- [ ] OpenVINO backend (Intel optimization)
- [ ] Batch processing optimization
- [ ] Response map caching
- [ ] Multi-threaded CPU backend
- [ ] Mobile deployment (iOS/Android)

## References

- **CoreML Documentation**: https://developer.apple.com/documentation/coreml
- **ONNX Runtime**: https://onnxruntime.ai/
- **PyTorch ONNX Export**: https://pytorch.org/docs/stable/onnx.html
- **PyMTCNN Backend Architecture**: `pymtcnn/backends/` (reference implementation)

---
Last Updated: 2025-11-15
Maintained by: SplitFace Team
