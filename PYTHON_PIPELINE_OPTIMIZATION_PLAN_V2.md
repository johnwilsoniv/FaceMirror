# Python AU Pipeline Optimization Plan v2.0

## Current Performance (Actual Benchmarks)

### Baseline Metrics
- **Accuracy**: 90.2% AU classification accuracy (vs OpenFace C++)
- **Landmark Error**: 2.05 pixels mean error
- **Speed**: 1.2 FPS (830ms/frame after warmup)
- **OpenFace C++**: 10.1 FPS (99ms/frame)
- **Speed Gap**: 8.4x slower than C++
- **Target**: Need 25x speedup for real-time (30 FPS)

### Component Timing Breakdown (estimated from 830ms total)
Based on typical pipeline distribution:
1. **MTCNN Detection**: ~250ms (30%)
2. **CLNF Landmark Fitting**: ~400ms (48%)
3. **AU Prediction**: ~150ms (18%)
4. **Overhead/Other**: ~30ms (4%)

## Optimization Strategy - Phased Approach

### 🚀 Phase 1: Quick Wins (2-3 days) - Target 2x Speedup

#### 1.1 Profile and Measure
```bash
# Run detailed profiling
python profile_au_pipeline.py

# Analyze with line_profiler for hot spots
kernprof -l -v benchmark_au_pipeline.py
```

#### 1.2 CLNF Optimizations (Primary Target)
**Current**: ~400ms per frame
**Target**: 200ms per frame

```python
# A. Add Numba JIT compilation for hot loops
from numba import jit, prange

@jit(nopython=True, parallel=True)
def compute_response_maps_parallel(image, patch_experts, landmarks):
    """Parallelize response map computation across landmarks."""
    responses = np.empty((68, height, width))
    for i in prange(68):  # Parallel loop
        responses[i] = compute_single_response(image, patch_experts[i])
    return responses

# B. Reduce CLNF iterations
class OptimizedCLNF:
    def __init__(self):
        self.max_iterations = 3  # Reduced from 5-10
        self.early_stop_threshold = 0.5  # Stop if converged
        self.use_previous_frame = True  # Better initialization
```

#### 1.3 MTCNN Acceleration
**Current**: ~250ms per frame
**Target**: 100ms per frame

```python
# A. Skip detection on consecutive frames
class TemporalMTCNN:
    def __init__(self):
        self.last_bbox = None
        self.skip_frames = 2  # Detect every 3rd frame

    def detect_with_tracking(self, frame, frame_idx):
        if frame_idx % self.skip_frames == 0:
            return self.full_detection(frame)
        else:
            return self.track_bbox(frame, self.last_bbox)

# B. Reduce MTCNN pyramid levels
detector = MTCNN(
    min_face_size=40,  # Increased from 20
    scale_factor=0.8,   # Reduced from 0.709
    thresholds=[0.7, 0.8, 0.9]  # More aggressive
)
```

#### 1.4 Caching and Memoization
```python
from functools import lru_cache

class CachedAUPipeline:
    @lru_cache(maxsize=128)
    def compute_hog_features(self, face_hash):
        """Cache HOG features for similar faces."""
        return extract_hog(face)

    def process_frame_cached(self, frame):
        # Use perceptual hashing to detect similar frames
        frame_hash = compute_dhash(frame)
        if self.is_similar(frame_hash, self.last_hash):
            return self.interpolate_aus(self.last_aus)
```

**Expected Impact**: 2x overall speedup (830ms → 415ms)

### 🔥 Phase 2: Backend Acceleration (1 week) - Target 4x Total Speedup

#### 2.1 ONNX/CoreML Conversion
```python
# Convert MTCNN to ONNX
import torch
import onnx

def convert_mtcnn_to_onnx():
    """Convert PyTorch MTCNN to ONNX format."""
    # Export P-Net, R-Net, O-Net separately
    dummy_input = torch.randn(1, 3, 12, 12)
    torch.onnx.export(pnet_model, dummy_input, "pnet.onnx")

    # Use ONNX Runtime for inference
    import onnxruntime as ort

    class ONNXMtcnn:
        def __init__(self):
            self.pnet = ort.InferenceSession("pnet.onnx")
            self.rnet = ort.InferenceSession("rnet.onnx")
            self.onet = ort.InferenceSession("onet.onnx")

# CoreML for Apple Silicon
import coremltools as ct

def convert_to_coreml():
    """Convert for Apple Silicon acceleration."""
    model = ct.convert(
        "mtcnn.onnx",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL  # Use Neural Engine
    )
    model.save("mtcnn.mlmodel")
```

#### 2.2 Cython Implementation for CLNF Core
```cython
# clnf_optimizer.pyx
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, exp

cpdef tuple optimize_landmarks_cython(
    cnp.ndarray[cnp.float64_t, ndim=2] image,
    cnp.ndarray[cnp.float64_t, ndim=2] landmarks,
    object patch_experts,
    int max_iterations=3
):
    """Cython-optimized CLNF fitting."""
    cdef int iteration
    cdef double convergence
    cdef cnp.ndarray[cnp.float64_t, ndim=2] responses

    for iteration in range(max_iterations):
        # Compute response maps in parallel
        responses = compute_responses_parallel(image, landmarks)

        # Update landmarks
        landmarks = update_landmarks_fast(responses, landmarks)

        # Check convergence
        convergence = compute_convergence(landmarks)
        if convergence < 0.5:
            break

    return landmarks, iteration
```

#### 2.3 GPU Acceleration (Optional)
```python
# Use CuPy for GPU arrays
import cupy as cp

class GPUAcceleratedCLNF:
    def compute_responses_gpu(self, image, landmarks):
        """Compute response maps on GPU."""
        image_gpu = cp.asarray(image)
        landmarks_gpu = cp.asarray(landmarks)

        # GPU kernel for response computation
        responses = cp.empty((68, h, w))
        # ... GPU computation ...

        return cp.asnumpy(responses)
```

**Expected Impact**: 4x total speedup (830ms → 207ms)

### ⚡ Phase 3: Algorithmic Optimizations (2 weeks) - Target 8x Total Speedup

#### 3.1 Temporal Coherence for Video
```python
class TemporalAUPipeline:
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.optical_flow = cv2.DISOpticalFlow_create()

    def process_video_temporal(self, frames):
        """Process video with temporal coherence."""
        results = []

        for i, frame in enumerate(frames):
            if i == 0:
                # Full processing for first frame
                result = self.full_process(frame)
            else:
                # Predict from previous frame
                predicted_landmarks = self.kalman_filter.predict()

                # Optical flow refinement
                flow = self.optical_flow.calc(frames[i-1], frame)
                refined_landmarks = self.refine_with_flow(
                    predicted_landmarks, flow
                )

                # Quick CLNF update (1-2 iterations only)
                final_landmarks = self.quick_clnf_update(
                    frame, refined_landmarks
                )

                # Update Kalman filter
                self.kalman_filter.update(final_landmarks)

                result = {'landmarks': final_landmarks}

        results.append(result)
        return results
```

#### 3.2 Multi-Scale Processing
```python
class MultiScaleProcessor:
    def process_adaptive(self, frame, quality_mode='balanced'):
        """Adaptive quality based on requirements."""

        if quality_mode == 'fast':
            # Downsample for speed
            small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            landmarks = self.detect_landmarks(small_frame)
            landmarks *= 2  # Scale back up

        elif quality_mode == 'balanced':
            # Standard processing
            landmarks = self.detect_landmarks(frame)

        elif quality_mode == 'accurate':
            # Multi-scale fusion for accuracy
            scales = [1.0, 0.75, 0.5]
            all_landmarks = []
            for scale in scales:
                scaled = cv2.resize(frame, None, fx=scale, fy=scale)
                lm = self.detect_landmarks(scaled)
                all_landmarks.append(lm / scale)
            landmarks = np.mean(all_landmarks, axis=0)

        return landmarks
```

#### 3.3 Model Quantization and Pruning
```python
# Quantize neural networks to INT8
def quantize_models():
    """Quantize models for faster inference."""

    # Dynamic quantization for CPU
    quantized_model = torch.quantization.quantize_dynamic(
        mtcnn_model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )

    # Static quantization for better performance
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    prepared = torch.quantization.prepare(model)
    quantized = torch.quantization.convert(prepared)

    return quantized
```

**Expected Impact**: 8x total speedup (830ms → 104ms), approaching C++ performance

## Implementation Roadmap

### Week 1: Foundation
- [ ] Day 1-2: Implement profiling and establish baseline
- [ ] Day 3-4: Apply Numba to CLNF hot loops
- [ ] Day 5: Implement frame skipping for MTCNN

### Week 2: Acceleration
- [ ] Day 1-2: Convert MTCNN to ONNX/CoreML
- [ ] Day 3-4: Implement Cython CLNF optimizer
- [ ] Day 5: Test and validate accuracy preservation

### Week 3: Advanced Optimizations
- [ ] Day 1-2: Implement temporal coherence
- [ ] Day 3-4: Add multi-scale processing modes
- [ ] Day 5: Final benchmarking and tuning

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Speed (FPS) | 1.2 | 10+ | ⏳ |
| vs C++ Gap | 8.4x | <2x | ⏳ |
| AU Accuracy | 90.2% | >90% | ✅ |
| Landmark Error | 2.05px | <3px | ✅ |
| Memory Usage | TBD | <1GB | ⏳ |

## Risk Mitigation

1. **Accuracy Loss**: Test after each optimization, rollback if >1% drop
2. **Platform Compatibility**: Maintain fallback to pure Python
3. **Complexity**: Keep optimized and reference implementations
4. **Debugging**: Comprehensive logging and profiling hooks

## Validation Strategy

```python
def validate_optimization(original_fn, optimized_fn, test_frames):
    """Ensure optimizations maintain accuracy."""

    for frame in test_frames:
        original_result = original_fn(frame)
        optimized_result = optimized_fn(frame)

        # Check AU accuracy
        au_diff = compare_aus(original_result, optimized_result)
        assert au_diff < 0.01, "AU accuracy degraded"

        # Check landmark accuracy
        lm_error = np.mean(np.abs(
            original_result['landmarks'] -
            optimized_result['landmarks']
        ))
        assert lm_error < 0.5, "Landmark accuracy degraded"

    print("✅ Optimization validated!")
```

## Next Immediate Actions

1. **Run profiling script** to identify exact bottlenecks
2. **Implement Numba optimization** for CLNF (biggest impact)
3. **Test ONNX conversion** for MTCNN
4. **Measure improvement** after each optimization
5. **Document performance gains** for each change

## Expected Final Performance

With all optimizations:
- **Target FPS**: 10-15 FPS (approaching C++ OpenFace)
- **Speedup**: 8-10x from current baseline
- **Accuracy**: Maintained at >90% AU classification
- **Use Cases**: Real-time for single face, batch processing for multiple faces