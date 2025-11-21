# Python AU Pipeline Optimization Strategy

## Current Performance Baseline
- **Accuracy:** 91.85% correlation with C++ OpenFace
- **Speed:** Significantly slower than C++ (needs benchmarking)
- **Memory:** High usage due to Python overhead

## Performance Bottlenecks (Ordered by Impact)

### 1. 🔥 CLNF Landmark Detection (60-70% of runtime)
**Current Issues:**
- Pure Python implementation of iterative optimization
- Multiple iterations (5-10) per frame
- Response map generation for 68 landmarks
- Matrix operations in NumPy instead of optimized C++

**Optimization Opportunities:**
- **Numba JIT compilation** for hot loops (30-50% speedup)
- **Cython implementation** of core CLNF optimizer (2-3x speedup)
- **Reduce iterations** with better initialization (20% speedup)
- **Vectorize response map generation** using NumPy broadcasting
- **Cache patch experts** between frames (10% speedup)

### 2. 🔥 HOG Feature Extraction (15-20% of runtime)
**Current Issues:**
- pyfhog uses dlib's implementation (already C++)
- Multiple image preprocessing steps
- Face alignment overhead

**Optimization Opportunities:**
- **Batch processing** for multiple faces (when applicable)
- **Skip redundant BGR conversions**
- **Use SIMD optimizations** via OpenCV's HOG
- **Precompute cell histograms** for video (temporal coherence)

### 3. 🔥 Face Detection (10-15% of runtime)
**Current Issues:**
- MTCNN runs 3 neural networks sequentially
- Pure Python implementation in pymtcnn
- No hardware acceleration

**Optimization Opportunities:**
- **ONNX Runtime** with GPU/CoreML acceleration (5-10x speedup)
- **TensorRT** for NVIDIA GPUs (10x speedup)
- **Batch multiple scales** in P-Net
- **Skip detection** for continuous video (track instead)
- **YOLOv8-face** as faster alternative (3x speedup, similar accuracy)

### 4. Running Median Tracker (5% of runtime)
**Current Issues:**
- Updates histogram every 2 frames
- Full histogram recomputation

**Optimization Opportunities:**
- **Incremental histogram updates** (50% speedup)
- **Skip updates** when face is stable
- **Use approximate median** for speed

## Proposed Optimization Phases

### Phase 1: Quick Wins (1-2 days)
1. **Profile the pipeline** to confirm bottlenecks
   ```python
   # Add cProfile/line_profiler to identify exact slow points
   python -m cProfile -o profile.stats run_pipeline.py
   ```

2. **Numba acceleration** for CLNF hot loops
   ```python
   @numba.jit(nopython=True, parallel=True)
   def compute_response_maps(...):
       # Parallelize landmark response computation
   ```

3. **Reduce redundant computations**
   - Cache face detection for video
   - Skip HOG recomputation for static frames
   - Reuse transformation matrices

### Phase 2: Backend Acceleration (3-5 days)
1. **ONNX/CoreML for neural networks**
   - Convert MTCNN to ONNX
   - Use CoreML on macOS for hardware acceleration
   - TensorRT on NVIDIA GPUs

2. **Cython for CLNF core**
   ```cython
   # Implement critical paths in Cython
   cdef double[:,:] optimize_landmarks(...)
   ```

3. **Parallel processing**
   - Process multiple faces in parallel
   - Use multiprocessing for batch videos

### Phase 3: Algorithmic Optimizations (1 week)
1. **Adaptive quality modes**
   ```python
   class QualityMode(Enum):
       FAST = "fast"        # 2 CLNF iterations, skip every 2nd frame
       BALANCED = "balanced" # 3 iterations, full processing
       ACCURATE = "accurate" # 5 iterations, maximum quality
   ```

2. **Temporal coherence for video**
   - Kalman filter for landmark tracking
   - Optical flow for inter-frame prediction
   - Skip CLNF on stable faces

3. **Model quantization**
   - INT8 quantization for neural networks
   - Reduced precision for non-critical computations

## Implementation Priority Matrix

| Optimization | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| Numba for CLNF | High | Low | 🟥 P0 |
| ONNX acceleration | High | Medium | 🟥 P0 |
| Skip redundant detection | Medium | Low | 🟧 P1 |
| Cython core | High | High | 🟧 P1 |
| Temporal coherence | High | High | 🟨 P2 |
| Model quantization | Medium | Medium | 🟨 P2 |

## Expected Performance Gains

### Conservative Estimates:
- **Phase 1:** 30-40% speedup
- **Phase 2:** 2-3x total speedup
- **Phase 3:** 3-5x total speedup

### Target Performance:
- **Goal:** Within 2x of C++ performance
- **Maintain:** >90% accuracy correlation
- **Real-time:** 30+ FPS for single face on modern CPU

## Benchmarking Strategy

```python
# Benchmark each component separately
def benchmark_pipeline():
    times = {
        'detection': [],
        'landmarks': [],
        'hog': [],
        'au_prediction': [],
        'total': []
    }

    for frame in video:
        t0 = time.perf_counter()

        # Measure each component
        face = detect_face(frame)  # Time this
        landmarks = detect_landmarks(face)  # Time this
        hog = extract_hog(aligned_face)  # Time this
        aus = predict_aus(hog, geom)  # Time this

        times['total'].append(time.perf_counter() - t0)

    return times
```

## Hardware Acceleration Options

### macOS (Apple Silicon)
- CoreML for neural networks
- Metal Performance Shaders for parallel compute
- Accelerate.framework for BLAS operations

### NVIDIA GPUs
- CUDA for parallel landmark optimization
- TensorRT for neural network inference
- cuDNN for optimized convolutions

### Intel CPUs
- MKL for optimized linear algebra
- OpenVINO for neural network inference
- AVX-512 for vectorized operations

## Next Steps

1. **Create benchmark suite** to measure current performance
2. **Implement Phase 1 optimizations** (Numba + caching)
3. **Test accuracy preservation** after each optimization
4. **Profile again** to identify new bottlenecks
5. **Iterate** based on results

## Success Metrics

- [ ] 3x speedup while maintaining >90% accuracy
- [ ] Real-time processing (30 FPS) for single face
- [ ] Memory usage under 1GB for video processing
- [ ] Clean API for quality/speed tradeoffs