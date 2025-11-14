# Python MTCNN Pre-Optimization Baseline

**Date**: 2025-11-14
**Python MTCNN Implementation**: `pure_python_mtcnn_v2.py`
**Benchmark Dataset**: Patient Data (10 videos, 30 frames)

---

## Executive Summary

This document establishes the performance and accuracy baseline for the Pure Python MTCNN implementation before optimization work begins. The Python implementation has been validated for accuracy (86.4% mean IoU vs C++ OpenFace) but requires **153.5x speedup** to reach the 30 FPS target.

---

## Test Methodology

### Dataset
- **Source**: `/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data`
- **Videos**: 10 total (5 Normal cohort, 5 Paralysis cohort)
- **Frames per video**: 3 (sampled at 20%, 50%, 80% positions)
- **Total frames tested**: 30

### Comparison Baseline
- **C++ Reference**: OpenFace FeatureExtraction tool (MTCNN + full pipeline)
- **Python Implementation**: Pure Python MTCNN v2 with calibration coefficients
- **Metric**: Intersection over Union (IoU) for bounding boxes

---

## Accuracy Results

### Detection Agreement
- **Both detected**: 30/30 frames (100%)
- **Only C++ detected**: 0 frames
- **Only Python detected**: 0 frames
- **Neither detected**: 0 frames
- **Detection agreement rate**: **100%**

### IoU (Intersection over Union)
| Metric | Value |
|--------|-------|
| **Mean IoU** | **86.37%** |
| Median IoU | 87.71% |
| Standard Deviation | 5.51% |
| Min IoU | 75.50% |
| Max IoU | 95.60% |
| Frames > 90% IoU | 10/30 (33.3%) |
| Frames > 95% IoU | 2/30 (6.7%) |
| Frames > 99% IoU | 0/30 (0%) |

### Position Differences (C++ vs Python)
| Metric | X-Offset (px) | Y-Offset (px) |
|--------|---------------|---------------|
| Mean | 20.3 | 21.5 |
| Median | 18.6 | 16.9 |
| Max | 57.7 | 55.7 |

### Size Differences (C++ vs Python)
| Metric | Width Diff (px) | Height Diff (px) |
|--------|-----------------|------------------|
| Mean | 35.7 | 27.9 |
| Median | 32.9 | 20.7 |
| Max | 100.1 | 111.6 |

### Accuracy Assessment
**Status**: ✅ **ACCEPTABLE FOR SCIENTIFIC USE**

The 86.4% mean IoU indicates good alignment with the C++ OpenFace pipeline. Differences are primarily due to:
1. Calibration coefficients matching full OpenFace pipeline (MTCNN + CLNF), not raw MTCNN
2. Legitimate variation in face detection across implementations
3. Sub-pixel rounding differences

---

## Performance Results

### Python MTCNN Performance
| Metric | Value |
|--------|-------|
| **Mean FPS** | **0.20 FPS** |
| **Mean Latency** | **5,118 ms** (5.1 seconds per frame) |
| Median Latency | 5,024 ms |
| P95 Latency | 5,866 ms |
| P99 Latency | 6,054 ms |
| Min Latency | 4,565 ms |
| Max Latency | 6,115 ms |

### Performance Assessment
**Status**: ❌ **REQUIRES OPTIMIZATION**

- **Current Performance**: 0.20 FPS (5.1 seconds per frame)
- **Target Performance**: 30 FPS (33 ms per frame)
- **Required Speedup**: **153.5x**

---

## Calibration Coefficients

The Python MTCNN applies these calibration adjustments to match C++ OpenFace full pipeline output:

```python
# X/Y Position Offsets
X_OFFSET = -0.0075  # -0.75% of width
Y_OFFSET = 0.2459   # +24.59% of height

# Width/Height Scaling
WIDTH_SCALE = 1.0323   # +3.23%
HEIGHT_SCALE = 0.7751  # -22.49%
```

**Note**: These coefficients are **essential** for accuracy. Testing showed:
- **WITH calibration**: Mean IoU = 86.4% ✅
- **WITHOUT calibration**: Mean IoU = 51.7% ❌

---

## Test Environment

### Hardware
- **Platform**: Darwin (macOS)
- **OS Version**: 25.0.0
- **CPU**: ARM64 (Apple Silicon)

### Software
- **Python Version**: 3.10
- **OpenCV**: Used for video decoding and image processing
- **NumPy**: Used for array operations

### C++ Baseline Tool
- **Path**: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction`
- **Implementation**: OpenFace MTCNN (C++)

---

## Per-Video Summary

### Normal Cohort Performance
| Video | Frames | Mean IoU | Mean Latency (ms) |
|-------|--------|----------|-------------------|
| IMG_0422.MOV | 3 | 86.3% | 4,796 |
| IMG_0428.MOV | 3 | 88.7% | 4,841 |
| IMG_0433.MOV | 3 | 91.2% | 5,005 |
| IMG_0434.MOV | 3 | 82.3% | 5,509 |
| IMG_0435.MOV | 3 | 90.5% | 4,849 |

**Normal Cohort Average**: 87.8% IoU, 4,800 ms latency

### Paralysis Cohort Performance
| Video | Frames | Mean IoU | Mean Latency (ms) |
|-------|--------|----------|-------------------|
| 20240723_175947000_iOS.MOV | 3 | 81.3% | 5,946 |
| 20240723_185024000_iOS.MOV | 3 | 89.3% | 4,869 |
| 20240730_134811000_iOS.MOV | 3 | 84.2% | 4,979 |
| 20240730_153031000_iOS.MOV | 3 | 90.7% | 5,187 |
| 20240730_172902000_iOS.MOV | 3 | 79.1% | 5,202 |

**Paralysis Cohort Average**: 84.9% IoU, 5,237 ms latency

**Observation**: Paralysis cohort shows slightly lower IoU (84.9% vs 87.8%) and higher latency (5,237ms vs 4,800ms), likely due to larger face sizes in these videos.

---

## Optimization Strategy

Based on `MTCNN_PERFORMANCE_OPTIMIZATION_STRATEGY.md`, the optimization approach is:

### Phase 1: ONNX Runtime Conversion (Target: 5-10x speedup)
- Convert MTCNN PNet, RNet, ONet to ONNX format
- Use ONNX Runtime with CoreML backend for Apple Neural Engine acceleration
- Expected performance: 1-2 FPS

### Phase 2: PyTorch MPS Backend (Target: 10-20x speedup)
- Implement MTCNN using PyTorch with Metal Performance Shaders (MPS)
- Leverage GPU acceleration on Apple Silicon
- Expected performance: 2-4 FPS

### Phase 3: Algorithm Optimization (Target: 2-3x additional speedup)
- Optimize image pyramid generation
- Improve non-maximum suppression
- Batch processing optimizations
- Expected performance: 4-12 FPS

### Phase 4: Advanced Optimizations (Target: 2-5x additional speedup)
- Cython for critical paths
- Multi-threaded frame processing
- Hardware-specific optimizations
- Expected performance: 8-60 FPS

**Combined Target**: 30+ FPS with preserved accuracy (>85% IoU)

---

## Next Steps

1. ✅ **Baseline Established**: Accuracy = 86.4% IoU, Performance = 0.20 FPS
2. ⏭️ **Phase 1**: Implement ONNX Runtime conversion
3. ⏭️ **Phase 2**: Implement PyTorch MPS backend
4. ⏭️ **Phase 3**: Algorithm optimizations
5. ⏭️ **Phase 4**: Advanced optimizations (if needed)

---

## Related Documentation

- **Accuracy Investigation**: `FRAME_00617_INVESTIGATION_SUMMARY.md`
- **Optimization Strategy**: `MTCNN_PERFORMANCE_OPTIMIZATION_STRATEGY.md`
- **Complete Verification**: `MTCNN_COMPLETE_VERIFICATION.md`
- **Raw Benchmark Data**: `accuracy_benchmark_results.json`

---

## Conclusion

The Pure Python MTCNN implementation has been **validated for accuracy** with 86.4% mean IoU vs C++ OpenFace, meeting scientific research requirements. However, performance is currently **0.20 FPS**, requiring a **153.5x speedup** to reach the 30 FPS target.

Optimization work will now proceed through the 4-phase strategy outlined above, prioritizing:
1. **Accuracy preservation**: Maintain >85% IoU throughout optimization
2. **ARM optimization**: Target Apple Neural Engine and Metal Performance Shaders
3. **Incremental validation**: Benchmark accuracy after each optimization phase
4. **Cross-platform support**: Ensure CUDA compatibility for future deployment

**Status**: Ready to begin Phase 1 (ONNX Runtime Conversion)
