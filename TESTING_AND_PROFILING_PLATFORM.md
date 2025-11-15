# Comprehensive Testing & Profiling Platform

## Overview

A unified testing and profiling infrastructure for PyMTCNN, PyFaceAU, and related face analysis projects.

## Goals

1. **Accuracy Validation** - Ensure implementations match reference implementations
2. **Performance Profiling** - Identify bottlenecks and optimization opportunities
3. **Cross-Platform Testing** - Verify functionality on CUDA/CoreML/CPU backends
4. **Regression Detection** - Catch performance/accuracy regressions early
5. **Benchmark Tracking** - Track performance over time

---

## Proposed Architecture

### 1. Test Dataset Repository

**Purpose:** Centralized test data for all projects

**Structure:**
```
face-analysis-test-data/
├── faces/
│   ├── single_face/           # Images with 1 face
│   ├── multiple_faces/        # Images with 2+ faces
│   ├── challenging/           # Occlusion, rotation, lighting
│   └── edge_cases/            # Very small, very large, profile views
├── videos/
│   ├── short_clips/           # 30-60 frames
│   ├── medium_clips/          # 300-600 frames
│   └── long_clips/            # 1800+ frames
├── ground_truth/
│   ├── face_detections.json   # Bbox annotations
│   ├── landmarks.json         # 68-point landmarks
│   └── action_units.csv       # OpenFace C++ AU outputs
└── reference_outputs/
    ├── openface_cpp/          # C++ OpenFace outputs
    ├── mtcnn_official/        # Official MTCNN outputs
    └── facenet_pytorch/       # FaceNet-PyTorch outputs
```

**Benefits:**
- Single source of truth for all projects
- Version controlled test data
- Easy to add new test cases
- Shared across PyMTCNN, PyFaceAU, PyFHOG

**Implementation:**
- Git LFS for large files
- Automated download scripts
- Checksums for data integrity

---

### 2. Unified Testing Framework

**Components:**

#### A. Accuracy Testing

**PyMTCNN Accuracy Tests:**
```python
# tests/accuracy/test_mtcnn_accuracy.py

class TestMTCNNAccuracy:
    """Compare PyMTCNN outputs vs reference implementations"""

    def test_pnet_vs_facenet_pytorch(self):
        """PNet layer outputs should match FaceNet-PyTorch"""
        assert correlation > 0.99

    def test_bbox_vs_official_mtcnn(self):
        """Bboxes should match official MTCNN"""
        assert iou > 0.95

    def test_landmark_accuracy(self):
        """5-point landmarks should be accurate"""
        assert mean_error < 2.0  # pixels
```

**PyFaceAU Accuracy Tests:**
```python
# tests/accuracy/test_pyfaceau_accuracy.py

class TestPyFaceAUAccuracy:
    """Compare PyFaceAU outputs vs C++ OpenFace"""

    def test_au_correlation(self):
        """AU predictions should correlate r > 0.92 with C++ OpenFace"""
        for au in ALL_AUS:
            assert correlation[au] > 0.85

    def test_calc_params_fidelity(self):
        """3D pose estimation should match C++ CalcParams"""
        assert params_correlation > 0.98

    def test_hog_features(self):
        """HOG features should match C++ OpenFace"""
        assert feature_correlation > 0.99
```

#### B. Performance Profiling

**Multi-Level Profiling:**

```python
# tests/profiling/profile_suite.py

class PerformanceProfiler:
    """Comprehensive performance profiling"""

    def profile_end_to_end(self):
        """Measure overall pipeline FPS"""
        # Returns: FPS, total time, frame breakdown

    def profile_per_stage(self):
        """Measure each pipeline stage"""
        # Returns: {
        #   'face_detection': 29.5ms,
        #   'landmarks': 8.3ms,
        #   'pose_estimation': 12.1ms,
        #   'alignment': 5.2ms,
        #   'hog_extraction': 15.7ms,
        #   'au_prediction': 3.2ms,
        # }

    def profile_memory_usage(self):
        """Track memory consumption"""
        # Returns: peak memory, allocations, leaks

    def profile_gpu_utilization(self):
        """CUDA/CoreML utilization"""
        # Returns: GPU usage %, memory bandwidth
```

**Profiling Tools:**
- `cProfile` for function-level profiling
- `line_profiler` for line-by-line profiling
- `memory_profiler` for memory usage
- `py-spy` for sampling profiler (zero overhead)
- `nvidia-smi` for GPU monitoring
- `instruments` for CoreML profiling (macOS)

#### C. Cross-Platform Testing

**Backend Matrix Testing:**

```python
# tests/backends/test_all_backends.py

@pytest.mark.parametrize("backend", ["cuda", "coreml", "cpu", "onnx"])
class TestBackends:
    """Verify all backends produce identical results"""

    def test_backend_consistency(self, backend):
        """All backends should produce same outputs"""
        detector = MTCNN(backend=backend)
        bboxes, landmarks = detector.detect(test_image)

        # Compare against reference
        assert_allclose(bboxes, reference_bboxes, rtol=1e-4)
        assert_allclose(landmarks, reference_landmarks, rtol=1e-4)

    def test_backend_performance(self, backend):
        """Record performance for each backend"""
        fps = benchmark_backend(backend)
        # Store in results database
```

**CI/CD Integration:**
- GitHub Actions matrix builds
- Test on: Ubuntu (CUDA), macOS (CoreML), Windows (CPU)
- Nightly performance benchmarks

---

### 3. Benchmark Tracking System

**Database Schema:**

```sql
-- benchmarks.db

CREATE TABLE benchmark_runs (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    commit_hash VARCHAR(40),
    project VARCHAR(50),  -- 'pymtcnn', 'pyfaceau', etc.
    platform VARCHAR(50),  -- 'macos-m3', 'ubuntu-cuda', etc.
    backend VARCHAR(20)    -- 'cuda', 'coreml', 'cpu'
);

CREATE TABLE accuracy_results (
    run_id INTEGER,
    metric VARCHAR(100),   -- 'au01_correlation', 'bbox_iou', etc.
    value FLOAT,
    FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
);

CREATE TABLE performance_results (
    run_id INTEGER,
    stage VARCHAR(100),    -- 'face_detection', 'full_pipeline', etc.
    fps FLOAT,
    latency_ms FLOAT,
    memory_mb FLOAT,
    FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
);
```

**Visualization Dashboard:**

```python
# dashboard/app.py (Streamlit or Gradio)

import streamlit as st
import plotly.express as px

st.title("Face Analysis Benchmark Dashboard")

# Plot 1: Performance over time
fig = px.line(
    performance_data,
    x='date', y='fps',
    color='backend',
    title='PyMTCNN Performance Trends'
)
st.plotly_chart(fig)

# Plot 2: Accuracy by AU
fig = px.bar(
    accuracy_data,
    x='action_unit', y='correlation',
    title='PyFaceAU Accuracy vs C++ OpenFace'
)
st.plotly_chart(fig)

# Plot 3: Cross-platform comparison
fig = px.box(
    backend_data,
    x='backend', y='fps',
    title='Backend Performance Distribution'
)
st.plotly_chart(fig)
```

---

### 4. Automated Testing Pipeline

**GitHub Actions Workflow:**

```yaml
# .github/workflows/comprehensive-testing.yml

name: Comprehensive Testing

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # Nightly benchmarks

jobs:
  accuracy-tests:
    strategy:
      matrix:
        platform: [ubuntu-latest, macos-latest, windows-latest]
        python: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
      - name: Download test data
        run: python scripts/download_test_data.py

      - name: Run accuracy tests
        run: pytest tests/accuracy/ -v --cov

      - name: Upload results
        run: python scripts/upload_benchmark_results.py

  performance-profiling:
    runs-on: [self-hosted, gpu]  # Use GPU runner

    steps:
      - name: Profile CUDA backend
        run: |
          python tests/profiling/profile_suite.py --backend cuda
          nvidia-smi --query-gpu=utilization.gpu --format=csv

      - name: Profile CPU backend
        run: python tests/profiling/profile_suite.py --backend cpu

      - name: Generate reports
        run: python scripts/generate_performance_report.py

  regression-detection:
    needs: [accuracy-tests, performance-profiling]

    steps:
      - name: Compare against baseline
        run: python scripts/detect_regressions.py

      - name: Post comment if regression detected
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: '⚠️ Performance regression detected!'
            })
```

---

### 5. Profiling Tools & Scripts

**A. Quick Profile Script:**

```python
# scripts/quick_profile.py

"""Quick profiling for development"""

import cProfile
import pstats
from pymtcnn import MTCNN

def profile_detection():
    detector = MTCNN(backend='cuda')

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        detector.detect(test_image)

    profiler.disable()

    # Print top 20 slowest functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == '__main__':
    profile_detection()
```

**B. Memory Profiler:**

```python
# scripts/memory_profile.py

from memory_profiler import profile

@profile
def process_video():
    pipeline = FullPythonAUPipeline(...)
    results = pipeline.process_video('test.mp4')
    return results

if __name__ == '__main__':
    process_video()
```

**C. GPU Profiler (CUDA):**

```python
# scripts/gpu_profile.py

import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Run detection
    detector.detect(image)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

### 6. Test Organization

**Directory Structure:**

```
tests/
├── accuracy/
│   ├── test_mtcnn_accuracy.py
│   ├── test_pyfaceau_accuracy.py
│   ├── test_calc_params.py
│   └── test_hog_features.py
├── performance/
│   ├── test_mtcnn_performance.py
│   ├── test_pyfaceau_performance.py
│   └── test_parallel_pipeline.py
├── backends/
│   ├── test_cuda_backend.py
│   ├── test_coreml_backend.py
│   └── test_cpu_backend.py
├── integration/
│   ├── test_mtcnn_pyfaceau_integration.py
│   └── test_end_to_end_pipeline.py
├── profiling/
│   ├── profile_suite.py
│   ├── memory_profiler.py
│   └── gpu_profiler.py
└── fixtures/
    ├── conftest.py
    ├── test_data.py
    └── reference_outputs.py
```

**Pytest Configuration:**

```ini
# pytest.ini

[pytest]
minversion = 7.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Custom markers
markers =
    accuracy: Accuracy validation tests
    performance: Performance benchmarks
    slow: Slow tests (>10s)
    gpu: Requires GPU
    integration: Integration tests

# Coverage
addopts =
    --cov=pymtcnn
    --cov=pyfaceau
    --cov-report=html
    --cov-report=term-missing
    --verbose
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Set up test data repository
- [ ] Create basic accuracy tests
- [ ] Set up benchmark database
- [ ] Create quick profiling scripts

### Phase 2: Automation (Week 3-4)
- [ ] Implement GitHub Actions workflows
- [ ] Set up cross-platform testing
- [ ] Create performance tracking system
- [ ] Build regression detection

### Phase 3: Visualization (Week 5-6)
- [ ] Build benchmark dashboard
- [ ] Create performance reports
- [ ] Set up alerting system
- [ ] Document testing procedures

### Phase 4: Optimization (Ongoing)
- [ ] Profile and optimize bottlenecks
- [ ] Add more test cases
- [ ] Improve accuracy metrics
- [ ] Expand backend coverage

---

## Key Metrics to Track

### Accuracy Metrics
- **PyMTCNN:**
  - Bbox IoU vs reference
  - Landmark error (pixels)
  - Detection recall/precision
  - False positive rate

- **PyFaceAU:**
  - AU correlation (per AU)
  - Overall AU correlation
  - CalcParams fidelity
  - HOG feature similarity

### Performance Metrics
- **Speed:**
  - Overall FPS
  - Per-stage latency
  - Throughput (faces/sec)
  - Batch processing speedup

- **Resources:**
  - CPU utilization
  - GPU utilization
  - Memory usage (peak/avg)
  - Disk I/O

- **Efficiency:**
  - Power consumption
  - Thermal throttling
  - Cache hit rates
  - Memory bandwidth

---

## Tools & Technologies

### Testing
- `pytest` - Test framework
- `pytest-benchmark` - Performance benchmarking
- `hypothesis` - Property-based testing
- `unittest.mock` - Mocking

### Profiling
- `cProfile` - Function profiling
- `line_profiler` - Line-by-line profiling
- `memory_profiler` - Memory profiling
- `py-spy` - Sampling profiler
- `scalene` - CPU+GPU+memory profiler

### Monitoring
- `nvidia-smi` - NVIDIA GPU monitoring
- `intel-gpu-tools` - Intel GPU monitoring
- `powermetrics` - macOS power monitoring
- `psutil` - System monitoring

### Visualization
- `streamlit` or `gradio` - Dashboard
- `plotly` - Interactive plots
- `pandas` - Data analysis
- `matplotlib`/`seaborn` - Static plots

### CI/CD
- GitHub Actions - Automation
- self-hosted runners - GPU testing
- Git LFS - Large test files

---

## Next Steps: Discussion Points

1. **Test Data Strategy:**
   - Where to host? (GitHub LFS, S3, etc.)
   - How much data? (balance coverage vs. size)
   - Licensing for test images/videos?

2. **Accuracy Targets:**
   - What correlation threshold is acceptable?
   - Which reference implementations to use?
   - How to handle platform-specific differences?

3. **Performance Baselines:**
   - What FPS targets per platform?
   - Acceptable memory usage?
   - Power consumption limits?

4. **Regression Policy:**
   - Auto-fail on X% performance drop?
   - Manual review process?
   - Rollback strategy?

5. **Dashboard Hosting:**
   - Self-hosted vs. cloud?
   - Public vs. private?
   - Real-time vs. static reports?

---

**Let's discuss which aspects are most important to prioritize!**
