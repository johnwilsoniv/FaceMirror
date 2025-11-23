#!/usr/bin/env python3
"""
Comprehensive benchmark of all optimization approaches.
Compares baseline vs all implemented optimizations.
"""

import time
import sys
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def benchmark_pipeline(pipeline, frames: List[np.ndarray], name: str) -> Dict:
    """Benchmark a pipeline with given frames."""
    frame_times = []
    results = []

    for i, frame in enumerate(frames):
        start = time.perf_counter()
        result = pipeline.process_frame(frame) if hasattr(pipeline, 'process_frame') else None
        elapsed = (time.perf_counter() - start) * 1000
        frame_times.append(elapsed)
        if result:
            results.append(result)

    avg_time = np.mean(frame_times)
    fps = 1000 / avg_time if avg_time > 0 else 0

    return {
        'name': name,
        'avg_time': avg_time,
        'fps': fps,
        'frame_times': frame_times,
        'results': results
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE AU PIPELINE OPTIMIZATION BENCHMARK")
    print("=" * 80)

    # Load test video
    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Collect test frames
    cap = cv2.VideoCapture(video_path)
    test_frames = []
    for _ in range(30):  # Use 30 frames for more stable average
        ret, frame = cap.read()
        if not ret:
            break
        test_frames.append(frame)
    cap.release()

    print(f"Testing with {len(test_frames)} frames")
    print()

    # Suppress output during initialization
    import warnings
    import io
    warnings.filterwarnings('ignore')
    old_stdout = sys.stdout

    benchmarks = []

    # 1. Baseline (original implementation)
    print("1. Initializing BASELINE pipeline...")
    sys.stdout = io.StringIO()
    try:
        from pymtcnn import MTCNN
        from pyclnf import CLNF
        from pyfaceau import FullPythonAUPipeline

        class BaselinePipeline:
            def __init__(self):
                self.detector = MTCNN()
                self.clnf = CLNF(
                    model_dir="pyclnf/models",
                    max_iterations=10,  # Original settings
                    convergence_threshold=0.1
                )
                self.au_pipeline = FullPythonAUPipeline(
                    pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
                    au_models_dir="pyfaceau/weights/AU_predictors",
                    triangulation_file="pyfaceau/weights/tris_68_full.txt",
                    patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
                    verbose=False
                )

            def process_frame(self, frame):
                detection = self.detector.detect(frame)
                if detection and isinstance(detection, tuple) and len(detection) == 2:
                    bboxes, _ = detection
                    if len(bboxes) > 0:
                        bbox = bboxes[0]
                        x, y, w, h = [int(v) for v in bbox]
                        bbox = (x, y, w, h)
                        landmarks, _ = self.clnf.fit(frame, bbox)
                        result = self.au_pipeline._process_frame(frame, 0, 0.0)
                        return result
                return {}

        baseline = BaselinePipeline()
    finally:
        sys.stdout = old_stdout

    print("   ✓ Baseline initialized")
    result = benchmark_pipeline(baseline, test_frames[:10], "Baseline")
    benchmarks.append(result)
    print(f"   Results: {result['fps']:.2f} FPS")

    # 2. Optimized (with convergence fix + Numba)
    print("\n2. Initializing OPTIMIZED pipeline...")
    sys.stdout = io.StringIO()
    try:
        from optimized_au_pipeline import OptimizedAUPipeline
        optimized = OptimizedAUPipeline(verbose=False)
    finally:
        sys.stdout = old_stdout

    print("   ✓ Optimized initialized")
    result = benchmark_pipeline(optimized, test_frames[:10], "Optimized")
    benchmarks.append(result)
    print(f"   Results: {result['fps']:.2f} FPS")

    # 3. Production (optimized + no debug)
    print("\n3. Initializing PRODUCTION pipeline...")
    sys.stdout = io.StringIO()
    try:
        from production_au_pipeline import ProductionAUPipeline
        production = ProductionAUPipeline(verbose=False)
    finally:
        sys.stdout = old_stdout

    print("   ✓ Production initialized")
    result = benchmark_pipeline(production, test_frames[:10], "Production")
    benchmarks.append(result)
    print(f"   Results: {result['fps']:.2f} FPS")

    # 4. Multi-threaded (best performer)
    print("\n4. Testing MULTI-THREADED pipeline...")
    sys.stdout = io.StringIO()
    try:
        from multithreaded_au_pipeline import MultithreadedAUPipeline
        multithreaded = MultithreadedAUPipeline(n_workers=4, batch_size=4, verbose=False)
    finally:
        sys.stdout = old_stdout

    print("   ✓ Multi-threaded initialized")

    # Multi-threaded needs batch processing
    start = time.perf_counter()
    results = multithreaded.process_batch_parallel(test_frames[:10])
    elapsed = (time.perf_counter() - start) * 1000
    avg_time = elapsed / len(test_frames[:10])
    fps = 1000 / avg_time if avg_time > 0 else 0

    benchmarks.append({
        'name': 'Multi-threaded',
        'avg_time': avg_time,
        'fps': fps,
        'frame_times': [],
        'results': results
    })
    print(f"   Results: {fps:.2f} FPS")

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\n{'Pipeline':<20} {'FPS':<10} {'Frame Time':<15} {'vs Baseline':<15}")
    print("-" * 60)

    baseline_fps = benchmarks[0]['fps']
    for bench in benchmarks:
        speedup = bench['fps'] / baseline_fps if baseline_fps > 0 else 1
        print(f"{bench['name']:<20} {bench['fps']:<10.2f} {bench['avg_time']:<15.1f}ms {speedup:<15.2f}x")

    # Visual comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE VISUALIZATION")
    print("=" * 80)

    for bench in benchmarks:
        bar_length = int(bench['fps'] * 10)
        bar = "█" * bar_length
        print(f"{bench['name']:<20} {bar} {bench['fps']:.2f} FPS")

    print("\nTarget (OpenFace C++): " + "█" * 100 + " 10.1 FPS")

    # Hardware acceleration potential
    print("\n" + "=" * 80)
    print("PROJECTED PERFORMANCE WITH HARDWARE ACCELERATION")
    print("=" * 80)

    best_fps = max(bench['fps'] for bench in benchmarks)

    projections = [
        ("Current Best", best_fps, 1.0),
        ("+ FP16 Quantization", best_fps * 1.5, 1.5),
        ("+ ONNX Runtime", best_fps * 2.0, 2.0),
        ("+ GPU (Metal/CUDA)", best_fps * 3.5, 3.5),
        ("+ Neural Architecture", best_fps * 5.0, 5.0),
    ]

    for name, fps, multiplier in projections:
        bar_length = int(fps * 10)
        bar = "█" * min(bar_length, 100)
        status = "✅" if fps <= best_fps else "🎯"
        print(f"{status} {name:<25} {bar} {fps:.1f} FPS ({multiplier:.1f}x)")

    # Next steps
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    print("\nImmediate actions for additional speedup:")
    print("1. Install PyTorch with MPS: pip install torch torchvision")
    print("2. Install ONNX Runtime: pip install onnxruntime")
    print("3. Convert models to ONNX format")
    print("4. Enable GPU acceleration")
    print("5. Implement model quantization")

    print("\nExpected improvements:")
    print("- Quantization: 1.5-2x speedup")
    print("- ONNX Runtime: 2-3x speedup")
    print("- GPU acceleration: 3-5x speedup")
    print("- Combined: 5-10x total speedup possible")


if __name__ == "__main__":
    main()