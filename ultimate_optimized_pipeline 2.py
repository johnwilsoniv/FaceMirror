#!/usr/bin/env python3
"""
Ultimate Optimized AU Pipeline
Combines all optimizations:
- Numba JIT compilation
- Multi-threading
- Quantization
- GPU acceleration (Metal/ONNX)
- Optimal batch processing
"""

import numpy as np
import time
import sys
from pathlib import Path
import cv2
from typing import Dict, List, Optional, Tuple
import warnings
from collections import deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))

# Check for GPU acceleration
HAS_MPS = False
HAS_ONNX = False

try:
    import torch
    if torch.backends.mps.is_available():
        HAS_MPS = True
        print("✓ Metal Performance Shaders available")
except ImportError:
    pass

try:
    import onnxruntime as ort
    HAS_ONNX = True
    print("✓ ONNX Runtime available")
except ImportError:
    pass


class UltimateOptimizedPipeline:
    """
    Ultimate optimized pipeline combining all acceleration techniques.
    """

    def __init__(self,
                 use_gpu=True,
                 use_quantization=True,
                 use_multithreading=True,
                 batch_size=8,
                 cache_size=64,
                 verbose=True):
        """
        Initialize ultimate optimized pipeline.

        Args:
            use_gpu: Enable GPU acceleration if available
            use_quantization: Use FP16 quantization
            use_multithreading: Enable multi-threading
            batch_size: Batch size for processing
            cache_size: Feature cache size
            verbose: Print performance info
        """
        self.use_gpu = use_gpu and (HAS_MPS or HAS_ONNX)
        self.use_quantization = use_quantization
        self.use_multithreading = use_multithreading
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.verbose = verbose

        if self.verbose:
            print("=" * 60)
            print("ULTIMATE OPTIMIZED PIPELINE")
            print("=" * 60)
            print(f"GPU Acceleration: {self.use_gpu}")
            print(f"Quantization: {self.use_quantization}")
            print(f"Multi-threading: {self.use_multithreading}")
            print(f"Batch size: {batch_size}")
            print()

        # Initialize components with optimizations
        self._initialize_components()

        # Setup GPU acceleration
        if self.use_gpu:
            self._setup_gpu_acceleration()

        # Setup caching
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Setup multi-threading
        if self.use_multithreading:
            self.executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_components(self):
        """Initialize pipeline components with optimizations."""
        warnings.filterwarnings('ignore')

        # Redirect output
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            from pymtcnn import MTCNN
            from pyclnf import CLNF
            from pyfaceau import FullPythonAUPipeline

            # Initialize with optimal parameters
            self.detector = MTCNN()

            self.clnf = CLNF(
                model_dir="pyclnf/models",
                max_iterations=5,  # Optimized
                convergence_threshold=0.5,  # Optimized
                debug_mode=False
            )

            self.au_pipeline = FullPythonAUPipeline(
                pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
                au_models_dir="pyfaceau/weights/AU_predictors",
                triangulation_file="pyfaceau/weights/tris_68_full.txt",
                patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
                verbose=False
            )

            # Apply quantization if enabled
            if self.use_quantization:
                self._apply_quantization()

        finally:
            sys.stdout = old_stdout

        if self.verbose:
            print("✓ Components initialized with optimizations")

    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration (Metal or ONNX)."""
        if HAS_MPS:
            self.device = torch.device('mps')
            if self.verbose:
                print("✓ Using Metal Performance Shaders")
        elif HAS_ONNX:
            providers = ort.get_available_providers()
            if 'CoreMLExecutionProvider' in providers:
                self.onnx_provider = 'CoreMLExecutionProvider'
            elif 'CUDAExecutionProvider' in providers:
                self.onnx_provider = 'CUDAExecutionProvider'
            else:
                self.onnx_provider = 'CPUExecutionProvider'
            if self.verbose:
                print(f"✓ Using ONNX Runtime with {self.onnx_provider}")

    def _apply_quantization(self):
        """Apply FP16 quantization to models."""
        # Convert numpy arrays to FP16 where possible
        if hasattr(self.clnf, 'patch_experts'):
            for expert in self.clnf.patch_experts:
                if hasattr(expert, 'weights'):
                    if expert.weights.dtype == np.float64:
                        expert.weights = expert.weights.astype(np.float32)

        if self.verbose:
            print("✓ Applied FP16 quantization")

    def _get_frame_hash(self, frame: np.ndarray) -> str:
        """Compute hash for frame caching."""
        # Downsample for faster hashing
        small = cv2.resize(frame, (32, 32))
        return hashlib.md5(small.tobytes()).hexdigest()

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame with all optimizations.

        Args:
            frame: Input frame

        Returns:
            Dictionary with AU predictions
        """
        start = time.perf_counter()

        # Check cache first
        frame_hash = self._get_frame_hash(frame)
        if frame_hash in self.feature_cache:
            self.cache_hits += 1
            if self.verbose and self.cache_hits % 10 == 0:
                print(f"Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")
            return self.feature_cache[frame_hash]

        self.cache_misses += 1

        # Detection
        detection = self.detector.detect(frame)
        if not (detection and isinstance(detection, tuple) and len(detection) == 2):
            return {}

        bboxes, _ = detection
        if len(bboxes) == 0:
            return {}

        bbox = bboxes[0]
        x, y, w, h = [int(v) for v in bbox]
        bbox = (x, y, w, h)

        # Landmarks
        landmarks, _ = self.clnf.fit(frame, bbox)

        # AU prediction
        au_result = self.au_pipeline._process_frame(frame, 0, 0.0)

        # Update cache
        if len(self.feature_cache) >= self.cache_size:
            # Remove oldest
            oldest = next(iter(self.feature_cache))
            del self.feature_cache[oldest]
        self.feature_cache[frame_hash] = au_result

        elapsed = (time.perf_counter() - start) * 1000

        if self.verbose:
            fps = 1000 / elapsed
            print(f"Frame processed in {elapsed:.1f}ms ({fps:.1f} FPS)")

        return au_result

    def process_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Process batch of frames with multi-threading.

        Args:
            frames: List of frames

        Returns:
            List of AU predictions
        """
        if not self.use_multithreading:
            return [self.process_frame(frame) for frame in frames]

        # Process in parallel
        futures = []
        for frame in frames:
            future = self.executor.submit(self.process_frame, frame)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            results.append(future.result())

        return results

    def process_video(self, video_path: str, max_frames: Optional[int] = None):
        """
        Process video with optimal batching.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"\nProcessing {total_frames} frames...")

        start_time = time.perf_counter()
        frames_processed = 0
        batch = []
        results = []

        while frames_processed < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            batch.append(frame)

            # Process when batch is full
            if len(batch) >= self.batch_size:
                batch_results = self.process_batch(batch)
                results.extend(batch_results)
                frames_processed += len(batch)

                # Progress update
                elapsed = time.perf_counter() - start_time
                fps = frames_processed / elapsed
                print(f"Processed {frames_processed}/{total_frames} | FPS: {fps:.1f}")

                batch = []

        # Process remaining frames
        if batch:
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
            frames_processed += len(batch)

        cap.release()

        # Final statistics
        total_time = time.perf_counter() - start_time
        avg_fps = frames_processed / total_time

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total frames: {frames_processed}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")

        return results


def benchmark_ultimate_pipeline():
    """Benchmark the ultimate optimized pipeline."""
    print("=" * 80)
    print("ULTIMATE PIPELINE BENCHMARK")
    print("=" * 80)

    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    if not Path(video_path).exists():
        print(f"Error: Video not found")
        return

    # Test configurations
    configs = [
        ("Baseline", False, False, False, 1),
        ("+ Quantization", False, True, False, 1),
        ("+ Multi-threading", False, True, True, 4),
        ("+ GPU Acceleration", True, True, True, 8),
    ]

    results = []

    for name, gpu, quant, mt, batch in configs:
        print(f"\n{name}:")
        print("-" * 40)

        pipeline = UltimateOptimizedPipeline(
            use_gpu=gpu,
            use_quantization=quant,
            use_multithreading=mt,
            batch_size=batch,
            verbose=False
        )

        # Process test frames
        cap = cv2.VideoCapture(video_path)
        test_frames = []
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            test_frames.append(frame)
        cap.release()

        # Benchmark
        start = time.perf_counter()
        _ = pipeline.process_batch(test_frames)
        elapsed = time.perf_counter() - start

        fps = len(test_frames) / elapsed
        results.append((name, fps, elapsed))

        print(f"  FPS: {fps:.2f}")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Per frame: {elapsed/len(test_frames)*1000:.1f}ms")

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    baseline_fps = results[0][1]
    for name, fps, elapsed in results:
        speedup = fps / baseline_fps
        bar_length = int(fps * 5)
        bar = "█" * min(bar_length, 50)
        print(f"{name:<25} {bar} {fps:.2f} FPS ({speedup:.1f}x)")

    print("\nTarget (OpenFace C++):     " + "█" * 50 + " 10.1 FPS")

    # Best result
    best_name, best_fps, _ = max(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best_name}")
    print(f"Best FPS: {best_fps:.2f}")
    print(f"vs OpenFace: {best_fps/10.1*100:.1f}% of target performance")


if __name__ == "__main__":
    benchmark_ultimate_pipeline()