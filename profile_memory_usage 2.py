#!/usr/bin/env python3
"""
Profile memory usage of the AU pipeline to identify optimization opportunities.

Uses memory_profiler and tracemalloc to track:
1. Peak memory usage
2. Memory allocations per function
3. Memory leaks
4. Optimization opportunities
"""

import tracemalloc
import psutil
import gc
import sys
import time
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def get_memory_usage() -> Dict:
    """Get current memory usage statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent()
    }


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def profile_memory_allocations():
    """Profile memory allocations during pipeline initialization and processing."""

    print("=" * 60)
    print("MEMORY PROFILING - AU PIPELINE")
    print("=" * 60)

    # Start memory tracking
    tracemalloc.start()

    # Baseline memory
    baseline_memory = get_memory_usage()
    print(f"\nBaseline memory: {baseline_memory['rss_mb']:.1f} MB")

    # Track memory during initialization
    print("\n1. INITIALIZATION PHASE")
    print("-" * 40)

    # Suppress output
    import io
    import warnings
    warnings.filterwarnings('ignore')
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    init_snapshots = []

    try:
        # MTCNN initialization
        snapshot1 = tracemalloc.take_snapshot()
        from pymtcnn import MTCNN
        detector = MTCNN()
        snapshot2 = tracemalloc.take_snapshot()
        mtcnn_memory = get_memory_usage()
        init_snapshots.append(('MTCNN', snapshot1, snapshot2))

        # CLNF initialization
        from pyclnf import CLNF
        clnf = CLNF(
            model_dir="pyclnf/models",
            max_iterations=5,
            convergence_threshold=0.5
        )
        snapshot3 = tracemalloc.take_snapshot()
        clnf_memory = get_memory_usage()
        init_snapshots.append(('CLNF', snapshot2, snapshot3))

        # AU Pipeline initialization
        from pyfaceau import FullPythonAUPipeline
        au_pipeline = FullPythonAUPipeline(
            pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
            au_models_dir="pyfaceau/weights/AU_predictors",
            triangulation_file="pyfaceau/weights/tris_68_full.txt",
            patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
            verbose=False
        )
        snapshot4 = tracemalloc.take_snapshot()
        au_memory = get_memory_usage()
        init_snapshots.append(('AU Pipeline', snapshot3, snapshot4))

    finally:
        sys.stdout = old_stdout

    # Print initialization memory usage
    print(f"MTCNN: {mtcnn_memory['rss_mb'] - baseline_memory['rss_mb']:.1f} MB")
    print(f"CLNF: {clnf_memory['rss_mb'] - mtcnn_memory['rss_mb']:.1f} MB")
    print(f"AU Pipeline: {au_memory['rss_mb'] - clnf_memory['rss_mb']:.1f} MB")
    print(f"Total initialization: {au_memory['rss_mb'] - baseline_memory['rss_mb']:.1f} MB")

    # Analyze top memory allocations
    print("\n2. TOP MEMORY ALLOCATIONS")
    print("-" * 40)

    snapshot_current = tracemalloc.take_snapshot()
    top_stats = snapshot_current.statistics('lineno')

    for stat in top_stats[:10]:
        print(f"{stat.traceback.format()[0]}")
        print(f"  Size: {format_bytes(stat.size)}, Count: {stat.count}")

    # Process frames and track memory
    print("\n3. FRAME PROCESSING MEMORY")
    print("-" * 40)

    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    if Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)

        frame_memories = []
        gc.collect()  # Force garbage collection

        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            before_memory = get_memory_usage()

            # Detection
            detection = detector.detect(frame)
            if detection and isinstance(detection, tuple) and len(detection) == 2:
                bboxes, _ = detection
                if len(bboxes) > 0:
                    bbox = bboxes[0]
                    x, y, w, h = [int(v) for v in bbox]
                    bbox = (x, y, w, h)

                    # Landmarks
                    landmarks, _ = clnf.fit(frame, bbox)

                    # AU prediction
                    au_result = au_pipeline._process_frame(
                        frame,
                        frame_idx=i,
                        timestamp=i/30.0
                    )

            after_memory = get_memory_usage()
            memory_delta = after_memory['rss_mb'] - before_memory['rss_mb']
            frame_memories.append(memory_delta)

            if i % 3 == 0:
                print(f"Frame {i}: {memory_delta:+.1f} MB (Total: {after_memory['rss_mb']:.1f} MB)")

        cap.release()

        # Memory statistics
        print(f"\nAverage memory per frame: {np.mean(frame_memories):.2f} MB")
        print(f"Peak memory increase: {np.max(frame_memories):.2f} MB")

        # Force garbage collection and measure
        gc.collect()
        after_gc_memory = get_memory_usage()
        print(f"Memory after GC: {after_gc_memory['rss_mb']:.1f} MB")
        print(f"Memory freed by GC: {au_memory['rss_mb'] - after_gc_memory['rss_mb']:.1f} MB")

    # Stop tracking
    tracemalloc.stop()

    # Optimization recommendations
    print("\n4. MEMORY OPTIMIZATION OPPORTUNITIES")
    print("-" * 40)
    print("✓ Model loading: Consider lazy loading or model sharing")
    print("✓ Frame buffers: Use memory pools for frame allocation")
    print("✓ Cache management: Implement LRU cache with size limits")
    print("✓ Garbage collection: Force GC after batch processing")
    print("✓ Data types: Use float16 instead of float32 where possible")


def analyze_memory_leaks():
    """Check for memory leaks in the pipeline."""

    print("\n" + "=" * 60)
    print("MEMORY LEAK ANALYSIS")
    print("=" * 60)

    from optimized_au_pipeline import OptimizedAUPipeline

    # Initialize pipeline
    pipeline = OptimizedAUPipeline(verbose=False)

    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    # Process frames in batches
    batch_size = 10
    memory_readings = []

    for batch_idx in range(5):
        gc.collect()
        before = get_memory_usage()

        # Process batch
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            _ = pipeline.process_frame(frame)

        after = get_memory_usage()
        memory_increase = after['rss_mb'] - before['rss_mb']
        memory_readings.append(memory_increase)

        print(f"Batch {batch_idx + 1}: {memory_increase:+.1f} MB "
              f"(Total: {after['rss_mb']:.1f} MB)")

    cap.release()

    # Analyze trend
    if len(memory_readings) > 1:
        memory_trend = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        if memory_trend > 0.1:
            print(f"\n⚠️  Potential memory leak detected: {memory_trend:.2f} MB/batch")
        else:
            print(f"\n✓ No significant memory leak detected")

    # Final cleanup
    del pipeline
    gc.collect()
    final_memory = get_memory_usage()
    print(f"\nFinal memory after cleanup: {final_memory['rss_mb']:.1f} MB")


def suggest_optimizations():
    """Suggest specific memory optimizations."""

    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION SUGGESTIONS")
    print("=" * 60)

    suggestions = [
        ("Use float16", "Convert models to half precision", "50% memory reduction"),
        ("Implement memory pools", "Reuse frame buffers", "Reduce allocations"),
        ("Lazy model loading", "Load models on demand", "Lower startup memory"),
        ("Optimize caches", "Set maximum cache sizes", "Bounded memory usage"),
        ("Batch processing", "Process multiple frames together", "Better memory locality"),
        ("Release unused objects", "Explicit deletion of large objects", "Faster GC"),
        ("Use generators", "Stream data instead of loading all", "Lower peak memory"),
        ("Optimize numpy arrays", "Use views instead of copies", "Reduce duplication"),
    ]

    for i, (optimization, description, benefit) in enumerate(suggestions, 1):
        print(f"\n{i}. {optimization}")
        print(f"   {description}")
        print(f"   Expected: {benefit}")


if __name__ == "__main__":
    # Run memory profiling
    profile_memory_allocations()

    # Check for leaks
    analyze_memory_leaks()

    # Provide suggestions
    suggest_optimizations()