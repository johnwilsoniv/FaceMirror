#!/usr/bin/env python3
"""
Test if Cython with OpenMP multithreading is actually working.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add pyclnf to path
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

try:
    from optimizer_cython import kde_mean_shift_cython, batch_kde_mean_shift
    print("✓ Cython module loaded successfully")
    CYTHON_AVAILABLE = True
except ImportError as e:
    print(f"✗ Cython module not available: {e}")
    CYTHON_AVAILABLE = False
    exit(1)

def test_single_vs_batch():
    """Test if batch processing shows multithreading benefits."""

    print("\n" + "="*60)
    print("TESTING CYTHON MULTITHREADING")
    print("="*60)

    # Create test data
    n_landmarks = 68
    window_size = 11

    # Create random response maps and KDE weights
    response_maps = []
    for _ in range(n_landmarks):
        resp_map = np.random.rand(window_size, window_size).astype(np.float32)
        response_maps.append(resp_map)

    kde_weights = np.ones((window_size, window_size), dtype=np.float32)
    kde_weights = kde_weights / kde_weights.sum()

    # Test 1: Sequential processing (calling kde_mean_shift_cython one by one)
    print("\n1. Sequential Processing (one by one):")
    print("-" * 40)

    start = time.perf_counter()
    sequential_results = []
    for resp_map in response_maps:
        result = kde_mean_shift_cython(resp_map, kde_weights, window_size)
        sequential_results.append(result)
    sequential_time = (time.perf_counter() - start) * 1000

    print(f"  Time: {sequential_time:.2f}ms")
    print(f"  Per landmark: {sequential_time/n_landmarks:.3f}ms")

    # Test 2: Batch processing (potentially uses parallel processing internally)
    print("\n2. Batch Processing (all at once):")
    print("-" * 40)

    start = time.perf_counter()
    batch_results = batch_kde_mean_shift(response_maps, kde_weights)
    batch_time = (time.perf_counter() - start) * 1000

    print(f"  Time: {batch_time:.2f}ms")
    print(f"  Per landmark: {batch_time/n_landmarks:.3f}ms")

    # Compare results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    speedup = sequential_time / batch_time if batch_time > 0 else 1.0
    print(f"\nSpeedup from batch processing: {speedup:.2f}x")

    if speedup > 1.1:
        print("✓ Batch processing is faster - some optimization is working!")
    else:
        print("✗ No speedup - multithreading may not be working")

    # Verify results match
    sequential_array = np.array(sequential_results)
    results_match = np.allclose(sequential_array, batch_results)
    print(f"\nResults match: {results_match}")

    # Test with increasing workload to see scaling
    print("\n" + "="*60)
    print("SCALING TEST")
    print("="*60)

    for n_test in [10, 50, 100, 200]:
        # Create test data
        test_maps = [np.random.rand(window_size, window_size).astype(np.float32)
                    for _ in range(n_test)]

        # Sequential
        start = time.perf_counter()
        for resp_map in test_maps:
            _ = kde_mean_shift_cython(resp_map, kde_weights, window_size)
        seq_time = (time.perf_counter() - start) * 1000

        # Batch
        start = time.perf_counter()
        _ = batch_kde_mean_shift(test_maps, kde_weights)
        batch_time = (time.perf_counter() - start) * 1000

        speedup = seq_time / batch_time if batch_time > 0 else 1.0
        print(f"  {n_test:3d} landmarks: Sequential={seq_time:6.1f}ms, "
              f"Batch={batch_time:6.1f}ms, Speedup={speedup:.2f}x")

def test_parallel_execution():
    """Test if we can actually use multiple threads."""

    print("\n" + "="*60)
    print("THREAD UTILIZATION TEST")
    print("="*60)

    # Check if OpenMP is available
    try:
        import os
        num_threads = os.cpu_count()
        print(f"\nCPU cores available: {num_threads}")

        # Set OpenMP threads
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        print(f"OMP_NUM_THREADS set to: {num_threads}")

    except Exception as e:
        print(f"Could not determine CPU count: {e}")

    # Heavy workload test
    print("\nHeavy workload test (should use multiple cores):")
    print("-" * 40)

    # Create a computationally intensive task
    n_iterations = 1000
    window_size = 11
    kde_weights = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)

    # Single large computation
    large_maps = [np.random.rand(window_size, window_size).astype(np.float32)
                  for _ in range(68)]

    print(f"Processing {n_iterations} iterations...")

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = batch_kde_mean_shift(large_maps, kde_weights)
    elapsed = time.perf_counter() - start

    throughput = n_iterations / elapsed
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} iterations/sec")
    print(f"  Per iteration: {elapsed/n_iterations*1000:.2f}ms")

    # Note about GIL
    print("\n" + "="*60)
    print("IMPORTANT NOTE")
    print("="*60)
    print("""
The Cython module was compiled with OpenMP support, but true parallel
execution in Python is limited by:

1. The GIL (Global Interpreter Lock) - Python can't truly run Python
   code in parallel, only C code that releases the GIL

2. Our Cython code still has Python object interactions (lists, None checks)
   which require the GIL

3. For true parallelism, we'd need pure C code with no Python interactions

Even so, Cython provides benefits:
- C-level loops are much faster than Python
- Memory access patterns are more efficient
- Math operations use C types instead of Python objects

Expected speedup: 2-3x from optimization, not from parallelism.
""")

if __name__ == "__main__":
    test_single_vs_batch()
    test_parallel_execution()