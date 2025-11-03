#!/usr/bin/env python3
"""
Benchmark CLNF optimizations (Numba JIT + vectorized im2col_bias).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import numpy as np
import time
from pyfaceau.clnf.cen_patch_experts import contrast_norm, im2col_bias, NUMBA_AVAILABLE

print("="*80)
print("CLNF Optimization Benchmark")
print("="*80)
print(f"Numba available: {NUMBA_AVAILABLE}")
print()

# Test parameters
test_patch_sizes = [(32, 32), (64, 64), (128, 128)]
window_size = (11, 11)
num_iterations = 100

print("Benchmarking contrast_norm()...")
print("-" * 80)
for size in test_patch_sizes:
    # Generate test data
    test_patch = np.random.randn(*size).astype(np.float32) * 50 + 128

    # Warm up (important for JIT compilation)
    for _ in range(5):
        _ = contrast_norm(test_patch)

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        result = contrast_norm(test_patch)
    elapsed = time.time() - start

    per_call = (elapsed / num_iterations) * 1000  # ms
    print(f"  {size[0]}x{size[1]} patch: {per_call:.3f} ms/call ({num_iterations} iterations)")

print()
print("Benchmarking im2col_bias()...")
print("-" * 80)
for size in test_patch_sizes:
    # Generate test data
    test_patch = np.random.randn(*size).astype(np.float32) * 50 + 128

    # Warm up
    for _ in range(5):
        _ = im2col_bias(test_patch, window_size[0], window_size[1])

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        result = im2col_bias(test_patch, window_size[0], window_size[1])
    elapsed = time.time() - start

    per_call = (elapsed / num_iterations) * 1000  # ms
    num_windows = (size[0] - window_size[0] + 1) * (size[1] - window_size[1] + 1)
    print(f"  {size[0]}x{size[1]} patch ({num_windows} windows): {per_call:.3f} ms/call ({num_iterations} iterations)")

print()
print("="*80)
print("Optimization Status:")
print("="*80)
print(f"✓ Numba JIT compilation: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED (fallback to NumPy)'}")
print(f"✓ Vectorized im2col_bias: ENABLED (stride tricks)")
print()

if NUMBA_AVAILABLE:
    print("Expected improvements:")
    print("  - contrast_norm: 5-10x faster (from 15% → ~2% of total time)")
    print("  - im2col_bias: 10-20x faster (from 30% → ~2% of total time)")
    print("  - Combined: 2-5x overall CLNF speedup (0.5 FPS → 1-2.5 FPS)")
else:
    print("⚠️  Install numba for additional 5-10x speedup on contrast_norm:")
    print("     pip install numba")
print("="*80)
