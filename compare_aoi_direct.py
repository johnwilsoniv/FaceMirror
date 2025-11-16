#!/usr/bin/env python3
"""
Compare C++ and Python AOIs directly.
"""
import numpy as np
import struct

# Load Python AOI
python_aoi = np.load('/tmp/python_aoi_lm36.npy')
print(f"Python AOI:")
print(f"  Shape: {python_aoi.shape}")
print(f"  Stats: min={python_aoi.min():.1f}, max={python_aoi.max():.1f}, mean={python_aoi.mean():.1f}")

# Load C++ AOI
with open('/tmp/cpp_aoi_lm36.bin', 'rb') as f:
    rows = struct.unpack('i', f.read(4))[0]
    cols = struct.unpack('i', f.read(4))[0]
    data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
    cpp_aoi = data.reshape((rows, cols))

print(f"\nC++ AOI:")
print(f"  Shape: {cpp_aoi.shape}")
print(f"  Stats: min={cpp_aoi.min():.1f}, max={cpp_aoi.max():.1f}, mean={cpp_aoi.mean():.1f}")

# Compare
print(f"\nComparison:")
correlation = np.corrcoef(python_aoi.flatten(), cpp_aoi.flatten())[0, 1]
print(f"  Correlation: {correlation:.6f}")

diff = np.abs(python_aoi - cpp_aoi)
print(f"  Mean absolute difference: {diff.mean():.3f}")
print(f"  Max absolute difference: {diff.max():.3f}")

# Show where they differ most
max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
print(f"\nMax difference at ({max_diff_idx[1]}, {max_diff_idx[0]}):")
print(f"  Python: {python_aoi[max_diff_idx]:.1f}")
print(f"  C++: {cpp_aoi[max_diff_idx]:.1f}")
print(f"  Diff: {diff[max_diff_idx]:.1f}")

# Visual comparison
print(f"\nFirst row comparison:")
print(f"  Python: {python_aoi[0, :].astype(int)}")
print(f"  C++:    {cpp_aoi[0, :].astype(int)}")

if correlation > 0.99:
    print("\n✓ AOIs MATCH!")
else:
    print(f"\n✗ AOIs differ significantly (correlation={correlation:.6f})")
    print("This explains why response maps differ!")
