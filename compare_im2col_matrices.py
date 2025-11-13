#!/usr/bin/env python3
"""
Compare C++ im2col matrix with our Python implementation.
This is THE KEY to finding the PNet divergence!
"""

import numpy as np
from implement_cpp_im2col_pnet import im2col_multimap_cpp

print("="*80)
print("COMPARING C++ IM2COL WITH PYTHON IM2COL")
print("="*80)

# Load C++ im2col matrix
with open('/tmp/cpp_pnet_layer0_im2col.bin', 'rb') as f:
    rows = np.fromfile(f, dtype=np.int32, count=1)[0]
    cols = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_im2col = np.fromfile(f, dtype=np.float32).reshape(rows, cols)

print(f"\nC++ Im2col Matrix:")
print(f"  Shape: {cpp_im2col.shape}")  # Should be (81748, 28)
print(f"  Sample values:")
print(f"    [0,0]: {cpp_im2col[0,0]}")
print(f"    [0,1]: {cpp_im2col[0,1]}")
print(f"    [0,27] (bias): {cpp_im2col[0,27]}")

# Load C++ PNet input
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)

# Create input maps for Python
# CRITICAL: OpenCV uses BGR order, so C++ inputs are [B, G, R] = [ch2, ch1, ch0]
input_maps = [cpp_input[:, :, c] for c in [2, 1, 0]]  # BGR order!

print(f"\nPython Input Maps:")
for i, inp_map in enumerate(input_maps):
    print(f"  Channel {i}: shape {inp_map.shape}, sample [0,0] = {inp_map[0,0]}")

# Generate Python im2col
py_im2col = im2col_multimap_cpp(input_maps, 3, 3)

print(f"\nPython Im2col Matrix:")
print(f"  Shape: {py_im2col.shape}")
print(f"  Sample values:")
print(f"    [0,0]: {py_im2col[0,0]}")
print(f"    [0,1]: {py_im2col[0,1]}")
print(f"    [0,27] (bias): {py_im2col[0,27]}")

# Compare
diff = np.abs(cpp_im2col - py_im2col)

print(f"\n{'='*80}")
print(f"IM2COL MATRIX COMPARISON:")
print(f"{'='*80}")
print(f"Max difference: {diff.max():.10f}")
print(f"Mean difference: {diff.mean():.10f}")
print(f"Shapes match: {cpp_im2col.shape == py_im2col.shape}")

# Distribution
print(f"\nDifference distribution:")
print(f"  < 1e-7: {(diff < 1e-7).sum()} / {diff.size} = {100*(diff < 1e-7).sum()/diff.size:.1f}%")
print(f"  < 1e-6: {(diff < 1e-6).sum()} / {diff.size} = {100*(diff < 1e-6).sum()/diff.size:.1f}%")
print(f"  < 1e-5: {(diff < 1e-5).sum()} / {diff.size} = {100*(diff < 1e-5).sum()/diff.size:.1f}%")

# Find largest differences
max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
print(f"\nLargest difference at index {max_diff_idx}:")
print(f"  C++: {cpp_im2col[max_diff_idx]}")
print(f"  Python: {py_im2col[max_diff_idx]}")
print(f"  Diff: {diff[max_diff_idx]}")

# Check first few positions in detail
print(f"\nFirst 3 positions, first 10 features:")
for pos in range(3):
    print(f"\n  Position {pos}:")
    for feat in range(10):
        cpp_val = cpp_im2col[pos, feat]
        py_val = py_im2col[pos, feat]
        match = "✓" if abs(cpp_val - py_val) < 1e-6 else "✗"
        print(f"    Feature {feat}: C++={cpp_val:.6f}, Py={py_val:.6f} {match}")

print(f"\n{'='*80}")
print(f"CONCLUSION:")
print(f"{'='*80}")
if diff.max() < 1e-5:
    print(f"✅ IM2COL MATRICES MATCH!")
    print(f"   The issue must be in the matrix multiplication or output reshaping.")
else:
    print(f"❌ IM2COL MATRICES DIFFER!")
    print(f"   Max diff: {diff.max():.6f}")
    print(f"   THIS IS THE ROOT CAUSE OF PNET DIVERGENCE!")
    print(f"\n   The im2col layout is different between C++ and Python.")
    print(f"   Need to fix the column ordering in im2col_multimap_cpp().")
