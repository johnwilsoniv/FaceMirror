#!/usr/bin/env python3
"""
Compare C++ weight matrix with our Python extracted weights.
This will reveal the exact difference!
"""

import numpy as np

print("="*80)
print("COMPARING C++ WEIGHT MATRIX WITH PYTHON EXTRACTED WEIGHTS")
print("="*80)

# Load C++ weight matrix (saved during model loading)
with open('/tmp/cpp_conv0_weight.bin', 'rb') as f:
    rows = np.fromfile(f, dtype=np.int32, count=1)[0]
    cols = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_weight_matrix = np.fromfile(f, dtype=np.float32).reshape(rows, cols)

print(f"\nC++ Weight Matrix:")
print(f"  Shape: {cpp_weight_matrix.shape}")  # Should be (28, 10) - transposed with bias
print(f"  Sample values:")
print(f"    [0,0]: {cpp_weight_matrix[0,0]}")
print(f"    [0,1]: {cpp_weight_matrix[0,1]}")

# Load our extracted weights
pnet_weights = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_weights.npy')
pnet_biases = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_biases.npy')

print(f"\nOur Extracted Weights:")
print(f"  Shape: {pnet_weights.shape}")  # (10, 3, 3, 3)
print(f"  Bias shape: {pnet_biases.shape}")  # (10,)

# Reconstruct weight matrix using our implementation
from implement_cpp_im2col_pnet import create_weight_matrix_cpp

py_weight_matrix = create_weight_matrix_cpp(pnet_weights, pnet_biases)

print(f"\nPython Reconstructed Weight Matrix:")
print(f"  Shape: {py_weight_matrix.shape}")
print(f"  Sample values:")
print(f"    [0,0]: {py_weight_matrix[0,0]}")
print(f"    [0,1]: {py_weight_matrix[0,1]}")

# Compare
diff = np.abs(cpp_weight_matrix - py_weight_matrix)

print(f"\n{'='*80}")
print(f"WEIGHT MATRIX COMPARISON:")
print(f"{'='*80}")
print(f"Max difference: {diff.max():.10f}")
print(f"Mean difference: {diff.mean():.10f}")
print(f"Shapes match: {cpp_weight_matrix.shape == py_weight_matrix.shape}")

# Distribution
print(f"\nDifference distribution:")
print(f"  < 1e-7: {(diff < 1e-7).sum()} / {diff.size} = {100*(diff < 1e-7).sum()/diff.size:.1f}%")
print(f"  < 1e-6: {(diff < 1e-6).sum()} / {diff.size} = {100*(diff < 1e-6).sum()/diff.size:.1f}%")
print(f"  < 1e-5: {(diff < 1e-5).sum()} / {diff.size} = {100*(diff < 1e-5).sum()/diff.size:.1f}%")
print(f"  < 0.1: {(diff < 0.1).sum()} / {diff.size} = {100*(diff < 0.1).sum()/diff.size:.1f}%")

# Find largest differences
max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
print(f"\nLargest difference at index {max_diff_idx}:")
print(f"  C++: {cpp_weight_matrix[max_diff_idx]}")
print(f"  Python: {py_weight_matrix[max_diff_idx]}")
print(f"  Diff: {diff[max_diff_idx]}")

# Check if matrices are just transposed
diff_transposed = np.abs(cpp_weight_matrix - py_weight_matrix.T)
print(f"\nTranspose check:")
print(f"  Diff with transpose: max={diff_transposed.max():.10f}, mean={diff_transposed.mean():.10f}")

# Check specific patterns
print(f"\n{'='*80}")
print(f"CONCLUSION:")
print(f"{'='*80}")
if diff.max() < 1e-5:
    print(f"✅ WEIGHT MATRICES MATCH!")
    print(f"   The issue must be elsewhere.")
elif diff_transposed.max() < 1e-5:
    print(f"⚠️  Weight matrices are TRANSPOSED!")
    print(f"   Need to transpose in Python implementation.")
else:
    print(f"❌ WEIGHT MATRICES DIFFER!")
    print(f"   Max diff: {diff.max():.6f}")
    print(f"   This is the source of PNet divergence!")
    print(f"\n   Next step: Investigate weight loading/transformation difference.")
