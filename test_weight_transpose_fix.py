#!/usr/bin/env python3
"""
The issue: C++ transposes kernels for im2col (column-major), but we extracted
them WITHOUT transpose for PyTorch. Let's test if we SHOULD transpose for PyTorch.

C++ im2col fills in column-major order: [col0[rows], col1[rows], col2[rows]]
PyTorch Conv2d expects row-major order: [row0[cols], row1[cols], row2[cols]]

For a kernel to work in both, we need to transpose it!
"""

import numpy as np
import torch
import torch.nn as nn
import struct

# Load C++ ONet input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))

# Load weights WITHOUT transpose (current approach)
weights_no_transpose = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')  # (32, 3, 3, 3)
biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')    # (32,)

# Create weights WITH transpose (each 3x3 kernel transposed)
weights_with_transpose = weights_no_transpose.copy()
for out_ch in range(32):
    for in_ch in range(3):
        # Transpose each 3x3 kernel
        weights_with_transpose[out_ch, in_ch] = weights_no_transpose[out_ch, in_ch].T

print("="*80)
print("KERNEL TRANSPOSE TEST")
print("="*80)
print("\nOriginal kernel [0, 0] (no transpose):")
print(weights_no_transpose[0, 0])
print("\nTransposed kernel [0, 0]:")
print(weights_with_transpose[0, 0])

# Convert input to CHW
input_chw = np.transpose(cpp_input, (2, 0, 1))
x = torch.from_numpy(input_chw).unsqueeze(0)

# Test 1: No transpose (current)
conv_no_transpose = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv_no_transpose.weight.data = torch.from_numpy(weights_no_transpose)
conv_no_transpose.bias.data = torch.from_numpy(biases)
output_no_transpose = conv_no_transpose(x).squeeze(0).detach().numpy()

# Test 2: With transpose
conv_with_transpose = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv_with_transpose.weight.data = torch.from_numpy(weights_with_transpose)
conv_with_transpose.bias.data = torch.from_numpy(biases)
output_with_transpose = conv_with_transpose(x).squeeze(0).detach().numpy()

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"Without transpose: output[0,0,0] = {output_no_transpose[0,0,0]:.6f}")
print(f"With transpose:    output[0,0,0] = {output_with_transpose[0,0,0]:.6f}")

# Now manually compute what C++'s im2col + weight matrix should give
# Extract 3x3 patch in column-major order
patch = []
for in_ch in range(3):
    channel = cpp_input[0:3, 0:3, in_ch]
    # Column-major: for each xx (width), for each yy (height)
    for xx in range(3):
        for yy in range(3):
            patch.append(channel[yy, xx])
patch.append(1.0)  # bias
patch = np.array(patch, dtype=np.float32)

# Reconstruct what C++'s weight matrix SHOULD be if we transpose kernels
weight_matrix_rows = []
for out_ch in range(32):
    row = []
    for in_ch in range(3):
        # C++ transposes the kernel, then flattens
        kernel_transposed = weights_no_transpose[out_ch, in_ch].T
        kernel_flat = kernel_transposed.flatten()  # row-major flatten after transpose
        row.extend(kernel_flat)
    row.append(biases[out_ch])
    weight_matrix_rows.append(row)

weight_matrix_transposed = np.array(weight_matrix_rows, dtype=np.float32).T  # (28, 32)

manual_output_transposed_kernel = patch @ weight_matrix_transposed[:, 0]

print(f"\n{'='*80}")
print("MANUAL COMPUTATION WITH TRANSPOSED KERNELS")
print(f"{'='*80}")
print(f"Manual (C++ style with transposed kernels): {manual_output_transposed_kernel:.6f}")
print(f"PyTorch with transpose: {output_with_transpose[0,0,0]:.6f}")
print(f"Difference: {abs(manual_output_transposed_kernel - output_with_transpose[0,0,0]):.6e}")

if abs(manual_output_transposed_kernel - output_with_transpose[0,0,0]) < 1e-5:
    print("✓ MATCH! PyTorch with transposed kernels matches C++ im2col approach!")
