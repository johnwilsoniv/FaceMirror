#!/usr/bin/env python3
"""
Verify the kernel transpose theory with simple synthetic data.

Theory: C++ im2col uses column-major ordering, which is equivalent to:
1. Transposing each kernel
2. Flattening in row-major order
3. Doing matrix multiplication

So for PyTorch to match C++'s behavior, we need to transpose each kernel.
"""

import numpy as np
import torch
import torch.nn as nn

print("="*80)
print("KERNEL TRANSPOSE THEORY VERIFICATION")
print("="*80)

# Create simple test data
input_3x3 = np.array([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]], dtype=np.float32)  # 1 channel, 3x3

kernel_3x3 = np.array([[[[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9]]]], dtype=np.float32)  # 1 output channel, 1 input channel, 3x3

bias = np.array([0.5], dtype=np.float32)

print("\nInput (1 channel, 3x3):")
print(input_3x3[0])

print("\nKernel (no transpose):")
print(kernel_3x3[0, 0])

# PyTorch Conv2d with original kernel
x = torch.from_numpy(input_3x3).unsqueeze(0)  # (1, 1, 3, 3)
conv_no_transpose = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
conv_no_transpose.weight.data = torch.from_numpy(kernel_3x3)
conv_no_transpose.bias.data = torch.from_numpy(bias)
output_no_transpose = conv_no_transpose(x).squeeze().detach().numpy()

print(f"\nPyTorch output (no transpose): {output_no_transpose:.6f}")

# PyTorch Conv2d with transposed kernel
kernel_transposed = np.transpose(kernel_3x3, (0, 1, 3, 2))  # Transpose last two dims
print("\nKernel (transposed):")
print(kernel_transposed[0, 0])

conv_with_transpose = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
conv_with_transpose.weight.data = torch.from_numpy(kernel_transposed)
conv_with_transpose.bias.data = torch.from_numpy(bias)
output_with_transpose = conv_with_transpose(x).squeeze().detach().numpy()

print(f"\nPyTorch output (with transpose): {output_with_transpose:.6f}")

# Manual computation - C++ im2col style (column-major)
print(f"\n{'='*80}")
print("MANUAL C++ IM2COL STYLE (column-major)")
print(f"{'='*80}")

# Extract patch in column-major order
patch_col_major = []
for xx in range(3):  # width first
    for yy in range(3):  # height
        patch_col_major.append(input_3x3[0, yy, xx])
patch_col_major.append(1.0)  # bias term

print(f"Patch (column-major): {patch_col_major[:9]}")

# Flatten kernel (no transpose, row-major)
kernel_flat_no_transpose = kernel_3x3[0, 0].flatten()  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
weight_col_no_transpose = np.concatenate([kernel_flat_no_transpose, bias])

output_manual_no_transpose = np.dot(patch_col_major, weight_col_no_transpose)
print(f"\nManual (column-major patch, no kernel transpose): {output_manual_no_transpose:.6f}")

# Flatten kernel WITH transpose first
kernel_transposed_then_flat = kernel_3x3[0, 0].T.flatten()  # Transpose then flatten
weight_col_with_transpose = np.concatenate([kernel_transposed_then_flat, bias])

output_manual_with_transpose = np.dot(patch_col_major, weight_col_with_transpose)
print(f"Manual (column-major patch, kernel transposed): {output_manual_with_transpose:.6f}")

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"PyTorch (no transpose):                   {output_no_transpose:.6f}")
print(f"PyTorch (with transpose):                 {output_with_transpose:.6f}")
print(f"Manual (col-major, no kernel transpose):  {output_manual_no_transpose:.6f}")
print(f"Manual (col-major, kernel transposed):    {output_manual_with_transpose:.6f}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

if abs(output_with_transpose - output_manual_with_transpose) < 1e-5:
    print("✓ PyTorch with transposed kernel == Manual with column-major + transposed kernel")
    print("  This means: For PyTorch to match C++'s im2col, we MUST transpose kernels!")
elif abs(output_no_transpose - output_manual_no_transpose) < 1e-5:
    print("✓ PyTorch without transpose == Manual with column-major + no transpose")
    print("  This means: PyTorch somehow accounts for the column-major ordering")
else:
    print("❌ Neither matches - there's something else going on")
