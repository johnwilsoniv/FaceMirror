#!/usr/bin/env python3
"""
Test if C++'s bias-in-weight-matrix trick is causing the 2.08x difference.
Replicate the exact C++ approach: add bias as extra column in weights,
add column of 1s in im2col, and multiply.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load C++ ONet input (48x48x3 HWC)
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))

# Convert to CHW for PyTorch
input_chw = np.transpose(cpp_input, (2, 0, 1))  # HWC -> CHW
x = torch.from_numpy(input_chw).unsqueeze(0)  # (1, 3, 48, 48)

print("="*80)
print("TESTING C++ BIAS-IN-WEIGHT-MATRIX TRICK")
print("="*80)

print(f"\nInput shape: {x.shape}")
print(f"Input sample [0,0,0,:3]: {x[0,0,0,:3]}")

# Load conv1 weights and biases
conv1_weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')  # (32, 3, 3, 3)
conv1_biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')    # (32,)

print(f"\nConv1 weights shape: {conv1_weights.shape}")
print(f"Conv1 biases shape: {conv1_biases.shape}")

# ============================================================================
# Method 1: Standard PyTorch Conv2d (what we currently do)
# ============================================================================
print(f"\n{'='*80}")
print("METHOD 1: Standard PyTorch Conv2d (current approach)")
print(f"{'='*80}")

conv1_standard = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv1_standard.weight.data = torch.from_numpy(conv1_weights)
conv1_standard.bias.data = torch.from_numpy(conv1_biases)

out_standard = conv1_standard(x)
print(f"Output shape: {out_standard.shape}")
print(f"Output [0,0,0,0]: {out_standard[0,0,0,0].item():.6f}")
print(f"Output range: [{out_standard.min().item():.6f}, {out_standard.max().item():.6f}]")

# ============================================================================
# Method 2: Manual im2col with bias-in-weight-matrix (C++ approach)
# ============================================================================
print(f"\n{'='*80}")
print("METHOD 2: Manual im2col + bias-in-weight-matrix (C++ approach)")
print(f"{'='*80}")

# Extract patches using unfold (equivalent to im2col)
# x is (1, 3, 48, 48), kernel is 3x3
patches = F.unfold(x, kernel_size=3, stride=1, padding=0)  # (1, 3*3*3, 46*46)
print(f"Patches shape: {patches.shape}")  # Should be (1, 27, 2116)

# Reshape to (num_patches, in_features)
num_patches = patches.shape[2]
in_features = patches.shape[1]
patches = patches.squeeze(0).transpose(0, 1)  # (2116, 27)
print(f"Patches reshaped: {patches.shape}")

# Add column of 1s (for bias term)
ones_column = torch.ones(num_patches, 1)
patches_with_ones = torch.cat([patches, ones_column], dim=1)  # (2116, 28)
print(f"Patches with 1s column: {patches_with_ones.shape}")

# Prepare weights with bias as extra row (before transpose)
# C++ does: W(rows, cols+1) where last column = biases, then W.t()
# So weight_matrix_extended.T is (cols+1, rows) = (28, 32)
out_channels = 32
weight_matrix = torch.from_numpy(conv1_weights)  # (32, 3, 3, 3)

# Flatten to (out_channels, in_features)
weight_flat = weight_matrix.view(out_channels, in_features)  # (32, 27)
print(f"Weight flat: {weight_flat.shape}")

# Add bias as extra column
bias_column = torch.from_numpy(conv1_biases).unsqueeze(1)  # (32, 1)
weight_with_bias = torch.cat([weight_flat, bias_column], dim=1)  # (32, 28)
print(f"Weight with bias column: {weight_with_bias.shape}")

# Transpose for matrix multiply (C++ does W.t())
weight_transposed = weight_with_bias.t()  # (28, 32)
print(f"Weight transposed: {weight_transposed.shape}")

# Matrix multiply: (2116, 28) @ (28, 32) = (2116, 32)
out_manual = patches_with_ones @ weight_transposed  # (2116, 32)
print(f"Output from matmul: {out_manual.shape}")

# Transpose to (32, 2116)
out_manual = out_manual.t()  # (32, 2116)

# Reshape each channel back to (46, 46)
out_manual_reshaped = out_manual.view(1, out_channels, 46, 46)
print(f"Output reshaped: {out_manual_reshaped.shape}")

print(f"Output [0,0,0,0]: {out_manual_reshaped[0,0,0,0].item():.6f}")
print(f"Output range: [{out_manual_reshaped.min().item():.6f}, {out_manual_reshaped.max().item():.6f}]")

# ============================================================================
# COMPARISON
# ============================================================================
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

diff = torch.abs(out_standard - out_manual_reshaped).max()
print(f"Max difference between methods: {diff.item():.6e}")

if diff < 1e-5:
    print("✓ METHODS MATCH! Bias-in-weight-matrix is mathematically equivalent")
else:
    print("❌ METHODS DIFFER! Something is wrong with the implementation")
    print(f"\nStandard [0,0,0,0]: {out_standard[0,0,0,0].item():.6f}")
    print(f"Manual [0,0,0,0]: {out_manual_reshaped[0,0,0,0].item():.6f}")

# Now test against C++ expected values
print(f"\n{'='*80}")
print("COMPARISON WITH C++")
print(f"{'='*80}")
print(f"We get: {out_standard[0,0,0,0].item():.6f}")
print(f"C++ gets: (unknown - need to extract from C++ layer debug)")
print(f"\nCurrent final outputs:")
print(f"  Python ONet: logit[0]=-1.637, logit[1]=+1.638")
print(f"  C++ ONet:    logit[0]=-3.414, logit[1]=+3.413")
print(f"  Ratio: ~2.08x")
