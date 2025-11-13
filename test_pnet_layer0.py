#!/usr/bin/env python3
"""
Test PNet layer 0 to verify the transpose fix works for PNet.
This replicates the test from BUGS_SQUASHED.md for PNet.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Load C++ PNet input (from earlier PNet run)
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)  # HWC

print("C++ PNet Input:")
print(f"  Shape: {cpp_input.shape}")
print(f"  Sample pixel [0,0]: {cpp_input[0,0,:]}")

# Load extracted PNet layer 0 weights
pnet_conv1_weights = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_weights.npy')
pnet_conv1_biases = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_biases.npy')

print(f"\nPNet Layer 0 Weights:")
print(f"  Shape: {pnet_conv1_weights.shape}")  # Should be (10, 3, 3, 3)
print(f"  Bias shape: {pnet_conv1_biases.shape}")  # Should be (10,)

# Convert to PyTorch format (HWC -> CHW)
py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
print(f"\nPyTorch input shape: {py_input.shape}")  # (1, 3, 384, 216)

# Run layer 0 convolution
conv_weights = torch.from_numpy(pnet_conv1_weights).float()
conv_biases = torch.from_numpy(pnet_conv1_biases).float()

with torch.no_grad():
    output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)

print(f"\nPyTorch Layer 0 Output:")
print(f"  Shape: {output.shape}")  # Should be (1, 10, 382, 214)
print(f"  Value at [0,0,0]: {output[0, 0, 0, 0].item()}")

# Load C++ layer 0 output file
cpp_layer0_file = '/tmp/cpp_pnet_layer0_after_conv_output.bin'
with open(cpp_layer0_file, 'rb') as f:
    # Read dimensions (3 x int32)
    num_channels = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    width = np.fromfile(f, dtype=np.int32, count=1)[0]

    print(f"\nC++ Layer 0 Output (from file):")
    print(f"  Dimensions: {num_channels}x{height}x{width}")

    # Read data (channels x H x W)
    cpp_layer0 = np.fromfile(f, dtype=np.float32).reshape(num_channels, height, width)

print(f"  Shape: {cpp_layer0.shape}")
print(f"  Value at [0,0,0]: {cpp_layer0[0, 0, 0]}")

# Compare
py_layer0 = output[0].numpy()  # Remove batch dimension
diff = np.abs(cpp_layer0 - py_layer0)

print(f"\n{'='*80}")
print(f"LAYER 0 COMPARISON:")
print(f"{'='*80}")
print(f"Max difference: {diff.max():.10f}")
print(f"Mean difference: {diff.mean():.10f}")
print(f"Value at [0,0,0] - C++: {cpp_layer0[0,0,0]:.10f}, Python: {py_layer0[0,0,0]:.10f}")

# Show distribution of differences
print(f"\nDifference distribution:")
print(f"  < 1e-7: {(diff < 1e-7).sum()} / {diff.size} = {100*(diff < 1e-7).sum()/diff.size:.1f}%")
print(f"  < 1e-6: {(diff < 1e-6).sum()} / {diff.size} = {100*(diff < 1e-6).sum()/diff.size:.1f}%")
print(f"  < 1e-5: {(diff < 1e-5).sum()} / {diff.size} = {100*(diff < 1e-5).sum()/diff.size:.1f}%")

if diff.max() < 1e-5:
    print(f"\n✅ PNet Layer 0 MATCHES! (max diff < 1e-5)")
    print(f"   Transpose fix is WORKING for PNet!")
    print(f"   This means weight extraction is correct.")
else:
    print(f"\n❌ PNet Layer 0 DOES NOT MATCH!")
    print(f"   Transpose fix is NOT working!")
    print(f"   Need to investigate weight loading.")
