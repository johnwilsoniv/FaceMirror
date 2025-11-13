#!/usr/bin/env python3
"""
Test ONet layer 0 to verify the transpose fix works for ONet.
This replicates the test from BUGS_SQUASHED.md.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Load C++ ONet input (48x48x3)
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(48, 48, 3)  # HWC

print("C++ ONet Input:")
print(f"  Shape: {cpp_input.shape}")
print(f"  Sample pixel [0,0]: {cpp_input[0,0,:]}")

# Load extracted ONet layer 0 weights
onet_conv1_weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')
onet_conv1_biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')

print(f"\nONet Layer 0 Weights:")
print(f"  Shape: {onet_conv1_weights.shape}")  # Should be (32, 3, 3, 3)
print(f"  Bias shape: {onet_conv1_biases.shape}")  # Should be (32,)

# Convert to PyTorch format (HWC -> CHW)
py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
print(f"\nPyTorch input shape: {py_input.shape}")  # (1, 3, 48, 48)

# Run layer 0 convolution
conv_weights = torch.from_numpy(onet_conv1_weights).float()
conv_biases = torch.from_numpy(onet_conv1_biases).float()

with torch.no_grad():
    output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)

print(f"\nPyTorch Layer 0 Output:")
print(f"  Shape: {output.shape}")  # Should be (1, 32, 46, 46)
print(f"  Value at [0,0,0]: {output[0, 0, 0, 0].item()}")

# Check if we have C++ layer 0 output file
import os
if os.path.exists('/tmp/cpp_layer0_after_conv_output.bin'):
    cpp_layer0 = np.fromfile('/tmp/cpp_layer0_after_conv_output.bin', dtype=np.float32)
    cpp_layer0 = cpp_layer0.reshape(32, 46, 46)  # C++ format: (channels, H, W)

    print(f"\nC++ Layer 0 Output:")
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

    if diff.max() < 1e-5:
        print(f"\n✅ ONet Layer 0 MATCHES! (max diff < 1e-5)")
        print(f"   Transpose fix is WORKING for ONet!")
    else:
        print(f"\n❌ ONet Layer 0 DOES NOT MATCH!")
        print(f"   Transpose fix is NOT working!")
else:
    print(f"\n⚠️  C++ layer 0 output file not found")
    print(f"   Need to regenerate C++ debug files with ONet layer 0 logging")
    print(f"   Expected location: /tmp/cpp_layer0_after_conv_output.bin")
