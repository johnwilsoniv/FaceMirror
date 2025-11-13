#!/usr/bin/env python3
"""
Compare C++ vs Python layer 0 (first conv) output to find where divergence starts.
"""

import numpy as np
import torch
import torch.nn as nn
import struct

# Load C++ ONet input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))

# Load C++ layer 0 output
with open('/tmp/cpp_layer0_after_conv_output.bin', 'rb') as f:
    cpp_num_channels = struct.unpack('<I', f.read(4))[0]
    cpp_height = struct.unpack('<I', f.read(4))[0]
    cpp_width = struct.unpack('<I', f.read(4))[0]
    cpp_layer0 = np.frombuffer(f.read(), dtype=np.float32).reshape(cpp_num_channels, cpp_height, cpp_width)

print("="*80)
print("C++ LAYER 0 OUTPUT")
print("="*80)
print(f"Shape: {cpp_layer0.shape}")
print(f"Range: [{cpp_layer0.min():.6f}, {cpp_layer0.max():.6f}]")
print(f"Sample [0,0,0]: {cpp_layer0[0,0,0]:.6f}")
print(f"Sample [0,0,1]: {cpp_layer0[0,0,1]:.6f}")
print(f"Sample [1,0,0]: {cpp_layer0[1,0,0]:.6f}")

# Compute Python layer 0 output
input_chw = np.transpose(cpp_input, (2, 0, 1))  # HWC -> CHW
x = torch.from_numpy(input_chw).unsqueeze(0)  # (1, 3, 48, 48)

conv1_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy'))
conv1_biases = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy'))

conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv1.weight.data = conv1_weights
conv1.bias.data = conv1_biases

python_layer0 = conv1(x).squeeze(0).detach().numpy()  # (32, 46, 46)

print(f"\n{'='*80}")
print("PYTHON LAYER 0 OUTPUT")
print(f"{'='*80}")
print(f"Shape: {python_layer0.shape}")
print(f"Range: [{python_layer0.min():.6f}, {python_layer0.max():.6f}]")
print(f"Sample [0,0,0]: {python_layer0[0,0,0]:.6f}")
print(f"Sample [0,0,1]: {python_layer0[0,0,1]:.6f}")
print(f"Sample [1,0,0]: {python_layer0[1,0,0]:.6f}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

if python_layer0.shape != cpp_layer0.shape:
    print(f"❌ SHAPE MISMATCH!")
    print(f"  C++:    {cpp_layer0.shape}")
    print(f"  Python: {python_layer0.shape}")
else:
    diff = np.abs(cpp_layer0 - python_layer0)
    print(f"Max absolute difference: {diff.max():.6e}")
    print(f"Mean absolute difference: {diff.mean():.6e}")

    if diff.max() < 1e-5:
        print("✓ LAYER 0 OUTPUTS MATCH!")
    else:
        print("❌ LAYER 0 OUTPUTS DIFFER!")

        # Find where the biggest differences are
        max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"\nBiggest difference at index {max_diff_idx}:")
        print(f"  C++:    {cpp_layer0[max_diff_idx]:.6f}")
        print(f"  Python: {python_layer0[max_diff_idx]:.6f}")
        print(f"  Diff:   {diff[max_diff_idx]:.6f}")

        # Show some sample differences
        print(f"\nSample differences [channel, row, col]:")
        for ch in range(min(3, cpp_num_channels)):
            for row in range(min(3, cpp_height)):
                for col in range(min(3, cpp_width)):
                    cpp_val = cpp_layer0[ch, row, col]
                    py_val = python_layer0[ch, row, col]
                    diff_val = abs(cpp_val - py_val)
                    if diff_val > 0.01:
                        print(f"  [{ch},{row},{col}]: C++={cpp_val:.6f}, Py={py_val:.6f}, diff={diff_val:.6f}")

        # Check if there's a systematic scaling
        ratio = cpp_layer0 / (python_layer0 + 1e-10)  # Avoid div by zero
        valid_ratios = ratio[np.abs(python_layer0) > 0.01]  # Only where Python has significant values
        if len(valid_ratios) > 0:
            print(f"\nScaling analysis (C++ / Python):")
            print(f"  Mean ratio: {valid_ratios.mean():.6f}")
            print(f"  Median ratio: {np.median(valid_ratios):.6f}")
            print(f"  Std ratio: {valid_ratios.std():.6f}")
            if abs(valid_ratios.mean() - 2.0) < 0.1:
                print(f"  🎯 FOUND IT! C++ output is ~2x Python output!")
