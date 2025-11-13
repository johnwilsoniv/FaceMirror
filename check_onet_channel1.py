#!/usr/bin/env python3
"""
Check if ONet channel 1 also has divergence issues.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Load C++ ONet input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(48, 48, 3)  # HWC

# Load extracted ONet layer 0 weights
weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')
biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')

# Python computation
py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
conv_weights = torch.from_numpy(weights).float()
conv_biases = torch.from_numpy(biases).float()

with torch.no_grad():
    py_output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)

py_output_np = py_output[0].numpy()

# Load C++ layer 0 output
with open('/tmp/cpp_layer0_after_conv_output.bin', 'rb') as f:
    num_channels = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_output = np.fromfile(f, dtype=np.float32).reshape(num_channels, height, width)

# Compute differences
diff = np.abs(cpp_output - py_output_np)

print("="*80)
print("ONET CHANNEL ANALYSIS")
print("="*80)
print(f"\nOutput shape: {cpp_output.shape}")

print(f"\nChannel-wise max differences:")
for ch in range(min(10, num_channels)):
    ch_max_diff = diff[ch].max()
    ch_mean_diff = diff[ch].mean()
    print(f"  Channel {ch}: max={ch_max_diff:.10f}, mean={ch_mean_diff:.10f}")

# Check if channel 1 has issues
if num_channels > 1:
    ch1_max = diff[1].max()
    ch1_mean = diff[1].mean()
    print(f"\n{'='*80}")
    print(f"CHANNEL 1 SPECIFIC:")
    print(f"{'='*80}")
    print(f"Max diff: {ch1_max:.10f}")
    print(f"Mean diff: {ch1_mean:.10f}")

    # Find max divergence in channel 1
    max_idx = np.unravel_index(np.argmax(diff[1]), diff[1].shape)
    print(f"Worst position: h={max_idx[0]}, w={max_idx[1]}")
    print(f"  C++ value: {cpp_output[1][max_idx]:.10f}")
    print(f"  Python value: {py_output_np[1][max_idx]:.10f}")
