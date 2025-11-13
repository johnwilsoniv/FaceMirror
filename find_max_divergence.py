#!/usr/bin/env python3
"""
Find where C++ and Python PNet layer 0 outputs diverge the most.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Load C++ PNet input
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)  # HWC

# Load extracted PNet layer 0 weights
weights = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_weights.npy')
biases = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_biases.npy')

# Python computation
py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
conv_weights = torch.from_numpy(weights).float()
conv_biases = torch.from_numpy(biases).float()

with torch.no_grad():
    py_output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)

py_output_np = py_output[0].numpy()

# Load C++ layer 0 output
with open('/tmp/cpp_pnet_layer0_after_conv_output.bin', 'rb') as f:
    num_channels = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_output = np.fromfile(f, dtype=np.float32).reshape(num_channels, height, width)

# Compute differences
diff = np.abs(cpp_output - py_output_np)

# Find max divergence
max_idx = np.unravel_index(np.argmax(diff), diff.shape)
max_diff = diff[max_idx]

print("="*80)
print("DIVERGENCE ANALYSIS")
print("="*80)
print(f"\nOutput shape: {cpp_output.shape}")
print(f"Total elements: {diff.size}")

print(f"\nMax difference: {max_diff:.6f}")
print(f"Max diff location: channel={max_idx[0]}, h={max_idx[1]}, w={max_idx[2]}")
print(f"  C++ value: {cpp_output[max_idx]:.6f}")
print(f"  Python value: {py_output_np[max_idx]:.6f}")

# Show distribution of differences
print(f"\nDifference distribution:")
print(f"  Mean: {diff.mean():.6f}")
print(f"  Median: {np.median(diff):.6f}")
print(f"  < 1e-5: {100*(diff < 1e-5).sum()/diff.size:.1f}%")
print(f"  < 1e-4: {100*(diff < 1e-4).sum()/diff.size:.1f}%")
print(f"  < 1e-3: {100*(diff < 1e-3).sum()/diff.size:.1f}%")
print(f"  < 1e-2: {100*(diff < 1e-2).sum()/diff.size:.1f}%")
print(f"  > 1.0:  {100*(diff > 1.0).sum()/diff.size:.1f}%")

# Find top 10 worst positions
worst_indices = np.argsort(diff.flatten())[-10:][::-1]
worst_indices_3d = [np.unravel_index(idx, diff.shape) for idx in worst_indices]

print(f"\nTop 10 worst divergences:")
for i, idx in enumerate(worst_indices_3d):
    d = diff[idx]
    cpp_val = cpp_output[idx]
    py_val = py_output_np[idx]
    print(f"  {i+1}. [{idx[0]:2d},{idx[1]:3d},{idx[2]:3d}]: diff={d:.6f}, C++={cpp_val:.6f}, Py={py_val:.6f}")

# Check if there's a pattern
print(f"\nPattern analysis:")
print(f"  Worst channel: {max_idx[0]}")
print(f"  Worst height: {max_idx[1]}")
print(f"  Worst width: {max_idx[2]}")

# Check channel-wise statistics
print(f"\nChannel-wise max differences:")
for ch in range(10):
    ch_max_diff = diff[ch].max()
    ch_mean_diff = diff[ch].mean()
    print(f"  Channel {ch}: max={ch_max_diff:.6f}, mean={ch_mean_diff:.6f}")
