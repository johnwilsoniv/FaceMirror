#!/usr/bin/env python3
"""
Test RNet layer 0 against C++ gold standard.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Load C++ RNet input (24x24x3)
cpp_input = np.fromfile('/tmp/cpp_rnet_input.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(24, 24, 3)  # HWC

print("="*80)
print("RNET LAYER 0 COMPARISON AGAINST C++ GOLD STANDARD")
print("="*80)

print(f"\nC++ RNet Input:")
print(f"  Shape: {cpp_input.shape}")
print(f"  Sample pixel [0,0]: {cpp_input[0,0,:]}")

# Load extracted RNet layer 0 weights
rnet_conv1_weights = np.load('cpp_mtcnn_weights/rnet/rnet_layer00_conv_weights.npy')
rnet_conv1_biases = np.load('cpp_mtcnn_weights/rnet/rnet_layer00_conv_biases.npy')

print(f"\nRNet Layer 0 Weights:")
print(f"  Shape: {rnet_conv1_weights.shape}")  # Should be (28, 3, 3, 3)
print(f"  Bias shape: {rnet_conv1_biases.shape}")  # Should be (28,)

# Convert to PyTorch format (HWC -> CHW)
py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
print(f"\nPyTorch input shape: {py_input.shape}")  # (1, 3, 24, 24)

# Run layer 0 convolution
conv_weights = torch.from_numpy(rnet_conv1_weights).float()
conv_biases = torch.from_numpy(rnet_conv1_biases).float()

with torch.no_grad():
    output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)

print(f"\nPyTorch Layer 0 Output:")
print(f"  Shape: {output.shape}")  # Should be (1, 28, 22, 22)
print(f"  Value at [0,0,0]: {output[0, 0, 0, 0].item()}")

# Load C++ layer 0 output file
cpp_layer0_file = '/tmp/cpp_rnet_layer0_after_conv_output.bin'
with open(cpp_layer0_file, 'rb') as f:
    # Read dimensions (3 x int32)
    num_channels = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    width = np.fromfile(f, dtype=np.int32, count=1)[0]

    print(f"\nC++ Layer 0 Output (GOLD STANDARD):")
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

# Channel-wise analysis
print(f"\nChannel-wise max differences:")
for ch in range(min(10, num_channels)):
    ch_max_diff = diff[ch].max()
    ch_mean_diff = diff[ch].mean()
    print(f"  Channel {ch}: max={ch_max_diff:.10f}, mean={ch_mean_diff:.10f}")

print(f"\n{'='*80}")
print(f"CONCLUSION:")
print(f"{'='*80}")
if diff.max() < 1e-5:
    print(f"✅ RNet Layer 0 MATCHES C++ GOLD STANDARD (max diff < 1e-5)")
    print(f"   Weight extraction is CORRECT for RNet!")
    print(f"   Same as ONet behavior.")
elif diff.max() < 1e-3:
    print(f"⚠️  RNet Layer 0 MOSTLY matches (max diff < 1e-3)")
    print(f"   Small numerical differences - likely acceptable.")
else:
    print(f"❌ RNet Layer 0 DIVERGES from C++ (max diff = {diff.max():.6f})")
    print(f"   Similar to PNet behavior - systematic difference from C++.")
