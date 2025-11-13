#!/usr/bin/env python3
"""
Manually compute one convolution output pixel using C++'s exact method:
1. Extract 3x3 patch from input (for each channel)
2. Flatten in column-major order (as C++ im2col_multimap does)
3. Multiply with weight matrix
4. Compare with PyTorch Conv2d output
"""

import numpy as np
import torch
import torch.nn as nn
import struct

# Load C++ ONet input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))

# Load weights
weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')  # (32, 3, 3, 3)
biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')    # (32,)

# Load C++ layer 0 output
with open('/tmp/cpp_layer0_after_conv_output.bin', 'rb') as f:
    cpp_num_channels = struct.unpack('<I', f.read(4))[0]
    cpp_height = struct.unpack('<I', f.read(4))[0]
    cpp_width = struct.unpack('<I', f.read(4))[0]
    cpp_layer0 = np.frombuffer(f.read(), dtype=np.float32).reshape(cpp_num_channels, cpp_height, cpp_width)

print("="*80)
print("MANUAL CONVOLUTION COMPUTATION")
print("="*80)

# Focus on output pixel [0, 0, 0] (channel 0, top-left)
out_ch = 0
out_y = 0
out_x = 0

# Input patch starts at (0, 0) for output (0, 0)
in_y = out_y
in_x = out_x

print(f"\nComputing output[{out_ch}, {out_y}, {out_x}]")
print(f"Input patch starts at ({in_y}, {in_x})")

# Extract 3x3 patches from each input channel (HWC format)
# C++ input_maps is a vector of channels
patch = []
for in_ch in range(3):
    channel = cpp_input[in_y:in_y+3, in_x:in_x+3, in_ch]  # 3x3 patch
    print(f"\nInput channel {in_ch} patch (HWC):")
    print(channel)

    # Flatten in COLUMN-MAJOR order (as C++ im2col_multimap does at line 487)
    # colIdx = xx*height + yy
    patch_flat = []
    for xx in range(3):  # width
        for yy in range(3):  # height
            patch_flat.append(channel[yy, xx])

    print(f"Flattened (column-major): {patch_flat}")
    patch.extend(patch_flat)

# Add bias term (im2col adds a column of 1.0 at the end)
patch.append(1.0)
patch = np.array(patch, dtype=np.float32)

print(f"\n{'='*80}")
print(f"FULL PATCH VECTOR (3 channels + bias)")
print(f"{'='*80}")
print(f"Length: {len(patch)} (should be 3*9 + 1 = 28)")
print(f"Values: {patch}")

# Now apply C++ weight matrix
# C++ weight matrix is (28, 32) = (in_features+1, out_channels)
# Load the weight matrix we saved from C++
with open('/tmp/cpp_conv6_weight.bin', 'rb') as f:
    rows = struct.unpack('<I', f.read(4))[0]
    cols = struct.unpack('<I', f.read(4))[0]
    cpp_weight = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)

print(f"\n{'='*80}")
print(f"C++ WEIGHT MATRIX")
print(f"{'='*80}")
print(f"Shape: {cpp_weight.shape}")

# Compute output as patch @ weight_matrix[:, out_ch]
manual_output = patch @ cpp_weight[:, out_ch]

print(f"\n{'='*80}")
print(f"MANUAL COMPUTATION (C++ method)")
print(f"{'='*80}")
print(f"patch @ weight[:, {out_ch}] = {manual_output:.6f}")
print(f"C++ layer 0 output[{out_ch}, {out_y}, {out_x}] = {cpp_layer0[out_ch, out_y, out_x]:.6f}")
print(f"Difference: {abs(manual_output - cpp_layer0[out_ch, out_y, out_x]):.6e}")

if abs(manual_output - cpp_layer0[out_ch, out_y, out_x]) < 1e-5:
    print("✓ MANUAL COMPUTATION MATCHES C++!")
else:
    print("❌ MANUAL COMPUTATION DOESN'T MATCH C++")

# Now compute with PyTorch Conv2d
print(f"\n{'='*80}")
print(f"PYTORCH CONV2D")
print(f"{'='*80}")

# Convert input to CHW
input_chw = np.transpose(cpp_input, (2, 0, 1))
x = torch.from_numpy(input_chw).unsqueeze(0)

conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv.weight.data = torch.from_numpy(weights)
conv.bias.data = torch.from_numpy(biases)

pytorch_output = conv(x).squeeze(0).detach().numpy()

print(f"PyTorch output[{out_ch}, {out_y}, {out_x}] = {pytorch_output[out_ch, out_y, out_x]:.6f}")
print(f"C++ output[{out_ch}, {out_y}, {out_x}] = {cpp_layer0[out_ch, out_y, out_x]:.6f}")
print(f"Difference: {abs(pytorch_output[out_ch, out_y, out_x] - cpp_layer0[out_ch, out_y, out_x]):.6e}")

# Now let's manually compute what PyTorch should be doing
print(f"\n{'='*80}")
print(f"MANUAL PYTORCH-STYLE COMPUTATION")
print(f"{'='*80}")

# PyTorch Conv2d computes: sum over all input channels of (input_patch * kernel)
# Extract patches in CHW order
pytorch_manual = 0.0
for in_ch in range(3):
    # Extract 3x3 patch from input (CHW format now)
    patch_chw = input_chw[in_ch, in_y:in_y+3, in_x:in_x+3]  # 3x3

    # Get corresponding kernel
    kernel = weights[out_ch, in_ch]  # 3x3

    print(f"\nInput channel {in_ch} patch (CHW):")
    print(patch_chw)
    print(f"Kernel weights[{out_ch}, {in_ch}]:")
    print(kernel)

    # Element-wise multiply and sum
    conv_result = np.sum(patch_chw * kernel)
    print(f"Sum(patch * kernel) = {conv_result:.6f}")

    pytorch_manual += conv_result

# Add bias
pytorch_manual += biases[out_ch]
print(f"\nAfter adding bias {biases[out_ch]:.6f}: {pytorch_manual:.6f}")
print(f"PyTorch output: {pytorch_output[out_ch, out_y, out_x]:.6f}")
print(f"Difference: {abs(pytorch_manual - pytorch_output[out_ch, out_y, out_x]):.6e}")

if abs(pytorch_manual - pytorch_output[out_ch, out_y, out_x]) < 1e-5:
    print("✓ MANUAL PYTORCH COMPUTATION MATCHES!")

# Compare manual methods
print(f"\n{'='*80}")
print(f"COMPARISON")
print(f"{'='*80}")
print(f"C++ method result:     {manual_output:.6f}")
print(f"PyTorch method result: {pytorch_manual:.6f}")
print(f"Difference:            {abs(manual_output - pytorch_manual):.6f}")

if abs(manual_output - pytorch_manual) < 1e-5:
    print("✓ BOTH METHODS GIVE SAME RESULT!")
else:
    print("❌ METHODS DIFFER - THIS IS THE ROOT CAUSE!")
