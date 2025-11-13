#!/usr/bin/env python3
"""
Test if flipping the kernels fixes the layer 0 output mismatch.

PyTorch Conv2d performs cross-correlation, not true convolution.
C++ might be doing true convolution which requires kernel flipping.
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
print("TESTING KERNEL FLIP HYPOTHESIS")
print("="*80)

# Load extracted weights
weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')  # (32, 3, 3, 3)
biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')    # (32,)

print(f"Original weights shape: {weights.shape}")
print(f"Original weights [0,0,:,:] (first kernel, first channel):")
print(weights[0, 0])

# Convert input to CHW
input_chw = np.transpose(cpp_input, (2, 0, 1))
x = torch.from_numpy(input_chw).unsqueeze(0)

# Test 1: Original (no flip)
print(f"\n{'='*80}")
print("TEST 1: NO FLIP (current approach)")
print(f"{'='*80}")

conv_no_flip = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv_no_flip.weight.data = torch.from_numpy(weights)
conv_no_flip.bias.data = torch.from_numpy(biases)

output_no_flip = conv_no_flip(x).squeeze(0).detach().numpy()
diff_no_flip = np.abs(cpp_layer0 - output_no_flip).max()

print(f"Max diff: {diff_no_flip:.6e}")
print(f"Python [0,0,0]: {output_no_flip[0,0,0]:.6f}")
print(f"C++    [0,0,0]: {cpp_layer0[0,0,0]:.6f}")

# Test 2: Flip both H and W dimensions
print(f"\n{'='*80}")
print("TEST 2: FLIP HEIGHT AND WIDTH")
print(f"{'='*80}")

weights_flipped_hw = np.flip(np.flip(weights, axis=2), axis=3).copy()  # .copy() to avoid negative strides
print(f"Flipped weights [0,0,:,:] (should be rotated 180°):")
print(weights_flipped_hw[0, 0])

conv_flip_hw = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv_flip_hw.weight.data = torch.from_numpy(weights_flipped_hw)
conv_flip_hw.bias.data = torch.from_numpy(biases)

output_flip_hw = conv_flip_hw(x).squeeze(0).detach().numpy()
diff_flip_hw = np.abs(cpp_layer0 - output_flip_hw).max()

print(f"Max diff: {diff_flip_hw:.6e}")
print(f"Python [0,0,0]: {output_flip_hw[0,0,0]:.6f}")
print(f"C++    [0,0,0]: {cpp_layer0[0,0,0]:.6f}")

if diff_flip_hw < 1e-5:
    print("🎯 KERNEL FLIP FIXES IT!")
elif diff_flip_hw < diff_no_flip:
    print(f"⚠️  Kernel flip reduces error by {(1 - diff_flip_hw/diff_no_flip)*100:.1f}%")
else:
    print("❌ Kernel flip doesn't help")

# Test 3: Try kernel transpose (rot90)
print(f"\n{'='*80}")
print("TEST 3: KERNEL TRANSPOSE (swap H and W)")
print(f"{'='*80}")

weights_transposed = np.transpose(weights, (0, 1, 3, 2))  # Swap last two dims
print(f"Transposed weights [0,0,:,:]:")
print(weights_transposed[0, 0])

conv_transpose = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv_transpose.weight.data = torch.from_numpy(weights_transposed)
conv_transpose.bias.data = torch.from_numpy(biases)

output_transpose = conv_transpose(x).squeeze(0).detach().numpy()
diff_transpose = np.abs(cpp_layer0 - output_transpose).max()

print(f"Max diff: {diff_transpose:.6e}")
print(f"Python [0,0,0]: {output_transpose[0,0,0]:.6f}")
print(f"C++    [0,0,0]: {cpp_layer0[0,0,0]:.6f}")

if diff_transpose < 1e-5:
    print("🎯 KERNEL TRANSPOSE FIXES IT!")
elif diff_transpose < diff_no_flip:
    print(f"⚠️  Kernel transpose reduces error by {(1 - diff_transpose/diff_no_flip)*100:.1f}%")
else:
    print("❌ Kernel transpose doesn't help")

# Test 4: Flip and transpose
print(f"\n{'='*80}")
print("TEST 4: FLIP + TRANSPOSE")
print(f"{'='*80}")

weights_flip_transpose = np.flip(np.flip(np.transpose(weights, (0, 1, 3, 2)), axis=2), axis=3).copy()

conv_flip_transpose = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv_flip_transpose.weight.data = torch.from_numpy(weights_flip_transpose)
conv_flip_transpose.bias.data = torch.from_numpy(biases)

output_flip_transpose = conv_flip_transpose(x).squeeze(0).detach().numpy()
diff_flip_transpose = np.abs(cpp_layer0 - output_flip_transpose).max()

print(f"Max diff: {diff_flip_transpose:.6e}")
print(f"Python [0,0,0]: {output_flip_transpose[0,0,0]:.6f}")
print(f"C++    [0,0,0]: {cpp_layer0[0,0,0]:.6f}")

if diff_flip_transpose < 1e-5:
    print("🎯 FLIP + TRANSPOSE FIXES IT!")
elif diff_flip_transpose < diff_no_flip:
    print(f"⚠️  Flip + transpose reduces error by {(1 - diff_flip_transpose/diff_no_flip)*100:.1f}%")
else:
    print("❌ Flip + transpose doesn't help")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"No flip:         {diff_no_flip:.6e}")
print(f"Flip H+W:        {diff_flip_hw:.6e}")
print(f"Transpose:       {diff_transpose:.6e}")
print(f"Flip+Transpose:  {diff_flip_transpose:.6e}")

best = min(diff_no_flip, diff_flip_hw, diff_transpose, diff_flip_transpose)
if best == diff_no_flip:
    print("\n✓ Best: NO FLIP (current approach)")
elif best == diff_flip_hw:
    print("\n🎯 Best: FLIP HEIGHT AND WIDTH")
elif best == diff_transpose:
    print("\n🎯 Best: TRANSPOSE")
else:
    print("\n🎯 Best: FLIP + TRANSPOSE")
