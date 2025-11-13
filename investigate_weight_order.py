#!/usr/bin/env python3
"""
Investigate which rows differ between C++ and Python weight matrices.
"""

import numpy as np
import struct

# Load C++ ONet conv1 weight matrix (layer 6)
with open('/tmp/cpp_conv6_weight.bin', 'rb') as f:
    rows = struct.unpack('<I', f.read(4))[0]
    cols = struct.unpack('<I', f.read(4))[0]
    cpp_weight = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)

# Load our extracted weights and reconstruct
extracted_weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')  # (32, 3, 3, 3)
extracted_biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')    # (32,)

# Reconstruct in C++ format
out_channels = 32
in_features = 27
weight_flat = extracted_weights.reshape(out_channels, in_features)
weight_with_bias = np.column_stack([weight_flat, extracted_biases])
reconstructed = weight_with_bias.T  # (28, 32)

print("="*80)
print("ROW-BY-ROW COMPARISON")
print("="*80)

for row in range(rows):
    diff = np.abs(cpp_weight[row, :] - reconstructed[row, :]).max()
    if diff > 0.01:
        print(f"Row {row}: MAX DIFF = {diff:.6f} ❌")
        print(f"  C++:           {cpp_weight[row, :5]}")
        print(f"  Reconstructed: {reconstructed[row, :5]}")
    else:
        print(f"Row {row}: MATCH ✓ (max diff: {diff:.6e})")

# Now let's check if the issue is in how we reshape the kernels
print(f"\n{'='*80}")
print("INVESTIGATING KERNEL RESHAPE ORDER")
print(f"{'='*80}")

# C++ does this (from line 388-390):
# k_flat = kernels_rearr[k][i].t()  # Transpose the kernel!
# k_flat = k_flat.reshape(0, 1).t()  # Reshape to column
# k_flat.copyTo(weight_matrix(Rect(k, i*kernel_size, 1, kernel_size)))

# Our code does:
# weights[out_ch, in_ch, :, :] = kernels[idx]

# The C++ code transposes each kernel before flattening!
# Let me try that:

print("Testing with kernel transpose...")
weights_kernel_transposed = extracted_weights.copy()
for out_ch in range(out_channels):
    for in_ch in range(3):
        # Transpose each 3x3 kernel
        weights_kernel_transposed[out_ch, in_ch] = extracted_weights[out_ch, in_ch].T

# Reconstruct with transposed kernels
weight_flat_kt = weights_kernel_transposed.reshape(out_channels, in_features)
weight_with_bias_kt = np.column_stack([weight_flat_kt, extracted_biases])
reconstructed_kt = weight_with_bias_kt.T

# Compare
diff_kt = np.abs(cpp_weight - reconstructed_kt).max()
print(f"\nMax diff with kernel transpose: {diff_kt:.6e}")

if diff_kt < 1e-5:
    print("✓ KERNEL TRANSPOSE FIXES IT!")
else:
    print(f"❌ Still doesn't match (max diff: {diff_kt})")
    print("\nRow-by-row with kernel transpose:")
    for row in range(min(10, rows)):
        diff = np.abs(cpp_weight[row, :] - reconstructed_kt[row, :]).max()
        if diff > 0.01:
            print(f"Row {row}: diff = {diff:.6f}")
        else:
            print(f"Row {row}: MATCH ✓")
