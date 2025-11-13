#!/usr/bin/env python3
"""
Compare C++'s actual weight matrix vs our extracted weights.
This will show if the transpose or other transforms are causing issues.
"""

import numpy as np
import struct

# Load C++ ONet conv1 weight matrix (layer 6)
with open('/tmp/cpp_conv6_weight.bin', 'rb') as f:
    rows = struct.unpack('<I', f.read(4))[0]
    cols = struct.unpack('<I', f.read(4))[0]
    cpp_weight = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)

print("="*80)
print("C++ ONET CONV1 WEIGHT MATRIX")
print("="*80)
print(f"Shape: {cpp_weight.shape}")  # Should be (28, 32)
print(f"First value: {cpp_weight[0, 0]}")
print(f"Bias column (last column) first value: {cpp_weight[0, cols-1]}")

# Load our extracted ONet conv1 weights and biases
extracted_weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')  # (32, 3, 3, 3)
extracted_biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')    # (32,)

print(f"\n{'='*80}")
print("OUR EXTRACTED WEIGHTS")
print(f"{'='*80}")
print(f"Weights shape: {extracted_weights.shape}")  # (32, 3, 3, 3)
print(f"Biases shape: {extracted_biases.shape}")    # (32,)
print(f"First weight value [0,0,0,0]: {extracted_weights[0,0,0,0]}")
print(f"First bias value [0]: {extracted_biases[0]}")

# Reconstruct C++'s format from our extracted weights
print(f"\n{'='*80}")
print("RECONSTRUCTING C++ FORMAT FROM OUR EXTRACTED WEIGHTS")
print(f"{'='*80}")

# C++ weight matrix is (28, 32) = (in_features+1, out_channels)
# where in_features = 3*3*3 = 27
out_channels = 32
in_features = 27

# Flatten our extracted weights to (out_channels, in_features)
weight_flat = extracted_weights.reshape(out_channels, in_features)  # (32, 27)
print(f"Flattened weights: {weight_flat.shape}")

# Add bias column to make (32, 28)
weight_with_bias = np.column_stack([weight_flat, extracted_biases])  # (32, 28)
print(f"With bias column: {weight_with_bias.shape}")

# Transpose to match C++ format (28, 32)
reconstructed = weight_with_bias.T  # (28, 32)
print(f"Transposed to C++ format: {reconstructed.shape}")
print(f"First value: {reconstructed[0, 0]}")
print(f"Bias column first value: {reconstructed[27, 0]}")  # Bias is now in row 27

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

diff = np.abs(cpp_weight - reconstructed).max()
print(f"Max difference: {diff:.6e}")

if diff < 1e-5:
    print("✓ WEIGHTS MATCH! Our extraction is correct!")
else:
    print("❌ WEIGHTS DIFFER! There's an issue with extraction or C++ assembly")

    # Show where they differ
    print(f"\nC++ first row: {cpp_weight[0, :5]}")
    print(f"Reconstructed first row: {reconstructed[0, :5]}")

    print(f"\nC++ last row (bias): {cpp_weight[27, :5]}")
    print(f"Reconstructed last row (bias): {reconstructed[27, :5]}")

    # Check if it's just a transpose issue
    diff_transposed = np.abs(cpp_weight - reconstructed.T).max()
    print(f"\nMax diff if we transpose reconstructed: {diff_transposed:.6e}")
    if diff_transposed < 1e-5:
        print("⚠️  Weights match if we transpose! There's a transpose mismatch")

    # Check if kernel weights need transpose
    # Try transposing each 3x3 kernel
    print(f"\n{'='*80}")
    print("TRYING WITH KERNEL TRANSPOSE")
    print(f"{'='*80}")

    # Transpose each 3x3 kernel within each channel
    weights_with_kernel_transpose = extracted_weights.copy()
    for out_ch in range(out_channels):
        for in_ch in range(3):
            # Transpose the 3x3 kernel
            weights_with_kernel_transpose[out_ch, in_ch] = extracted_weights[out_ch, in_ch].T

    # Flatten and reconstruct
    weight_flat_transposed = weights_with_kernel_transpose.reshape(out_channels, in_features)
    weight_with_bias_transposed = np.column_stack([weight_flat_transposed, extracted_biases])
    reconstructed_transposed = weight_with_bias_transposed.T

    diff_with_kernel_transpose = np.abs(cpp_weight - reconstructed_transposed).max()
    print(f"Max diff with kernel transpose: {diff_with_kernel_transpose:.6e}")

    if diff_with_kernel_transpose < 1e-5:
        print("✓ WEIGHTS MATCH WITH KERNEL TRANSPOSE!")
        print("🎯 ROOT CAUSE: Each 3x3 kernel needs to be transposed!")
    else:
        print(f"Still doesn't match. Need to investigate further.")
        print(f"\nC++ first few values: {cpp_weight[:3, 0]}")
        print(f"Reconstructed (kernel transposed) first few values: {reconstructed_transposed[:3, 0]}")
