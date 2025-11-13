#!/usr/bin/env python3
"""
Compare C++ model weights vs ONNX model weights to find differences.
"""

import numpy as np
import onnx
import onnxruntime as ort

print("="*80)
print("C++ vs ONNX MODEL COMPARISON")
print("="*80)

# Load C++ PNet layer 0 weights
print("\n1. Loading C++ PNet Conv0 weights...")
with open('/tmp/cpp_conv0_weight.bin', 'rb') as f:
    rows = np.fromfile(f, dtype=np.int32, count=1)[0]
    cols = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_conv0 = np.fromfile(f, dtype=np.float32).reshape(rows, cols)

print(f"   C++ Conv0 shape: {cpp_conv0.shape}")
print(f"   First few values: {cpp_conv0[0, :5]}")
print(f"   Bias (last column): {cpp_conv0[:, -1]}")

# Load ONNX PNet model
print("\n2. Loading ONNX PNet model...")
pnet_model = onnx.load('cpp_mtcnn_onnx/pnet.onnx')
pnet_session = ort.InferenceSession('cpp_mtcnn_onnx/pnet.onnx')

# Extract ONNX weights
print("\n3. Extracting ONNX PNet Conv0 weights...")
initializers = {init.name: init for init in pnet_model.graph.initializer}

# Find first conv layer
conv0_weight_name = None
conv0_bias_name = None
for node in pnet_model.graph.node:
    if node.op_type == 'Conv':
        conv0_weight_name = node.input[1]
        conv0_bias_name = node.input[2] if len(node.input) > 2 else None
        break

if conv0_weight_name:
    print(f"   Found Conv0 weight: {conv0_weight_name}")
    weight_tensor = onnx.numpy_helper.to_array(initializers[conv0_weight_name])
    print(f"   ONNX Conv0 weight shape: {weight_tensor.shape}")
    print(f"   Format: (out_channels, in_channels, kernel_h, kernel_w)")

    if conv0_bias_name:
        bias_tensor = onnx.numpy_helper.to_array(initializers[conv0_bias_name])
        print(f"   ONNX Conv0 bias shape: {bias_tensor.shape}")
        print(f"   ONNX Conv0 bias values: {bias_tensor}")

    # Reconstruct C++ format from ONNX
    # ONNX: (out_ch, in_ch, kh, kw) → C++: (num_in * k_h * k_w + 1, num_out)
    num_out, num_in, kh, kw = weight_tensor.shape

    # C++ format: rows = num_in * kh * kw + 1 (for bias), cols = num_out
    onnx_as_cpp = np.zeros((num_in * kh * kw + 1, num_out), dtype=np.float32)

    for out_idx in range(num_out):
        for in_idx in range(num_in):
            # C++ stores transposed and flattened kernels
            kernel = weight_tensor[out_idx, in_idx, :, :].T  # Transpose
            kernel_flat = kernel.flatten()  # Flatten
            start_row = in_idx * kh * kw
            onnx_as_cpp[start_row:start_row + kh * kw, out_idx] = kernel_flat

        # Add bias
        if conv0_bias_name:
            onnx_as_cpp[-1, out_idx] = bias_tensor[out_idx]
        else:
            onnx_as_cpp[-1, out_idx] = 0.0

    print(f"\n4. Comparing C++ vs ONNX (reconstructed to C++ format)...")
    print(f"   Reconstructed ONNX shape: {onnx_as_cpp.shape}")
    print(f"   C++ shape: {cpp_conv0.shape}")

    if cpp_conv0.shape == onnx_as_cpp.shape:
        diff = np.abs(cpp_conv0 - onnx_as_cpp)
        print(f"   Max difference: {diff.max()}")
        print(f"   Mean difference: {diff.mean()}")
        print(f"   Values match: {np.allclose(cpp_conv0, onnx_as_cpp, atol=1e-5)}")

        if not np.allclose(cpp_conv0, onnx_as_cpp, atol=1e-5):
            print(f"\n   ⚠ WEIGHTS DO NOT MATCH!")
            print(f"   Showing first 5 mismatched values:")
            mismatch_idx = np.where(diff > 1e-5)
            for i in range(min(5, len(mismatch_idx[0]))):
                r, c = mismatch_idx[0][i], mismatch_idx[1][i]
                print(f"     [{r}, {c}]: C++={cpp_conv0[r,c]:.6f}, ONNX={onnx_as_cpp[r,c]:.6f}, diff={diff[r,c]:.6f}")
    else:
        print(f"   ⚠ SHAPE MISMATCH!")

# Check if there are PReLU or other activation layers
print(f"\n5. Checking ONNX model architecture...")
print(f"   Layers in ONNX PNet:")
for i, node in enumerate(pnet_model.graph.node):
    print(f"     {i}: {node.op_type} - {node.name}")

print(f"\n6. Checking for PReLU parameters...")
has_prelu = False
for node in pnet_model.graph.node:
    if node.op_type == 'PRelu':
        has_prelu = True
        prelu_slope_name = node.input[1]
        if prelu_slope_name in initializers:
            slope = onnx.numpy_helper.to_array(initializers[prelu_slope_name])
            print(f"   Found PReLU: {node.name}")
            print(f"   Slope shape: {slope.shape}")
            print(f"   Slope values (first 5): {slope.flatten()[:5]}")

if not has_prelu:
    print(f"   No PReLU layers found (may use ReLU instead)")

print(f"\n{'='*80}")
print("DIAGNOSIS:")
print(f"{'='*80}")
if conv0_weight_name and cpp_conv0.shape == onnx_as_cpp.shape:
    if np.allclose(cpp_conv0, onnx_as_cpp, atol=1e-5):
        print("✅ Conv0 weights match perfectly!")
        print("   The divergence must be in later layers or activation functions.")
    else:
        print("❌ Conv0 weights DO NOT match!")
        print("   The ONNX model uses different weights than C++.")
        print("   Need to export ONNX from the same source as C++.")
else:
    print("❌ Cannot compare - shape mismatch or missing weights")
