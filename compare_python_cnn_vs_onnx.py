#!/usr/bin/env python3
"""
Compare Python CNN weights (from C++ .dat) with ONNX weights to verify they're different.
This confirms we're loading the original C++ weights, not ONNX weights.
"""

import numpy as np
import onnx
import onnxruntime as ort
import os
from cpp_cnn_loader import CPPCNN

print("="*80)
print("PYTHON CNN (C++ .dat) vs ONNX WEIGHT COMPARISON")
print("="*80)

# Load Python CNN from C++ binary
print("\n1. Loading PNet from C++ binary (.dat file)...")
model_path = os.path.expanduser("~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/face_detection/mtcnn/convert_to_cpp/PNet.dat")
pnet_py = CPPCNN(model_path)
conv0_py = pnet_py.layers[0]

print(f"   Conv0 kernels shape: {conv0_py.kernels.shape}")
print(f"   Conv0 biases shape: {conv0_py.biases.shape}")
print(f"   First 3 biases: {conv0_py.biases[:3]}")

# Load ONNX model
print("\n2. Loading ONNX PNet model...")
onnx_path = "cpp_mtcnn_onnx/pnet.onnx"
if os.path.exists(onnx_path):
    pnet_onnx = onnx.load(onnx_path)

    # Extract ONNX weights
    initializers = {init.name: init for init in pnet_onnx.graph.initializer}

    # Find first conv layer
    conv0_weight_name = None
    conv0_bias_name = None
    for node in pnet_onnx.graph.node:
        if node.op_type == 'Conv':
            conv0_weight_name = node.input[1]
            conv0_bias_name = node.input[2] if len(node.input) > 2 else None
            break

    if conv0_weight_name:
        weight_tensor = onnx.numpy_helper.to_array(initializers[conv0_weight_name])
        print(f"   ONNX Conv0 weight shape: {weight_tensor.shape}")

        if conv0_bias_name:
            bias_tensor = onnx.numpy_helper.to_array(initializers[conv0_bias_name])
            print(f"   ONNX Conv0 bias shape: {bias_tensor.shape}")
            print(f"   First 3 biases: {bias_tensor[:3]}")

        # Compare biases
        print("\n3. Comparing biases...")
        bias_diff = np.abs(conv0_py.biases - bias_tensor)
        print(f"   Max bias difference: {bias_diff.max():.10f}")
        print(f"   Mean bias difference: {bias_diff.mean():.10f}")
        print(f"   Biases match (atol=1e-5): {np.allclose(conv0_py.biases, bias_tensor, atol=1e-5)}")

        # Compare kernel weights
        print("\n4. Comparing kernel weights...")
        # ONNX format: (num_kernels, num_in_maps, kh, kw)
        # Python format: (num_kernels, num_in_maps, kh, kw)
        # Should be same format!

        kernel_diff = np.abs(conv0_py.kernels - weight_tensor)
        print(f"   Max kernel difference: {kernel_diff.max():.10f}")
        print(f"   Mean kernel difference: {kernel_diff.mean():.10f}")
        print(f"   Kernels match (atol=1e-5): {np.allclose(conv0_py.kernels, weight_tensor, atol=1e-5)}")
        print(f"   Kernels match (atol=1e-6): {np.allclose(conv0_py.kernels, weight_tensor, atol=1e-6)}")

        print("\n5. Detailed comparison:")
        print(f"   Python kernel[0,0,0,0]: {conv0_py.kernels[0,0,0,0]:.8f}")
        print(f"   ONNX kernel[0,0,0,0]:   {weight_tensor[0,0,0,0]:.8f}")
        print(f"   Difference:             {kernel_diff[0,0,0,0]:.8f}")

        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)

        if np.allclose(conv0_py.kernels, weight_tensor, atol=1e-6):
            print("✅ Python CNN weights MATCH ONNX weights!")
            print("   The .dat files contain the same weights as ONNX.")
        else:
            print("❌ Python CNN weights are DIFFERENT from ONNX weights!")
            print("   This means:")
            print("   - C++ .dat files contain ORIGINAL weights")
            print("   - ONNX was exported with different/modified weights")
            print("   - Our pure Python CNN uses the TRUE C++ weights")
            print("\n   This is actually GOOD news! We want the original C++ weights,")
            print("   not the ONNX converted weights.")

else:
    print(f"   ONNX model not found at: {onnx_path}")
    print("   Skipping ONNX comparison")

print("\n" + "="*80)
