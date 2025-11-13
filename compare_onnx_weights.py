#!/usr/bin/env python3
"""
Compare weights in ONNX model vs extracted .npy weights.
This will tell us if the ONNX conversion preserved the weights correctly.
"""

import numpy as np
import onnx

# Load ONNX model
model = onnx.load('cpp_mtcnn_onnx/onet.onnx')

print("="*80)
print("COMPARING ONNX MODEL WEIGHTS vs EXTRACTED .NPY WEIGHTS")
print("="*80)

# Extract weights from ONNX model
onnx_weights = {}
for initializer in model.graph.initializer:
    name = initializer.name
    dims = initializer.dims
    data = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(dims)
    onnx_weights[name] = data
    print(f"\nONNX weight '{name}':")
    print(f"  Shape: {data.shape}")
    print(f"  Range: [{data.min():.6f}, {data.max():.6f}]")
    if data.size > 0:
        print(f"  Sample [0]: {data.flat[0]:.6f}")

# Load extracted weights
print("\n" + "="*80)
print("EXTRACTED WEIGHTS FROM .NPY FILES")
print("="*80)

conv1_weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')
conv1_biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')
prelu1_weights = np.load('cpp_mtcnn_weights/onet/onet_layer01_prelu_weights.npy')

print(f"\nExtracted conv1_weights:")
print(f"  Shape: {conv1_weights.shape}")
print(f"  Range: [{conv1_weights.min():.6f}, {conv1_weights.max():.6f}]")
print(f"  Sample [0,0,0,0]: {conv1_weights[0,0,0,0]:.6f}")

print(f"\nExtracted conv1_biases:")
print(f"  Shape: {conv1_biases.shape}")
print(f"  Range: [{conv1_biases.min():.6f}, {conv1_biases.max():.6f}]")
print(f"  Sample [0]: {conv1_biases[0]:.6f}")

print(f"\nExtracted prelu1_weights:")
print(f"  Shape: {prelu1_weights.shape}")
print(f"  Range: [{prelu1_weights.min():.6f}, {prelu1_weights.max():.6f}]")
print(f"  Sample [0]: {prelu1_weights[0]:.6f}")

# Now compare ONNX weights to extracted weights
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# We need to figure out which ONNX weight corresponds to which layer
# List all ONNX weight names to see the structure
print("\nAll ONNX weight names:")
for name in sorted(onnx_weights.keys()):
    print(f"  {name}: shape {onnx_weights[name].shape}")

# Try to match them up
print("\n" + "="*80)
print("MATCHING WEIGHTS")
print("="*80)

# First conv layer is likely the first conv weight in the model
# Let's find weights that match the expected shapes
for name, weight in onnx_weights.items():
    if weight.shape == conv1_weights.shape:
        print(f"\nFound potential conv1_weights match: '{name}'")
        print(f"  Shape: {weight.shape} (matches extracted)")
        diff = np.abs(weight - conv1_weights).max()
        print(f"  Max difference: {diff:.6e}")
        if diff < 1e-5:
            print(f"  ✓ EXACT MATCH!")
        else:
            print(f"  ❌ WEIGHTS DIFFER!")
            print(f"  ONNX [0,0,0,0]: {weight[0,0,0,0]:.6f}")
            print(f"  Extracted [0,0,0,0]: {conv1_weights[0,0,0,0]:.6f}")

    if weight.shape == conv1_biases.shape and len(weight.shape) == 1 and weight.shape[0] == 32:
        print(f"\nFound potential conv1_biases match: '{name}'")
        print(f"  Shape: {weight.shape} (matches extracted)")
        diff = np.abs(weight - conv1_biases).max()
        print(f"  Max difference: {diff:.6e}")
        if diff < 1e-5:
            print(f"  ✓ EXACT MATCH!")
        else:
            print(f"  ❌ WEIGHTS DIFFER!")
            print(f"  ONNX [0]: {weight[0]:.6f}")
            print(f"  Extracted [0]: {conv1_biases[0]:.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If weights match: ONNX model was built correctly from extracted weights")
print("If weights differ: Bug in ONNX model building process")
print("If weights match but outputs differ: Bug in ONNX graph structure")
