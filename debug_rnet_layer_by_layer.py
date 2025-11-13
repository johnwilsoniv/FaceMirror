#!/usr/bin/env python3
"""
Debug Pure Python RNet layer-by-layer to find where it diverges from C++.
We'll manually step through each layer and inspect outputs.
"""

import cv2
import numpy as np
from cpp_cnn_loader import CPPCNN
import os

# Load RNet
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)

print("=" * 80)
print("RNET LAYER-BY-LAYER DEBUG")
print("=" * 80)

rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))

# Create a test input (24x24x3 face crop, normalized)
# Use a simple pattern we can trace
test_input = np.random.randn(3, 24, 24).astype(np.float32) * 0.1

print(f"\nTest Input:")
print(f"  Shape: {test_input.shape}")
print(f"  Range: [{test_input.min():.4f}, {test_input.max():.4f}]")
print(f"  Mean: {test_input.mean():.4f}")
print(f"  Std: {test_input.std():.4f}")

# Manually run through each layer
print(f"\n{'=' * 80}")
print(f"LAYER-BY-LAYER FORWARD PASS")
print(f"{'=' * 80}")

current = test_input

for i, layer in enumerate(rnet.layers):
    print(f"\n--- Layer {i}: {layer.__class__.__name__} ---")

    # Print layer details
    if hasattr(layer, 'num_kernels'):
        print(f"  Conv: {layer.num_in_maps}→{layer.num_kernels}, kernel={layer.kernel_h}x{layer.kernel_w}")
        print(f"  Weights shape: {layer.kernels.shape}")
        print(f"  Weights range: [{layer.kernels.min():.4f}, {layer.kernels.max():.4f}]")
        print(f"  Biases range: [{layer.biases.min():.4f}, {layer.biases.max():.4f}]")
    elif hasattr(layer, 'slopes'):
        print(f"  PReLU slopes shape: {layer.slopes.shape}")
        print(f"  Slopes range: [{layer.slopes.min():.4f}, {layer.slopes.max():.4f}]")
    elif hasattr(layer, 'kernel_size'):
        print(f"  MaxPool: kernel={layer.kernel_size}, stride={layer.stride}")
    elif hasattr(layer, 'weights'):
        print(f"  FC: {layer.weights.shape[1]}→{layer.weights.shape[0]}")
        print(f"  Weights range: [{layer.weights.min():.4f}, {layer.weights.max():.4f}]")
        print(f"  Biases range: [{layer.biases.min():.4f}, {layer.biases.max():.4f}]")

    # Forward pass
    print(f"  Input shape: {current.shape}")

    current = layer.forward(current)

    print(f"  Output shape: {current.shape}")
    print(f"  Output range: [{current.min():.4f}, {current.max():.4f}]")
    print(f"  Output mean: {current.mean():.4f}")
    print(f"  Output std: {current.std():.4f}")

    # Check for NaNs or Infs
    if np.isnan(current).any():
        print(f"  ⚠️  WARNING: NaN detected!")
    if np.isinf(current).any():
        print(f"  ⚠️  WARNING: Inf detected!")

    # For final FC output, calculate face score
    if i == len(rnet.layers) - 1:
        # Final output should be (6,): [logit_not_face, logit_face, reg_x, reg_y, reg_w, reg_h]
        logit_not_face = current[0]
        logit_face = current[1]
        score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

        print(f"\n  Final RNet output:")
        print(f"    Logit not-face: {logit_not_face:.4f}")
        print(f"    Logit face: {logit_face:.4f}")
        print(f"    Score: {score:.4f}")
        print(f"    Regression: [{current[2]:.4f}, {current[3]:.4f}, {current[4]:.4f}, {current[5]:.4f}]")

print(f"\n{'=' * 80}")
print(f"ANALYSIS")
print(f"{'=' * 80}")

print(f"\nExpected behavior:")
print(f"  - Scores should be in range [0, 1]")
print(f"  - For face images, score should be > 0.7 (official threshold)")
print(f"  - Regression values typically in range [-1, 1]")

print(f"\nLook for:")
print(f"  1. Weights that are all zeros (not loaded correctly)")
print(f"  2. Outputs that don't change (layer not working)")
print(f"  3. Exploding/vanishing values")
print(f"  4. Wrong output dimensions")
