#!/usr/bin/env python3
"""Debug RNet FC layer issue"""

import cv2
import numpy as np
from cpp_cnn_loader import CPPCNN
import os

# Load RNet
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)

print("Loading RNet...")
rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))

# Create test 24x24 input
test_img = np.random.rand(3, 24, 24).astype(np.float32)

print(f"\nInput shape: {test_img.shape}")
print("\nRunning through layers...")

current = test_img
for i, layer in enumerate(rnet.layers):
    print(f"\n  Layer {i}: {type(layer).__name__}")
    print(f"    Input shape: {current.shape}")

    if hasattr(layer, 'weights'):
        print(f"    Expected input size: {layer.weights.shape[1]}")
        print(f"    Actual flat size: {current.flatten().shape[0]}")

    current = layer.forward(current)
    print(f"    Output shape: {current.shape}")

print(f"\nFinal output shape: {current.shape}")
print(f"Expected: (6,) for RNet")
