#!/usr/bin/env python3
"""
Test MTCNN with workaround for torch+numpy segfault.
"""

import sys
import numpy as np
import cv2
import torch

print("Loading image...")
image = cv2.imread("/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg")
print(f"✓ Image: {image.shape}")

print("\nTest 1: Direct torch.from_numpy() - THIS CRASHES")
sys.stdout.flush()
try:
    # img_tensor = torch.from_numpy(image)  # THIS SEGFAULTS
    print("  Skipping direct conversion (known to crash)")
except:
    print("  CRASHED")

print("\nTest 2: Copy first, then torch.from_numpy()")
sys.stdout.flush()
try:
    image_copy = image.copy()
    img_tensor = torch.from_numpy(image_copy)
    print(f"✓ Success! Tensor shape: {img_tensor.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 3: Manual conversion via list/array")
sys.stdout.flush()
try:
    # Convert to float first
    image_float = image.astype(np.float32)
    img_tensor = torch.tensor(image_float)
    print(f"✓ Success! Tensor shape: {img_tensor.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 4: Use torch.as_tensor() instead")
sys.stdout.flush()
try:
    img_tensor = torch.as_tensor(image)
    print(f"✓ Success! Tensor shape: {img_tensor.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n✓ Tests complete!")
