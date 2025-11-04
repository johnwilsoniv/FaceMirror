#!/usr/bin/env python3
"""
Minimal MTCNN test - try to isolate the segfault.
"""

import sys
print("Step 1: Importing numpy and cv2...")
import numpy as np
import cv2
print("✓ numpy and cv2 imported")

print("Step 2: Loading image...")
image = cv2.imread("/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg")
print(f"✓ Image loaded: {image.shape}")

print("Step 3: Importing torch...")
import torch
print(f"✓ torch imported, version: {torch.__version__}")

print("Step 4: Testing simple torch operation...")
x = torch.rand(3, 3)
print(f"✓ Created tensor: {x.shape}")

print("Step 5: Testing torch + numpy...")
y = x.numpy()
print(f"✓ Converted to numpy: {y.shape}")

print("Step 6: Testing cv2 with numpy...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"✓ Converted to grayscale: {gray.shape}")

print("Step 7: Testing torch tensor from numpy...")
img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
print(f"✓ Created image tensor: {img_tensor.shape}")

print("Step 8: Importing OpenFaceMTCNN...")
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
print("✓ OpenFaceMTCNN imported")

print("Step 9: Initializing MTCNN...")
mtcnn = OpenFaceMTCNN()
print("✓ MTCNN initialized")

print("Step 10: Calling mtcnn.detect()...")
sys.stdout.flush()  # Force flush before potential crash

result = mtcnn.detect(image, return_landmarks=True)

print("✓✓✓ MTCNN DETECTION SUCCEEDED! ✓✓✓")
print(f"Result: {result}")
