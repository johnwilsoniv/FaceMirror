#!/usr/bin/env python3
"""
Super granular MTCNN test to find segfault.
"""

import sys
from pathlib import Path

print("Step 0: Adding paths...")
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')
print("  ✓ Paths added")

print("\nStep 1: Import cv2...")
import cv2
print("  ✓ cv2 imported")

print("\nStep 2: Import numpy...")
import numpy as np
print("  ✓ numpy imported")

print("\nStep 3: Load image...")
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
image = cv2.imread(TEST_IMAGE)
print(f"  ✓ Image loaded: {image.shape}")

print("\nStep 4: Import torch...")
import torch
print(f"  ✓ torch imported (version {torch.__version__})")

print("\nStep 5: Import OpenFaceMTCNN class...")
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
print("  ✓ OpenFaceMTCNN class imported")

print("\nStep 6: Create MTCNN instance...")
print("  (If segfault occurs here, it's during __init__)")
mtcnn = OpenFaceMTCNN(device='cpu')
print("  ✓ MTCNN instance created")

print("\nStep 7: Call detect method...")
print("  (If segfault occurs here, it's during forward pass)")
bboxes, landmarks = mtcnn.detect(image)
print(f"  ✓ Detection complete: {len(bboxes)} faces")

print("\nALL STEPS COMPLETED - NO SEGFAULT!")
