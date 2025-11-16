#!/usr/bin/env python3
"""Test CEN integration with pyclnf"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pyclnf.clnf import CLNF

# Load test image
image_path = "calibration_frames/patient1_frame1.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load {image_path}")
    sys.exit(1)

print(f"Loaded image: {image.shape}")

# Initialize CLNF with CEN
print("\nInitializing CLNF with CEN patch experts...")
try:
    clnf = CLNF(
        model_dir="pyclnf/models",
        debug_mode=True,
        tracked_landmarks=[36, 48, 30, 8]
    )
    print("✓ CLNF initialized successfully with CEN!")
except Exception as e:
    print(f"✗ Error initializing CLNF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Use exact C++ bbox
bbox = (301.938, 782.149, 400.586, 400.585)
print(f"\nUsing exact C++ bbox: {bbox}")

# Run detection
print("\nRunning CLNF landmark detection...")
try:
    landmarks, info = clnf.fit(image, face_bbox=bbox)
    print(f"✓ Detection complete!")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Optimization info: {info}")
    print(f"\nKey landmarks:")
    for idx in [36, 48, 30, 8]:
        print(f"  Landmark {idx}: ({landmarks[idx, 0]:.2f}, {landmarks[idx, 1]:.2f})")
except Exception as e:
    print(f"✗ Error during detection: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ CEN integration test PASSED!")
