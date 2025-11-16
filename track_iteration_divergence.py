#!/usr/bin/env python3
"""
Track where Python and C++ start to diverge across iterations.
Save iteration-by-iteration landmark positions from both implementations.
"""
import cv2
import numpy as np
from pyclnf.clnf import CLNF

# Test image
image_path = "calibration_frames/patient1_frame1.jpg"
cpp_bbox = (301.938, 782.149, 400.586, 400.585)

print("="*70)
print("ITERATION-BY-ITERATION DIVERGENCE TRACKING")
print("="*70)
print(f"Image: {image_path}")
print(f"Bbox: {cpp_bbox}")
print()

# Initialize CLNF with iteration tracking
clnf = CLNF(
    model_dir="pyclnf/models",
    regularization=35,
    max_iterations=10,  # Match C++
    convergence_threshold=0.005,
    sigma=1.5,
    weight_multiplier=0.0,
    window_sizes=[11, 9, 7, 5],
    detector=None,
    debug_mode=True,  # Enable to capture iteration details
    tracked_landmarks=[36, 48, 30, 8]
)

# Load image
image = cv2.imread(image_path)

# Run Python CLNF
print("Running Python CLNF...")
landmarks, info = clnf.fit(image, face_bbox=cpp_bbox)

print("\nPython completed")
print(f"Final landmarks:")
for lm_idx in [36, 48, 30, 8]:
    print(f"  Landmark_{lm_idx}: ({landmarks[lm_idx][0]:.4f}, {landmarks[lm_idx][1]:.4f})")

print("\n" + "="*70)
print("Next step: Add C++ instrumentation to save iteration-by-iteration data")
print("Then compare where the first divergence occurs")
print("="*70)
