#!/usr/bin/env python3
"""
Compare C++ and Python final landmarks with same bbox.
"""
import cv2
import numpy as np
from pyclnf.clnf import CLNF

# Test image
image_path = "calibration_frames/patient1_frame1.jpg"
cpp_bbox = (301.938, 782.149, 400.586, 400.585)

# Initialize CLNF
clnf = CLNF(
    model_dir="pyclnf/models",
    regularization=35,
    max_iterations=10,
    convergence_threshold=0.005,
    sigma=1.5,
    weight_multiplier=0.0,
    window_sizes=[11, 9, 7, 5],
    detector=None,
    debug_mode=False  # Turn off debug for cleaner output
)

# Load image
image = cv2.imread(image_path)
print(f"Testing with image: {image_path}")
print(f"Using C++ bbox: {cpp_bbox}")
print()

# Fit model
landmarks, info = clnf.fit(image, face_bbox=cpp_bbox)

print(f"Python fit completed:")
print(f"  landmarks shape: {landmarks.shape if landmarks is not None else 'None'}")
print(f"  iterations: {info.get('total_iterations', 'N/A')}")
print()

if landmarks is not None:
    print("Python Final landmarks:")
    for lm_idx in [36, 48, 30, 8]:
        if lm_idx < len(landmarks):
            print(f"  Landmark_{lm_idx}: ({landmarks[lm_idx][0]:.4f}, {landmarks[lm_idx][1]:.4f})")

    print("\nC++ Final landmarks (from previous run):")
    print("  Landmark_36: (364.3000, 866.1000)")
    print("  Landmark_48: (420.6000, 1053.5000)")
    print("  Landmark_30: (483.8000, 944.3000)")
    print("  Landmark_8: (503.0000, 1164.3000)")

    print("\nDifferences:")
    cpp_lms = {
        36: (364.3000, 866.1000),
        48: (420.6000, 1053.5000),
        30: (483.8000, 944.3000),
        8: (503.0000, 1164.3000)
    }

    for lm_idx in [36, 48, 30, 8]:
        py_x, py_y = landmarks[lm_idx]
        cpp_x, cpp_y = cpp_lms[lm_idx]
        dx = py_x - cpp_x
        dy = py_y - cpp_y
        dist = np.sqrt(dx**2 + dy**2)
        print(f"  Landmark_{lm_idx}: dx={dx:7.3f}, dy={dy:7.3f}, dist={dist:7.3f} px")

    # Calculate mean error
    errors = []
    for lm_idx in [36, 48, 30, 8]:
        py_x, py_y = landmarks[lm_idx]
        cpp_x, cpp_y = cpp_lms[lm_idx]
        dist = np.sqrt((py_x - cpp_x)**2 + (py_y - cpp_y)**2)
        errors.append(dist)

    mean_error = np.mean(errors)
    print(f"\nMean error: {mean_error:.3f} px")

    if mean_error > 5.0:
        print(f"\n✗ Significant difference detected ({mean_error:.3f} px)")
        print("   Need to investigate convergence/parameter updates/numerical precision")
    elif mean_error > 1.0:
        print(f"\n⚠ Small difference detected ({mean_error:.3f} px)")
        print("   Likely due to numerical precision")
    else:
        print(f"\n✓ Results match within tolerance ({mean_error:.3f} px)")
else:
    print("ERROR: landmarks is None!")
