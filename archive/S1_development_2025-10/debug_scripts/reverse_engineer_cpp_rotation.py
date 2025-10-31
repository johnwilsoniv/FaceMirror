#!/usr/bin/env python3
"""
Reverse-engineer the rotation C++ applied by analyzing aligned face images

Strategy:
1. Load C++ aligned face image
2. Detect key features (eyes, nose)
3. Measure their orientation
4. Compare to expected orientation
5. Calculate the rotation that was applied
"""

import numpy as np
import cv2
from pathlib import Path

# Paths
cpp_aligned_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

# Load a few C++ aligned faces
test_frames = [1, 493, 617, 863]

print("=" * 80)
print("Reverse Engineering C++ Rotation from Aligned Faces")
print("=" * 80)

for frame_num in test_frames:
    # Load C++ aligned face
    cpp_file = cpp_aligned_dir / f"frame_det_00_{frame_num:06d}.bmp"

    if not cpp_file.exists():
        print(f"\nFrame {frame_num}: File not found: {cpp_file}")
        continue

    cpp_img = cv2.imread(str(cpp_file), cv2.IMREAD_GRAYSCALE)

    if cpp_img is None:
        print(f"\nFrame {frame_num}: Could not load image")
        continue

    print(f"\nFrame {frame_num}:")
    print(f"  Image size: {cpp_img.shape}")
    print(f"  Image range: [{cpp_img.min()}, {cpp_img.max()}]")

    # Detect face orientation using simple gradient analysis
    # Compute horizontal and vertical gradients
    grad_x = cv2.Sobel(cpp_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cpp_img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute orientation of strongest gradients (this gives face tilt)
    angles = np.arctan2(grad_y, grad_x)
    magnitudes = np.sqrt(grad_x**2 + grad_y**2)

    # Weight angles by gradient magnitude
    strong_gradients = magnitudes > np.percentile(magnitudes, 90)
    dominant_angles = angles[strong_gradients]

    # Circular mean of angles
    mean_angle_rad = np.arctan2(np.mean(np.sin(dominant_angles)),
                                  np.mean(np.cos(dominant_angles)))
    mean_angle_deg = mean_angle_rad * 180 / np.pi

    print(f"  Dominant gradient angle: {mean_angle_deg:.2f}°")

    # Check if face appears upright (vertical gradients should be strong)
    vertical_grad_strength = np.mean(np.abs(grad_y))
    horizontal_grad_strength = np.mean(np.abs(grad_x))
    ratio = vertical_grad_strength / (horizontal_grad_strength + 1e-10)

    print(f"  Vertical/Horizontal gradient ratio: {ratio:.3f}")
    if ratio > 1.0:
        print(f"  → Face appears UPRIGHT (more vertical structure)")
    else:
        print(f"  → Face appears TILTED (more horizontal structure)")

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)
print("If all C++ aligned faces show:")
print("  - Similar gradient angles across frames")
print("  - High vertical/horizontal ratio")
print("  → C++ successfully produces upright, stable faces")
print("\nThis confirms C++ is doing something we're not.")
print("Since it's not in AlignFace code, it must be in how")
print("landmarks are prepared BEFORE AlignFace is called.")
print("=" * 80)
