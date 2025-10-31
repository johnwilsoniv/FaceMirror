#!/usr/bin/env python3
"""
Measure the actual rotation in C++ aligned face images

If C++ faces are truly upright, eyes should be perfectly horizontal
"""

import cv2
import numpy as np
from pathlib import Path

cpp_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

# Test frames
test_frames = [1, 493, 617, 863]

print("=" * 70)
print("Measuring C++ Face Rotation from Images")
print("=" * 70)

# In aligned 112×112 faces, eyes should be at approximately these positions if upright:
# Left eye outer corner (landmark 36): around (30, 45)
# Right eye outer corner (landmark 45): around (82, 45)
# If eyes are horizontal, Y coordinates should be equal

print("\n[Method 1] Visual inspection of eye line orientation...")
print("(Looking for horizontal lines in upper face region)")

for frame_num in test_frames:
    cpp_file = cpp_dir / f"frame_det_00_{frame_num:06d}.bmp"
    img = cv2.imread(str(cpp_file))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Look at the eye region (roughly upper half)
    eye_region = gray[30:60, :]

    # Find horizontal edges (eyes should create strong horizontal features)
    # Use Sobel to detect edges
    sobel_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient direction
    gradient_angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # Strong edges (high magnitude)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    threshold = np.percentile(magnitude, 90)  # Top 10% of edges
    strong_edges = magnitude > threshold

    # Average angle of strong edges (these should be eye contours)
    strong_angles = gradient_angle[strong_edges]

    # Eyes are horizontal lines, so their gradient should be vertical (~90°)
    # If face is rotated, gradient will be tilted
    mean_gradient = np.median(strong_angles)

    # Face rotation is perpendicular to gradient
    face_rotation = mean_gradient - 90

    print(f"Frame {frame_num}: gradient={mean_gradient:6.1f}°, implied face rotation={face_rotation:6.1f}°")

print("\n[Method 2] Check if faces look identical when overlaid...")

# Load all frames
images = {}
for frame_num in test_frames:
    cpp_file = cpp_dir / f"frame_det_00_{frame_num:06d}.bmp"
    images[frame_num] = cv2.imread(str(cpp_file))

# Compare pixel differences (if rotation is identical, should be minimal)
frame_pairs = [(1, 493), (493, 617), (617, 863)]

print("\nPixel differences between frames (lower = more similar orientation):")
for f1, f2 in frame_pairs:
    diff = np.mean(np.abs(images[f1].astype(float) - images[f2].astype(float)))
    print(f"  Frames {f1} vs {f2}: MAE = {diff:.2f} pixels")

print("\n[Method 3] Analyze nose-to-chin vertical alignment...")

# In a 112×112 aligned face, nose tip and chin should be vertically aligned
# If rotated, they'll have horizontal offset

# For OpenFace 112×112 aligned faces, approximate landmark positions:
# Nose tip (30) is around (56, 65)
# Chin (8) is around (56, 95)
# If perfectly vertical: same X coordinate

# We can't extract exact landmarks, but we can check if dark features (eyes, mouth)
# are horizontally distributed or tilted

for frame_num in test_frames:
    img = images[frame_num]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find darkest pixels (eyes, eyebrows, mouth)
    dark_threshold = np.percentile(gray, 20)
    dark_pixels = np.where(gray < dark_threshold)

    # Check if dark pixels form a vertical line (upright face)
    # vs. tilted line (rotated face)
    if len(dark_pixels[0]) > 0:
        # Fit a line through dark pixels
        coords = np.column_stack([dark_pixels[1], dark_pixels[0]])  # X, Y
        vx, vy, x0, y0 = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)

        # Angle of this line from vertical
        angle_from_vertical = np.arctan2(vx[0], vy[0]) * 180 / np.pi

        print(f"Frame {frame_num}: facial feature line angle from vertical = {angle_from_vertical:.2f}°")

print("\n" + "=" * 70)
print("Interpretation:")
print("=" * 70)
print("If all C++ frames show:")
print("  - Face rotation ~0° (within ±2°)")
print("  - Similar pixel differences between frames")
print("  - Feature lines vertical (within ±2°)")
print("Then C++ truly produces consistently upright faces.")
print()
print("Our Python alignment should match these characteristics!")
print("=" * 70)
