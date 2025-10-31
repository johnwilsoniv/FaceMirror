#!/usr/bin/env python3
"""
Analyze the PDM mean shape orientation
"""

import numpy as np
from pdm_parser import PDMParser

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")

# Get mean shape
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_2d = mean_shape_scaled[:136].reshape(68, 2)

# Rigid indices
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]
rigid_points = mean_shape_2d[RIGID_INDICES]

print("=" * 70)
print("PDM Mean Shape Orientation Analysis")
print("=" * 70)

# Compute orientation from nose bridge (landmarks 27-30)
# These should be vertical in an upright face
nose_bottom = mean_shape_2d[30]  # Nose tip
nose_top = mean_shape_2d[27]     # Nose bridge top

# Vector from nose tip to bridge
nose_vector = nose_top - nose_bottom
nose_angle = np.arctan2(nose_vector[0], nose_vector[1]) * 180 / np.pi

print(f"\nNose orientation:")
print(f"  Bottom (landmark 30): ({nose_bottom[0]:.2f}, {nose_bottom[1]:.2f})")
print(f"  Top (landmark 27):    ({nose_top[0]:.2f}, {nose_top[1]:.2f})")
print(f"  Vector: ({nose_vector[0]:.2f}, {nose_vector[1]:.2f})")
print(f"  Angle from vertical: {nose_angle:.2f}°")

# Compute orientation from eye line (landmarks 36-45 are eyes)
left_eye_outer = mean_shape_2d[36]   # Left eye outer corner
right_eye_outer = mean_shape_2d[45]  # Right eye outer corner

eye_vector = right_eye_outer - left_eye_outer
eye_angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi

print(f"\nEye line orientation:")
print(f"  Left eye outer (36):  ({left_eye_outer[0]:.2f}, {left_eye_outer[1]:.2f})")
print(f"  Right eye outer (45): ({right_eye_outer[0]:.2f}, {right_eye_outer[1]:.2f})")
print(f"  Vector: ({eye_vector[0]:.2f}, {eye_vector[1]:.2f})")
print(f"  Angle from horizontal: {eye_angle:.2f}°")

# Check jaw line (landmarks 0-16)
left_jaw = mean_shape_2d[0]
right_jaw = mean_shape_2d[16]
jaw_vector = right_jaw - left_jaw
jaw_angle = np.arctan2(jaw_vector[1], jaw_vector[0]) * 180 / np.pi

print(f"\nJaw line orientation:")
print(f"  Left jaw (0):  ({left_jaw[0]:.2f}, {left_jaw[1]:.2f})")
print(f"  Right jaw (16): ({right_jaw[0]:.2f}, {right_jaw[1]:.2f})")
print(f"  Angle from horizontal: {jaw_angle:.2f}°")

# Overall bounding box
print(f"\nBounding box:")
print(f"  X range: {mean_shape_2d[:, 0].min():.2f} to {mean_shape_2d[:, 0].max():.2f}")
print(f"  Y range: {mean_shape_2d[:, 1].min():.2f} to {mean_shape_2d[:, 1].max():.2f}")
print(f"  Center: ({mean_shape_2d[:, 0].mean():.2f}, {mean_shape_2d[:, 1].mean():.2f})")

print("\n" + "=" * 70)
print("INTERPRETATION:")
if abs(nose_angle) < 2 and abs(eye_angle) < 2:
    print("✓ PDM mean shape is properly oriented (upright)")
else:
    print(f"⚠ PDM mean shape has built-in tilt:")
    print(f"   Nose: {nose_angle:.2f}° from vertical")
    print(f"   Eyes: {eye_angle:.2f}° from horizontal")
print("=" * 70)
