#!/usr/bin/env python3
"""
Verify that Python is using the same bbox as C++ for initialization.
"""

import sys
import numpy as np

# The bbox we're using in Python test
python_bbox = [310, 780, 370, 370]

print("Python bbox (hardcoded in test):")
print(f"  [x, y, w, h] = {python_bbox}")
print(f"  Center: ({python_bbox[0] + python_bbox[2]/2:.1f}, {python_bbox[1] + python_bbox[3]/2:.1f})")

print("\nQuestions:")
print("1. Where did C++ get its bbox from?")
print("   - MTCNN detector")
print("   - Manual specification")
print("   - Previous frame tracking")
print("")
print("2. To verify, we should:")
print("   - Add C++ debug to print the initial bbox used")
print("   - OR: Use C++'s final landmarks to reverse-engineer the bbox")
print("")

# Load C++ final landmarks
import csv
with open('/tmp/cpp_baseline/patient1_frame1.csv', 'r') as f:
    reader = csv.DictReader(f)
    row = next(reader)

    # Get all landmarks
    cpp_landmarks = []
    for i in range(68):
        x = float(row[f'x_{i}'])
        y = float(row[f'y_{i}'])
        cpp_landmarks.append([x, y])

    cpp_landmarks = np.array(cpp_landmarks)

print("C++ landmarks bbox (from final landmarks):")
x_min, y_min = cpp_landmarks.min(axis=0)
x_max, y_max = cpp_landmarks.max(axis=0)
width = x_max - x_min
height = y_max - y_min
cx = (x_min + x_max) / 2
cy = (y_min + y_max) / 2

print(f"  Bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
print(f"  Size: {width:.1f} x {height:.1f}")
print(f"  Center: ({cx:.1f}, {cy:.1f})")

# Estimate initial bbox (assuming some margin)
# OpenFace typically expands bbox by a factor for robustness
estimated_bbox_x = x_min - width * 0.1
estimated_bbox_y = y_min - height * 0.1
estimated_bbox_w = width * 1.2
estimated_bbox_h = height * 1.2

print(f"\nEstimated C++ initial bbox (with 10% margin):")
print(f"  [x, y, w, h] ≈ [{estimated_bbox_x:.1f}, {estimated_bbox_y:.1f}, {estimated_bbox_w:.1f}, {estimated_bbox_h:.1f}]")
print(f"  Center: ({estimated_bbox_x + estimated_bbox_w/2:.1f}, {estimated_bbox_y + estimated_bbox_h/2:.1f})")

print(f"\nComparison:")
print(f"  Python center: ({python_bbox[0] + python_bbox[2]/2:.1f}, {python_bbox[1] + python_bbox[3]/2:.1f})")
print(f"  C++ est. center: ({estimated_bbox_x + estimated_bbox_w/2:.1f}, {estimated_bbox_y + estimated_bbox_h/2:.1f})")
print(f"  Difference: ({(python_bbox[0] + python_bbox[2]/2) - (estimated_bbox_x + estimated_bbox_w/2):.1f}, {(python_bbox[1] + python_bbox[3]/2) - (estimated_bbox_y + estimated_bbox_h/2):.1f})")
