#!/usr/bin/env python3
"""
Test the fixed Pure Python MTCNN V2 with simplified cropping.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("TESTING FIXED PURE PYTHON MTCNN V2")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
print(f"\nTest image: {img.shape}")

# Test with official MATLAB/C++ thresholds
print("\n" + "=" * 80)
print("TEST: Official MATLAB/C++ thresholds [0.6, 0.7, 0.7]")
print("=" * 80)

detector = PurePythonMTCNN_V2()
print(f"Thresholds: {detector.thresholds}")

bboxes, landmarks = detector.detect(img, debug=True)

print(f"\nDetected: {len(bboxes)} faces")
if len(bboxes) > 0:
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        print(f"  Face {i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
else:
    print("  (No detections)")

# C++ Gold Standard
cpp_bbox = np.array([331.6, 753.5, 367.9, 422.8])

print("\n" + "=" * 80)
print("COMPARISON TO C++ GOLD STANDARD")
print("=" * 80)

print(f"\nC++ MTCNN (reference):")
print(f"  Position: x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}")
print(f"  Size: w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")

if len(bboxes) > 0:
    pp_bbox = bboxes[0]

    print(f"\nPure Python CNN:")
    print(f"  Position: x={pp_bbox[0]:.1f}, y={pp_bbox[1]:.1f}")
    print(f"  Size: w={pp_bbox[2]:.1f}, h={pp_bbox[3]:.1f}")

    # Calculate differences
    dx = abs(pp_bbox[0] - cpp_bbox[0])
    dy = abs(pp_bbox[1] - cpp_bbox[1])
    dw = abs(pp_bbox[2] - cpp_bbox[2])
    dh = abs(pp_bbox[3] - cpp_bbox[3])

    print(f"\nAbsolute Differences:")
    print(f"  Position: dx={dx:.1f}px, dy={dy:.1f}px")
    print(f"  Size: dw={dw:.1f}px, dh={dh:.1f}px")

    print(f"\nRelative Differences:")
    print(f"  Position: {dx/cpp_bbox[0]*100:.1f}% x, {dy/cpp_bbox[1]*100:.1f}% y")
    print(f"  Size: {dw/cpp_bbox[2]*100:.1f}% w, {dh/cpp_bbox[3]*100:.1f}% h")

    # Success criteria
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if len(bboxes) > 0:
        print("\n✓ Detection SUCCESS with official thresholds!")
        if dw < 50 and dh < 50:
            print("✓ BBox size close to C++ reference")
        else:
            print("⚠️  BBox size differs from C++ reference")
    else:
        print("\n✗ Detection FAILED")
else:
    print("\n✗ Detection FAILED with official thresholds")
