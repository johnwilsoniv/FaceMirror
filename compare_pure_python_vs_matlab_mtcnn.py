#!/usr/bin/env python3
"""
Compare Pure Python CNN MTCNN to MATLAB/C++ MTCNN specification.
Based on detect_face_mtcnn.m official implementation.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')

print("=" * 80)
print("PURE PYTHON CNN MTCNN vs MATLAB/C++ SPECIFICATION")
print("=" * 80)

print("\nMAT LAB/C++ MTCNN Official Parameters:")
print("  Thresholds: [0.6, 0.7, 0.7] (PNet, RNet, ONet)")
print("  Factor: 0.709")
print("  Min face size: 30 (default in MATLAB, 40 in our tests)")
print("  Normalization: (img - 127.5) * 0.0078125")

# Test 1: Official thresholds [0.6, 0.7, 0.7]
print("\n" + "=" * 80)
print("TEST 1: Official MATLAB thresholds [0.6, 0.7, 0.7]")
print("=" * 80)

detector1 = PurePythonMTCNN_V2()
detector1.thresholds = [0.6, 0.7, 0.7]
bboxes1, landmarks1 = detector1.detect(img)

print(f"\nDetected: {len(bboxes1)} faces")
if len(bboxes1) > 0:
    for i, bbox in enumerate(bboxes1):
        x, y, w, h = bbox
        print(f"  Face {i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
else:
    print("  (No detections with official thresholds)")

# Test 2: Lowered RNet threshold [0.6, 0.6, 0.7]
print("\n" + "=" * 80)
print("TEST 2: Lowered RNet threshold [0.6, 0.6, 0.7]")
print("=" * 80)

detector2 = PurePythonMTCNN_V2()
detector2.thresholds = [0.6, 0.6, 0.7]
bboxes2, landmarks2 = detector2.detect(img)

print(f"\nDetected: {len(bboxes2)} faces")
if len(bboxes2) > 0:
    for i, bbox in enumerate(bboxes2):
        x, y, w, h = bbox
        print(f"  Face {i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")

# Test 3: All lowered [0.6, 0.6, 0.6]
print("\n" + "=" * 80)
print("TEST 3: All lowered thresholds [0.6, 0.6, 0.6]")
print("=" * 80)

detector3 = PurePythonMTCNN_V2()
detector3.thresholds = [0.6, 0.6, 0.6]
bboxes3, landmarks3 = detector3.detect(img)

print(f"\nDetected: {len(bboxes3)} faces")
if len(bboxes3) > 0:
    for i, bbox in enumerate(bboxes3):
        x, y, w, h = bbox
        print(f"  Face {i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")

# C++ Gold Standard (from previous testing)
cpp_bbox = np.array([331.6, 753.5, 367.9, 422.8])

print("\n" + "=" * 80)
print("COMPARISON TO C++ GOLD STANDARD")
print("=" * 80)

print(f"\nC++ MTCNN (reference):")
print(f"  Position: x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}")
print(f"  Size: w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")

# Compare best result
best_bboxes = bboxes3 if len(bboxes3) > 0 else (bboxes2 if len(bboxes2) > 0 else bboxes1)
best_test = "Test 3 [0.6, 0.6, 0.6]" if len(bboxes3) > 0 else ("Test 2 [0.6, 0.6, 0.7]" if len(bboxes2) > 0 else "Test 1 [0.6, 0.7, 0.7]")

if len(best_bboxes) > 0:
    pp_bbox = best_bboxes[0]

    print(f"\nPure Python CNN ({best_test}):")
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
    print(f"  Size ratio: {pp_bbox[2]/cpp_bbox[2]:.3f}x width, {pp_bbox[3]/cpp_bbox[3]:.3f}x height")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if len(bboxes1) == 0:
    print("\n⚠️  Pure Python MTCNN cannot detect faces with official MATLAB thresholds [0.6, 0.7, 0.7]")
    print("This indicates Pure Python RNet is producing lower scores than C++ RNet.")
    print("\nPossible causes:")
    print("  1. Weight loading bug in cpp_cnn_loader.py")
    print("  2. PReLU implementation difference")
    print("  3. Convolution implementation bug")
    print("  4. FC layer bug")
    print("  5. im2col transformation bug")

if len(best_bboxes) > 0 and (best_bboxes[0][2] < 100 or best_bboxes[0][3] < 100):
    print("\n⚠️  Detection bbox is very small compared to C++ (>90% smaller)")
    print("This indicates a bbox scaling/transformation issue in the pipeline.")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Debug RNet: Compare Pure Python RNet output vs C++ RNet on same input")
print("2. Layer-by-layer validation: Check each layer output matches C++")
print("3. Weight validation: Verify weights loaded from .dat match C++ exactly")
