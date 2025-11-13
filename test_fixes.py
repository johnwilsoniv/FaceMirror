"""Test that the fixes are actually being applied."""

import numpy as np

# Test 1: Check that the +1 fix is in the ONet bbox regression
print("Test 1: Checking ONet +1 fix...")
print("Manual calculation:")
x1, x2 = 100, 200
w_old = x2 - x1  # 100
w_new = x2 - x1 + 1  # 101
print(f"  Without +1: w = {w_old}")
print(f"  With +1: w = {w_new}")
print(f"  Difference: {w_new - w_old}")

# Test 2: Check that _square_bbox is being called after RNet
print("\nTest 2: Checking _square_bbox after RNet...")
# We can't easily test this without modifying the code, but we added it at line 482

# Test 3: Let's manually calculate what the bbox should be
print("\nTest 3: Manual calculation of expected bbox...")
# Starting from Python Face 2: x=247.26, y=819.67, w=378.28, h=370.76
# C++ Face: x=301.94, y=782.15, w=400.59, h=400.59

# If the issue was missing +1 in ONet, what would the difference be?
# Regression formula: new_x1 = old_x1 + reg_x * w
# If w was calculated WITHOUT +1, then we'd be applying less regression

print("\nLet's just run the detector and see if it's using the new code...")
from cpp_mtcnn_detector import CPPMTCNNDetector
import cv2

img = cv2.imread('calibration_frames/patient1_frame1.jpg')
detector = CPPMTCNNDetector()
bboxes, landmarks = detector.detect(img)

print(f"\nPython MTCNN Results:")
for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox
    print(f"  Face {i+1}: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")
    print(f"           Square? {abs(w - h) < 1.0}")

print("\nC++ Result: x=301.94, y=782.15, w=400.59, h=400.59")
print(f"           Square? True")

# If Python bbox is still rectangular, the fixes aren't working or we're missing something else
if len(bboxes) >= 2:
    x, y, w, h = bboxes[1]  # Face 2 (largest)
    if abs(w - h) > 5:
        print(f"\n❌ FAIL: Python bbox is still rectangular (w={w:.2f}, h={h:.2f})")
        print("This suggests:")
        print("  1. The fixes aren't being applied, OR")
        print("  2. We're missing another transformation")
    else:
        print(f"\n✓ PASS: Python bbox is square (w={w:.2f}, h={h:.2f})")
