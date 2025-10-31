#!/usr/bin/env python3
"""
Test warpAffine coordinate system to understand if we need to invert the transform
"""

import numpy as np
import cv2

# Create a simple test image with a known pattern
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(img, (50, 50), 10, (255, 0, 0), -1)  # Blue circle at (50, 50)
cv2.circle(img, (150, 150), 10, (0, 255, 0), -1)  # Green circle at (150, 150)

# Create a transformation that should move (50,50) to (25,25) and scale down by 0.5
# This is: newpos = scale * oldpos
# (25, 25) = 0.5 * (50, 50)
scale = 0.5
M_forward = np.array([
    [scale, 0, 0],
    [0, scale, 0]
], dtype=np.float32)

# Apply warpAffine
result_forward = cv2.warpAffine(img, M_forward, (100, 100))

# Save images
cv2.imwrite("/tmp/test_original.png", img)
cv2.imwrite("/tmp/test_forward.png", result_forward)

print("Forward transform M =")
print(M_forward)
print("\nFor warpAffine, this means:")
print("output(x,y) = input(M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2])")
print("output(x,y) = input(0.5*x, 0.5*y)")
print("\nSo output(50,50) samples from input(25,25)")
print("Expected: Blue circle moves from (50,50) to (100,100) [scaled up]")

# Now test the inverse
M_inverse = np.array([
    [1/scale, 0, 0],
    [0, 1/scale, 0]
], dtype=np.float32)

result_inverse = cv2.warpAffine(img, M_inverse, (100, 100))
cv2.imwrite("/tmp/test_inverse.png", result_inverse)

print("\nInverse transform M =")
print(M_inverse)
print("output(x,y) = input(2*x, 2*y)")
print("Expected: Blue circle moves from (50,50) to (25,25) [scaled down]")

print("\nImages saved to /tmp/test_*.png")
print("Blue circle was at (50,50) in original")
print("Green circle was at (150,150) in original")
