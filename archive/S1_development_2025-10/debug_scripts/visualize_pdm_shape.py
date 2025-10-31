#!/usr/bin/env python3
"""
Visualize the PDM mean shape to understand its orientation
"""

import numpy as np
import cv2
from pdm_parser import PDMParser

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")

# Get mean shape
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_2d = mean_shape_scaled[:136].reshape(68, 2)

# Create image
img_size = 400
img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# Find bounding box
min_x, min_y = mean_shape_2d.min(axis=0)
max_x, max_y = mean_shape_2d.max(axis=0)

# Scale to fit image with margin
margin = 50
scale = min((img_size - 2*margin) / (max_x - min_x),
            (img_size - 2*margin) / (max_y - min_y))

# Center in image
offset_x = img_size / 2 - (max_x + min_x) / 2 * scale
offset_y = img_size / 2 - (max_y + min_y) / 2 * scale

# Draw landmarks
for i, (x, y) in enumerate(mean_shape_2d):
    px = int(x * scale + offset_x)
    py = int(y * scale + offset_y)

    # Color code different regions
    if i < 17:  # Jaw
        color = (255, 0, 0)  # Blue
    elif i < 27:  # Eyebrows
        color = (0, 255, 0)  # Green
    elif i < 36:  # Nose
        color = (0, 255, 255)  # Yellow
    elif i < 48:  # Eyes
        color = (255, 0, 255)  # Magenta
    else:  # Mouth
        color = (0, 165, 255)  # Orange

    cv2.circle(img, (px, py), 3, color, -1)
    cv2.putText(img, str(i), (px+5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

# Draw reference lines
# Nose bridge (27-30)
for i in range(27, 30):
    p1 = mean_shape_2d[i]
    p2 = mean_shape_2d[i+1]
    px1 = int(p1[0] * scale + offset_x)
    py1 = int(p1[1] * scale + offset_y)
    px2 = int(p2[0] * scale + offset_x)
    py2 = int(p2[1] * scale + offset_y)
    cv2.line(img, (px1, py1), (px2, py2), (0, 255, 255), 2)

# Draw coordinate axes
center = (img_size // 2, img_size // 2)
cv2.line(img, center, (center[0] + 50, center[1]), (0, 0, 255), 2)  # X axis (red)
cv2.line(img, center, (center[0], center[1] + 50), (255, 0, 0), 2)  # Y axis (blue)
cv2.putText(img, "+X", (center[0] + 55, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(img, "+Y", (center[0] + 5, center[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imwrite("pdm_mean_shape_visualization.png", img)
print("✓ Saved pdm_mean_shape_visualization.png")
print("\nColor coding:")
print("  Blue:    Jaw (0-16)")
print("  Green:   Eyebrows (17-26)")
print("  Yellow:  Nose (27-35)")
print("  Magenta: Eyes (36-47)")
print("  Orange:  Mouth (48-67)")
print("\nCoordinate system:")
print("  Red line:  +X axis (right)")
print("  Blue line: +Y axis (down)")
