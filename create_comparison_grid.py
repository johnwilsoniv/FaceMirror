#!/usr/bin/env python3
"""
Create 2x3 comparison grid of all 6 test videos.
"""

import cv2
import numpy as np
from pathlib import Path

output_dir = Path("test_output/extended_comparison")

# Order: paralysis cases first, then normal cohort
image_files = [
    "IMG_8401_source.jpg",
    "IMG_9330_source.jpg",
    "IMG_0434_source.jpg",
    "IMG_0437_source.jpg",
    "IMG_0441_source.jpg",
    "IMG_0942_source.jpg",
]

print("Creating 2x3 comparison grid...")

# Load all images
imgs = []
for fname in image_files:
    img_path = output_dir / fname
    if img_path.exists():
        imgs.append(cv2.imread(str(img_path)))
    else:
        print(f"Warning: {img_path} not found")

if len(imgs) != 6:
    print(f"Error: Expected 6 images, found {len(imgs)}")
    exit(1)

# Resize to consistent height
target_h = 640
imgs_resized = []
for img in imgs:
    aspect = img.shape[1] / img.shape[0]
    target_w = int(target_h * aspect)
    imgs_resized.append(cv2.resize(img, (target_w, target_h)))

# Pad widths to match within each row
max_w_row1 = max(imgs_resized[0].shape[1], imgs_resized[1].shape[1], imgs_resized[2].shape[1])
max_w_row2 = max(imgs_resized[3].shape[1], imgs_resized[4].shape[1], imgs_resized[5].shape[1])
max_w = max(max_w_row1, max_w_row2)

imgs_padded = []
for img in imgs_resized:
    if img.shape[1] < max_w:
        pad_w = max_w - img.shape[1]
        img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    imgs_padded.append(img)

# Create 2x3 grid
row1 = np.hstack([imgs_padded[0], imgs_padded[1], imgs_padded[2]])
row2 = np.hstack([imgs_padded[3], imgs_padded[4], imgs_padded[5]])
grid = np.vstack([row1, row2])

# Add title
title_height = 100
grid_with_title = np.zeros((grid.shape[0] + title_height, grid.shape[1], 3), dtype=np.uint8)
grid_with_title[title_height:, :] = grid

# Title text
title = "Extended Comparison: Smart Bbox Selection (5/6 GOOD)"
subtitle = "Only IMG_8401 (surgical markings) failing - all others working correctly"
cv2.putText(grid_with_title, title, (30, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
cv2.putText(grid_with_title, title, (30, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

cv2.putText(grid_with_title, subtitle, (30, 85),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(grid_with_title, subtitle, (30, 85),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

# Save
grid_path = output_dir / "comparison_grid_6videos.jpg"
cv2.imwrite(str(grid_path), grid_with_title)
print(f"✓ Saved: {grid_path}")
print(f"  Grid size: {grid_with_title.shape[1]}x{grid_with_title.shape[0]}")
