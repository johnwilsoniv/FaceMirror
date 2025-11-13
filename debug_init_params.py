#!/usr/bin/env python3
"""
Debug init_params calculation step by step.
"""

import numpy as np
import cv2
from pathlib import Path
from pyclnf.core import PDM

# BBox from C++ output
bbox = (1399.95, 854.129, 413.5, 377.308)
x, y, width, height = bbox

print("="*60)
print("DEBUGGING INIT_PARAMS CALCULATION")
print("="*60)
print(f"\nInput BBox: x={x}, y={y}, width={width}, height={height}")

# Load PDM
model_dir = Path("pyclnf/models/exported_pdm")
pdm = PDM(str(model_dir))

# Get mean shape
mean_shape_3d = pdm.mean_shape.reshape(-1, 3).T  # Shape: (3, 68)

print(f"\nMean shape: {mean_shape_3d.shape}")
print(f"Mean shape stats:")
print(f"  X: min={mean_shape_3d[0, :].min():.4f}, max={mean_shape_3d[0, :].max():.4f}")
print(f"  Y: min={mean_shape_3d[1, :].min():.4f}, max={mean_shape_3d[1, :].max():.4f}")
print(f"  Z: min={mean_shape_3d[2, :].min():.4f}, max={mean_shape_3d[2, :].max():.4f}")

# Rotate with identity (zero rotation)
rotation = np.array([0.0, 0.0, 0.0])
R = cv2.Rodrigues(rotation)[0]
rotated_shape = R @ mean_shape_3d

print(f"\nRotated shape (should be same as mean shape):")
print(f"  X: min={rotated_shape[0, :].min():.4f}, max={rotated_shape[0, :].max():.4f}")
print(f"  Y: min={rotated_shape[1, :].min():.4f}, max={rotated_shape[1, :].max():.4f}")

# Find model bounds
min_x = rotated_shape[0, :].min()
max_x = rotated_shape[0, :].max()
min_y = rotated_shape[1, :].min()
max_y = rotated_shape[1, :].max()

model_width = abs(max_x - min_x)
model_height = abs(max_y - min_y)

print(f"\nModel dimensions:")
print(f"  Width: {model_width:.4f}")
print(f"  Height: {model_height:.4f}")
print(f"  Aspect ratio: {model_width/model_height:.4f}")

# Calculate scaling
width_scale = width / model_width
height_scale = height / model_height
scaling = (width_scale + height_scale) / 2.0

print(f"\nScaling calculation:")
print(f"  width_scale = {width} / {model_width:.4f} = {width_scale:.6f}")
print(f"  height_scale = {height} / {model_height:.4f} = {height_scale:.6f}")
print(f"  scaling = ({width_scale:.6f} + {height_scale:.6f}) / 2 = {scaling:.6f}")

# Calculate translation
tx_before = x + width / 2.0
ty_before = y + height / 2.0

model_center_x = (min_x + max_x) / 2.0
model_center_y = (min_y + max_y) / 2.0

tx = tx_before - scaling * model_center_x
ty = ty_before - scaling * model_center_y

print(f"\nTranslation calculation:")
print(f"  BBox center: ({tx_before:.2f}, {ty_before:.2f})")
print(f"  Model center: ({model_center_x:.4f}, {model_center_y:.4f})")
print(f"  Correction: ({scaling * model_center_x:.2f}, {scaling * model_center_y:.2f})")
print(f"  Final TX = {tx_before:.2f} - {scaling * model_center_x:.2f} = {tx:.2f}")
print(f"  Final TY = {ty_before:.2f} - {scaling * model_center_y:.2f} = {ty:.2f}")

print(f"\nFinal parameters:")
print(f"  Scale: {scaling:.6f}")
print(f"  TX: {tx:.2f}")
print(f"  TY: {ty:.2f}")

print(f"\nC++ reported:")
print(f"  Scale: 2.777210")
print(f"  TX: 1609.50")
print(f"  TY: 1022.95")

print(f"\nDifferences:")
print(f"  Scale diff: {2.77721 - scaling:.6f}")
print(f"  TX diff: {1609.5 - tx:.2f}")
print(f"  TY diff: {1022.95 - ty:.2f}")
