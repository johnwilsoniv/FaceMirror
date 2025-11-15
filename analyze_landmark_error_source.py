#!/usr/bin/env python3
"""Analyze where the 36 pixel landmark error comes from"""

import numpy as np
import pandas as pd

# C++ data
df = pd.read_csv("/tmp/mtcnn_debug.csv")
cpp_bbox_x = 301.9380
cpp_bbox_y = 782.1490
cpp_bbox_w = 400.5860
cpp_bbox_h = 400.5850

# PyMTCNN data  
pymtcnn_bbox_x = 296.9593
pymtcnn_bbox_y = 778.0527
pymtcnn_bbox_w = 404.2962
pymtcnn_bbox_h = 407.3596

# C++ raw ONet landmarks (normalized)
cpp_raw = np.array([
    [0.324284, 0.672122],
    [0.496269, 0.367142],
    [0.662997, 0.356616],
    [0.344393, 0.559805],
    [0.756904, 0.745246]
])

# PyMTCNN raw ONet landmarks (normalized)
pymtcnn_raw = np.array([
    [0.324707, 0.657227],
    [0.484375, 0.360107],
    [0.648926, 0.368164],
    [0.356934, 0.556152],
    [0.749512, 0.738281]
])

print("="*80)
print("Scenario 1: Use SAME bbox for both (C++ bbox)")
print("="*80)

# Denormalize both using C++ bbox
cpp_denorm = np.zeros((5, 2))
pymtcnn_denorm_cpp_bbox = np.zeros((5, 2))

for i in range(5):
    cpp_denorm[i, 0] = cpp_bbox_x + cpp_raw[i, 0] * cpp_bbox_w
    cpp_denorm[i, 1] = cpp_bbox_y + cpp_raw[i, 1] * cpp_bbox_h
    
    pymtcnn_denorm_cpp_bbox[i, 0] = cpp_bbox_x + pymtcnn_raw[i, 0] * cpp_bbox_w
    pymtcnn_denorm_cpp_bbox[i, 1] = cpp_bbox_y + pymtcnn_raw[i, 1] * cpp_bbox_h

diffs_same_bbox = pymtcnn_denorm_cpp_bbox - cpp_denorm
errors_same_bbox = np.sqrt(np.sum(diffs_same_bbox**2, axis=1))

print("\nUsing C++ bbox for both:")
for i, err in enumerate(errors_same_bbox):
    print(f"  Point {i}: {err:.2f} pixels")
print(f"Mean: {np.mean(errors_same_bbox):.2f} pixels")

print("\n" + "="*80)
print("Scenario 2: Use DIFFERENT bboxes (actual situation)")
print("="*80)

# Denormalize using respective bboxes
pymtcnn_denorm = np.zeros((5, 2))
for i in range(5):
    pymtcnn_denorm[i, 0] = pymtcnn_bbox_x + pymtcnn_raw[i, 0] * pymtcnn_bbox_w
    pymtcnn_denorm[i, 1] = pymtcnn_bbox_y + pymtcnn_raw[i, 1] * pymtcnn_bbox_h

diffs_diff_bbox = pymtcnn_denorm - cpp_denorm
errors_diff_bbox = np.sqrt(np.sum(diffs_diff_bbox**2, axis=1))

print("\nUsing respective bboxes:")
for i, err in enumerate(errors_diff_bbox):
    print(f"  Point {i}: {err:.2f} pixels")
print(f"Mean: {np.mean(errors_diff_bbox):.2f} pixels")

print("\n" + "="*80)
print("Analysis")
print("="*80)
print(f"\nError from ONet model differences alone: {np.mean(errors_same_bbox):.2f} pixels")
print(f"Error from bbox + model differences: {np.mean(errors_diff_bbox):.2f} pixels")
print(f"Additional error from bbox mismatch: {np.mean(errors_diff_bbox) - np.mean(errors_same_bbox):.2f} pixels")

print("\nBBox differences:")
print(f"  dx: {pymtcnn_bbox_x - cpp_bbox_x:.2f} pixels")
print(f"  dy: {pymtcnn_bbox_y - cpp_bbox_y:.2f} pixels")
print(f"  dw: {pymtcnn_bbox_w - cpp_bbox_w:.2f} pixels")
print(f"  dh: {pymtcnn_bbox_h - cpp_bbox_h:.2f} pixels")
