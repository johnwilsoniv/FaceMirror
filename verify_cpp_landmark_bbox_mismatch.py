#!/usr/bin/env python3
"""Verify C++ landmark/bbox mismatch"""

import numpy as np
import pandas as pd

# C++ POST-calibration bbox (from CSV)
cpp_postcal_x = 301.9380
cpp_postcal_y = 782.1490
cpp_postcal_w = 400.5860
cpp_postcal_h = 400.5850

# Reverse calibration to get PRE-calibration bbox
cpp_precal_w = cpp_postcal_w / 1.0323
cpp_precal_h = cpp_postcal_h / 0.7751
cpp_precal_x = cpp_postcal_x + cpp_precal_w * 0.0075
cpp_precal_y = cpp_postcal_y - cpp_precal_h * 0.2459

print("C++ BBoxes:")
print(f"PRE-calibration:  x={cpp_precal_x:.4f}, y={cpp_precal_y:.4f}, w={cpp_precal_w:.4f}, h={cpp_precal_h:.4f}")
print(f"POST-calibration: x={cpp_postcal_x:.4f}, y={cpp_postcal_y:.4f}, w={cpp_postcal_w:.4f}, h={cpp_postcal_h:.4f}")

# PyMTCNN POST-calibration bbox
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

print("\n" + "="*80)
print("Method 1: Denormalize C++ landmarks with PRE-cal bbox (CORRECT)")
print("="*80)

# Denormalize C++ landmarks with PRE-cal bbox (what they were stored with)
cpp_landmarks_correct = np.zeros((5, 2))
for i in range(5):
    cpp_landmarks_correct[i, 0] = cpp_precal_x + cpp_raw[i, 0] * cpp_precal_w
    cpp_landmarks_correct[i, 1] = cpp_precal_y + cpp_raw[i, 1] * cpp_precal_h

# Denormalize PyMTCNN landmarks with POST-cal bbox
pymtcnn_landmarks = np.zeros((5, 2))
for i in range(5):
    pymtcnn_landmarks[i, 0] = pymtcnn_bbox_x + pymtcnn_raw[i, 0] * pymtcnn_bbox_w
    pymtcnn_landmarks[i, 1] = pymtcnn_bbox_y + pymtcnn_raw[i, 1] * pymtcnn_bbox_h

diffs = pymtcnn_landmarks - cpp_landmarks_correct
errors = np.sqrt(np.sum(diffs**2, axis=1))

print(f"\nMean error: {np.mean(errors):.2f} pixels")
for i, err in enumerate(errors):
    print(f"  Point {i}: {err:.2f} pixels")

print("\n" + "="*80)
print("Method 2: Denormalize C++ landmarks with POST-cal bbox (WRONG - current bug)")
print("="*80)

# Denormalize C++ landmarks with POST-cal bbox (WRONG - what CSV implies)
cpp_landmarks_wrong = np.zeros((5, 2))
for i in range(5):
    cpp_landmarks_wrong[i, 0] = cpp_postcal_x + cpp_raw[i, 0] * cpp_postcal_w
    cpp_landmarks_wrong[i, 1] = cpp_postcal_y + cpp_raw[i, 1] * cpp_postcal_h

diffs_wrong = pymtcnn_landmarks - cpp_landmarks_wrong
errors_wrong = np.sqrt(np.sum(diffs_wrong**2, axis=1))

print(f"\nMean error: {np.mean(errors_wrong):.2f} pixels (INCORRECT DUE TO C++ CSV BUG)")
for i, err in enumerate(errors_wrong):
    print(f"  Point {i}: {err:.2f} pixels")
