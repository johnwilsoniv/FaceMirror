#!/usr/bin/env python3
"""
Compare PyCLNF vs OpenFace C++ initialization from the same bounding box.

OpenFace C++ initialization (PDM.cpp:193-231):
1. Computes mean shape with shape parameters applied
2. Rotates shape by initial rotation
3. Finds min/max x,y of rotated shape
4. Scaling = ((bbox.width / model_width) + (bbox.height / model_height)) / 2.0
5. Centers: tx = bbox.x + bbox.width/2 - scaling * (min_x + max_x)/2
          ty = bbox.y + bbox.height/2 - scaling * (min_y + max_y)/2

PyCLNF initialization (pdm.py:370-384):
1. Uses empirical scale: width / 200.0
2. Simple centering: tx = x + width/2, ty = y + height/2
3. Has 54-pixel hack: ty += 54.0

This difference could explain poor convergence!
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core.pdm import PDM
import subprocess
from pathlib import Path
import pandas as pd

# Load test frame
video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_bbox = (241, 555, 532, 532)

print("=" * 80)
print("INITIALIZATION COMPARISON: PyCLNF vs OpenFace C++")
print("=" * 80)
print()

print(f"Face bounding box: {face_bbox}")
print(f"  x={face_bbox[0]}, y={face_bbox[1]}, width={face_bbox[2]}, height={face_bbox[3]}")
print()

# ============================================================================
# 1. PyCLNF initialization
# ============================================================================
print("-" * 80)
print("1. PyCLNF INITIALIZATION")
print("-" * 80)

clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)
py_initial_params = clnf.pdm.init_params(face_bbox)
py_initial_landmarks = clnf.pdm.params_to_landmarks_2d(py_initial_params)

print(f"PyCLNF initial parameters:")
print(f"  Scale:       {py_initial_params[0]:.6f}")
print(f"  Rotation:    [{py_initial_params[1]:.6f}, {py_initial_params[2]:.6f}, {py_initial_params[3]:.6f}]")
print(f"  Translation: [{py_initial_params[4]:.2f}, {py_initial_params[5]:.2f}]")
print(f"  Shape params: {len(py_initial_params) - 6} values (all zero)")
print()

print(f"PyCLNF initialization formula:")
print(f"  Scale:  width / 200.0 = {face_bbox[2]} / 200.0 = {face_bbox[2] / 200.0:.6f}")
print(f"  tx:     x + width/2 = {face_bbox[0]} + {face_bbox[2]/2} = {face_bbox[0] + face_bbox[2]/2:.2f}")
print(f"  ty:     y + height/2 + 54 = {face_bbox[1]} + {face_bbox[3]/2} + 54 = {face_bbox[1] + face_bbox[3]/2 + 54:.2f}")
print()

# ============================================================================
# 2. Compute OpenFace-style initialization in Python
# ============================================================================
print("-" * 80)
print("2. OPENFACE-STYLE INITIALIZATION (Reimplemented in Python)")
print("-" * 80)

# Get mean shape from PDM
mean_shape_3d = clnf.pdm.mean_shape.reshape(-1, 3).T  # Shape: (3, 68)

# With zero rotation and zero shape params, shape is just mean_shape
# Rotate by identity (zero rotation)
rotation = np.array([0.0, 0.0, 0.0])
R = cv2.Rodrigues(rotation)[0]  # 3x3 rotation matrix

# Rotate shape
rotated_shape = R @ mean_shape_3d  # (3, 68)

# Find min/max
min_x = rotated_shape[0, :].min()
max_x = rotated_shape[0, :].max()
min_y = rotated_shape[1, :].min()
max_y = rotated_shape[1, :].max()

model_width = abs(max_x - min_x)
model_height = abs(max_y - min_y)

# OpenFace formula for scale
x, y, width, height = face_bbox
scale_openface = ((width / model_width) + (height / model_height)) / 2.0

# OpenFace translation with correction
tx_openface = x + width / 2.0 - scale_openface * (min_x + max_x) / 2.0
ty_openface = y + height / 2.0 - scale_openface * (min_y + max_y) / 2.0

print(f"Mean shape dimensions:")
print(f"  Model width:  {model_width:.2f}")
print(f"  Model height: {model_height:.2f}")
print(f"  min_x: {min_x:.2f}, max_x: {max_x:.2f}")
print(f"  min_y: {min_y:.2f}, max_y: {max_y:.2f}")
print()

print(f"OpenFace-style scale computation:")
print(f"  scale_width  = bbox.width / model_width  = {width} / {model_width:.2f} = {width / model_width:.6f}")
print(f"  scale_height = bbox.height / model_height = {height} / {model_height:.2f} = {height / model_height:.6f}")
print(f"  scale = (scale_width + scale_height) / 2 = {scale_openface:.6f}")
print()

print(f"OpenFace-style translation:")
print(f"  tx_center = x + width/2  = {x} + {width/2} = {x + width/2:.2f}")
print(f"  ty_center = y + height/2 = {y} + {height/2} = {y + height/2:.2f}")
print(f"  model_center_x = (min_x + max_x)/2 = {(min_x + max_x)/2:.2f}")
print(f"  model_center_y = (min_y + max_y)/2 = {(min_y + max_y)/2:.2f}")
print(f"  tx = tx_center - scale * model_center_x = {x + width/2:.2f} - {scale_openface:.6f} * {(min_x + max_x)/2:.2f} = {tx_openface:.2f}")
print(f"  ty = ty_center - scale * model_center_y = {y + height/2:.2f} - {scale_openface:.6f} * {(min_y + max_y)/2:.2f} = {ty_openface:.2f}")
print()

# Create OpenFace-style params
of_params = py_initial_params.copy()
of_params[0] = scale_openface
of_params[4] = tx_openface
of_params[5] = ty_openface

of_initial_landmarks = clnf.pdm.params_to_landmarks_2d(of_params)

print(f"OpenFace-style initial parameters:")
print(f"  Scale:       {of_params[0]:.6f}")
print(f"  Rotation:    [{of_params[1]:.6f}, {of_params[2]:.6f}, {of_params[3]:.6f}]")
print(f"  Translation: [{of_params[4]:.2f}, {of_params[5]:.2f}]")
print()

# ============================================================================
# 3. Compare
# ============================================================================
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

print(f"{'Parameter':<20} {'PyCLNF':>15} {'OpenFace-style':>15} {'Difference':>15}")
print("-" * 80)
print(f"{'Scale':<20} {py_initial_params[0]:>15.6f} {of_params[0]:>15.6f} {py_initial_params[0] - of_params[0]:>15.6f}")
print(f"{'tx':<20} {py_initial_params[4]:>15.2f} {of_params[4]:>15.2f} {py_initial_params[4] - of_params[4]:>15.2f}")
print(f"{'ty':<20} {py_initial_params[5]:>15.2f} {of_params[5]:>15.2f} {py_initial_params[5] - of_params[5]:>15.2f}")
print()

# Compute landmark differences
landmark_diff = py_initial_landmarks - of_initial_landmarks
landmark_dist = np.linalg.norm(landmark_diff, axis=1)

print(f"Initial landmark positions (PyCLNF vs OpenFace-style):")
print(f"  Mean distance: {landmark_dist.mean():.2f} pixels")
print(f"  Max distance:  {landmark_dist.max():.2f} pixels")
print()

if landmark_dist.mean() > 5.0:
    print("⚠️  WARNING: Initial landmarks differ significantly!")
    print("   This will require more iterations to converge.")
    print()

# ============================================================================
# 4. Test convergence from both initializations
# ============================================================================
print("=" * 80)
print("CONVERGENCE COMPARISON")
print("=" * 80)
print()

print("Testing convergence with PyCLNF initialization...")
clnf1 = CLNF(model_dir='pyclnf/models', max_iterations=20)
py_landmarks, py_info = clnf1.fit(gray, face_bbox)
print(f"  PyCLNF init: Converged={py_info['converged']}, Iterations={py_info['iterations']}, Final update={py_info['final_update']:.6f}")

# Test with OpenFace-style initialization
# We can manually set the initial params by providing them to the optimizer
print("Testing convergence with OpenFace-style initialization...")
clnf2 = CLNF(model_dir='pyclnf/models', max_iterations=20)
# Override initialization by getting patch experts and calling optimizer directly
view_idx = 0
patch_experts = clnf2._get_patch_experts(view_idx, 0.25)
of_landmarks, of_info = clnf2.optimizer.optimize(
    clnf2.pdm,
    of_params,  # Start from OpenFace-style params
    patch_experts,
    gray,
    weights=None,
    window_size=11,
    patch_scaling=0.25,
    sigma_components=clnf2.ccnf.sigma_components
)
# Convert params to landmarks
of_final_landmarks = clnf2.pdm.params_to_landmarks_2d(of_info['params'])
print(f"  OpenFace init: Final update={of_info['iteration_history'][-1]['update_magnitude']:.6f}, Iterations={len(of_info['iteration_history'])}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)

print()
if abs(py_initial_params[0] - of_params[0]) > 0.1:
    print("❌ INITIALIZATION DIFFERENCE FOUND!")
    print(f"   Scale differs by {abs(py_initial_params[0] - of_params[0]):.6f}")
    print(f"   PyCLNF uses empirical scale (width / 200.0)")
    print(f"   OpenFace computes scale from actual model dimensions")
    print()
    print("   Recommendation: Implement OpenFace-style initialization in pdm.py")
else:
    print("✓ Initializations are similar")

print()
print("=" * 80)
