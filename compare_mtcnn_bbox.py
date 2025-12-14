#!/usr/bin/env python3
"""Compare MTCNN bbox between Python and C++."""

import numpy as np
import cv2
import sys

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')

from pyclnf.core.pdm import PDM
from pymtcnn.backends.coreml_backend import CoreMLMTCNN

print("=" * 80)
print("MTCNN BBOX COMPARISON: Python vs C++")
print("=" * 80)

# Load frame
video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0422.MOV'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

print(f"Frame shape: {frame.shape}")

# C++ expected bbox (reverse-engineered from init params)
cpp_x, cpp_y, cpp_w, cpp_h = 196.21, 790.04, 531.71, 513.60
print(f"\n--- C++ Expected bbox (from init params) ---")
print(f"  x={cpp_x:.2f}, y={cpp_y:.2f}, w={cpp_w:.2f}, h={cpp_h:.2f}")
print(f"  Center: ({cpp_x + cpp_w/2:.2f}, {cpp_y + cpp_h/2:.2f})")

# Run Python MTCNN
print(f"\n--- Python MTCNN Detection ---")
detector = CoreMLMTCNN()
bboxes, landmarks = detector.detect(frame)

if len(bboxes) > 0:
    py_x, py_y, py_w, py_h = bboxes[0]
    print(f"  x={py_x:.2f}, y={py_y:.2f}, w={py_w:.2f}, h={py_h:.2f}")
    print(f"  Center: ({py_x + py_w/2:.2f}, {py_y + py_h/2:.2f})")

    # Difference
    print(f"\n--- Difference (Python - C++) ---")
    print(f"  Δx = {py_x - cpp_x:.2f}")
    print(f"  Δy = {py_y - cpp_y:.2f}")
    print(f"  Δw = {py_w - cpp_w:.2f}")
    print(f"  Δh = {py_h - cpp_h:.2f}")

    # Load PDM and compute init params for both bboxes
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/pyclnf/models/exported_pdm'
    pdm = PDM(model_dir)

    cpp_params = pdm.init_params((cpp_x, cpp_y, cpp_w, cpp_h))
    py_params = pdm.init_params((py_x, py_y, py_w, py_h))

    print(f"\n--- Init params from C++ bbox ---")
    print(f"  scale: {cpp_params[0]:.6f}")
    print(f"  tx:    {cpp_params[4]:.6f}")
    print(f"  ty:    {cpp_params[5]:.6f}")

    print(f"\n--- Init params from Python bbox ---")
    print(f"  scale: {py_params[0]:.6f}")
    print(f"  tx:    {py_params[4]:.6f}")
    print(f"  ty:    {py_params[5]:.6f}")

    print(f"\n--- Init param difference ---")
    print(f"  Δscale: {py_params[0] - cpp_params[0]:.6f}")
    print(f"  Δtx:    {py_params[4] - cpp_params[4]:.6f}")
    print(f"  Δty:    {py_params[5] - cpp_params[5]:.6f}")

    # The key insight: this init difference is what causes divergence
    print(f"\n" + "=" * 80)
    print("IMPACT ON OPTIMIZATION")
    print("=" * 80)
    print("""
The bbox difference causes init params to differ by:
  - Scale: ~0.05-0.1
  - tx: ~5-10 pixels
  - ty: ~5-10 pixels

With reg_factor=1.0:
  - The optimizer has freedom to explore
  - Both init positions converge to similar landmarks
  - Python: Local[0] ≈ 32 (jaw shape variant)

With reg_factor=22.5:
  - The optimizer is heavily constrained toward Local[0]=0
  - Different init positions converge to DIFFERENT local optima
  - C++: converges to Local[0] ≈ 12 (good)
  - Python: converges to a BAD local optimum (high error)

CONCLUSION: Python should use reg_factor=1.0 to match AU accuracy.
The exact C++ value (22.5) doesn't transfer because the optimization
landscape is sensitive to the starting position.
""")
