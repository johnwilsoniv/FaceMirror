#!/usr/bin/env python3
"""
Compare one iteration of PyCLNF vs OpenFace C++ to identify where they diverge.
"""

import cv2
import numpy as np
from pyclnf import CLNF
import subprocess
import json
from pathlib import Path

# Test frame
video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
frame_idx = 50
face_bbox = (241, 555, 532, 532)

# Load frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to load frame")
    exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print("="*80)
print("ONE-ITERATION COMPARISON: PyCLNF vs OpenFace C++")
print("="*80)
print(f"Frame: {frame_idx} from {video_path}")
print(f"Face bbox: {face_bbox}")
print()

# Run PyCLNF with just 1 iteration
print("Running PyCLNF (1 iteration)...")
clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)
py_landmarks, py_info = clnf.fit(gray, face_bbox, return_params=True)

print(f"  Final update: {py_info['final_update']:.6f}")
print(f"  Landmarks shape: {py_landmarks.shape}")
print()

# Run OpenFace C++ on the same frame
print("Running OpenFace C++ (for comparison)...")
temp_dir = Path("/tmp/openface_compare")
temp_dir.mkdir(exist_ok=True)

# Save frame to temp file
temp_frame_path = temp_dir / "frame.png"
cv2.imwrite(str(temp_frame_path), frame)

# Run OpenFace
openface_bin = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
cmd = [
    openface_bin,
    "-f", str(temp_frame_path),
    "-out_dir", str(temp_dir),
    "-2Dfp"  # Output 2D landmarks
]

result = subprocess.run(cmd, capture_output=True, text=True)

# Read OpenFace output
csv_file = temp_dir / "frame.csv"
if csv_file.exists():
    import pandas as pd
    df = pd.read_csv(csv_file)

    # Extract landmark columns
    x_cols = [c for c in df.columns if c.startswith('x_')]
    y_cols = [c for c in df.columns if c.startswith('y_')]

    cpp_landmarks = np.zeros((len(x_cols), 2))
    for i, (x_col, y_col) in enumerate(zip(sorted(x_cols), sorted(y_cols))):
        cpp_landmarks[i] = [df[x_col].values[0], df[y_col].values[0]]

    print(f"  OpenFace landmarks shape: {cpp_landmarks.shape}")
    print()

    # Compare landmarks
    print("="*80)
    print("LANDMARK COMPARISON")
    print("="*80)

    # Compute differences
    diff = py_landmarks - cpp_landmarks
    diff_mag = np.linalg.norm(diff, axis=1)

    print(f"Mean landmark error: {diff_mag.mean():.2f} pixels")
    print(f"Max landmark error: {diff_mag.max():.2f} pixels")
    print(f"Min landmark error: {diff_mag.min():.2f} pixels")
    print()

    # Show worst landmarks
    worst_idxs = np.argsort(diff_mag)[-5:]
    print("5 worst landmarks:")
    for idx in worst_idxs:
        print(f"  Landmark {idx}: PyCLNF=({py_landmarks[idx,0]:.1f}, {py_landmarks[idx,1]:.1f}) "
              f"OpenFace=({cpp_landmarks[idx,0]:.1f}, {cpp_landmarks[idx,1]:.1f}) "
              f"Error={diff_mag[idx]:.1f}px")
else:
    print("  ERROR: OpenFace did not produce output")
    print(f"  stdout: {result.stdout}")
    print(f"  stderr: {result.stderr}")

print()
print("="*80)
print("Next steps:")
print("  1. Check if initial PDM parameters match")
print("  2. Check if response map values match")
print("  3. Check if mean-shift vectors match")
print("="*80)
