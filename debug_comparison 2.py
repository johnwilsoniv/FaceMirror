"""
Debug comparison script to find where C++ and Python diverge.
Focuses on first iteration, window size 11.
"""

import cv2
import numpy as np
import subprocess
from pyclnf import CLNF
from pyclnf.core.pdm import PDM

# Load frame 0
video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
ret, frame = video.read()
video.release()

# Use bbox from analyze_convergence frame 0
bbox = [311.4, 681.17, 444., 470.76]

print("=" * 70)
print("DEBUG COMPARISON: C++ vs Python")
print("=" * 70)

# Step 1: Get C++ landmarks and iteration data
print("\n[1] Running C++ OpenFace...")
subprocess.run([
    '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
    '-f', 'Patient Data/Normal Cohort/Shorty.mov',
    '-facedetect', str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
    '-out_dir', '/tmp',
    '-frame_number', '0',
    '-2Dfp'
], capture_output=True, text=True)

# Read C++ final landmarks
with open('/tmp/Shorty.csv', 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    values = lines[1].strip().split(',')
    
cpp_landmarks = np.zeros((68, 2))
for i in range(68):
    x_idx = header.index(f'x_{i}')
    y_idx = header.index(f'y_{i}')
    cpp_landmarks[i] = [float(values[x_idx]), float(values[y_idx])]

print(f"    C++ final landmarks loaded")

# Step 2: Run Python with debug mode
print("\n[2] Running Python CLNF with debug mode...")
clnf = CLNF(use_eye_refinement=False, debug_mode=True)

# Get initial landmarks from PDM
pdm = PDM('pyclnf/models/exported_pdm')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initialize params from bbox (like C++)
x, y, w, h = bbox
center_x = x + w / 2
center_y = y + h / 2

# C++ CalcParams uses bbox to get initial params
mean_2d = pdm.mean_shape.flatten()[:68*2].reshape(68, 2)
mean_min = mean_2d.min(axis=0)
mean_max = mean_2d.max(axis=0)
mean_width = mean_max[0] - mean_min[0]
mean_height = mean_max[1] - mean_min[1]

scale = ((w / mean_width) + (h / mean_height)) / 2.0
tx = center_x
ty = center_y

initial_params = np.zeros(pdm.n_params)
initial_params[0] = scale
initial_params[4] = tx
initial_params[5] = ty

initial_landmarks = pdm.params_to_landmarks_2d(initial_params)

print(f"\n[3] Initial Parameters:")
print(f"    scale: {scale:.4f}")
print(f"    tx: {tx:.2f}, ty: {ty:.2f}")

# Compare initial landmarks with C++ ground truth
initial_errors = np.sqrt(np.sum((initial_landmarks - cpp_landmarks) ** 2, axis=1))
print(f"\n[4] Initial Landmarks vs C++ Final:")
print(f"    Mean error: {np.mean(initial_errors):.2f} px")
print(f"    This is expected to be high since we haven't optimized yet")

# Run Python fit
landmarks = clnf.fit(frame, bbox)[0]
final_errors = np.sqrt(np.sum((landmarks - cpp_landmarks) ** 2, axis=1))

print(f"\n[5] Python Final vs C++ Final:")
print(f"    Mean error: {np.mean(final_errors):.2f} px")

# Show sample landmark comparisons
print(f"\n[6] Sample Landmark Comparison (Python vs C++):")
for idx in [0, 27, 36, 48]:
    py = landmarks[idx]
    cpp = cpp_landmarks[idx]
    err = np.sqrt((py[0]-cpp[0])**2 + (py[1]-cpp[1])**2)
    print(f"    {idx}: Py({py[0]:.1f}, {py[1]:.1f}) vs C++({cpp[0]:.1f}, {cpp[1]:.1f}) err={err:.2f}px")

# Check debug files
print(f"\n[7] Debug files generated:")
import os
debug_files = [
    '/tmp/python_param_update_iter0.txt',
    '/tmp/python_ws5_debug.txt',
]
for f in debug_files:
    if os.path.exists(f):
        print(f"    ✓ {f}")
    else:
        print(f"    ✗ {f} (not found)")

print("\n" + "=" * 70)
print("Next: Compare response maps and mean-shifts")
print("=" * 70)
