#!/usr/bin/env python3
"""
Find the first point of divergence between C++ and Python eye refinement.
Compare step by step: response maps -> mean-shift -> transform
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

# Check if C++ response maps exist
try:
    with open('/tmp/cpp_eye_response_maps.txt', 'r') as f:
        cpp_content = f.read()
    print("C++ Eye Response Maps found")
except:
    print("ERROR: Run C++ OpenFace first to generate /tmp/cpp_eye_response_maps.txt")
    sys.exit(1)

# Parse C++ response maps for Eye_8
import re
match = re.search(r'Eye landmark 8 response map:.*?3x3 response map:\s*\n((?:\s+[-\d.]+\s*)+)', cpp_content, re.DOTALL)
if match:
    values = [float(x) for x in match.group(1).split()]
    cpp_resp_8 = np.array(values).reshape(3, 3)
    print("\nC++ Eye_8 response map (3x3):")
    for row in cpp_resp_8:
        print(f"  {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}")
else:
    print("Could not parse C++ Eye_8 response map")
    cpp_resp_8 = None

# Get Python response maps
try:
    with open('/tmp/python_eye_response_maps.txt', 'r') as f:
        py_content = f.read()

    # Parse Python Eye_8 response map (5x5 in file, need window_size=3)
    match = re.search(r'Eye landmark 8 response map:.*?5x5 response map:\s*\n((?:\s+[-\d.]+\s*)+)', py_content, re.DOTALL)
    if match:
        values = [float(x) for x in match.group(1).split()]
        py_resp_8_5x5 = np.array(values).reshape(5, 5)
        print("\nPython Eye_8 response map (5x5):")
        for row in py_resp_8_5x5:
            print(f"  {' '.join([f'{x:8.4f}' for x in row])}")
except:
    print("Python response maps not found")
    py_resp_8_5x5 = None

# The issue: Python has 5x5 but C++ has 3x3 for window_size=3
# Need to run Python with same window size

print("\n" + "=" * 60)
print("STEP 1: Compare response maps at window_size=3")
print("=" * 60)

if cpp_resp_8 is not None:
    # Find peak location in C++
    cpp_peak = np.unravel_index(np.argmax(cpp_resp_8), cpp_resp_8.shape)
    print(f"\nC++ Eye_8 peak at: row={cpp_peak[0]}, col={cpp_peak[1]}")
    print(f"C++ peak value: {cpp_resp_8[cpp_peak]:.4f}")

    # Compute mean-shift from C++ response map
    center = 1.0  # (3-1)/2
    ws = 3
    sigma = 1.0
    a_kde = -0.5 / (sigma * sigma)

    total = 0.0
    mx = 0.0
    my = 0.0
    for ii in range(ws):
        for jj in range(ws):
            dist_sq = (ii - center)**2 + (jj - center)**2
            kde = np.exp(a_kde * dist_sq)
            w = cpp_resp_8[ii, jj] * kde
            total += w
            mx += w * jj
            my += w * ii

    if total > 1e-10:
        cpp_ms_x = mx / total - center
        cpp_ms_y = my / total - center
    else:
        cpp_ms_x = cpp_ms_y = 0.0

    print(f"\nC++ Eye_8 mean-shift (computed from response map):")
    print(f"  ms_x = {cpp_ms_x:.6f}, ms_y = {cpp_ms_y:.6f}")

print("\n" + "=" * 60)
print("STEP 2: Run Python with window_size=3 and compare")
print("=" * 60)

# Now run Python eye refinement and get its 3x3 response map
from pyclnf.core.eye_patch_expert import HierarchicalEyeModel
from pyclnf.core.eye_pdm import EyePDM
import cv2

# Load image
video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Load PDM and get landmarks
model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_pdm_left'
pdm = EyePDM(model_dir)

params = np.zeros(pdm.n_params)
params[0] = 3.371204
params[1] = -0.118319
params[2] = 0.176098
params[3] = -0.099366
params[4] = 425.031629
params[5] = 820.112023

eye_landmarks = pdm.params_to_landmarks_2d(params)

print(f"\nEye_8 position: ({eye_landmarks[8, 0]:.4f}, {eye_landmarks[8, 1]:.4f})")

# Load patch expert for landmark 8
from pyclnf.core.eye_patch_expert import EyeCCNFPatchExpert, align_shapes_with_scale
ccnf_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_ccnf_left/scale_1.00/view_00'
patch_expert = EyeCCNFPatchExpert(f'{ccnf_dir}/patch_08')

# Compute similarity transform
ref_params = params.copy()
ref_params[0] = 1.0
ref_params[1:4] = 0
ref_params[4:6] = 0
reference_shape = pdm.params_to_landmarks_2d(ref_params)

sim_img_to_ref = align_shapes_with_scale(eye_landmarks, reference_shape)
sim_ref_to_img = np.linalg.inv(sim_img_to_ref)

a1 = sim_ref_to_img[0, 0]
b1 = -sim_ref_to_img[0, 1]

print(f"a1 = {a1:.6f}, b1 = {b1:.6f}")

# Extract AOI for window_size=3
window_size = 3
patch_size = 11
aoi_size = window_size + patch_size - 1  # 13
half_aoi = (aoi_size - 1) / 2.0

x, y = eye_landmarks[8]
tx = x - a1 * half_aoi + b1 * half_aoi
ty = y - a1 * half_aoi - b1 * half_aoi

sim = np.array([[a1, -b1, tx],
                [b1, a1, ty]], dtype=np.float32)

aoi = cv2.warpAffine(gray.astype(np.float32), sim, (aoi_size, aoi_size),
                     flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)

print(f"\nAOI extracted: {aoi_size}x{aoi_size}")
print(f"AOI stats: min={aoi.min():.1f}, max={aoi.max():.1f}, mean={aoi.mean():.1f}")

# Compute 3x3 response map
py_resp_8 = np.zeros((window_size, window_size), dtype=np.float32)
for i in range(window_size):
    for j in range(window_size):
        patch = aoi[i:i+patch_size, j:j+patch_size]
        py_resp_8[i, j] = patch_expert.compute_response(patch.astype(np.uint8))

# Make non-negative
min_val = py_resp_8.min()
if min_val < 0:
    py_resp_8 = py_resp_8 - min_val

print(f"\nPython Eye_8 response map (3x3):")
for row in py_resp_8:
    print(f"  {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}")

# Find peak
py_peak = np.unravel_index(np.argmax(py_resp_8), py_resp_8.shape)
print(f"\nPython Eye_8 peak at: row={py_peak[0]}, col={py_peak[1]}")
print(f"Python peak value: {py_resp_8[py_peak]:.4f}")

# Compute mean-shift
total = 0.0
mx = 0.0
my = 0.0
for ii in range(ws):
    for jj in range(ws):
        dist_sq = (ii - center)**2 + (jj - center)**2
        kde = np.exp(a_kde * dist_sq)
        w = py_resp_8[ii, jj] * kde
        total += w
        mx += w * jj
        my += w * ii

if total > 1e-10:
    py_ms_x = mx / total - center
    py_ms_y = my / total - center
else:
    py_ms_x = py_ms_y = 0.0

print(f"\nPython Eye_8 mean-shift (from response map):")
print(f"  ms_x = {py_ms_x:.6f}, ms_y = {py_ms_y:.6f}")

print("\n" + "=" * 60)
print("DIVERGENCE ANALYSIS")
print("=" * 60)

if cpp_resp_8 is not None:
    print("\nResponse map comparison:")
    diff = py_resp_8 - cpp_resp_8
    print(f"Max absolute difference: {np.max(np.abs(diff)):.4f}")

    if np.max(np.abs(diff)) > 0.01:
        print(">>> DIVERGENCE FOUND: Response maps differ!")
        print("\nDifference (Python - C++):")
        for row in diff:
            print(f"  {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}")
    else:
        print("Response maps match!")

    print(f"\nMean-shift comparison:")
    print(f"  C++:    ({cpp_ms_x:8.4f}, {cpp_ms_y:8.4f})")
    print(f"  Python: ({py_ms_x:8.4f}, {py_ms_y:8.4f})")
    print(f"  Diff:   ({py_ms_x - cpp_ms_x:8.4f}, {py_ms_y - cpp_ms_y:8.4f})")
