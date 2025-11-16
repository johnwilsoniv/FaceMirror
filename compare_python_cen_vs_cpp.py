#!/usr/bin/env python3
"""Compare Python CEN output vs C++ CEN output"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pyclnf.clnf import CLNF
import struct
import os

# Load C++ response map (saved from LandmarkDetectorModel.cpp)
def load_cpp_response_map(filepath):
    with open(filepath, 'rb') as f:
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
        return data.reshape(rows, cols)

# Load test image
image_path = "calibration_frames/patient1_frame1.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

print("=" * 80)
print("PYTHON CEN vs C++ CEN VALIDATION")
print("=" * 80)

# Initialize CLNF with CEN
print("\n1. Loading Python CEN...")
clnf = CLNF(
    model_dir="pyclnf/models",
    debug_mode=True,
    tracked_landmarks=[36, 48, 30, 8]
)

# Use exact C++ bbox
bbox = (301.938, 782.149, 400.586, 400.585)
print(f"\n2. Using exact C++ bbox: {bbox}")

# Run Python CLNF
print("\n3. Running Python CLNF with CEN...")
landmarks_py, info_py = clnf.fit(image, face_bbox=bbox)

print(f"\n4. Python optimization complete")
print(f"   Final landmarks for validation:")
for idx in [36, 48, 30, 8]:
    print(f"   Landmark {idx}: ({landmarks_py[idx, 0]:.4f}, {landmarks_py[idx, 1]:.4f})")

# Load and compare response maps
print("\n5. Comparing response maps...")

# Check if Python saved response map
py_response_file = '/tmp/python_response_map_lm36_iter0_ws11_BEFORE_SIGMA.npy'
cpp_response_file = '/tmp/cpp_response_map_lm36_iter0.bin'

try:
    if os.path.exists(py_response_file):
        py_resp = np.load(py_response_file)
        print(f"   Python response map: shape={py_resp.shape}, min={py_resp.min():.6f}, max={py_resp.max():.6f}, mean={py_resp.mean():.6f}")
    else:
        print(f"   ⚠ Python response map not found at {py_response_file}")
        py_resp = None

    if os.path.exists(cpp_response_file):
        cpp_resp = load_cpp_response_map(cpp_response_file)
        print(f"   C++ response map:    shape={cpp_resp.shape}, min={cpp_resp.min():.6f}, max={cpp_resp.max():.6f}, mean={cpp_resp.mean():.6f}")
    else:
        print(f"   ⚠ C++ response map not found at {cpp_response_file}")
        cpp_resp = None

    if py_resp is not None and cpp_resp is not None:
        # Compare
        diff = np.abs(py_resp - cpp_resp)
        corr = np.corrcoef(py_resp.flatten(), cpp_resp.flatten())[0, 1]
        rms = np.sqrt(np.mean(diff**2))
        max_diff = diff.max()

        print(f"\n6. Response Map Comparison (Landmark 36, ITER0, WS=11):")
        print(f"   Correlation:  {corr:.6f}")
        print(f"   RMS diff:     {rms:.6f}")
        print(f"   Max diff:     {max_diff:.6f}")
        print(f"   Mean diff:    {diff.mean():.6f}")

        # Find peak locations
        py_peak = np.unravel_index(py_resp.argmax(), py_resp.shape)
        cpp_peak = np.unravel_index(cpp_resp.argmax(), cpp_resp.shape)

        print(f"\n   Peak locations:")
        print(f"   Python peak: {py_peak} with value {py_resp[py_peak]:.6f}")
        print(f"   C++ peak:    {cpp_peak} with value {cpp_resp[cpp_peak]:.6f}")

        if py_peak == cpp_peak and corr > 0.99 and max_diff < 0.01:
            print(f"\n   ✅ Response maps MATCH!")
        elif corr > 0.95:
            print(f"\n   ⚠ Response maps are similar but not identical")
        else:
            print(f"\n   ❌ Response maps differ significantly")

except Exception as e:
    print(f"   Error comparing response maps: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
