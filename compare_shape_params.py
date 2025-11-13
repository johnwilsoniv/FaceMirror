"""
Compare shape parameter values between PyCLNF and OpenFace C++.
"""
import numpy as np
import csv
from pyclnf import CLNF
import cv2

# Load test video
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read video: {video_path}")

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_bbox = (241, 555, 532, 532)

# Run PyCLNF
print("=" * 80)
print("Running PyCLNF...")
clnf = CLNF(model_dir="pyclnf/models")
py_landmarks, py_info = clnf.fit(gray, face_bbox, return_params=True)
py_params = py_info['params']

print(f"  PyCLNF shape parameters (first 15):")
for i in range(min(15, len(py_params) - 6)):
    print(f"    Mode {i:2d}: {py_params[6+i]:8.3f}")

print(f"\n  PyCLNF shape parameter statistics:")
print(f"    Min: {py_params[6:].min():8.3f}")
print(f"    Max: {py_params[6:].max():8.3f}")
print(f"    Mean: {py_params[6:].mean():8.3f}")
print(f"    Std: {py_params[6:].std():8.3f}")
print(f"    Mean |value|: {np.abs(py_params[6:]).mean():8.3f}")

# Load OpenFace C++ parameters from CSV
print("\n" + "=" * 80)
print("Loading OpenFace C++ parameters...")

# OpenFace CSV has columns: frame, face_id, timestamp, confidence, success,
# followed by p_scale, p_rx, p_ry, p_rz, p_tx, p_ty, p_0, p_1, ..., p_33
cpp_csv_path = "/tmp/openface_test/IMG_0433.csv"

try:
    with open(cpp_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find parameter columns
        param_cols = [i for i, col in enumerate(header) if col.startswith('p_')]
        print(f"  Found {len(param_cols)} parameter columns")

        # Read first frame
        row = next(reader)

        # Extract parameters (p_scale, p_rx, p_ry, p_rz, p_tx, p_ty, p_0, ..., p_33)
        cpp_params = np.array([float(row[i]) for i in param_cols])

        # OpenFace order: scale, rx, ry, rz, tx, ty, shape_params
        cpp_shape_params = cpp_params[6:]  # Skip first 6 (scale, rotation, translation)

        print(f"  OpenFace C++ shape parameters (first 15):")
        for i in range(min(15, len(cpp_shape_params))):
            print(f"    Mode {i:2d}: {cpp_shape_params[i]:8.3f}")

        print(f"\n  OpenFace C++ shape parameter statistics:")
        print(f"    Min: {cpp_shape_params.min():8.3f}")
        print(f"    Max: {cpp_shape_params.max():8.3f}")
        print(f"    Mean: {cpp_shape_params.mean():8.3f}")
        print(f"    Std: {cpp_shape_params.std():8.3f}")
        print(f"    Mean |value|: {np.abs(cpp_shape_params).mean():8.3f}")

except FileNotFoundError:
    print(f"  ERROR: OpenFace CSV not found at {cpp_csv_path}")
    print(f"  Please run OpenFace C++ first to generate comparison data.")
    exit(1)

# Compare
print("\n" + "=" * 80)
print("Comparison:")
print("  Mode | PyCLNF | OpenFace | Difference | Ratio")
print("  " + "-" * 60)
py_shape = py_params[6:]
for i in range(min(20, len(py_shape))):
    diff = py_shape[i] - cpp_shape_params[i]
    if abs(cpp_shape_params[i]) > 0.01:
        ratio = py_shape[i] / cpp_shape_params[i]
    else:
        ratio = float('nan')
    print(f"  {i:4d} | {py_shape[i]:7.2f} | {cpp_shape_params[i]:8.2f} | {diff:10.2f} | {ratio:6.2f}")

print("\n" + "=" * 80)
print("Key differences:")

# Find modes with large differences
diffs = np.abs(py_shape - cpp_shape_params[:len(py_shape)])
large_diff_indices = np.argsort(diffs)[-5:][::-1]

print(f"\nTop 5 modes with largest differences:")
for idx in large_diff_indices:
    diff = py_shape[idx] - cpp_shape_params[idx]
    print(f"  Mode {idx:2d}: PyCLNF={py_shape[idx]:7.2f}, OpenFace={cpp_shape_params[idx]:7.2f}, Diff={diff:7.2f}")

print("\n" + "=" * 80)
