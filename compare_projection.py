"""
Compare PyCLNF vs OpenFace C++ landmark projection step-by-step.
"""
import numpy as np
from pyclnf import CLNF
import subprocess
import csv

# Initialize PyCLNF
clnf = CLNF(model_dir="pyclnf/models")

# Use the same parameters from the diagnostic
face_bbox = (241, 555, 532, 532)
initial_params = clnf.pdm.init_params(face_bbox)

print("=" * 80)
print("Projection Comparison: PyCLNF vs OpenFace C++")
print("=" * 80)

# Get PyCLNF landmarks
py_landmarks_3d = clnf.pdm.params_to_landmarks_3d(initial_params)
py_landmarks_2d = clnf.pdm.params_to_landmarks_2d(initial_params)

print(f"\nPyCLNF Initial Parameters:")
print(f"  scale={initial_params[0]:.3f}")
print(f"  rotation: pitch={initial_params[1]:.3f}, yaw={initial_params[2]:.3f}, roll={initial_params[3]:.3f}")
print(f"  translation: tx={initial_params[4]:.1f}, ty={initial_params[5]:.1f}")

print(f"\nPyCLNF 3D Landmarks:")
print(f"  Shape: {py_landmarks_3d.shape}")
print(f"  X range: [{py_landmarks_3d[:, 0].min():.1f}, {py_landmarks_3d[:, 0].max():.1f}]")
print(f"  Y range: [{py_landmarks_3d[:, 1].min():.1f}, {py_landmarks_3d[:, 1].max():.1f}]")
print(f"  Z range: [{py_landmarks_3d[:, 2].min():.1f}, {py_landmarks_3d[:, 2].max():.1f}]")

print(f"\nPyCLNF 2D Landmarks (projection):")
print(f"  Shape: {py_landmarks_2d.shape}")
print(f"  X range: [{py_landmarks_2d[:, 0].min():.1f}, {py_landmarks_2d[:, 0].max():.1f}]")
print(f"  Y range: [{py_landmarks_2d[:, 1].min():.1f}, {py_landmarks_2d[:, 1].max():.1f}]")
print(f"  Center: ({py_landmarks_2d[:, 0].mean():.1f}, {py_landmarks_2d[:, 1].mean():.1f})")

# Load OpenFace landmarks from the diagnostic output
print(f"\n" + "=" * 80)
print("Loading OpenFace C++ landmarks...")
csv_file = "/tmp/openface_diagnostic/frame.csv"
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    row = next(reader)

    cpp_landmarks = np.zeros((68, 2))
    for i in range(68):
        cpp_landmarks[i, 0] = float(row[f'x_{i}'])
        cpp_landmarks[i, 1] = float(row[f'y_{i}'])

print(f"\nOpenFace C++ 2D Landmarks:")
print(f"  X range: [{cpp_landmarks[:, 0].min():.1f}, {cpp_landmarks[:, 0].max():.1f}]")
print(f"  Y range: [{cpp_landmarks[:, 1].min():.1f}, {cpp_landmarks[:, 1].max():.1f}]")
print(f"  Center: ({cpp_landmarks[:, 0].mean():.1f}, {cpp_landmarks[:, 1].mean():.1f})")

# Compare
print(f"\n" + "=" * 80)
print("Comparison (PyCLNF - OpenFace C++):")
diff = py_landmarks_2d - cpp_landmarks
print(f"  Mean X offset: {diff[:, 0].mean():.1f} pixels")
print(f"  Mean Y offset: {diff[:, 1].mean():.1f} pixels")
print(f"  Mean L2 error: {np.linalg.norm(diff, axis=1).mean():.1f} pixels")

# Check specific landmarks
print(f"\nSpecific Landmark Comparison (INITIAL, before optimization):")
landmarks_to_check = {
    0: "Jaw left",
    8: "Chin",
    16: "Jaw right",
    27: "Nose tip",
    36: "Left eye left",
    45: "Right eye right"
}

for idx, name in landmarks_to_check.items():
    py_pt = py_landmarks_2d[idx]
    cpp_pt = cpp_landmarks[idx]
    error = np.linalg.norm(py_pt - cpp_pt)
    print(f"  {name:20s} (#{idx:2d}): PY=({py_pt[0]:6.1f}, {py_pt[1]:6.1f})  CPP=({cpp_pt[0]:6.1f}, {cpp_pt[1]:6.1f})  Error={error:6.2f}px")

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)

# Calculate the systematic offset
mean_x_offset = diff[:, 0].mean()
mean_y_offset = diff[:, 1].mean()

if abs(mean_y_offset) > 40:
    print(f"\n⚠️  SYSTEMATIC VERTICAL OFFSET DETECTED: {mean_y_offset:.1f} pixels")
    print("   This suggests:")
    if mean_y_offset > 0:
        print("   - PyCLNF landmarks are BELOW OpenFace landmarks")
    else:
        print("   - PyCLNF landmarks are ABOVE OpenFace landmarks")

    print("\n   Possible causes:")
    print("   1. Y-axis orientation difference (top-left vs bottom-left origin)")
    print("   2. Camera calibration parameter mismatch")
    print("   3. Translation parameter interpretation difference")

if abs(mean_x_offset) > 20:
    print(f"\n⚠️  SYSTEMATIC HORIZONTAL OFFSET DETECTED: {mean_x_offset:.1f} pixels")
    if mean_x_offset > 0:
        print("   - PyCLNF landmarks are to the RIGHT of OpenFace landmarks")
    else:
        print("   - PyCLNF landmarks are to the LEFT of OpenFace landmarks")

# Check if the error is consistent across all landmarks (systematic vs random)
std_x = diff[:, 0].std()
std_y = diff[:, 1].std()
print(f"\nError variability:")
print(f"  X std dev: {std_x:.1f} pixels (systematic if low)")
print(f"  Y std dev: {std_y:.1f} pixels (systematic if low)")

if std_x < 10 and std_y < 10:
    print("\n✓ Low variability suggests SYSTEMATIC offset (fixable with calibration)")
else:
    print("\n✗ High variability suggests SHAPE/PROJECTION mismatch")

print("\n" + "=" * 80)
