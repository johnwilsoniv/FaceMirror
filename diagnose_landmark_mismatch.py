"""
Diagnose why PyCLNF landmarks don't match OpenFace C++ landmarks.
"""

import numpy as np
import cv2
from pyclnf import CLNF
import subprocess
import os

# Test frame from IMG_0433
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
frame_num = 50

# Load frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

print("=" * 80)
print("Landmark Mismatch Diagnostic")
print("=" * 80)

# Get OpenFace C++ landmarks
print("\n1. Running OpenFace C++ for ground truth...")
temp_dir = "/tmp/openface_diagnostic"
os.makedirs(temp_dir, exist_ok=True)

# Save frame
temp_frame = f"{temp_dir}/frame.jpg"
cv2.imwrite(temp_frame, frame)

# Run OpenFace
cmd = f"~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction -f {temp_frame} -out_dir {temp_dir} -2Dfp"
subprocess.run(cmd, shell=True, capture_output=True)

# Load OpenFace landmarks
import csv
csv_file = f"{temp_dir}/frame.csv"
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    row = next(reader)

    # Extract landmarks
    cpp_landmarks = np.zeros((68, 2))
    for i in range(68):
        cpp_landmarks[i, 0] = float(row[f'x_{i}'])
        cpp_landmarks[i, 1] = float(row[f'y_{i}'])

print(f"  OpenFace C++ landmarks loaded: {cpp_landmarks.shape}")
print(f"  OpenFace C++ landmark range: x=[{cpp_landmarks[:, 0].min():.1f}, {cpp_landmarks[:, 0].max():.1f}], y=[{cpp_landmarks[:, 1].min():.1f}, {cpp_landmarks[:, 1].max():.1f}]")

# Get face bbox from OpenFace
cpp_x_min = float(row['x_0'])
cpp_x_max = float(row['x_16'])  # Jawline
cpp_y_min = min(float(row['y_19']), float(row['y_24']))  # Eyebrows
cpp_y_max = float(row['y_8'])  # Chin

cpp_width = cpp_x_max - cpp_x_min
cpp_height = cpp_y_max - cpp_y_min
cpp_bbox = (cpp_x_min, cpp_y_min, cpp_width, cpp_height)
print(f"  OpenFace C++ estimated bbox: x={cpp_x_min:.1f}, y={cpp_y_min:.1f}, w={cpp_width:.1f}, h={cpp_height:.1f}")

# Initialize PyCLNF
print("\n2. Initializing PyCLNF...")
clnf = CLNF(model_dir="pyclnf/models")

# Use a reasonable face bbox (from face detector or manual)
# Let's use the bbox from our test
face_bbox = (241, 555, 532, 532)
print(f"  Using face bbox: x={face_bbox[0]}, y={face_bbox[1]}, w={face_bbox[2]}, h={face_bbox[3]}")

# Get initial params
initial_params = clnf.pdm.init_params(face_bbox)
print(f"\n3. Initial PyCLNF parameters:")
print(f"  scale={initial_params[0]:.3f}")
print(f"  rotation: wx={initial_params[1]:.3f}, wy={initial_params[2]:.3f}, wz={initial_params[3]:.3f}")
print(f"  translation: tx={initial_params[4]:.1f}, ty={initial_params[5]:.1f}")

# Check initial landmarks
initial_landmarks = clnf.pdm.params_to_landmarks_2d(initial_params)
print(f"\n4. Initial PyCLNF landmarks (before optimization):")
print(f"  Range: x=[{initial_landmarks[:, 0].min():.1f}, {initial_landmarks[:, 0].max():.1f}], y=[{initial_landmarks[:, 1].min():.1f}, {initial_landmarks[:, 1].max():.1f}]")
print(f"  Width: {initial_landmarks[:, 0].max() - initial_landmarks[:, 0].min():.1f}px")
print(f"  Height: {initial_landmarks[:, 1].max() - initial_landmarks[:, 1].min():.1f}px")
print(f"  Center: ({initial_landmarks[:, 0].mean():.1f}, {initial_landmarks[:, 1].mean():.1f})")

# Run PyCLNF optimization
print("\n5. Running PyCLNF optimization...")
py_landmarks, py_info = clnf.fit(gray, face_bbox, return_params=True)
py_params = py_info['params']

print(f"  Converged: {py_info['converged']}, Iterations: {py_info['iterations']}")
print(f"  Final params:")
print(f"    scale={py_params[0]:.3f}")
print(f"    rotation: wx={py_params[1]:.3f}, wy={py_params[2]:.3f}, wz={py_params[3]:.3f}")
print(f"    translation: tx={py_params[4]:.1f}, ty={py_params[5]:.1f}")

print(f"\n6. Final PyCLNF landmarks (after optimization):")
print(f"  Range: x=[{py_landmarks[:, 0].min():.1f}, {py_landmarks[:, 0].max():.1f}], y=[{py_landmarks[:, 1].min():.1f}, {py_landmarks[:, 1].max():.1f}]")
print(f"  Width: {py_landmarks[:, 0].max() - py_landmarks[:, 0].min():.1f}px")
print(f"  Height: {py_landmarks[:, 1].max() - py_landmarks[:, 1].min():.1f}px")
print(f"  Center: ({py_landmarks[:, 0].mean():.1f}, {py_landmarks[:, 1].mean():.1f})")

# Compare with OpenFace
print(f"\n7. Comparison with OpenFace C++:")
print(f"  CPP landmark range: x=[{cpp_landmarks[:, 0].min():.1f}, {cpp_landmarks[:, 0].max():.1f}], y=[{cpp_landmarks[:, 1].min():.1f}, {cpp_landmarks[:, 1].max():.1f}]")
print(f"  CPP Width: {cpp_landmarks[:, 0].max() - cpp_landmarks[:, 0].min():.1f}px")
print(f"  CPP Height: {cpp_landmarks[:, 1].max() - cpp_landmarks[:, 1].min():.1f}px")
print(f"  CPP Center: ({cpp_landmarks[:, 0].mean():.1f}, {cpp_landmarks[:, 1].mean():.1f})")

# Compute errors
landmark_diff = py_landmarks - cpp_landmarks
l2_errors = np.linalg.norm(landmark_diff, axis=1)
mean_l2_error = l2_errors.mean()
max_l2_error = l2_errors.max()

print(f"\n8. Landmark errors (PyCLNF - OpenFace C++):")
print(f"  Mean L2 error: {mean_l2_error:.2f} pixels")
print(f"  Max L2 error: {max_l2_error:.2f} pixels")
print(f"  Mean X error: {landmark_diff[:, 0].mean():.2f} pixels")
print(f"  Mean Y error: {landmark_diff[:, 1].mean():.2f} pixels")

# Check specific landmarks
print(f"\n9. Specific landmark comparison:")
landmark_names = {
    0: "Jaw left",
    8: "Chin",
    16: "Jaw right",
    27: "Nose tip",
    36: "Left eye left corner",
    45: "Right eye right corner",
    48: "Mouth left corner",
    54: "Mouth right corner"
}

for idx, name in landmark_names.items():
    cpp_pt = cpp_landmarks[idx]
    py_pt = py_landmarks[idx]
    error = np.linalg.norm(py_pt - cpp_pt)
    print(f"  {name:25s} (#{idx:2d}): CPP=({cpp_pt[0]:6.1f}, {cpp_pt[1]:6.1f})  PY=({py_pt[0]:6.1f}, {py_pt[1]:6.1f})  Error={error:6.2f}px")

# Visualize
print(f"\n10. Creating visualization...")
vis = frame.copy()

# Draw OpenFace C++ landmarks in GREEN
for pt in cpp_landmarks:
    cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

# Draw PyCLNF landmarks in RED
for pt in py_landmarks:
    cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

# Draw face bbox in BLUE
x, y, w, h = face_bbox
cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Add legend
cv2.putText(vis, "GREEN = OpenFace C++", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(vis, "RED = PyCLNF", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(vis, f"Mean Error: {mean_l2_error:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

output_path = "diagnostic_landmarks_comparison.jpg"
cv2.imwrite(output_path, vis)
print(f"  Saved visualization to: {output_path}")

print("\n" + "=" * 80)
print("Diagnostic complete!")
print("=" * 80)
