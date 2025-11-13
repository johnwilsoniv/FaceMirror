"""
Debug vertical offset issue in PyCLNF landmarks.

Compares initialization vs final landmarks to understand the ~50px vertical offset.
"""

import numpy as np
import cv2
from pyclnf import CLNF

# Load test image
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

# Face bbox from OpenFace C++ (exact same bbox OpenFace used)
bbox = (293.145, 702.034, 418.033, 404.659)

# Initialize CLNF
clnf = CLNF(model_dir="pyclnf/models", scale=0.25)

# Get initial parameters
initial_params = clnf.pdm.init_params(bbox)
print("=" * 60)
print("Initial Parameters")
print("=" * 60)
print(f"Scale: {initial_params[0]:.3f}")
print(f"Rotation: wx={initial_params[1]:.3f}, wy={initial_params[2]:.3f}, wz={initial_params[3]:.3f}")
print(f"Translation: tx={initial_params[4]:.1f}, ty={initial_params[5]:.1f}")

# Get initial landmarks (before optimization)
initial_landmarks = clnf.pdm.params_to_landmarks_2d(initial_params)
print(f"\nInitial landmark range:")
print(f"  x=[{initial_landmarks[:, 0].min():.1f}, {initial_landmarks[:, 0].max():.1f}]")
print(f"  y=[{initial_landmarks[:, 1].min():.1f}, {initial_landmarks[:, 1].max():.1f}]")

init_center_y = initial_landmarks[:, 1].mean()
bbox_center_y = bbox[1] + bbox[3] / 2
print(f"\nInitial landmark center Y: {init_center_y:.1f}")
print(f"Bbox center Y: {bbox_center_y:.1f}")
print(f"Initial offset from bbox center: {init_center_y - bbox_center_y:.1f}px")

# Run CLNF fitting
landmarks, info = clnf.fit(frame, bbox, return_params=True)
final_params = info['params']

print("\n" + "=" * 60)
print("Final Parameters (after optimization)")
print("=" * 60)
print(f"Scale: {final_params[0]:.3f}")
print(f"Rotation: wx={final_params[1]:.3f}, wy={final_params[2]:.3f}, wz={final_params[3]:.3f}")
print(f"Translation: tx={final_params[4]:.1f}, ty={final_params[5]:.1f}")
print(f"Converged: {info['converged']}, Iterations: {info['iterations']}")

print(f"\nFinal landmark range:")
print(f"  x=[{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}]")
print(f"  y=[{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")

final_center_y = landmarks[:, 1].mean()
print(f"\nFinal landmark center Y: {final_center_y:.1f}")
print(f"Bbox center Y: {bbox_center_y:.1f}")
print(f"Final offset from bbox center: {final_center_y - bbox_center_y:.1f}px")

# Check mean shape center
mean_shape_flat = clnf.pdm.mean_shape.flatten()
n = clnf.pdm.n_points
mean_y = mean_shape_flat[n:2*n]  # Y coordinates
print("\n" + "=" * 60)
print("Mean Shape Analysis")
print("=" * 60)
print(f"Mean shape Y range: [{mean_y.min():.1f}, {mean_y.max():.1f}]")
print(f"Mean shape Y center: {mean_y.mean():.1f}")
print(f"Mean shape centered at origin: {abs(mean_y.mean()) < 1.0}")

# Compare to OpenFace C++ (known good values)
print("\n" + "=" * 60)
print("Comparison to OpenFace C++")
print("=" * 60)
print("OpenFace C++ landmark range (from previous test):")
print("  x=[300.7, 703.6], y=[691.4, 1087.6]")
print(f"  Center Y: {(691.4 + 1087.6) / 2:.1f}")
print(f"\nPyCLNF vs OpenFace Y offset: {final_center_y - (691.4 + 1087.6) / 2:.1f}px")
