"""
Debug reference transform computation.
"""

import numpy as np
import cv2
from pyclnf import CLNF
from pyclnf.core.utils import align_shapes_with_scale

# Load test image
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# OpenFace C++ bbox
bbox = (286.5, 686.2, 425.8, 412.6)

# Initialize CLNF
clnf = CLNF(model_dir="pyclnf/models")

# Get initial params
params = clnf.pdm.init_params(bbox)

print("=" * 60)
print("Transform Debug")
print("=" * 60)

# Get landmarks in image coordinates
landmarks_image = clnf.pdm.params_to_landmarks_2d(params)
print(f"\nImage landmarks (first 3 points):")
for i in range(min(3, len(landmarks_image))):
    print(f"  Point {i}: ({landmarks_image[i, 0]:.1f}, {landmarks_image[i, 1]:.1f})")

# Get reference shape at patch_scaling=0.25
patch_scaling = 0.25
reference_shape = clnf.pdm.get_reference_shape(patch_scaling, params[6:])
print(f"\nReference landmarks at scale={patch_scaling} (first 3 points):")
for i in range(min(3, len(reference_shape))):
    print(f"  Point {i}: ({reference_shape[i, 0]:.1f}, {reference_shape[i, 1]:.1f})")

# Compute similarity transform
sim_img_to_ref = align_shapes_with_scale(landmarks_image, reference_shape)
print(f"\nSimilarity transform (image -> reference):")
print(f"  [[{sim_img_to_ref[0, 0]:.6f}, {sim_img_to_ref[0, 1]:.6f}, {sim_img_to_ref[0, 2]:.6f}]")
print(f"   [{sim_img_to_ref[1, 0]:.6f}, {sim_img_to_ref[1, 1]:.6f}, {sim_img_to_ref[1, 2]:.6f}]]")

# Extract scale factor from transform
# For similarity transform [[a -b tx][b a ty]], scale = sqrt(a^2 + b^2)
a = sim_img_to_ref[0, 0]
b = sim_img_to_ref[1, 0]
transform_scale = np.sqrt(a**2 + b**2)
print(f"\nTransform scale factor: {transform_scale:.6f}")
print(f"Expected: {patch_scaling / params[0]:.6f}")
print(f"Ratio: {transform_scale / (patch_scaling / params[0]):.6f}")
