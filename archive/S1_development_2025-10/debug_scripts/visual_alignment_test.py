#!/usr/bin/env python3
"""
Create visual side-by-side comparison of C++ vs Python alignment
"""

import cv2
import numpy as np
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner

# Load aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load test video
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]

# Read first frame
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"Failed to read frame from {video_path}")
    exit(1)

print(f"✓ Loaded frame from video: {frame.shape}")

# Extract landmarks
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)

pose_tx = row['p_tx']
pose_ty = row['p_ty']

# Align with Python
python_aligned = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty)
print(f"✓ Python aligned face: {python_aligned.shape}")

# Draw landmarks on Python aligned face
python_with_landmarks = python_aligned.copy()
# Transform landmarks to aligned space
source_rigid = aligner._extract_rigid_points(landmarks_68)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
warp_matrix = aligner._build_warp_matrix(scale_rot, pose_tx, pose_ty)

# Transform all 68 landmarks
for i in range(68):
    pt = landmarks_68[i]
    # Apply affine transform: [x', y'] = M @ [x, y, 1]
    pt_homog = np.array([pt[0], pt[1], 1.0])
    pt_transformed = warp_matrix @ pt_homog
    x_t, y_t = int(pt_transformed[0]), int(pt_transformed[1])
    cv2.circle(python_with_landmarks, (x_t, y_t), 1, (0, 255, 0), -1)

# Draw reference landmarks on Python aligned
python_with_ref = python_aligned.copy()
ref_center_x = 112 / 2
ref_center_y = 112 / 2
for i in range(68):
    pt = aligner.reference_shape[i]
    x_r = int(pt[0] + ref_center_x)
    y_r = int(pt[1] + ref_center_y)
    cv2.circle(python_with_ref, (x_r, y_r), 1, (0, 0, 255), -1)

# Create comparison grid
comparison = np.zeros((112 * 2, 112 * 2, 3), dtype=np.uint8)
comparison[0:112, 0:112] = python_aligned
comparison[0:112, 112:224] = python_with_landmarks
comparison[112:224, 0:112] = python_with_ref
# Create overlay of landmarks + reference
python_overlay = python_aligned.copy()
for i in range(68):
    pt = landmarks_68[i]
    pt_homog = np.array([pt[0], pt[1], 1.0])
    pt_transformed = warp_matrix @ pt_homog
    x_t, y_t = int(pt_transformed[0]), int(pt_transformed[1])
    cv2.circle(python_overlay, (x_t, y_t), 1, (0, 255, 0), -1)
for i in range(68):
    pt = aligner.reference_shape[i]
    x_r = int(pt[0] + 112/2)
    y_r = int(pt[1] + 112/2)
    cv2.circle(python_overlay, (x_r, y_r), 1, (0, 0, 255), -1)
comparison[112:224, 112:224] = python_overlay

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "Python Aligned", (5, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(comparison, "Python+Landmarks", (117, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(comparison, "Python+Reference", (5, 127), font, 0.4, (255, 255, 255), 1)
cv2.putText(comparison, "Overlay (G=det,R=ref)", (117, 127), font, 0.3, (255, 255, 255), 1)

# Save
cv2.imwrite("alignment_visual_comparison.png", comparison)
print("✓ Saved alignment_visual_comparison.png")

# Also create large version for better viewing
large = cv2.resize(comparison, (112 * 4, 112 * 4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("alignment_visual_comparison_large.png", large)
print("✓ Saved alignment_visual_comparison_large.png")

# Try to detect rotation by checking specific landmark positions
# Check nose tip (landmark 30) position
nose_idx = 30
cpp_nose_expected = (56, 56)  # Approximate center
nose_pt = landmarks_68[nose_idx]
nose_homog = np.array([nose_pt[0], nose_pt[1], 1.0])
nose_transformed = warp_matrix @ nose_homog
print(f"\nNose tip (landmark 30) position:")
print(f"  In aligned face: ({nose_transformed[0]:.1f}, {nose_transformed[1]:.1f})")
print(f"  Expected: ~(56, 56)")
