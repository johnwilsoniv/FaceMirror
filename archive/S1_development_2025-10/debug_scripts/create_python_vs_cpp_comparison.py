#!/usr/bin/env python3
"""
Create clear Python vs C++ comparison to verify if rotations match
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner

# Initialize aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cpp_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

cap = cv2.VideoCapture(video_path)

# Test key frames
test_frames = [1, 493, 617, 863]

print("=" * 70)
print("Creating Python vs C++ Comparison")
print("=" * 70)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    if not ret:
        continue

    # Extract landmarks and pose
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)
    pose_tx = row['p_tx']
    pose_ty = row['p_ty']

    # Align with Python (current implementation)
    python_aligned = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty, apply_mask=False)

    # Load C++ aligned
    cpp_file = cpp_dir / f"frame_det_00_{frame_num:06d}.bmp"
    cpp_aligned = cv2.imread(str(cpp_file))

    # Compute rotation angle for Python
    source_rigid = aligner._extract_rigid_points(landmarks_68)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
    python_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi

    # Create comparison
    comparison = np.hstack([cpp_aligned, python_aligned])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "C++", (10, 20), font, 0.5, (0, 255, 0), 1)
    cv2.putText(comparison, f"Python (angle={python_angle:.1f}deg)", (122, 20), font, 0.4, (0, 255, 0), 1)

    # Save
    cv2.imwrite(f"clear_comparison_frame_{frame_num}.png", comparison)
    print(f"Frame {frame_num}: Python angle = {python_angle:.2f}°")

cap.release()

# Create large grid comparison
print("\nCreating grid comparison...")
comparisons = []
for frame_num in test_frames:
    img = cv2.imread(f"clear_comparison_frame_{frame_num}.png")
    comparisons.append(img)

grid = np.vstack(comparisons)
cv2.imwrite("PYTHON_VS_CPP_GRID.png", grid)

print("\n✓ Created PYTHON_VS_CPP_GRID.png")
print("\nPlease visually inspect:")
print("  - Are Python faces rotated compared to C++?")
print("  - If so, by approximately how many degrees?")
print("  - Is the rotation consistent across all 4 frames?")
print("=" * 70)
