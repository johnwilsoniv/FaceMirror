#!/usr/bin/env python3
"""
Visualize the transformation to understand what's happening
"""

import numpy as np
import cv2
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner

# Load aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load first frame
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]
frame_num = int(row['frame'])

# Extract landmarks
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)

pose_tx = row['p_tx']
pose_ty = row['p_ty']

# Load video frame
video_file = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
ret, frame = cap.read()
cap.release()

# Draw landmarks on original image
frame_with_landmarks = frame.copy()
for i, (x, y) in enumerate(landmarks_68):
    cv2.circle(frame_with_landmarks, (int(x), int(y)), 2, (0, 255, 0), -1)
    if i % 5 == 0:  # Label every 5th landmark
        cv2.putText(frame_with_landmarks, str(i), (int(x)+5, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

# Extract rigid points for visualization
source_rigid = aligner._extract_rigid_points(landmarks_68)
for i, (x, y) in enumerate(source_rigid):
    cv2.circle(frame_with_landmarks, (int(x), int(y)), 3, (0, 0, 255), -1)

# Align face
python_aligned = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty)

# Draw reference shape on a blank image (scaled to output size)
ref_vis = np.zeros((112, 112, 3), dtype=np.uint8) + 128
ref_shape_scaled = aligner.reference_shape.copy()
# Reference shape is centered around 0, need to shift to image center
ref_shape_vis = ref_shape_scaled + np.array([56, 56])  # Center at 56,56
for i, (x, y) in enumerate(ref_shape_vis):
    if 0 <= x < 112 and 0 <= y < 112:
        cv2.circle(ref_vis, (int(x), int(y)), 1, (0, 255, 0), -1)

# Draw rigid points
ref_rigid = aligner._extract_rigid_points(ref_shape_vis)
for x, y in ref_rigid:
    if 0 <= x < 112 and 0 <= y < 112:
        cv2.circle(ref_vis, (int(x), int(y)), 2, (0, 0, 255), -1)

# Load C++ aligned face
cpp_aligned_file = f"pyfhog_validation_output/IMG_0942_left_mirrored_aligned/frame_det_00_{frame_num:06d}.bmp"
cpp_aligned = cv2.imread(cpp_aligned_file)

# Create comprehensive comparison
# Row 1: Original with landmarks | Reference shape visualization
# Row 2: Python aligned | C++ aligned | Difference
frame_resized = cv2.resize(frame_with_landmarks, (224, 224))
ref_vis_large = cv2.resize(ref_vis, (224, 224))
row1 = np.hstack([frame_resized, ref_vis_large])

python_large = cv2.resize(python_aligned, (224, 224))
cpp_large = cv2.resize(cpp_aligned, (224, 224))
diff = np.abs(python_aligned.astype(float) - cpp_aligned.astype(float))
diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
diff_large = cv2.resize(diff_vis, (224, 224))

row2 = np.hstack([python_large, cpp_large, diff_large])

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(row1, "Original + Landmarks", (10, 20), font, 0.6, (0, 255, 0), 2)
cv2.putText(row1, "Reference Shape", (234, 20), font, 0.6, (0, 255, 0), 2)
cv2.putText(row2, "Python Aligned", (10, 20), font, 0.6, (0, 255, 0), 2)
cv2.putText(row2, "C++ Aligned", (234, 20), font, 0.6, (0, 255, 0), 2)
cv2.putText(row2, "Diff x5", (458, 20), font, 0.6, (0, 255, 0), 2)

# Combine rows
visualization = np.vstack([row1, row2])

cv2.imwrite("alignment_debug_visualization.png", visualization)
print("Saved visualization to: alignment_debug_visualization.png")
print(f"\nTransform details:")
print(f"  pose_tx: {pose_tx:.4f}, pose_ty: {pose_ty:.4f}")
print(f"  Landmarks center: ({landmarks_68[:,0].mean():.2f}, {landmarks_68[:,1].mean():.2f})")
print(f"  Reference shape center: ({aligner.reference_shape[:,0].mean():.4f}, {aligner.reference_shape[:,1].mean():.4f})")
