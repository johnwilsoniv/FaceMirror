#!/usr/bin/env python3
"""
Test face masking functionality
"""

import cv2
import numpy as np
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner
from triangulation_parser import TriangulationParser

# Initialize aligner and triangulation
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")
triangulation = TriangulationParser("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/tris_68.txt")

# Load validation data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]

# Extract landmarks and pose
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)
pose_tx = row['p_tx']
pose_ty = row['p_ty']

# Load frame
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame")
    exit(1)

print("=" * 70)
print("Testing Face Masking")
print("=" * 70)

# Align without mask
aligned_no_mask = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty, apply_mask=False)

# Align with mask
aligned_with_mask = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty, apply_mask=True, triangulation=triangulation)

# Load C++ aligned face
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

# Compute correlations
corr_no_mask = np.corrcoef(cpp_aligned.flatten(), aligned_no_mask.flatten())[0, 1]
corr_with_mask = np.corrcoef(cpp_aligned.flatten(), aligned_with_mask.flatten())[0, 1]

print(f"\n[1] Correlation Results:")
print(f"  Without mask: r={corr_no_mask:.6f}")
print(f"  With mask:    r={corr_with_mask:.6f}")

if corr_with_mask > corr_no_mask:
    print(f"  ✓ Masking improves correlation by {corr_with_mask - corr_no_mask:.6f}")
else:
    print(f"  ⚠ Masking reduces correlation by {corr_no_mask - corr_with_mask:.6f}")

# Create comparison visualization
vis = np.zeros((112 * 2, 112 * 2, 3), dtype=np.uint8)
vis[:112, :112] = cpp_aligned
vis[:112, 112:224] = aligned_no_mask
vis[112:, :112] = aligned_with_mask

# Compute difference
diff_no_mask = cv2.absdiff(cpp_aligned, aligned_no_mask)
diff_with_mask = cv2.absdiff(cpp_aligned, aligned_with_mask)
diff_no_mask_gray = cv2.cvtColor(diff_no_mask, cv2.COLOR_BGR2GRAY)
diff_with_mask_gray = cv2.cvtColor(diff_with_mask, cv2.COLOR_BGR2GRAY)
diff_colored = cv2.applyColorMap(diff_with_mask_gray, cv2.COLORMAP_JET)
vis[112:, 112:224] = diff_colored

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(vis, "C++ OpenFace", (5, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "Python No Mask", (117, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "Python With Mask", (5, 127), font, 0.3, (255, 255, 255), 1)
cv2.putText(vis, "Difference (JET)", (117, 127), font, 0.3, (255, 255, 255), 1)

cv2.imwrite("mask_comparison.png", vis)
print(f"\n✓ Saved mask_comparison.png")

# Large version
large = cv2.resize(vis, (112 * 4, 112 * 4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("mask_comparison_large.png", large)
print("✓ Saved mask_comparison_large.png")

print(f"\n[2] Difference Statistics:")
print(f"  Without mask - Mean diff: {diff_no_mask_gray.mean():.2f}, Max: {diff_no_mask_gray.max()}")
print(f"  With mask    - Mean diff: {diff_with_mask_gray.mean():.2f}, Max: {diff_with_mask_gray.max()}")

print("\n" + "=" * 70)
