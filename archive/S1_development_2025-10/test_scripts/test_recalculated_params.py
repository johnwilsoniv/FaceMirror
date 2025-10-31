#!/usr/bin/env python3
"""
Test alignment using recalculated tx, ty (like C++ CalcParams does)
"""

import numpy as np
import pandas as pd
import cv2
from openface22_face_aligner import OpenFace22FaceAligner

# Load aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load test data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

# Test frames including eyes-closed frame 617
test_frames = [1, 493, 617, 740, 863]

print("=" * 70)
print("Testing with Recalculated Translation Parameters")
print("=" * 70)

# Load C++ aligned for comparison
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

cap = cv2.VideoCapture(video_path)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()

    # Extract landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Method 1: Use CSV tx, ty (current method)
    pose_tx_csv = row['p_tx']
    pose_ty_csv = row['p_ty']

    aligned_csv = aligner.align_face(frame, landmarks_68, pose_tx_csv, pose_ty_csv)

    # Method 2: Recalculate tx, ty from landmark centroid (like C++ CalcParams)
    # CalcParams computes translation as centroid of landmarks:
    # translation = (min_x + max_x) / 2, (min_y + max_y) / 2
    min_x = landmarks_68[:, 0].min()
    max_x = landmarks_68[:, 0].max()
    min_y = landmarks_68[:, 1].min()
    max_y = landmarks_68[:, 1].max()

    pose_tx_recalc = (min_x + max_x) / 2.0
    pose_ty_recalc = (min_y + max_y) / 2.0

    aligned_recalc = aligner.align_face(frame, landmarks_68, pose_tx_recalc, pose_ty_recalc)

    # Measure rotation angles
    source_rigid = aligner._extract_rigid_points(landmarks_68)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
    angle = np.arctan2(scale_rot[1, 0], scale_rot[0, 0]) * 180 / np.pi

    print(f"\nFrame {frame_num}:")
    print(f"  Rotation angle: {angle:7.2f}°")
    print(f"  CSV tx, ty:    ({pose_tx_csv:.2f}, {pose_ty_csv:.2f})")
    print(f"  Recalc tx, ty: ({pose_tx_recalc:.2f}, {pose_ty_recalc:.2f})")
    print(f"  Difference:    ({pose_tx_recalc - pose_tx_csv:.2f}, {pose_ty_recalc - pose_ty_csv:.2f})")

    if frame_num == 1:
        corr_csv = np.corrcoef(cpp_aligned.flatten(), aligned_csv.flatten())[0, 1]
        corr_recalc = np.corrcoef(cpp_aligned.flatten(), aligned_recalc.flatten())[0, 1]
        print(f"  Corr using CSV tx,ty:    {corr_csv:.6f}")
        print(f"  Corr using recalc tx,ty: {corr_recalc:.6f}")

        # Save comparison
        vis = np.hstack([cpp_aligned, aligned_csv, aligned_recalc])
        cv2.putText(vis, "C++", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "CSV tx,ty", (117, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(vis, "Recalc tx,ty", (229, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imwrite("test_recalc_params_frame1.png", vis)

cap.release()

print("\n" + "=" * 70)
print("If recalculated tx, ty improves results, this confirms C++ CalcParams finding!")
print("=" * 70)
