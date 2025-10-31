#!/usr/bin/env python3
"""
Test alignment with rotation correction applied

Compare THREE approaches:
1. With eyes (24 points) - original
2. Without eyes (16 points) + correction
3. Without eyes (16 points) raw (for reference)
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner
from scipy.stats import pearsonr

# Load data
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cpp_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

cap = cv2.VideoCapture(video_path)

# Test on key frames
test_frames = [1, 493, 617, 863]

NON_EYE_RIGID = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35]
ORIGINAL_RIGID = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

# Correction angle from previous analysis
CORRECTION_ANGLE = -30.98  # degrees

def apply_rotation_correction(scale_rot, angle_deg):
    """Apply additional rotation to scale-rotation matrix"""
    angle_rad = angle_deg * np.pi / 180
    R_corr = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ], dtype=np.float32)
    return R_corr @ scale_rot

print("=" * 70)
print("Rotation Correction Test: WITH eyes vs WITHOUT eyes + correction")
print("=" * 70)

results = []

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

    # Load C++ reference
    cpp_file = cpp_dir / f"frame_det_00_{frame_num:06d}.bmp"
    cpp_aligned = cv2.imread(str(cpp_file))

    # Method 1: With eyes (original - 24 points)
    aligner.RIGID_INDICES = ORIGINAL_RIGID
    source_rigid = aligner._extract_rigid_points(landmarks_68)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
    scale_rot_with_eyes = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
    warp_matrix = aligner._build_warp_matrix(scale_rot_with_eyes, pose_tx, pose_ty)
    aligned_with_eyes = cv2.warpAffine(frame, warp_matrix, (112, 112), flags=cv2.INTER_LINEAR)

    angle_with_eyes = np.arctan2(scale_rot_with_eyes[1,0], scale_rot_with_eyes[0,0]) * 180 / np.pi
    corr_with_eyes, _ = pearsonr(aligned_with_eyes.flatten(), cpp_aligned.flatten())

    # Method 2: Without eyes (16 points)
    aligner.RIGID_INDICES = NON_EYE_RIGID
    source_rigid = aligner._extract_rigid_points(landmarks_68)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
    scale_rot_no_eyes = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
    warp_matrix = aligner._build_warp_matrix(scale_rot_no_eyes, pose_tx, pose_ty)
    aligned_no_eyes = cv2.warpAffine(frame, warp_matrix, (112, 112), flags=cv2.INTER_LINEAR)

    angle_no_eyes = np.arctan2(scale_rot_no_eyes[1,0], scale_rot_no_eyes[0,0]) * 180 / np.pi
    corr_no_eyes, _ = pearsonr(aligned_no_eyes.flatten(), cpp_aligned.flatten())

    # Method 3: Without eyes + correction
    scale_rot_corrected = apply_rotation_correction(scale_rot_no_eyes, CORRECTION_ANGLE)
    warp_matrix = aligner._build_warp_matrix(scale_rot_corrected, pose_tx, pose_ty)
    aligned_corrected = cv2.warpAffine(frame, warp_matrix, (112, 112), flags=cv2.INTER_LINEAR)

    angle_corrected = np.arctan2(scale_rot_corrected[1,0], scale_rot_corrected[0,0]) * 180 / np.pi
    corr_corrected, _ = pearsonr(aligned_corrected.flatten(), cpp_aligned.flatten())

    print(f"\nFrame {frame_num}:")
    print(f"  With eyes:      angle={angle_with_eyes:6.2f}°, r={corr_with_eyes:.4f}")
    print(f"  No eyes (raw):  angle={angle_no_eyes:6.2f}°, r={corr_no_eyes:.4f}")
    print(f"  No eyes + corr: angle={angle_corrected:6.2f}°, r={corr_corrected:.4f}")

    results.append({
        'frame': frame_num,
        'with_eyes_r': corr_with_eyes,
        'no_eyes_r': corr_no_eyes,
        'corrected_r': corr_corrected,
        'with_eyes_angle': angle_with_eyes,
        'no_eyes_angle': angle_no_eyes,
        'corrected_angle': angle_corrected
    })

    # Save comparison
    comparison = np.hstack([cpp_aligned, aligned_with_eyes, aligned_corrected])
    cv2.putText(comparison, "C++", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(comparison, "Python+eyes", (117, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(comparison, "Python-eyes+corr", (229, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(f"rotation_correction_frame_{frame_num}.png", comparison)

cap.release()

# Restore original
aligner.RIGID_INDICES = ORIGINAL_RIGID

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)

results_df = pd.DataFrame(results)
print(f"\nCorrelation with C++:")
print(f"  With eyes:            mean r = {results_df['with_eyes_r'].mean():.4f}")
print(f"  No eyes (raw):        mean r = {results_df['no_eyes_r'].mean():.4f}")
print(f"  No eyes + correction: mean r = {results_df['corrected_r'].mean():.4f}")

print(f"\nRotation angle stability:")
print(f"  With eyes:            std = {results_df['with_eyes_angle'].std():.2f}°")
print(f"  No eyes (raw):        std = {results_df['no_eyes_angle'].std():.2f}°")
print(f"  No eyes + correction: std = {results_df['corrected_angle'].std():.2f}°")

print("\n" + "=" * 70)
print("Verdict:")
print("=" * 70)

if results_df['corrected_r'].mean() > results_df['with_eyes_r'].mean():
    improvement = ((results_df['corrected_r'].mean() - results_df['with_eyes_r'].mean())
                   / results_df['with_eyes_r'].mean() * 100)
    print(f"  ✓✓✓ NO EYES + CORRECTION WINS!")
    print(f"  ✓ {improvement:.1f}% better correlation than with eyes")
    print(f"  ✓ {results_df['corrected_angle'].std():.1f}° rotation stability")
    print(f"\n  SOLUTION: Use 16 rigid points (no eyes) + {CORRECTION_ANGLE:.1f}° correction")
    print(f"  Pure Python, no C++ dependency needed!")
else:
    print(f"  ~ With eyes still slightly better")
    print(f"  ~ But corrected version has better stability")
    print(f"  ~ Decision depends on priority: correlation vs stability")

print("=" * 70)
