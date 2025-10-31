#!/usr/bin/env python3
"""
Debug face alignment to identify scale/transformation issues
"""

import numpy as np
import cv2
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner


def main():
    # Initialize aligner
    aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

    # Load one frame of data
    df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
    row = df.iloc[0]

    # Extract landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    pose_tx = row['pose_Tx']
    pose_ty = row['pose_Ty']

    print("=" * 70)
    print("Alignment Debug Information")
    print("=" * 70)

    print("\n[1] Input Landmarks")
    print(f"  Shape: {landmarks_68.shape}")
    print(f"  X range: [{landmarks_68[:,0].min():.2f}, {landmarks_68[:,0].max():.2f}]")
    print(f"  Y range: [{landmarks_68[:,1].min():.2f}, {landmarks_68[:,1].max():.2f}]")
    print(f"  Center: ({landmarks_68[:,0].mean():.2f}, {landmarks_68[:,1].mean():.2f})")

    print("\n[2] Pose Translation")
    print(f"  pose_Tx: {pose_tx:.6f}")
    print(f"  pose_Ty: {pose_ty:.6f}")

    print("\n[3] Reference Shape (from PDM)")
    print(f"  Shape: {aligner.reference_shape.shape}")
    print(f"  X range: [{aligner.reference_shape[:,0].min():.4f}, {aligner.reference_shape[:,0].max():.4f}]")
    print(f"  Y range: [{aligner.reference_shape[:,1].min():.4f}, {aligner.reference_shape[:,1].max():.4f}]")
    print(f"  Center: ({aligner.reference_shape[:,0].mean():.4f}, {aligner.reference_shape[:,1].mean():.4f})")

    # Extract rigid points
    source_rigid = aligner._extract_rigid_points(landmarks_68)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)

    print("\n[4] Rigid Points (24 points)")
    print("  Source (detected):")
    print(f"    X range: [{source_rigid[:,0].min():.2f}, {source_rigid[:,0].max():.2f}]")
    print(f"    Y range: [{source_rigid[:,1].min():.2f}, {source_rigid[:,1].max():.2f}]")
    print(f"    Center: ({source_rigid[:,0].mean():.2f}, {source_rigid[:,1].mean():.2f})")
    print("  Destination (reference):")
    print(f"    X range: [{dest_rigid[:,0].min():.4f}, {dest_rigid[:,0].max():.4f}]")
    print(f"    Y range: [{dest_rigid[:,1].min():.4f}, {dest_rigid[:,1].max():.4f}]")
    print(f"    Center: ({dest_rigid[:,0].mean():.4f}, {dest_rigid[:,1].mean():.4f})")

    # Compute similarity transform
    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)

    print("\n[5] Similarity Transform (2x2 scale-rotation matrix)")
    print(f"  {scale_rot}")
    print(f"  Determinant (scale²): {np.linalg.det(scale_rot):.6f}")
    print(f"  Estimated scale: {np.sqrt(np.abs(np.linalg.det(scale_rot))):.6f}")

    # Build warp matrix
    warp_matrix = aligner._build_warp_matrix(scale_rot, pose_tx, pose_ty)

    print("\n[6] Warp Matrix (2x3 affine)")
    print(f"  {warp_matrix}")
    print(f"  Translation: ({warp_matrix[0,2]:.2f}, {warp_matrix[1,2]:.2f})")

    # Test transformation on a few landmarks
    print("\n[7] Test Transform on First 3 Landmarks")
    for i in range(3):
        pt = landmarks_68[i]
        # Apply affine: pt_new = M[:2,:2] @ pt + M[:,2]
        pt_transformed = scale_rot @ pt + warp_matrix[:, 2]
        print(f"  Landmark {i}: ({pt[0]:.2f}, {pt[1]:.2f}) → ({pt_transformed[0]:.2f}, {pt_transformed[1]:.2f})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
