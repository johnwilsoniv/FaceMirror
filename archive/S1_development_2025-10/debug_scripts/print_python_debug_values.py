#!/usr/bin/env python3
"""
Print Python debug values in same format as C++ debug output
for easy comparison
"""

import numpy as np
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner

# Load aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

test_frames = [1, 493, 617, 863]

print("=" * 80)
print("Python Debug Output (matches C++ format)")
print("=" * 80)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get CSV landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    csv_x = row[x_cols].values.astype(np.float32)
    csv_y = row[y_cols].values.astype(np.float32)
    csv_landmarks = np.stack([csv_x, csv_y], axis=1)  # (68, 2)

    # Extract rigid points
    source_rigid = aligner._extract_rigid_points(csv_landmarks)  # (24, 2)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)  # (24, 2)

    # Transpose to match C++ format: (2, N) instead of (N, 2)
    source_cpp_format = source_rigid.T  # (2, 24)
    dest_cpp_format = dest_rigid.T  # (2, 24)

    # Compute scale-rot matrix
    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
    angle_rad = np.arctan2(scale_rot[1,0], scale_rot[0,0])
    angle_deg = angle_rad * 180 / np.pi

    # Get params_global
    params_global = [
        row['p_scale'],
        row['p_rx'],
        row['p_ry'],
        row['p_rz'],
        row['p_tx'],
        row['p_ty']
    ]

    print(f"\n=== DEBUG Frame {frame_num} ===")

    # Print first 3 source landmarks
    print("Source landmarks (first 3):")
    for i in range(3):
        print(f"  [{i}]: ({source_cpp_format[0, i]:.6f}, {source_cpp_format[1, i]:.6f})")

    # Print first 3 destination landmarks
    print("Dest landmarks (first 3):")
    for i in range(3):
        print(f"  [{i}]: ({dest_cpp_format[0, i]:.6f}, {dest_cpp_format[1, i]:.6f})")

    # Print scale-rot matrix
    print("Scale-rot matrix:")
    print(f"  [{scale_rot[0,0]:.6f}, {scale_rot[0,1]:.6f}]")
    print(f"  [{scale_rot[1,0]:.6f}, {scale_rot[1,1]:.6f}]")

    # Print rotation angle
    print(f"Rotation angle: {angle_deg:.6f}°")

    # Print params_global
    print("params_global:")
    print(f"  scale={params_global[0]:.6f} "
          f"rx={params_global[1]:.6f} "
          f"ry={params_global[2]:.6f} "
          f"rz={params_global[3]:.6f} "
          f"tx={params_global[4]:.6f} "
          f"ty={params_global[5]:.6f}")

    print("==========================\n")

print("=" * 80)
print("Instructions:")
print("=" * 80)
print("1. Run: ./apply_cpp_debug_patch.sh")
print("2. Run instrumented C++:")
print("   cd /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin")
print("   ./FeatureExtraction -f /path/to/IMG_0942_left_mirrored.mp4 2>&1 | grep -A 30 'DEBUG Frame'")
print("3. Compare C++ output above to this Python output")
print("4. Look for differences in:")
print("   - Source landmarks (should be IDENTICAL - from same CSV)")
print("   - Dest landmarks (should be IDENTICAL - from same PDM)")
print("   - Scale-rot matrix (THIS is where we expect the difference)")
print("   - Rotation angle (C++ should be ~0°, Python is -8.79° to +2.17°)")
print("=" * 80)
