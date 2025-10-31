#!/usr/bin/env python3
"""
Test using PDM rotation parameters directly instead of computing from landmarks
"""

import numpy as np
import pandas as pd
import cv2

# Load test data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)

test_frames = [1, 617, 740]

print("=" * 70)
print("Testing PDM Rotation Parameters")
print("=" * 70)

# Load C++ aligned for comparison
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

def euler_to_rotation_2d(rz):
    """Convert Z-axis rotation (yaw) to 2D rotation matrix"""
    cos_rz = np.cos(rz)
    sin_rz = np.sin(rz)
    return np.array([[cos_rz, -sin_rz],
                     [sin_rz,  cos_rz]], dtype=np.float32)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()

    # Get PDM parameters
    p_scale = row['p_scale']
    p_rx = row['p_rx']  # Roll (radians)
    p_ry = row['p_ry']  # Pitch (radians)
    p_rz = row['p_rz']  # Yaw (radians)
    p_tx = row['p_tx']
    p_ty = row['p_ty']

    # Convert to degrees for display
    rx_deg = np.rad2deg(p_rx)
    ry_deg = np.rad2deg(p_ry)
    rz_deg = np.rad2deg(p_rz)

    print(f"\nFrame {frame_num}:")
    print(f"  PDM params: scale={p_scale:.4f}, rx={rx_deg:.2f}°, ry={ry_deg:.2f}°, rz={rz_deg:.2f}°")

    # Build warp matrix using PDM rotation
    # For 2D alignment, we primarily care about rz (yaw/in-plane rotation)
    R_2d = euler_to_rotation_2d(-p_rz)  # Negative to undo the rotation

    # Scale the rotation
    scale_factor = 1.0 / p_scale  # Inverse because we're warping the image, not the points
    scale_rot = scale_factor * R_2d

    # Build warp matrix
    warp = np.zeros((2, 3), dtype=np.float32)
    warp[:2, :2] = scale_rot

    # Transform translation
    T = scale_rot @ np.array([p_tx, p_ty])
    warp[0, 2] = -T[0] + 112/2 + 2
    warp[1, 2] = -T[1] + 112/2 - 2

    aligned = cv2.warpAffine(frame, warp, (112, 112), flags=cv2.INTER_LINEAR)

    # Check rotation angle
    actual_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi
    print(f"  Resulting rotation: {actual_angle:.2f}°")

    if frame_num == 1:
        corr = np.corrcoef(cpp_aligned.flatten(), aligned.flatten())[0, 1]
        print(f"  Correlation with C++: {corr:.6f}")

cap.release()

print("\n" + "=" * 70)
print("If rotation angles are consistent (close to 0°), this approach is correct")
print("=" * 70)
