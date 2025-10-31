#!/usr/bin/env python3
"""
Check if CSV landmarks show eye movement between frames 493 (open) and 617 (closed)

This will tell us if the landmarks themselves are changing, or if C++ is filtering them somehow.
"""

import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Eye landmark indices (from OpenFace 68-point model)
EYE_LANDMARKS = [36, 39, 40, 41, 42, 45, 46, 47]

frames = {
    493: "eyes open",
    617: "eyes closed",
    863: "eyes open"
}

print("=" * 70)
print("CSV Landmark Analysis: Eye Movement Detection")
print("=" * 70)

# Extract eye landmarks for each frame
eye_positions = {}
for frame_num, description in frames.items():
    row = df[df['frame'] == frame_num].iloc[0]

    # Extract eye landmark positions
    x_cols = [f'x_{i}' for i in EYE_LANDMARKS]
    y_cols = [f'y_{i}' for i in EYE_LANDMARKS]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)

    eye_positions[frame_num] = np.stack([x, y], axis=1)

    # Compute eye height (vertical span)
    eye_height = y.max() - y.min()

    print(f"\nFrame {frame_num} ({description}):")
    print(f"  Eye landmarks X range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"  Eye landmarks Y range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  Eye vertical span: {eye_height:.1f} pixels")

# Compute differences
print("\n" + "=" * 70)
print("Landmark Movement Analysis:")
print("=" * 70)

diff_493_617 = np.mean(np.abs(eye_positions[493] - eye_positions[617]))
diff_493_863 = np.mean(np.abs(eye_positions[493] - eye_positions[863]))
diff_617_863 = np.mean(np.abs(eye_positions[617] - eye_positions[863]))

print(f"\nMean absolute movement of eye landmarks:")
print(f"  493 (open) vs 617 (closed): {diff_493_617:.2f} pixels")
print(f"  493 (open) vs 863 (open):   {diff_493_863:.2f} pixels")
print(f"  617 (closed) vs 863 (open): {diff_617_863:.2f} pixels")

# Vertical movement specifically (more sensitive to eye closure)
v_diff_493_617 = np.mean(np.abs(eye_positions[493][:, 1] - eye_positions[617][:, 1]))
v_diff_493_863 = np.mean(np.abs(eye_positions[493][:, 1] - eye_positions[863][:, 1]))

print(f"\nVertical movement of eye landmarks:")
print(f"  493 (open) vs 617 (closed): {v_diff_493_617:.2f} pixels")
print(f"  493 (open) vs 863 (open):   {v_diff_493_863:.2f} pixels")

print("\n" + "=" * 70)
print("Conclusion:")
print("=" * 70)

if v_diff_493_617 > 5:
    print(f"  ✓ Eye landmarks DO move significantly when eyes close ({v_diff_493_617:.1f}px)")
    print("  → C++ must be filtering/weighting to ignore this movement")
    print("  → Our Python implementation is correctly detecting the movement")
    print("  → But we need to handle it like C++ does")
elif v_diff_493_617 < 2:
    print(f"  ⚠ Eye landmarks DON'T move much when eyes close ({v_diff_493_617:.1f}px)")
    print("  → The CLNF landmark detector might be filtering eye closure")
    print("  → This would explain why C++ is stable")
else:
    print(f"  ~ Eye landmarks move moderately ({v_diff_493_617:.1f}px)")
    print("  → Unclear if this explains the rotation difference")

print("=" * 70)
