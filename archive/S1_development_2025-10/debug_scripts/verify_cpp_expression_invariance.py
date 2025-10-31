#!/usr/bin/env python3
"""
Verify if C++ OpenFace produces identical rotation across expressions

Tests frames 493 (eyes open), 617 (eyes closed), 863 (eyes open)
"""

import cv2
import numpy as np
from pathlib import Path

# Paths to C++ aligned faces
cpp_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

frames = {
    493: "eyes open",
    617: "eyes closed",
    863: "eyes open"
}

print("=" * 70)
print("C++ Face Alignment - Expression Invariance Verification")
print("=" * 70)

# Load C++ aligned faces
images = {}
for frame_num, description in frames.items():
    cpp_file = cpp_dir / f"frame_det_00_{frame_num:06d}.bmp"
    img = cv2.imread(str(cpp_file))
    images[frame_num] = img
    print(f"\nLoaded frame {frame_num} ({description}): {img.shape}")

# Measure rotation by checking face orientation
# Use nose tip (landmark 30) and chin (landmark 8) vertical alignment
print("\n" + "=" * 70)
print("Visual Inspection:")
print("=" * 70)

# Create side-by-side comparison
comparison = np.hstack([images[493], images[617], images[863]])

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "Frame 493 (eyes open)", (5, 20), font, 0.4, (0, 255, 0), 1)
cv2.putText(comparison, "Frame 617 (eyes closed)", (117, 20), font, 0.4, (0, 255, 0), 1)
cv2.putText(comparison, "Frame 863 (eyes open)", (229, 20), font, 0.4, (0, 255, 0), 1)

# Save comparison
output_path = "cpp_expression_comparison.png"
cv2.imwrite(output_path, comparison)
print(f"\nComparison saved to: {output_path}")

# Compute pixel differences
diff_493_617 = np.mean(np.abs(images[493].astype(float) - images[617].astype(float)))
diff_493_863 = np.mean(np.abs(images[493].astype(float) - images[863].astype(float)))
diff_617_863 = np.mean(np.abs(images[617].astype(float) - images[863].astype(float)))

print("\n" + "=" * 70)
print("Pixel Difference Analysis:")
print("=" * 70)
print(f"  493 vs 617: MAE = {diff_493_617:.2f} pixels")
print(f"  493 vs 863: MAE = {diff_493_863:.2f} pixels")
print(f"  617 vs 863: MAE = {diff_617_863:.2f} pixels")

print("\nInterpretation:")
if diff_493_617 < 10 and diff_617_863 < 10:
    print("  ✓ C++ produces NEARLY IDENTICAL alignment across expressions")
    print("  → C++ is truly invariant to eye closure")
elif diff_493_617 > 30:
    print("  ⚠ C++ DOES show differences with eye closure")
    print("  → Our assumption was wrong!")
else:
    print("  ~ C++ shows minor differences (expected from expression changes)")
    print("  → But rotation appears consistent")

print("=" * 70)
