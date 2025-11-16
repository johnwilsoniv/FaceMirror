#!/usr/bin/env python3
"""
Debug the warping transformation to understand why response maps are so different.
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

from pathlib import Path
from pyclnf.core.pdm import PDM
from pyclnf.core.utils import align_shapes_with_scale, invert_similarity_transform

# Load image
image = cv2.imread('calibration_frames/patient1_frame1.jpg', cv2.IMREAD_GRAYSCALE)

# Load PDM
pdm_dir = Path('pyclnf/models/exported_pdm')
pdm = PDM(str(pdm_dir))

# Initialize from bbox (same as CLNF test)
bbox = [310, 780, 370, 370]
initial_params = pdm.init_params(tuple(bbox))

# Get current landmarks (ITER0 positions)
landmarks_2d = pdm.params_to_landmarks_2d(initial_params)

print("ITER0 landmark positions (tracked):")
for lm_idx in [36, 48, 30, 8]:
    x, y = landmarks_2d[lm_idx]
    print(f"  Landmark_{lm_idx}: ({x:.4f}, {y:.4f})")

# Get reference shape at patch_scaling=0.25
patch_scaling = 0.25
reference_shape = pdm.get_reference_shape(patch_scaling, initial_params[6:])

print(f"\nReference shape at scale={patch_scaling} (tracked):")
for lm_idx in [36, 48, 30, 8]:
    x, y = reference_shape[lm_idx]
    print(f"  Landmark_{lm_idx}: ({x:.4f}, {y:.4f})")

# Compute similarity transform: IMAGE ↔ REFERENCE
sim_img_to_ref = align_shapes_with_scale(landmarks_2d, reference_shape)
sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

print(f"\nSimilarity transform IMAGE → REFERENCE:")
print(f"  {sim_img_to_ref}")
print(f"\nSimilarity transform REFERENCE → IMAGE:")
print(f"  {sim_ref_to_img}")

# Extract transformation parameters
a_mat = sim_ref_to_img[0, 0]
b_mat = sim_ref_to_img[1, 0]
tx = sim_ref_to_img[0, 2]
ty = sim_ref_to_img[1, 2]

scale_transform = np.sqrt(a_mat**2 + b_mat**2)
rotation_transform = np.arctan2(b_mat, a_mat) * 180 / np.pi

print(f"\nTransform parameters (REF → IMG):")
print(f"  a_mat: {a_mat:.6f}, b_mat: {b_mat:.6f}")
print(f"  Scale: {scale_transform:.6f}")
print(f"  Rotation: {rotation_transform:.2f}°")
print(f"  Translation: ({tx:.2f}, {ty:.2f})")

# Now let's see what happens when we extract a window around landmark 36
lm_x, lm_y = landmarks_2d[36]
print(f"\nLandmark 36 IMAGE position: ({lm_x:.4f}, {lm_y:.4f})")

# Transform landmark to reference coordinates
lm_ref = sim_img_to_ref @ np.array([lm_x, lm_y, 1.0])
print(f"Landmark 36 REFERENCE position: ({lm_ref[0]:.4f}, {lm_ref[1]:.4f})")

# In CLNF optimization, we extract a window in image space,
# then warp it to reference space

window_size = 11
half_window = window_size // 2

# Define window in IMAGE coordinates
cx_int = int(lm_x)
cy_int = int(lm_y)

print(f"\nWindow in IMAGE space:")
print(f"  Center: ({cx_int}, {cy_int})")
print(f"  Bounds: x=[{cx_int-half_window}, {cx_int+half_window+1}], y=[{cy_int-half_window}, {cy_int+half_window+1}]")

# Extract the window
window = image[cy_int-half_window:cy_int+half_window+1,
               cx_int-half_window:cx_int+half_window+1].copy()

print(f"  Window shape: {window.shape}")
print(f"  Window stats: min={window.min()}, max={window.max()}, mean={window.mean():.1f}")

# Save the original window
cv2.imwrite('/tmp/window_original.png', window)
print(f"  Saved original window to /tmp/window_original.png")

# Now warp this window to reference coordinates using sim_img_to_ref
# The warping should map the window to the canonical pose where patches were trained

# For reference: In OpenFace C++ (CCNF_patch_expert.cpp lines 313-332):
# 1. Define warped_img size based on reference shape bounds
# 2. cv2.warpAffine(image, sim_img_to_ref, warped_img.size)
# 3. Extract patches from warped_img at reference landmark positions

# Compute reference shape bounds
ref_x_min = reference_shape[:, 0].min()
ref_x_max = reference_shape[:, 0].max()
ref_y_min = reference_shape[:, 1].min()
ref_y_max = reference_shape[:, 1].max()

ref_width = int(np.ceil(ref_x_max - ref_x_min))
ref_height = int(np.ceil(ref_y_max - ref_y_min))

print(f"\nReference shape bounds:")
print(f"  X: [{ref_x_min:.2f}, {ref_x_max:.2f}] width={ref_width}")
print(f"  Y: [{ref_y_min:.2f}, {ref_y_max:.2f}] height={ref_height}")

# Warp the entire image to reference coordinates
warped_img = cv2.warpAffine(image, sim_img_to_ref[:2, :], (ref_width + 20, ref_height + 20))

print(f"\nWarped image:")
print(f"  Shape: {warped_img.shape}")
print(f"  Stats: min={warped_img.min()}, max={warped_img.max()}, mean={warped_img.mean():.1f}")

# Save warped image
cv2.imwrite('/tmp/warped_image.png', warped_img)
print(f"  Saved to /tmp/warped_image.png")

# Extract window around landmark 36 in reference coordinates
lm_ref_x = int(lm_ref[0])
lm_ref_y = int(lm_ref[1])

print(f"\nExtracting window at reference position ({lm_ref_x}, {lm_ref_y}):")

if (lm_ref_y - half_window >= 0 and lm_ref_y + half_window + 1 < warped_img.shape[0] and
    lm_ref_x - half_window >= 0 and lm_ref_x + half_window + 1 < warped_img.shape[1]):
    warped_window = warped_img[lm_ref_y - half_window:lm_ref_y + half_window + 1,
                                lm_ref_x - half_window:lm_ref_x + half_window + 1]
    print(f"  Warped window shape: {warped_window.shape}")
    print(f"  Warped window stats: min={warped_window.min()}, max={warped_window.max()}, mean={warped_window.mean():.1f}")

    cv2.imwrite('/tmp/window_warped.png', warped_window)
    print(f"  Saved warped window to /tmp/window_warped.png")
else:
    print(f"  ERROR: Reference position out of bounds!")

print("\nSummary:")
print("The warping transforms the image to a canonical pose/scale.")
print("Patches should be extracted from the warped image, not the original.")
print("Check if CLNF is correctly extracting from warped image or if there's a bug.")
