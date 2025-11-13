#!/usr/bin/env python3
"""
Test mean-shift calculation to see if it correctly identifies the peak offset.

Given a response map with a peak at offset (+3, +2), the mean-shift should
return approximately (+3, +2) indicating the landmark should move in that direction.
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core.optimizer import NURLMSOptimizer

# Load test frame
video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_bbox = (241, 555, 532, 532)

# Initialize CLNF
clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)
params = clnf.pdm.init_params(face_bbox)
landmarks = clnf.pdm.params_to_landmarks_2d(params)

# Get patch experts
patch_experts = clnf._get_patch_experts(view_idx=0, scale=0.25)

# Test landmark 48 (known to have peak at offset +3, +2)
landmark_idx = 48
patch_expert = patch_experts[landmark_idx]
lm_x, lm_y = landmarks[landmark_idx]

window_size = 11
print("=" * 80)
print("TESTING MEAN-SHIFT CALCULATION")
print("=" * 80)
print(f"Landmark {landmark_idx}: position ({lm_x:.1f}, {lm_y:.1f})")
print(f"Window size: {window_size}x{window_size}")
print()

# Create optimizer
optimizer = NURLMSOptimizer()

# Compute response map (without sigma for simplicity)
response_map = optimizer._compute_response_map(
    gray, lm_x, lm_y, patch_expert, window_size,
    sim_img_to_ref=None,
    sim_ref_to_img=None,
    sigma_components=None
)

# Find peak location
peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
peak_y, peak_x = peak_idx
center = (window_size - 1) / 2.0

print("Response map analysis:")
print(f"  Peak location: ({peak_x}, {peak_y})")
print(f"  Center: ({center:.1f}, {center:.1f})")
print(f"  Peak offset from center: ({peak_x - center:+.1f}, {peak_y - center:+.1f})")
print(f"  Peak value: {response_map[peak_y, peak_x]:.4f}")
print(f"  Center value: {response_map[int(center), int(center)]:.4f}")
print()

# Now compute mean-shift
# In the optimizer, dx and dy are computed as:
dx_frac = lm_x - int(lm_x)
dy_frac = lm_y - int(lm_y)
dx = dx_frac + center
dy = dy_frac + center

print(f"Current landmark position:")
print(f"  lm_x = {lm_x:.6f}, lm_y = {lm_y:.6f}")
print(f"  Fractional part: dx_frac = {dx_frac:.6f}, dy_frac = {dy_frac:.6f}")
print(f"  Position within response map: dx = {dx:.6f}, dy = {dy:.6f}")
print()

# Compute mean-shift with default sigma
sigma = 1.5
a = -0.5 / (sigma ** 2)
ms_x, ms_y = optimizer._kde_mean_shift(response_map, dx, dy, a)

print(f"Mean-shift calculation:")
print(f"  Sigma: {sigma}")
print(f"  KDE parameter a: {a:.6f}")
print(f"  Mean-shift result: ({ms_x:+.6f}, {ms_y:+.6f})")
print(f"  Mean-shift magnitude: {np.sqrt(ms_x**2 + ms_y**2):.6f}")
print()

# Expected result: mean-shift should point toward the peak
expected_ms_x = (peak_x - center) - dx_frac
expected_ms_y = (peak_y - center) - dy_frac

print("=" * 80)
print("VERIFICATION:")
print("=" * 80)
print(f"Peak is at offset ({peak_x - center:+.1f}, {peak_y - center:+.1f}) from center")
print(f"Expected mean-shift: ({expected_ms_x:+.6f}, {expected_ms_y:+.6f})")
print(f"Actual mean-shift:   ({ms_x:+.6f}, {ms_y:+.6f})")
print()

# Check if they match
ms_error = np.sqrt((ms_x - expected_ms_x)**2 + (ms_y - expected_ms_y)**2)
print(f"Error: {ms_error:.6f} pixels")
print()

if ms_error < 0.5:
    print("✓ Mean-shift is CORRECT - points toward the peak")
elif abs(ms_x) < 0.1 and abs(ms_y) < 0.1:
    print("❌ BUG FOUND: Mean-shift is near ZERO even though peak is offset!")
    print("   This means KDE is too smooth or response map is being flattened")
elif ms_error > 2.0:
    print("❌ BUG FOUND: Mean-shift points in WRONG direction!")
    print("   There may be a sign error or coordinate system mismatch")
else:
    print("⚠️  Mean-shift is somewhat inaccurate (error > 0.5 px)")
    print("   This could be due to KDE smoothing or sub-pixel positioning")

print()
print("=" * 80)
print("DETAILED MEAN-SHIFT COMPUTATION:")
print("=" * 80)

# Manual computation to debug
mx = 0.0
my = 0.0
total_weight = 0.0

for ii in range(window_size):
    for jj in range(window_size):
        # Distance from current position (dx, dy) to grid point (jj, ii)
        dist_sq = (dy - ii)**2 + (dx - jj)**2

        # Gaussian weight
        kde_weight = np.exp(a * dist_sq)

        # Combined weight
        weight = kde_weight * response_map[ii, jj]

        total_weight += weight
        mx += weight * jj
        my += weight * ii

if total_weight > 1e-10:
    weighted_mean_x = mx / total_weight
    weighted_mean_y = my / total_weight
    ms_x_manual = weighted_mean_x - dx
    ms_y_manual = weighted_mean_y - dy
else:
    weighted_mean_x = 0.0
    weighted_mean_y = 0.0
    ms_x_manual = 0.0
    ms_y_manual = 0.0

print(f"Total weight: {total_weight:.6f}")
print(f"Weighted mean position: ({weighted_mean_x:.6f}, {weighted_mean_y:.6f})")
print(f"Current position (dx, dy): ({dx:.6f}, {dy:.6f})")
print(f"Mean-shift (manual): ({ms_x_manual:+.6f}, {ms_y_manual:+.6f})")
print()

# Check KDE weights at a few key positions
print("KDE weights at key positions:")
print(f"  At center ({center:.0f}, {center:.0f}): {np.exp(a * 0):.6f} (should be 1.0)")
print(f"  At peak ({peak_x}, {peak_y}): {np.exp(a * ((dy - peak_y)**2 + (dx - peak_x)**2)):.6f}")
print(f"  At 1 pixel away: {np.exp(a * 1):.6f}")
print(f"  At 2 pixels away: {np.exp(a * 4):.6f}")
print()

# Print response map cross-sections
print("Response map cross-section (Y=center):")
center_int = int(center)
for jj in range(window_size):
    val = response_map[center_int, jj]
    if jj == peak_x:
        print(f"  X={jj}: {val:8.4f} ← PEAK")
    elif jj == int(center):
        print(f"  X={jj}: {val:8.4f} ← CENTER")
    else:
        print(f"  X={jj}: {val:8.4f}")

print()
print("=" * 80)
