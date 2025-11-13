#!/usr/bin/env python3
"""
Test if Sigma transformation causes the response map peak offset.

Compare response maps computed WITH and WITHOUT Sigma transformation for the same landmark.
This will definitively show if Sigma is the bug.
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core.optimizer import NURLMSOptimizer
import matplotlib.pyplot as plt
from pathlib import Path

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

# Get initial landmarks and parameters
params = clnf.pdm.init_params(face_bbox)
landmarks = clnf.pdm.params_to_landmarks_2d(params)

# Get patch experts and sigma components
patch_experts = clnf._get_patch_experts(view_idx=0, scale=0.25)

# Load sigma components
sigma_components = {}
for ws in [7, 9, 11, 15]:
    sigma_file = f'pyclnf/models/sigma_components/ccnf/sigma_ws_{ws}.dat'
    try:
        sigma_data = np.loadtxt(sigma_file)
        sigma_components[ws] = sigma_data
    except:
        pass

# Test landmark
landmark_idx = 48  # Mouth corner (showed 3.6px offset in visualization)
patch_expert = patch_experts[landmark_idx]
lm_x, lm_y = landmarks[landmark_idx]

window_size = 11
print("=" * 80)
print("TESTING SIGMA TRANSFORMATION EFFECT ON RESPONSE MAPS")
print("=" * 80)
print(f"Landmark {landmark_idx}: position ({lm_x:.1f}, {lm_y:.1f})")
print(f"Window size: {window_size}x{window_size}")
print(f"Patch expert: {patch_expert.width}x{patch_expert.height}, {len(patch_expert.neurons)} neurons")
print()

# Create optimizer instance to access _compute_response_map
optimizer = NURLMSOptimizer()

# Test 1: Response map WITHOUT Sigma transformation
print("Test 1: Computing response map WITHOUT Sigma...")
response_no_sigma = optimizer._compute_response_map(
    gray, lm_x, lm_y, patch_expert, window_size,
    sim_img_to_ref=None,
    sim_ref_to_img=None,
    sigma_components=None  # No Sigma
)

# Find peak
peak_idx_no_sigma = np.unravel_index(np.argmax(response_no_sigma), response_no_sigma.shape)
peak_y_no_sigma, peak_x_no_sigma = peak_idx_no_sigma
center = (window_size - 1) / 2.0
offset_x_no_sigma = peak_x_no_sigma - center
offset_y_no_sigma = peak_y_no_sigma - center
offset_dist_no_sigma = np.sqrt(offset_x_no_sigma**2 + offset_y_no_sigma**2)

print(f"  Response range: [{response_no_sigma.min():.4f}, {response_no_sigma.max():.4f}]")
print(f"  Peak at: ({peak_x_no_sigma}, {peak_y_no_sigma})")
print(f"  Offset from center: ({offset_x_no_sigma:+.1f}, {offset_y_no_sigma:+.1f}) - {offset_dist_no_sigma:.2f} px")
print(f"  Peak value: {response_no_sigma[peak_y_no_sigma, peak_x_no_sigma]:.4f}")
print(f"  Center value: {response_no_sigma[int(center), int(center)]:.4f}")
print()

# Test 2: Response map WITH Sigma transformation
print("Test 2: Computing response map WITH Sigma...")
response_with_sigma = optimizer._compute_response_map(
    gray, lm_x, lm_y, patch_expert, window_size,
    sim_img_to_ref=None,
    sim_ref_to_img=None,
    sigma_components=sigma_components  # With Sigma
)

# Find peak
peak_idx_with_sigma = np.unravel_index(np.argmax(response_with_sigma), response_with_sigma.shape)
peak_y_with_sigma, peak_x_with_sigma = peak_idx_with_sigma
offset_x_with_sigma = peak_x_with_sigma - center
offset_y_with_sigma = peak_y_with_sigma - center
offset_dist_with_sigma = np.sqrt(offset_x_with_sigma**2 + offset_y_with_sigma**2)

print(f"  Response range: [{response_with_sigma.min():.4f}, {response_with_sigma.max():.4f}]")
print(f"  Peak at: ({peak_x_with_sigma}, {peak_y_with_sigma})")
print(f"  Offset from center: ({offset_x_with_sigma:+.1f}, {offset_y_with_sigma:+.1f}) - {offset_dist_with_sigma:.2f} px")
print(f"  Peak value: {response_with_sigma[peak_y_with_sigma, peak_x_with_sigma]:.4f}")
print(f"  Center value: {response_with_sigma[int(center), int(center)]:.4f}")
print()

# Analysis
print("=" * 80)
print("COMPARISON:")
print("=" * 80)
print(f"Peak offset WITHOUT Sigma: {offset_dist_no_sigma:.2f} px at ({offset_x_no_sigma:+.1f}, {offset_y_no_sigma:+.1f})")
print(f"Peak offset WITH Sigma:    {offset_dist_with_sigma:.2f} px at ({offset_x_with_sigma:+.1f}, {offset_y_with_sigma:+.1f})")
print()

peak_moved = (peak_x_with_sigma != peak_x_no_sigma) or (peak_y_with_sigma != peak_y_no_sigma)
if peak_moved:
    dx = peak_x_with_sigma - peak_x_no_sigma
    dy = peak_y_with_sigma - peak_y_no_sigma
    movement = np.sqrt(dx**2 + dy**2)
    print(f"⚠️  Peak MOVED by ({dx:+.0f}, {dy:+.0f}) = {movement:.2f} pixels after Sigma transformation!")
else:
    print("✓ Peak stayed at same location")

print()

if offset_dist_with_sigma > offset_dist_no_sigma + 1.0:
    print("❌ BUG FOUND: Sigma transformation INCREASES the peak offset!")
    print("   This is the source of the convergence problem.")
elif abs(offset_dist_with_sigma - offset_dist_no_sigma) < 0.5:
    print("✓ Sigma transformation does NOT significantly change peak offset")
    print("  The bug is elsewhere (mean-shift, parameter updates, or Jacobian)")
else:
    print("⚠️  Sigma transformation slightly affects peak offset")

print()
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Without Sigma
# Plot 1: Heatmap
im1 = axes[0, 0].imshow(response_no_sigma, cmap='hot', interpolation='nearest')
axes[0, 0].plot(peak_x_no_sigma, peak_y_no_sigma, 'b+', markersize=15, markeredgewidth=3)
axes[0, 0].plot(center, center, 'gx', markersize=15, markeredgewidth=3)
axes[0, 0].set_title(f'WITHOUT Sigma\nOffset={offset_dist_no_sigma:.2f}px')
axes[0, 0].set_xlabel('X position')
axes[0, 0].set_ylabel('Y position')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(im1, ax=axes[0, 0])

# Plot 2: 3D surface
from mpl_toolkits.mplot3d import Axes3D
ax1 = fig.add_subplot(232, projection='3d')
X, Y = np.meshgrid(range(window_size), range(window_size))
surf1 = ax1.plot_surface(X, Y, response_no_sigma, cmap='hot', alpha=0.8)
ax1.scatter([peak_x_no_sigma], [peak_y_no_sigma], [response_no_sigma[peak_y_no_sigma, peak_x_no_sigma]],
          color='blue', s=100, label='Peak')
ax1.scatter([center], [center], [response_no_sigma[int(center), int(center)]],
          color='green', s=100, label='Center')
ax1.set_title('WITHOUT Sigma (3D)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Response')

# Plot 3: Cross-section
center_row = response_no_sigma[int(center), :]
axes[0, 2].plot(center_row, 'b-', linewidth=2)
axes[0, 2].axvline(center, color='g', linestyle='--', label='Center')
axes[0, 2].axvline(peak_x_no_sigma, color='r', linestyle='--', label='Peak')
axes[0, 2].set_title('Cross-section (Y=center)')
axes[0, 2].set_xlabel('X position')
axes[0, 2].set_ylabel('Response')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# Row 2: With Sigma
# Plot 4: Heatmap
im2 = axes[1, 0].imshow(response_with_sigma, cmap='hot', interpolation='nearest')
axes[1, 0].plot(peak_x_with_sigma, peak_y_with_sigma, 'b+', markersize=15, markeredgewidth=3)
axes[1, 0].plot(center, center, 'gx', markersize=15, markeredgewidth=3)
axes[1, 0].set_title(f'WITH Sigma\nOffset={offset_dist_with_sigma:.2f}px')
axes[1, 0].set_xlabel('X position')
axes[1, 0].set_ylabel('Y position')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(im2, ax=axes[1, 0])

# Plot 5: 3D surface
ax2 = fig.add_subplot(235, projection='3d')
surf2 = ax2.plot_surface(X, Y, response_with_sigma, cmap='hot', alpha=0.8)
ax2.scatter([peak_x_with_sigma], [peak_y_with_sigma], [response_with_sigma[peak_y_with_sigma, peak_x_with_sigma]],
          color='blue', s=100, label='Peak')
ax2.scatter([center], [center], [response_with_sigma[int(center), int(center)]],
          color='green', s=100, label='Center')
ax2.set_title('WITH Sigma (3D)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Response')

# Plot 6: Cross-section
center_row_sigma = response_with_sigma[int(center), :]
axes[1, 2].plot(center_row_sigma, 'b-', linewidth=2)
axes[1, 2].axvline(center, color='g', linestyle='--', label='Center')
axes[1, 2].axvline(peak_x_with_sigma, color='r', linestyle='--', label='Peak')
axes[1, 2].set_title('Cross-section (Y=center)')
axes[1, 2].set_xlabel('X position')
axes[1, 2].set_ylabel('Response')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()

out_path = Path('/tmp/response_maps') / f'sigma_comparison_landmark_{landmark_idx}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved: {out_path}")
plt.close()

print()
print("Next steps based on results:")
print("  - If Sigma causes the offset: Bug is in Sigma transformation or sigma component data")
print("  - If Sigma doesn't affect offset: Bug is in mean-shift or parameter update logic")
print("  - Either way, we've narrowed down the source!")
