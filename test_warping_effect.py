#!/usr/bin/env python3
"""
Test if image warping causes the response map peak offset.

Compare response maps computed with and without warping for the same landmark.
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

# Get patch experts
patch_experts = clnf._get_patch_experts(view_idx=0, scale=0.25)

# Test landmark
landmark_idx = 48  # Mouth corner (showed 3.6px offset in visualization)
patch_expert = patch_experts[landmark_idx]
lm_x, lm_y = landmarks[landmark_idx]

window_size = 11
print("=" * 80)
print("TESTING WARPING EFFECT ON RESPONSE MAPS")
print("=" * 80)
print(f"Landmark {landmark_idx}: position ({lm_x:.1f}, {lm_y:.1f})")
print(f"Window size: {window_size}x{window_size}")
print()

# For this test, we'll just skip the warping test since computing transforms is complex.
# Instead, let's focus on the simple case: does the offset exist WITHOUT warping?
# If yes, then warping is not the primary cause.

sim_img_to_ref = None
sim_ref_to_img = None

print("NOTE: Testing WITHOUT warping first to see if offset exists in simple case")
print()

# Create optimizer instance to access _compute_response_map
optimizer = NURLMSOptimizer()

# Test 1: Response map WITHOUT warping
print("Test 1: Computing response map WITHOUT warping...")
response_no_warp = optimizer._compute_response_map(
    gray, lm_x, lm_y, patch_expert, window_size,
    sim_img_to_ref=None,  # No warping
    sim_ref_to_img=None,
    sigma_components=None  # Also disable sigma for fair comparison
)

# Find peak
peak_idx_no_warp = np.unravel_index(np.argmax(response_no_warp), response_no_warp.shape)
peak_y_no_warp, peak_x_no_warp = peak_idx_no_warp
center = (window_size - 1) / 2.0
offset_x_no_warp = peak_x_no_warp - center
offset_y_no_warp = peak_y_no_warp - center
offset_dist_no_warp = np.sqrt(offset_x_no_warp**2 + offset_y_no_warp**2)

print(f"  Peak at: ({peak_x_no_warp}, {peak_y_no_warp})")
print(f"  Offset from center: ({offset_x_no_warp:+.1f}, {offset_y_no_warp:+.1f}) - {offset_dist_no_warp:.2f} px")
print(f"  Peak value: {response_no_warp[peak_y_no_warp, peak_x_no_warp]:.4f}")
print()

# Analysis
print("=" * 80)
print("RESULT:")
print("=" * 80)
print(f"Peak offset WITHOUT warping: {offset_dist_no_warp:.2f} px")
print()

if offset_dist_no_warp > 2.0:
    print("The offset is LARGE even without warping!")
    print("This means the bug is NOT caused by image warping.")
    print("The bug is in either:")
    print("  1. Patch extraction (wrong coordinates)")
    print("  2. Patch expert forward pass (wrong preprocessing/normalization)")
    print("  3. The trained patch expert models themselves")
else:
    print("The offset is small without warping.")
    print("This means warping may be the primary cause of the offset.")

print()
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Response map heatmap
im1 = axes[0].imshow(response_no_warp, cmap='hot', interpolation='nearest')
axes[0].plot(peak_x_no_warp, peak_y_no_warp, 'b+', markersize=15, markeredgewidth=3)
axes[0].plot(center, center, 'gx', markersize=15, markeredgewidth=3)
axes[0].set_title(f'Response Map (No Warping)\nOffset={offset_dist_no_warp:.2f}px')
axes[0].set_xlabel('X position')
axes[0].set_ylabel('Y position')
axes[0].grid(True, alpha=0.3)
plt.colorbar(im1, ax=axes[0])

# Plot 2: Response map 3D
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(122, projection='3d')
X, Y = np.meshgrid(range(window_size), range(window_size))
surf = ax.plot_surface(X, Y, response_no_warp, cmap='hot', alpha=0.8)
ax.scatter([peak_x_no_warp], [peak_y_no_warp], [response_no_warp[peak_y_no_warp, peak_x_no_warp]],
          color='blue', s=100, label='Peak')
ax.scatter([center], [center], [response_no_warp[int(center), int(center)]],
          color='green', s=100, label='Center')
ax.set_title('Response Map (3D)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Response')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()

out_path = Path('/tmp/response_maps') / f'warping_comparison_landmark_{landmark_idx}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved: {out_path}")
plt.close()

print()
print("Next steps based on results:")
print("  - If offset is similar with/without warping: Bug is in patch extraction or patch expert")
print("  - If warping makes it worse: Bug is in warping transform computation")
print("  - Either way, we've narrowed down where to look!")
