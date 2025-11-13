#!/usr/bin/env python3
"""
Visualize response maps to understand why peaks are offset from center.
Extract and save response maps for a few landmarks to inspect visually.
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core import pdm
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

# Get initial landmarks
params = clnf.pdm.init_params(face_bbox)
landmarks = clnf.pdm.params_to_landmarks_2d(params)

# Get patch experts for scale 0.25, view 0
patch_experts = clnf._get_patch_experts(view_idx=0, scale=0.25)

# Create output directory
out_dir = Path('/tmp/response_maps')
out_dir.mkdir(exist_ok=True)

print("=" * 80)
print("RESPONSE MAP VISUALIZATION")
print("=" * 80)
print(f"Frame: {video_path} frame 50")
print(f"Face bbox: {face_bbox}")
print(f"Patch experts available: {len(patch_experts)}")
print()

# Test a few landmarks that showed large peak offsets
test_landmarks = [1, 6, 48]  # From our debug output

window_size = 11
half_window = window_size // 2

for landmark_idx in test_landmarks:
    if landmark_idx not in patch_experts:
        print(f"Landmark {landmark_idx}: No patch expert")
        continue

    patch_expert = patch_experts[landmark_idx]
    lm_x, lm_y = landmarks[landmark_idx]

    print(f"\nLandmark {landmark_idx}: position ({lm_x:.1f}, {lm_y:.1f})")
    print(f"  Patch size: {patch_expert.width}x{patch_expert.height}")

    # Compute response map WITHOUT warping first (simpler)
    response_map = np.zeros((window_size, window_size))

    start_x = int(lm_x) - half_window
    start_y = int(lm_y) - half_window

    for i in range(window_size):
        for j in range(window_size):
            patch_x = start_x + j
            patch_y = start_y + i

            # Extract patch
            patch_half_w = patch_expert.width // 2
            patch_half_h = patch_expert.height // 2

            y1 = max(0, patch_y - patch_half_h)
            y2 = min(gray.shape[0], patch_y + patch_half_h + 1)
            x1 = max(0, patch_x - patch_half_w)
            x2 = min(gray.shape[1], patch_x + patch_half_w + 1)

            if y2 - y1 == patch_expert.height and x2 - x1 == patch_expert.width:
                patch = gray[y1:y2, x1:x2]
                response_map[i, j] = patch_expert.compute_response(patch)
            else:
                response_map[i, j] = -1e10

    # Normalize response map for visualization
    min_val = response_map.min()
    if min_val < 0:
        response_map_viz = response_map - min_val
    else:
        response_map_viz = response_map.copy()

    # Find peak
    peak_idx = np.unravel_index(np.argmax(response_map_viz), response_map_viz.shape)
    peak_y, peak_x = peak_idx
    peak_value = response_map_viz[peak_y, peak_x]

    # Compute offset from center
    center = (window_size - 1) / 2.0
    offset_x = peak_x - center
    offset_y = peak_y - center
    offset_dist = np.sqrt(offset_x**2 + offset_y**2)

    print(f"  Response range: [{response_map_viz.min():.4f}, {response_map_viz.max():.4f}]")
    print(f"  Peak location: ({peak_x}, {peak_y}) - should be near ({center:.1f}, {center:.1f})")
    print(f"  Peak offset: ({offset_x:+.1f}, {offset_y:+.1f}) - distance={offset_dist:.2f} pixels")
    print(f"  Peak value: {peak_value:.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Response map heatmap
    im1 = axes[0].imshow(response_map_viz, cmap='hot', interpolation='nearest')
    axes[0].plot(peak_x, peak_y, 'b+', markersize=15, markeredgewidth=3)
    axes[0].plot(center, center, 'gx', markersize=15, markeredgewidth=3)
    axes[0].set_title(f'Response Map (Landmark {landmark_idx})')
    axes[0].set_xlabel('X position')
    axes[0].set_ylabel('Y position')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0], label='Response')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='+', color='w', markerfacecolor='b', markersize=10, label='Peak'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='g', markersize=10, label='Center (landmark position)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')

    # 2. Response map 3D surface
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(132, projection='3d')
    X, Y = np.meshgrid(range(window_size), range(window_size))
    surf = ax.plot_surface(X, Y, response_map_viz, cmap='hot', alpha=0.8)
    ax.scatter([peak_x], [peak_y], [peak_value], color='blue', s=100, label='Peak')
    ax.scatter([center], [center], [response_map_viz[int(center), int(center)]], color='green', s=100, label='Center')
    ax.set_title('Response Map (3D)')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Response')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 3. Image patch at landmark location
    patch_size = 50  # Show larger region for context
    y1 = max(0, int(lm_y) - patch_size // 2)
    y2 = min(gray.shape[0], int(lm_y) + patch_size // 2)
    x1 = max(0, int(lm_x) - patch_size // 2)
    x2 = min(gray.shape[1], int(lm_x) + patch_size // 2)

    img_patch = gray[y1:y2, x1:x2]
    axes[2].imshow(img_patch, cmap='gray')
    axes[2].plot(lm_x - x1, lm_y - y1, 'g+', markersize=15, markeredgewidth=3)
    axes[2].set_title(f'Image Region (Landmark {landmark_idx})')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')

    plt.tight_layout()

    # Save figure
    out_path = out_dir / f'response_map_landmark_{landmark_idx}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()

print()
print("=" * 80)
print(f"Visualizations saved to: {out_dir}")
print("=" * 80)
print()
print("KEY OBSERVATIONS TO LOOK FOR:")
print("1. Is the peak truly offset from center, or is it at center?")
print("2. Are there multiple peaks, or is there one clear peak?")
print("3. Does the image patch show the correct facial feature?")
print("4. Is the response map smooth or noisy?")
