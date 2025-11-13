"""
Analyze PDM shape parameters to understand jaw landmark spacing.
"""
import numpy as np
from pyclnf import CLNF
import cv2
import matplotlib.pyplot as plt

# Initialize
clnf = CLNF(model_dir="pyclnf/models")

# Load test video and extract first frame
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read video: {video_path}")

# Convert to grayscale
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_bbox = (241, 555, 532, 532)

# Get initial and final parameters
initial_params = clnf.pdm.init_params(face_bbox)
print("Initial parameters:")
print(f"  Scale: {initial_params[0]:.3f}")
print(f"  Rotation: [{initial_params[1]:.3f}, {initial_params[2]:.3f}, {initial_params[3]:.3f}]")
print(f"  Translation: [{initial_params[4]:.1f}, {initial_params[5]:.1f}]")
print(f"  Shape params (first 10): {initial_params[6:16]}")
print(f"  Shape params range: [{initial_params[6:].min():.3f}, {initial_params[6:].max():.3f}]")
print(f"  Shape params std: {initial_params[6:].std():.3f}")

# Run optimization
landmarks, info = clnf.fit(image, face_bbox, return_params=True)
params_final = info['params']

print("\nFinal parameters:")
print(f"  Scale: {params_final[0]:.3f}")
print(f"  Rotation: [{params_final[1]:.3f}, {params_final[2]:.3f}, {params_final[3]:.3f}]")
print(f"  Translation: [{params_final[4]:.1f}, {params_final[5]:.1f}]")
print(f"  Shape params (first 10): {params_final[6:16]}")
print(f"  Shape params range: [{params_final[6:].min():.3f}, {params_final[6:].max():.3f}]")
print(f"  Shape params std: {params_final[6:].std():.3f}")

print("\nShape parameter changes:")
shape_delta = params_final[6:] - initial_params[6:]
print(f"  Max change: {np.abs(shape_delta).max():.3f}")
print(f"  Mean |change|: {np.abs(shape_delta).mean():.3f}")
print(f"  Top 5 changed params (index, delta):")
top_5_indices = np.argsort(np.abs(shape_delta))[-5:][::-1]
for idx in top_5_indices:
    print(f"    Mode {idx}: {shape_delta[idx]:.3f}")

# Visualize shape mode effects on jawline
print("\n" + "="*80)
print("Analyzing jaw landmark spacing...")

# Get jawline landmarks (indices 0-16)
jaw_indices = list(range(0, 17))

# Get landmarks with mean shape only (no deformation)
mean_params = initial_params.copy()
mean_params[6:] = 0  # Zero out all shape parameters
mean_landmarks = clnf.pdm.params_to_landmarks_2d(mean_params)
mean_jaw = mean_landmarks[jaw_indices]

# Get landmarks with final shape
final_jaw = landmarks[jaw_indices]

# Compute spacing between consecutive jaw points
def compute_spacing(jaw_points):
    distances = []
    for i in range(len(jaw_points) - 1):
        dist = np.linalg.norm(jaw_points[i+1] - jaw_points[i])
        distances.append(dist)
    return np.array(distances)

mean_spacing = compute_spacing(mean_jaw)
final_spacing = compute_spacing(final_jaw)

print("\nJaw point spacing (mean shape vs final):")
print("  Index | Mean | Final | Ratio")
for i in range(len(mean_spacing)):
    ratio = final_spacing[i] / mean_spacing[i] if mean_spacing[i] > 0 else 0
    print(f"  {i:2d}-{i+1:2d} | {mean_spacing[i]:5.1f} | {final_spacing[i]:5.1f} | {ratio:.2f}")

# Visualize the top shape modes and their effect on jawline
print("\nVisualizing top 3 shape modes...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for plot_idx, mode_idx in enumerate(top_5_indices[:3]):
    ax = axes[plot_idx]

    # Create params with only this mode activated
    mode_params = initial_params.copy()
    mode_params[6:] = 0
    eigen_vals_flat = clnf.pdm.eigen_values.flatten()
    mode_params[6 + mode_idx] = 3.0 * np.sqrt(eigen_vals_flat[mode_idx])  # 3 std devs

    mode_landmarks = clnf.pdm.params_to_landmarks_2d(mode_params)
    mode_jaw = mode_landmarks[jaw_indices]

    # Plot mean shape jaw
    ax.plot(mean_jaw[:, 0], mean_jaw[:, 1], 'b-o', label='Mean shape', markersize=4)

    # Plot this mode's effect
    ax.plot(mode_jaw[:, 0], mode_jaw[:, 1], 'r-o', label=f'Mode {mode_idx}', markersize=4)

    ax.set_title(f'Shape Mode {mode_idx}\n(weight={params_final[6+mode_idx]:.2f})')
    ax.legend()
    ax.invert_yaxis()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('jaw_shape_modes.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: jaw_shape_modes.png")

# Analyze which shape modes affect jaw asymmetry
print("\n" + "="*80)
print("Analyzing jaw asymmetry from shape modes...")

# For each shape mode, measure how it affects left vs right jaw
left_jaw_indices = list(range(0, 9))  # Left side of face (right side of image)
right_jaw_indices = list(range(8, 17))  # Right side of face (left side of image)

asymmetry_scores = []
eigen_vals_flat = clnf.pdm.eigen_values.flatten()
for mode_idx in range(min(20, clnf.pdm.n_modes)):
    # Activate mode at +3 std
    test_params = initial_params.copy()
    test_params[6:] = 0
    test_params[6 + mode_idx] = 3.0 * np.sqrt(eigen_vals_flat[mode_idx])

    test_landmarks = clnf.pdm.params_to_landmarks_2d(test_params)

    # Measure average Y displacement for left vs right jaw
    left_y = test_landmarks[left_jaw_indices, 1].mean()
    right_y = test_landmarks[right_jaw_indices, 1].mean()
    mean_y = mean_landmarks[jaw_indices, 1].mean()

    left_delta = left_y - mean_y
    right_delta = right_y - mean_y
    asymmetry = abs(left_delta - right_delta)

    asymmetry_scores.append((mode_idx, asymmetry, left_delta, right_delta))

# Sort by asymmetry
asymmetry_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 shape modes causing jaw asymmetry:")
print("  Mode | Asymmetry | Left ΔY | Right ΔY | Final Weight")
for mode_idx, asym, left_d, right_d in asymmetry_scores[:5]:
    weight = params_final[6 + mode_idx]
    print(f"  {mode_idx:4d} | {asym:9.2f} | {left_d:7.2f} | {right_d:8.2f} | {weight:12.3f}")

print("\n" + "="*80)
