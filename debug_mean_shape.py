"""Debug mean shape dimensions."""
import numpy as np
from pathlib import Path

# Load PDM mean shape
model_dir = Path("pyclnf/models/exported_pdm")
mean_shape = np.load(model_dir / 'mean_shape.npy')

print(f"Mean shape shape: {mean_shape.shape}")
print(f"Mean shape dtype: {mean_shape.dtype}")

# Flatten and extract coordinates
shape_3d = mean_shape.flatten()
n = len(shape_3d) // 3

print(f"Number of points: {n}")

# Extract x, y, z
x_coords = shape_3d[:n]
y_coords = shape_3d[n:2*n]
z_coords = shape_3d[2*n:3*n]

print(f"\nX range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
print(f"Y range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
print(f"Z range: [{z_coords.min():.3f}, {z_coords.max():.3f}]")

print(f"\nX width: {x_coords.max() - x_coords.min():.3f}")
print(f"Y height: {y_coords.max() - y_coords.min():.3f}")
print(f"Z depth: {z_coords.max() - z_coords.min():.3f}")

# Test rotation with zero rotation (identity)
from pyclnf import CLNF
clnf = CLNF(model_dir="pyclnf/models")

rotation = np.array([0.0, 0.0, 0.0])
R = clnf.pdm._euler_to_rotation_matrix(rotation)
print(f"\nRotation matrix (should be identity):")
print(R)

# Reshape and rotate
shape_3d_mat = np.array([x_coords, y_coords, z_coords])  # (3, n_points)
rotated_shape = R @ shape_3d_mat

print(f"\nRotated X range: [{rotated_shape[0, :].min():.3f}, {rotated_shape[0, :].max():.3f}]")
print(f"Rotated Y range: [{rotated_shape[1, :].min():.3f}, {rotated_shape[1, :].max():.3f}]")
print(f"Rotated Z range: [{rotated_shape[2, :].min():.3f}, {rotated_shape[2, :].max():.3f}]")

print(f"\nRotated X width: {rotated_shape[0, :].max() - rotated_shape[0, :].min():.3f}")
print(f"Rotated Y height: {rotated_shape[1, :].max() - rotated_shape[1, :].min():.3f}")
