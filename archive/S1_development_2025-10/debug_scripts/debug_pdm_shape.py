#!/usr/bin/env python3
"""
Debug PDM mean shape format
"""

import numpy as np
from pdm_parser import PDMParser

pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")

print("=" * 70)
print("PDM Mean Shape Analysis")
print("=" * 70)

print("\n[1] Raw Mean Shape")
print(f"  Shape: {pdm.mean_shape.shape}")
print(f"  First 10 values: {pdm.mean_shape[:10].flatten()}")
print(f"  Values 136-146: {pdm.mean_shape[136:146].flatten()}")

print("\n[2] Scaled Mean Shape (sim_scale = 0.7)")
mean_scaled = pdm.mean_shape * 0.7
print(f"  Shape: {mean_scaled.shape}")
print(f"  First 10 values: {mean_scaled[:10].flatten()}")

print("\n[3] Discard Z (first 136 values = 68 * 2)")
mean_2d = mean_scaled[:136]
print(f"  Shape: {mean_2d.shape}")
print(f"  First 10 values: {mean_2d[:10].flatten()}")

print("\n[4] Reshape to (68, 2)")
mean_reshaped = mean_2d.reshape(68, 2)
print(f"  Shape: {mean_reshaped.shape}")
print(f"  First 5 landmarks:")
for i in range(5):
    print(f"    [{i}] ({mean_reshaped[i,0]:.4f}, {mean_reshaped[i,1]:.4f})")

print("\n[5] Alternative: Reshape then discard")
# Maybe OpenFace does: reshape (204,) -> (68, 3), then keep [:, :2]?
mean_3d = pdm.mean_shape.reshape(68, 3) * 0.7
mean_2d_alt = mean_3d[:, :2]
print(f"  Shape: {mean_2d_alt.shape}")
print(f"  First 5 landmarks:")
for i in range(5):
    print(f"    [{i}] ({mean_2d_alt[i,0]:.4f}, {mean_2d_alt[i,1]:.4f})")

print("\n[6] Check if values match C++ expected format")
print(f"  X range: [{mean_reshaped[:,0].min():.4f}, {mean_reshaped[:,0].max():.4f}]")
print(f"  Y range: [{mean_reshaped[:,1].min():.4f}, {mean_reshaped[:,1].max():.4f}]")
print(f"  Center: ({mean_reshaped[:,0].mean():.4f}, {mean_reshaped[:,1].mean():.4f})")

print("\n" + "=" * 70)
