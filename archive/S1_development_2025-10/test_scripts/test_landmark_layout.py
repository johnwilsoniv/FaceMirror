#!/usr/bin/env python3
"""
Test to understand landmark data layout difference between C++ and Python
"""

import numpy as np
import pandas as pd

# Load first frame
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]

# Extract landmarks the way we currently do it
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)

print("=" * 70)
print("Landmark Data Layout Test")
print("=" * 70)

print("\n[1] Current Python approach: np.stack([x, y], axis=1)")
landmarks_current = np.stack([x, y], axis=1)
print(f"Shape: {landmarks_current.shape}")
print("First 3 points:")
for i in range(3):
    print(f"  Point {i}: ({landmarks_current[i,0]:10.2f}, {landmarks_current[i,1]:10.2f})")

print("\n[2] Alternative: Interleave then reshape")
# Create [x0, y0, x1, y1, ..., x67, y67]
interleaved = np.empty(136, dtype=np.float32)
interleaved[0::2] = x  # Even indices = X
interleaved[1::2] = y  # Odd indices = Y
print(f"Interleaved shape: {interleaved.shape}")
print(f"First 6 elements: {interleaved[:6]}")

# C++ does: detected_landmarks.reshape(1, 2).t()
# This means: reshape(136,) to (2, 68), then transpose to (68, 2)
# Simulate this:
reshaped_2_68 = interleaved.reshape(2, 68)
print(f"\nAfter reshape(2, 68):")
print(f"  Row 0 (first 3): {reshaped_2_68[0, :3]}")
print(f"  Row 1 (first 3): {reshaped_2_68[1, :3]}")

transposed_68_2 = reshaped_2_68.T
print(f"\nAfter transpose to (68, 2):")
print("First 3 points:")
for i in range(3):
    print(f"  Point {i}: ({transposed_68_2[i,0]:10.2f}, {transposed_68_2[i,1]:10.2f})")

print("\n[3] Check if they're the same:")
if np.allclose(landmarks_current, transposed_68_2):
    print("✓ MATCH - layouts are identical")
else:
    print("✗ DIFFERENT - this could be the bug!")
    diff = np.abs(landmarks_current - transposed_68_2)
    print(f"  Max difference: {diff.max()}")
    print(f"  Mean difference: {diff.mean()}")

print("\n[4] What if C++ landmarks are stored differently?")
# Maybe C++ has them as [x0, x1, ..., x67, y0, y1, ..., y67]?
all_x_then_y = np.concatenate([x, y])
reshaped_2_68_v2 = all_x_then_y.reshape(2, 68)
print(f"If stored as [all X, all Y]:")
print(f"  Row 0 (first 3 X): {reshaped_2_68_v2[0, :3]}")
print(f"  Row 1 (first 3 Y): {reshaped_2_68_v2[1, :3]}")

transposed_68_2_v2 = reshaped_2_68_v2.T
print(f"\nAfter transpose:")
print("First 3 points:")
for i in range(3):
    print(f"  Point {i}: ({transposed_68_2_v2[i,0]:10.2f}, {transposed_68_2_v2[i,1]:10.2f})")

if np.allclose(landmarks_current, transposed_68_2_v2):
    print("\n✓ MATCH with [all X, all Y] layout")
else:
    print("\n✗ DIFFERENT from [all X, all Y] layout")
