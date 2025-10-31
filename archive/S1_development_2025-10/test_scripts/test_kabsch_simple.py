#!/usr/bin/env python3
"""
Test Kabsch algorithm with a known simple rotation
"""

import numpy as np

def kabsch_current(src, dst):
    """Current implementation"""
    U, S, Vt = np.linalg.svd(src.T @ dst)
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    corr[1, 1] = 1 if d > 0 else -1
    R = Vt.T @ corr @ U.T
    return R

def kabsch_alternative(src, dst):
    """Alternative: swap U and Vt"""
    U, S, Vt = np.linalg.svd(src.T @ dst)
    d = np.linalg.det(U @ Vt)
    corr = np.eye(2)
    corr[1, 1] = 1 if d > 0 else -1
    R = U @ corr @ Vt
    return R

# Test with 45° rotation
theta = np.pi / 4
R_true = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])

# Create test points (square)
src = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
dst = (R_true @ src.T).T

print("="*70)
print("Kabsch Algorithm Test: 45° Rotation")
print("="*70)

print("\nTrue rotation matrix (45°):")
print(R_true)

print("\n[1] Current implementation:")
R1 = kabsch_current(src, dst)
print(R1)
error1 = np.linalg.norm(R_true - R1)
print(f"Error: {error1:.6f}")

print("\n[2] Alternative (swapped U and Vt):")
R2 = kabsch_alternative(src, dst)
print(R2)
error2 = np.linalg.norm(R_true - R2)
print(f"Error: {error2:.6f}")

print("\n" + "="*70)
if error1 < error2:
    print("Current implementation is CORRECT")
else:
    print("Alternative implementation is CORRECT - SWAP U and Vt!")
print("="*70)
