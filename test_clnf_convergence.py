#!/usr/bin/env python3
"""
Quick test to verify CLNF convergence fix.
"""

import numpy as np
import sys
from pathlib import Path

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

# Import the optimizer module
from pyclnf.core.optimizer import NURLMSOptimizer

# Create optimizer and check convergence threshold
optimizer = NURLMSOptimizer(
    pdm_file='pyfaceau/weights/In-the-wild_aligned_PDM_68.txt',
    patch_expert_file='pyfaceau/weights/svr_patches_0.25_general.txt',
    convergence_threshold=0.1
)

print(f"Optimizer convergence_threshold: {optimizer.convergence_threshold}")
print(f"Expected: 0.1")

# Test convergence calculation
# Simulate small landmark changes
current_landmarks = np.random.randn(68, 2) * 100
prev_landmarks = current_landmarks + np.random.randn(68, 2) * 0.05  # Small change

# Calculate convergence as the optimizer does
shape_change = np.linalg.norm(current_landmarks - prev_landmarks, 'fro')
mean_change = shape_change / np.sqrt(len(current_landmarks))

print(f"\nTest convergence calculation:")
print(f"Total shape change (Frobenius norm): {shape_change:.4f}")
print(f"Mean per-landmark change: {mean_change:.4f}")
print(f"Would converge with threshold 0.1? {mean_change < 0.1}")
print(f"Would converge with old threshold 0.01? {shape_change < 0.01}")