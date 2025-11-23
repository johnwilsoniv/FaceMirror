#!/usr/bin/env python3
"""
Compare exported Python eye PDM with C++ values.
"""

import sys
sys.path.insert(0, 'pyclnf')

import numpy as np
from pathlib import Path

def main():
    # Load left eye PDM
    pdm_dir = Path('pyclnf/models/exported_eye_pdm_left')

    mean_shape = np.load(pdm_dir / 'mean_shape.npy')
    eigenvectors = np.load(pdm_dir / 'eigenvectors.npy')
    eigenvalues = np.load(pdm_dir / 'eigenvalues.npy')

    print("=== Left Eye PDM ===")
    print(f"mean_shape: {mean_shape.shape}")
    print(f"eigenvectors: {eigenvectors.shape}")
    print(f"eigenvalues: {eigenvalues.shape}")

    # For 28 landmarks, mean_shape should be (84, 1) = 28*3
    n = mean_shape.shape[0] // 3
    print(f"\nn_points: {n}")
    print(f"n_modes: {eigenvectors.shape[1]}")

    # Extract 2D coordinates
    X = mean_shape[:n].flatten()
    Y = mean_shape[n:2*n].flatten()

    # Show the 6 visible landmarks that correspond to eyelid
    # Eye model indices [8, 10, 12, 14, 16, 18] are the visible ones
    eye_indices = [8, 10, 12, 14, 16, 18]

    print("\n=== Visible Landmark Positions (mean shape) ===")
    for i, idx in enumerate(eye_indices):
        print(f"Eye_{idx}: ({X[idx]:.6f}, {Y[idx]:.6f})")

    # Compute bounding box of visible landmarks
    vis_X = X[eye_indices]
    vis_Y = Y[eye_indices]

    bbox_width = np.max(vis_X) - np.min(vis_X)
    bbox_height = np.max(vis_Y) - np.min(vis_Y)

    print(f"\n=== Bounding Box (visible landmarks) ===")
    print(f"Width: {bbox_width:.6f}")
    print(f"Height: {bbox_height:.6f}")

    # Show eigenvalues
    print(f"\n=== Eigenvalues ===")
    print(eigenvalues.flatten())

    # Also check first eigenvector for eye_8
    print(f"\n=== First Eigenvector for Eye_8 ===")
    ev0 = eigenvectors[:, 0]
    print(f"X: {ev0[8]:.6f}")
    print(f"Y: {ev0[n + 8]:.6f}")
    print(f"Z: {ev0[2*n + 8]:.6f}")

if __name__ == '__main__':
    main()
