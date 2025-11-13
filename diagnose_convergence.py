#!/usr/bin/env python3
"""
Detailed convergence diagnostic for PyCLNF.
Captures intermediate values to understand why convergence fails.
"""

import cv2
import numpy as np
from pyclnf import CLNF

# Load test frame
video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to load frame")
    exit(1)

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Face bbox
face_bbox = (241, 555, 532, 532)

print("="*80)
print("PyCLNF Convergence Diagnostic")
print("="*80)
print(f"Image size: {gray.shape}")
print(f"Face bbox: {face_bbox}")
print()

# Initialize CLNF
clnf = CLNF(model_dir='pyclnf/models', max_iterations=10)  # Fewer iterations for detailed analysis

# Monkey-patch the optimizer to capture intermediate values
original_optimize = clnf.optimizer.optimize
intermediate_data = []

def capturing_optimize(pdm, initial_params, patch_experts, image,
                      weights=None, window_size=11, patch_scaling=0.25,
                      sigma_components=None):
    """Wrapped optimize that captures intermediate values."""

    # Import needed functions
    from pyclnf.core.utils import align_shapes_with_scale, invert_similarity_transform

    # Convergence tracking
    converged = False
    iteration_info = []

    # Start with initial params
    params = initial_params.copy()

    # Use default window sizes progression
    window_sizes = [11, 9, 7]  # Removed ws=5 (no sigma components)
    max_iterations = 10  # Fixed for diagnostic purposes

    for iteration in range(max_iterations):
        # Determine window size based on iteration
        if iteration < 3:
            current_window_size = window_sizes[0]
        elif iteration < 7:
            current_window_size = window_sizes[1]
        elif iteration < 12:
            current_window_size = window_sizes[2]
        else:
            current_window_size = window_sizes[3]

        # Get current 2D landmarks
        landmarks_2d = pdm.params_to_landmarks_2d(params)

        # Compute similarity transforms
        patch_scaling = 0.25
        reference_shape = pdm.get_reference_shape(patch_scaling, params[6:])
        sim_img_to_ref = align_shapes_with_scale(landmarks_2d, reference_shape)
        sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

        # Compute mean-shift
        mean_shift = clnf.optimizer._compute_mean_shift(
            landmarks_2d, patch_experts, image, pdm, current_window_size,
            sim_img_to_ref, sim_ref_to_img, sigma_components
        )

        # Compute Jacobian
        J = pdm.compute_jacobian(params)

        # Compute weight matrix
        W = clnf.optimizer._compute_weights(landmarks_2d, patch_experts)

        # Compute Lambda_inv (regularization)
        Lambda_inv = clnf.optimizer._compute_lambda_inv(pdm, params)

        # Solve for parameter update
        delta_p = clnf.optimizer._solve_update(J, mean_shift, W, Lambda_inv, params)

        # Store intermediate data
        iter_data = {
            'iteration': iteration,
            'window_size': current_window_size,
            'mean_shift_magnitude': np.linalg.norm(mean_shift),
            'mean_shift_max': np.abs(mean_shift).max(),
            'mean_shift_mean': np.abs(mean_shift).mean(),
            'delta_p_magnitude': np.linalg.norm(delta_p),
            'delta_p_max': np.abs(delta_p).max(),
            'delta_p': delta_p.copy(),
            'params': params.copy(),
            'mean_shift': mean_shift.copy(),
            'W_mean': np.mean(np.diag(W)),
            'W_min': np.min(np.diag(W)),
            'W_max': np.max(np.diag(W)),
        }
        iteration_info.append(iter_data)

        # Update parameters
        params = pdm.update_params(params, delta_p)
        params = pdm.clamp_params(params)

        # Check convergence
        update_magnitude = np.linalg.norm(delta_p)
        convergence_threshold = 0.005  # Default threshold
        if update_magnitude < convergence_threshold:
            converged = True
            break

    info = {
        'converged': converged,
        'iterations': len(iteration_info),
        'final_update': iteration_info[-1]['delta_p_magnitude'] if iteration_info else 0.0,
        'iteration_history': iteration_info
    }

    return params, info

# Replace optimize method
clnf.optimizer.optimize = capturing_optimize

# Run CLNF
print("Running CLNF with detailed diagnostics...")
landmarks, info = clnf.fit(gray, face_bbox, return_params=True)

print(f"\nConverged: {info['converged']}")
print(f"Iterations: {info['iterations']}")
print(f"Final update: {info['final_update']:.6f}")
print()

# Analyze iteration history
print("="*80)
print("Iteration Analysis")
print("="*80)
print(f"{'Iter':<6} {'WinSz':<6} {'MeanShift':<12} {'DeltaP':<12} {'W_mean':<10} {'W_range':<15}")
print("-"*80)

for iter_data in info['iteration_history']:
    w_range = f"{iter_data['W_min']:.2e}-{iter_data['W_max']:.2e}"
    print(f"{iter_data['iteration']:<6} "
          f"{iter_data['window_size']:<6} "
          f"{iter_data['mean_shift_magnitude']:<12.4f} "
          f"{iter_data['delta_p_magnitude']:<12.4f} "
          f"{iter_data['W_mean']:<10.4f} "
          f"{w_range:<15}")

print()
print("="*80)
print("Parameter Evolution")
print("="*80)

# Look at first and last iteration
first = info['iteration_history'][0]
last = info['iteration_history'][-1]

print("\nFirst iteration (params):")
print(f"  Scale: {first['params'][0]:.4f}")
print(f"  Rotation: [{first['params'][1]:.4f}, {first['params'][2]:.4f}, {first['params'][3]:.4f}]")
print(f"  Translation: [{first['params'][4]:.4f}, {first['params'][5]:.4f}]")
print(f"  Shape params (first 5): {first['params'][6:11]}")

print("\nLast iteration (params):")
print(f"  Scale: {last['params'][0]:.4f}")
print(f"  Rotation: [{last['params'][1]:.4f}, {last['params'][2]:.4f}, {last['params'][3]:.4f}]")
print(f"  Translation: [{last['params'][4]:.4f}, {last['params'][5]:.4f}]")
print(f"  Shape params (first 5): {last['params'][6:11]}")

print("\nParameter changes:")
param_change = last['params'] - first['params']
print(f"  Scale change: {param_change[0]:.4f}")
print(f"  Rotation change: [{param_change[1]:.4f}, {param_change[2]:.4f}, {param_change[3]:.4f}]")
print(f"  Translation change: [{param_change[4]:.4f}, {param_change[5]:.4f}]")
print(f"  Shape params change magnitude: {np.linalg.norm(param_change[6:]):.4f}")

print()
print("="*80)
print("Convergence Issues")
print("="*80)

# Check if mean-shift is decreasing
mean_shifts = [d['mean_shift_magnitude'] for d in info['iteration_history']]
print(f"\nMean-shift trend:")
print(f"  First: {mean_shifts[0]:.4f}")
print(f"  Last: {mean_shifts[-1]:.4f}")
print(f"  Change: {mean_shifts[-1] - mean_shifts[0]:.4f}")

if mean_shifts[-1] > mean_shifts[0]:
    print("  ⚠ WARNING: Mean-shift INCREASING (diverging!)")
elif mean_shifts[-1] > mean_shifts[0] * 0.5:
    print("  ⚠ WARNING: Mean-shift not decreasing fast enough")
else:
    print("  ✓ Mean-shift decreasing")

# Check if updates are decreasing
updates = [d['delta_p_magnitude'] for d in info['iteration_history']]
print(f"\nParameter update trend:")
print(f"  First: {updates[0]:.4f}")
print(f"  Last: {updates[-1]:.4f}")
print(f"  Change: {updates[-1] - updates[0]:.4f}")

if updates[-1] > updates[0]:
    print("  ⚠ WARNING: Updates INCREASING (diverging!)")
elif updates[-1] > updates[0] * 0.5:
    print("  ⚠ WARNING: Updates not decreasing fast enough")
else:
    print("  ✓ Updates decreasing")

# Check weight consistency
weights = [d['W_mean'] for d in info['iteration_history']]
print(f"\nWeight matrix trend:")
print(f"  First: {weights[0]:.4f}")
print(f"  Last: {weights[-1]:.4f}")
print(f"  Range: {min(weights):.4f} - {max(weights):.4f}")

print()
print("="*80)
print("Diagnostic Complete")
print("="*80)
