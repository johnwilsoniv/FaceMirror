#!/usr/bin/env python3
"""
Test the Jacobian fix to verify analytical rotation derivatives work correctly.
"""

import numpy as np
import cv2
from pathlib import Path
from pyclnf import CLNF

print("=" * 80)
print("Testing Jacobian Fix: Analytical Rotation Derivatives")
print("=" * 80)

# Test 1: Verify PDM Jacobian can be computed without errors
print("\nTest 1: PDM Jacobian Computation")
print("-" * 80)

clnf = CLNF(model_dir="pyclnf/models", max_iterations=5)

# Create test parameters
params = clnf.pdm.init_params(bbox=(100, 100, 200, 250))
print(f"Test parameters shape: {params.shape}")
print(f"Scale: {params[0]:.3f}, Rotation: [{params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}]")

# Compute Jacobian with the fixed analytical derivatives
try:
    J = clnf.pdm.compute_jacobian(params)
    print(f"✓ Jacobian computed successfully!")
    print(f"  Shape: {J.shape}")
    print(f"  Expected: ({2 * clnf.pdm.n_points}, {clnf.pdm.n_params}) = (136, 40)")

    # Check for NaN or Inf values
    if np.any(np.isnan(J)):
        print(f"  ⚠ WARNING: Jacobian contains NaN values")
    elif np.any(np.isinf(J)):
        print(f"  ⚠ WARNING: Jacobian contains Inf values")
    else:
        print(f"  ✓ No NaN/Inf values detected")

    # Print statistics for rotation derivative columns
    print(f"\n  Rotation derivative statistics:")
    for i, name in enumerate(['pitch (wx)', 'yaw (wy)', 'roll (wz)']):
        col = J[:, 1 + i]
        print(f"    {name}: min={col.min():.6f}, max={col.max():.6f}, mean={col.mean():.6f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Verify Jacobian is continuous (small changes in params → small changes in J)
print("\nTest 2: Jacobian Continuity Check")
print("-" * 80)

try:
    # Compute Jacobian at original params
    J1 = clnf.pdm.compute_jacobian(params)

    # Perturb rotation slightly
    params_perturbed = params.copy()
    params_perturbed[1] += 0.01  # Small change in pitch

    # Compute Jacobian at perturbed params
    J2 = clnf.pdm.compute_jacobian(params_perturbed)

    # Check difference
    diff = np.linalg.norm(J2 - J1)
    print(f"Jacobian difference for Δpitch=0.01: {diff:.6f}")

    if diff < 1.0:  # Should be small but non-zero
        print(f"✓ Jacobian is continuous (expected small difference)")
    else:
        print(f"⚠ WARNING: Jacobian difference seems large")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compare analytical vs numerical derivatives (sanity check)
print("\nTest 3: Analytical vs Numerical Derivative Comparison")
print("-" * 80)

try:
    # Use neutral pose for fair comparison
    params_neutral = clnf.pdm.init_params()

    # Compute analytical Jacobian
    J_analytical = clnf.pdm.compute_jacobian(params_neutral)

    # Compute numerical Jacobian for rotation parameters only
    h = 1e-6
    J_numerical_rotation = np.zeros((2 * clnf.pdm.n_points, 3))

    for i in range(3):  # For each rotation parameter
        params_plus = params_neutral.copy()
        params_plus[1 + i] += h
        landmarks_plus = clnf.pdm.params_to_landmarks_2d(params_plus)

        params_minus = params_neutral.copy()
        params_minus[1 + i] -= h
        landmarks_minus = clnf.pdm.params_to_landmarks_2d(params_minus)

        numerical_deriv = (landmarks_plus - landmarks_minus) / (2 * h)
        J_numerical_rotation[:, i] = numerical_deriv.flatten()

    # Compare
    J_analytical_rotation = J_analytical[:, 1:4]  # Columns 1-3 are rotation

    error = np.linalg.norm(J_analytical_rotation - J_numerical_rotation)
    rel_error = error / np.linalg.norm(J_numerical_rotation)

    print(f"Rotation derivatives comparison (neutral pose):")
    print(f"  Absolute error: {error:.6e}")
    print(f"  Relative error: {rel_error:.6e}")

    if rel_error < 1e-4:
        print(f"✓ Analytical derivatives match numerical (within tolerance)")
    else:
        print(f"⚠ WARNING: Analytical derivatives differ from numerical")
        print(f"  This is EXPECTED if rotation is non-zero (different linearizations)")
        print(f"  For neutral pose, should match closely")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Full CLNF optimization test
print("\nTest 4: CLNF Optimization Test")
print("-" * 80)

try:
    # Load test image
    video_path = Path("Patient Data/Normal Cohort/IMG_0434.MOV")
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # Set up face bbox
            face_size = min(width, height) // 2
            face_bbox = (
                width // 2 - face_size // 2,
                height // 2 - face_size // 2,
                face_size,
                face_size
            )

            print(f"Running CLNF optimization on {gray.shape} image...")

            # Run CLNF
            clnf_opt = CLNF(model_dir="pyclnf/models", max_iterations=10)
            landmarks, info = clnf_opt.fit(gray, face_bbox, return_params=True)

            print(f"\nOptimization Results:")
            print(f"  Converged: {info['converged']}")
            print(f"  Iterations: {info['iterations']}")
            print(f"  Final update magnitude: {info.get('final_update', 'N/A')}")
            print(f"  Final rotation: [{info['params'][1]:.3f}, {info['params'][2]:.3f}, {info['params'][3]:.3f}]")

            if info['converged']:
                print(f"  ✓ CLNF converged successfully with fixed Jacobian!")
            else:
                print(f"  ⚠ CLNF did not converge (may need more investigation)")

        else:
            print("⚠ Could not read video frame")
    else:
        print(f"⚠ Test video not found at {video_path}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)
