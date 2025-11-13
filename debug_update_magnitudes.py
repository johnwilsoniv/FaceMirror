"""
Debug update magnitudes to find the scaling issue.
"""
import cv2
import numpy as np
from pyclnf import CLNF
import pyclnf.core.optimizer as opt_module

# Monkey-patch optimizer to print diagnostic information
original_optimize = opt_module.NURLMSOptimizer.optimize

def debug_optimize(self, pdm, initial_params, patch_experts, image, weights=None, window_size=11, patch_scaling=1.0):
    """Modified optimize with diagnostic output."""
    import numpy as np

    params = initial_params.copy()
    n_landmarks = pdm.n_points
    n_params = len(params)

    # Create weight matrix
    if weights is None:
        weights = np.ones(n_landmarks)

    if self.weight_multiplier > 0:
        W = self.weight_multiplier * np.diag(np.repeat(weights, 2))
    else:
        W = np.eye(n_landmarks * 2)

    # Create regularization matrix
    Lambda_inv = self._compute_lambda_inv(pdm, n_params)

    # Run ONLY ONE iteration to see magnitudes
    print(f"\n  Window {window_size} - Iteration 1 Diagnostics:")

    # Get current landmarks
    landmarks_2d = pdm.params_to_landmarks_2d(params)

    # Get reference shape
    reference_shape = pdm.get_reference_shape(patch_scaling, params[6:])

    # Compute similarity transform
    from pyclnf.core.utils import align_shapes_with_scale, invert_similarity_transform
    sim_img_to_ref = align_shapes_with_scale(landmarks_2d, reference_shape)
    sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

    # Compute mean-shift
    mean_shift = self._compute_mean_shift(
        landmarks_2d, patch_experts, image, pdm, window_size,
        sim_img_to_ref, sim_ref_to_img
    )

    # Compute Jacobian
    J = pdm.compute_jacobian(params)

    # Solve for parameter update
    delta_p = self._solve_update(J, mean_shift, W, Lambda_inv, params)

    # Print diagnostics
    print(f"    Mean-shift vector (v):")
    print(f"      Shape: {mean_shift.shape}")
    print(f"      Mean |v|: {np.abs(mean_shift).mean():.6f}")
    print(f"      Max |v|: {np.abs(mean_shift).max():.6f}")
    print(f"      Norm ||v||: {np.linalg.norm(mean_shift):.6f}")

    print(f"    Jacobian (J):")
    print(f"      Shape: {J.shape}")
    print(f"      Mean |J|: {np.abs(J).mean():.6f}")
    print(f"      Max |J|: {np.abs(J).max():.6f}")

    print(f"    Parameter update (Δp):")
    print(f"      Shape: {delta_p.shape}")
    print(f"      Scale Δp: {delta_p[0]:.6f}")
    print(f"      Rotation Δp: [{delta_p[1]:.6f}, {delta_p[2]:.6f}, {delta_p[3]:.6f}]")
    print(f"      Translation Δp: [{delta_p[4]:.3f}, {delta_p[5]:.3f}]")
    print(f"      Shape Δp (first 5): {delta_p[6:11]}")
    print(f"      Mean |Shape Δp|: {np.abs(delta_p[6:]).mean():.6f}")
    print(f"      Max |Shape Δp|: {np.abs(delta_p[6:]).max():.6f}")
    print(f"      Norm ||Δp||: {np.linalg.norm(delta_p):.6f}")

    # Update parameters
    params_updated = pdm.update_params(params, delta_p)
    params_updated = pdm.clamp_params(params_updated)

    # Compute shape change
    landmarks_new = pdm.params_to_landmarks_2d(params_updated)
    shape_change = np.linalg.norm(landmarks_new - landmarks_2d)

    print(f"    Resulting shape change: {shape_change:.3f} pixels")
    print(f"    Per-landmark change: {shape_change / np.sqrt(n_landmarks * 2):.3f} pixels")

    # Return dummy result (we only care about first iteration)
    info = {
        'converged': False,
        'iterations': 1,
        'final_update': np.linalg.norm(delta_p),
        'iteration_info': []
    }

    return params_updated, info

opt_module.NURLMSOptimizer.optimize = debug_optimize

# Load test data
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read frame from {video_path}")

face_bbox = (241, 555, 532, 532)

print("=" * 80)
print("Debugging Update Magnitudes - First Iteration Only")
print("=" * 80)

clnf = CLNF(model_dir="pyclnf/models", max_iterations=1)  # Only 1 iteration
landmarks, info = clnf.fit(frame, face_bbox)

print("\n" + "=" * 80)
print("Analysis:")
print("  OpenFace typical values:")
print("    - Mean-shift: ~0.1-1.0 pixels")
print("    - Shape change: 0.01-0.1 pixels per iteration")
print("  ")
print("  If PyCLNF values are 10-100× larger, we have a scaling bug!")
print("=" * 80)
