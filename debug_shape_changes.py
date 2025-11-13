"""
Debug shape changes per iteration to understand why convergence isn't happening.
"""
import cv2
import numpy as np
from pyclnf import CLNF

# Temporarily modify optimizer to print shape changes
import pyclnf.core.optimizer as opt_module

# Monkey-patch the optimize method to add debug output
original_optimize = opt_module.NURLMSOptimizer.optimize

def debug_optimize(self, pdm, initial_params, patch_experts, image, weights=None, window_size=11, patch_scaling=1.0):
    """Modified optimize with shape change debugging."""
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

    # Optimization loop with debug output
    previous_landmarks = None
    converged = False
    iteration_info = []

    print(f"\n  Starting optimization with window_size={window_size}")

    for iteration in range(self.max_iterations):
        # Get current landmarks
        landmarks_2d = pdm.params_to_landmarks_2d(params)

        # Check shape-based convergence
        if previous_landmarks is not None:
            shape_change = np.linalg.norm(landmarks_2d - previous_landmarks)
            per_landmark_change = shape_change / np.sqrt(n_landmarks * 2)
            print(f"    Iter {iteration}: shape_change={shape_change:.6f}, per_landmark={per_landmark_change:.6f}")

            if shape_change < 0.01:
                print(f"    Converged! (shape_change={shape_change:.6f} < 0.01)")
                converged = True
                break

        previous_landmarks = landmarks_2d.copy()

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

        # Update parameters
        params = pdm.update_params(params, delta_p)
        params = pdm.clamp_params(params)

        # Check parameter convergence
        update_magnitude = np.linalg.norm(delta_p)
        iteration_info.append({
            'iteration': iteration,
            'update_magnitude': update_magnitude,
            'params': params.copy()
        })

        if update_magnitude < self.convergence_threshold:
            print(f"    Converged! (update_magnitude={update_magnitude:.6f} < {self.convergence_threshold})")
            converged = True
            break

    # Return results
    info = {
        'converged': converged,
        'iterations': len(iteration_info),
        'final_update': iteration_info[-1]['update_magnitude'] if iteration_info else 0,
        'iteration_info': iteration_info
    }

    return params, info

opt_module.NURLMSOptimizer.optimize = debug_optimize

# Now run the test
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read frame from {video_path}")

face_bbox = (241, 555, 532, 532)

print("=" * 80)
print("Debugging Shape Changes Per Iteration")
print("=" * 80)

clnf = CLNF(model_dir="pyclnf/models", max_iterations=10)
landmarks, info = clnf.fit(frame, face_bbox, return_params=True)

print("\n" + "=" * 80)
print(f"Final result: converged={info['converged']}, total_iterations={info['iterations']}")
print("=" * 80)
