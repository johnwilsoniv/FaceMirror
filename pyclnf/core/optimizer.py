"""
NU-RLMS Optimizer - Parameter optimization for CLNF

Implements the Normalized Unconstrained Regularized Least Mean Squares optimizer
used in OpenFace CLNF for fitting the Point Distribution Model to detected landmarks.

The optimizer minimizes:
    E(p) = ||v - J·Δp||² + λ||Λ^(-1/2)·Δp||²

Where:
    - p: Current parameter vector [scale, tx, ty, wx, wy, wz, q0, ..., qm]
    - Δp: Parameter update
    - v: Mean-shift vector (from patch expert responses)
    - J: Jacobian matrix (∂landmarks/∂params)
    - λ: Regularization weight
    - Λ: Diagonal matrix of parameter variances (eigenvalues for shape params)

Update rule:
    Δp = (J^T·W·J + λ·Λ^(-1))^(-1) · (J^T·W·v - λ·Λ^(-1)·p)

Where W is a diagonal weight matrix (typically identity for uniform weighting).
"""

import numpy as np
from typing import Tuple, Optional
import cv2


class NURLMSOptimizer:
    """
    NU-RLMS optimizer for CLNF parameter estimation.

    This optimizer iteratively refines the PDM parameters to fit detected landmarks
    using patch expert responses and shape model constraints.
    """

    def __init__(self,
                 regularization: float = 1.0,
                 max_iterations: int = 10,
                 convergence_threshold: float = 0.01):
        """
        Initialize NU-RLMS optimizer.

        Args:
            regularization: Regularization weight λ (higher = stronger shape prior)
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for parameter change
        """
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def optimize(self,
                 pdm,
                 initial_params: np.ndarray,
                 patch_experts: dict,
                 image: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Optimize PDM parameters to fit landmarks to image.

        Args:
            pdm: PDM instance with compute_jacobian and params_to_landmarks methods
            initial_params: Initial parameter guess [s, tx, ty, wx, wy, wz, q...]
            patch_experts: Dict mapping landmark_idx -> CCNFPatchExpert
            image: Grayscale image to fit to
            weights: Optional per-landmark weights (default: uniform)

        Returns:
            optimized_params: Optimized parameter vector
            info: Dictionary with optimization info (iterations, convergence, etc.)
        """
        params = initial_params.copy()
        n_params = len(params)
        n_landmarks = pdm.n_points

        # Initialize weights (default: uniform)
        if weights is None:
            weights = np.ones(n_landmarks)

        # Create diagonal weight matrix W for 2D landmarks (2n × 2n)
        W = np.diag(np.repeat(weights, 2))

        # Create regularization matrix Λ^(-1)
        Lambda_inv = self._compute_lambda_inv(pdm, n_params)

        # Optimization loop
        iteration_info = []
        converged = False

        for iteration in range(self.max_iterations):
            # 1. Get current landmark positions
            landmarks_2d = pdm.params_to_landmarks_2d(params)

            # 2. Compute mean-shift vector from patch responses
            mean_shift = self._compute_mean_shift(
                landmarks_2d, patch_experts, image, pdm
            )

            # 3. Compute Jacobian at current parameters
            J = pdm.compute_jacobian(params)

            # 4. Solve for parameter update: Δp
            delta_p = self._solve_update(J, mean_shift, W, Lambda_inv, params)

            # 5. Update parameters
            params = params + delta_p

            # 6. Clamp shape parameters to valid range
            params = pdm.clamp_shape_params(params)

            # 7. Check convergence
            update_magnitude = np.linalg.norm(delta_p)
            iteration_info.append({
                'iteration': iteration,
                'update_magnitude': update_magnitude,
                'params': params.copy()
            })

            if update_magnitude < self.convergence_threshold:
                converged = True
                break

        # Return optimized parameters and info
        info = {
            'converged': converged,
            'iterations': len(iteration_info),
            'final_update': iteration_info[-1]['update_magnitude'] if iteration_info else 0.0,
            'iteration_history': iteration_info
        }

        return params, info

    def _compute_lambda_inv(self, pdm, n_params: int) -> np.ndarray:
        """
        Compute inverse regularization matrix Λ^(-1).

        For rigid parameters (scale, translation, rotation): no regularization (set to 0)
        For shape parameters: use inverse eigenvalues

        Args:
            pdm: PDM instance
            n_params: Total number of parameters

        Returns:
            Lambda_inv: Diagonal matrix (n_params,)
        """
        Lambda_inv = np.zeros(n_params)

        # No regularization for rigid parameters (indices 0-5)
        # These are: scale, tx, ty, wx, wy, wz
        Lambda_inv[:6] = 0.0

        # Shape parameters (indices 6+): use inverse eigenvalues
        eigenvalues = pdm.eigen_values.flatten()
        Lambda_inv[6:] = 1.0 / (eigenvalues + 1e-8)  # Add small epsilon for stability

        return Lambda_inv

    def _compute_mean_shift(self,
                           landmarks_2d: np.ndarray,
                           patch_experts: dict,
                           image: np.ndarray,
                           pdm) -> np.ndarray:
        """
        Compute mean-shift vector from patch expert responses.

        The mean-shift vector indicates the direction each landmark should move
        to better fit the image based on patch expert responses.

        Args:
            landmarks_2d: Current 2D landmark positions (n_points, 2)
            patch_experts: Dict mapping landmark_idx -> CCNFPatchExpert
            image: Grayscale image
            pdm: PDM instance

        Returns:
            mean_shift: Mean-shift vector, shape (2 * n_points,)
        """
        n_points = landmarks_2d.shape[0]
        mean_shift = np.zeros(2 * n_points)

        # For each landmark with a patch expert
        for landmark_idx, patch_expert in patch_experts.items():
            if landmark_idx >= n_points:
                continue

            # Get current landmark position
            lm_x, lm_y = landmarks_2d[landmark_idx]

            # Sample responses in a small neighborhood around current position
            # to estimate gradient of response function
            search_radius = 3  # pixels
            best_response = -np.inf
            best_offset = np.array([0.0, 0.0])

            # Extract patch at current position
            patch = self._extract_patch(
                image, int(lm_x), int(lm_y),
                patch_expert.width, patch_expert.height
            )

            if patch is not None:
                center_response = patch_expert.compute_response(patch)

                # Sample neighborhood
                for dx in range(-search_radius, search_radius + 1):
                    for dy in range(-search_radius, search_radius + 1):
                        if dx == 0 and dy == 0:
                            continue

                        test_x = int(lm_x + dx)
                        test_y = int(lm_y + dy)

                        test_patch = self._extract_patch(
                            image, test_x, test_y,
                            patch_expert.width, patch_expert.height
                        )

                        if test_patch is not None:
                            response = patch_expert.compute_response(test_patch)

                            if response > best_response:
                                best_response = response
                                best_offset = np.array([float(dx), float(dy)])

                # If we found a better response, set mean-shift toward it
                if best_response > center_response:
                    mean_shift[2 * landmark_idx] = best_offset[0]
                    mean_shift[2 * landmark_idx + 1] = best_offset[1]

        return mean_shift

    def _extract_patch(self,
                      image: np.ndarray,
                      center_x: int,
                      center_y: int,
                      patch_width: int,
                      patch_height: int) -> Optional[np.ndarray]:
        """
        Extract image patch centered at (center_x, center_y).

        Args:
            image: Source image
            center_x, center_y: Patch center coordinates
            patch_width, patch_height: Patch dimensions

        Returns:
            patch: Extracted patch, or None if out of bounds
        """
        half_w = patch_width // 2
        half_h = patch_height // 2

        # Compute patch bounds
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + patch_width
        y2 = y1 + patch_height

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            return None

        # Extract patch
        patch = image[y1:y2, x1:x2]

        return patch

    def _solve_update(self,
                     J: np.ndarray,
                     v: np.ndarray,
                     W: np.ndarray,
                     Lambda_inv: np.ndarray,
                     params: np.ndarray) -> np.ndarray:
        """
        Solve for parameter update using NU-RLMS equation.

        Solves: (J^T·W·J + λ·Λ^(-1))·Δp = J^T·W·v - λ·Λ^(-1)·p

        Args:
            J: Jacobian matrix (2n, m)
            v: Mean-shift vector (2n,)
            W: Weight matrix (2n, 2n)
            Lambda_inv: Inverse regularization matrix (m,)
            params: Current parameters (m,)

        Returns:
            delta_p: Parameter update (m,)
        """
        # Compute left-hand side: A = J^T·W·J + λ·Λ^(-1)
        JtWJ = J.T @ W @ J  # (m, m)
        Lambda_inv_diag = np.diag(self.regularization * Lambda_inv)  # (m, m)
        A = JtWJ + Lambda_inv_diag

        # Compute right-hand side: b = J^T·W·v - λ·Λ^(-1)·p
        JtWv = J.T @ W @ v  # (m,)
        reg_term = self.regularization * Lambda_inv * params  # (m,)
        b = JtWv - reg_term

        # Solve linear system: A·Δp = b
        try:
            delta_p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            delta_p = np.linalg.lstsq(A, b, rcond=None)[0]

        return delta_p


def test_optimizer():
    """Test NU-RLMS optimizer."""
    print("=" * 60)
    print("Testing NU-RLMS Optimizer")
    print("=" * 60)

    # Import dependencies
    from pyclnf.core.pdm import PDM
    from pyclnf.core.patch_expert import CCNFPatchExpert

    # Test 1: Load PDM
    print("\nTest 1: Initialize optimizer and PDM")
    pdm = PDM("pyclnf/models/exported_pdm")
    optimizer = NURLMSOptimizer(
        regularization=1.0,
        max_iterations=5,
        convergence_threshold=0.1
    )
    print(f"  PDM loaded: {pdm.n_points} landmarks, {pdm.n_params} params")
    print(f"  Optimizer: max_iter={optimizer.max_iterations}, λ={optimizer.regularization}")

    # Test 2: Initialize parameters
    print("\nTest 2: Initialize parameters from bbox")
    bbox = (100, 100, 200, 250)
    initial_params = pdm.init_params(bbox)
    print(f"  Initial params shape: {initial_params.shape}")
    print(f"  Scale: {initial_params[0]:.3f}")
    print(f"  Translation: ({initial_params[1]:.1f}, {initial_params[2]:.1f})")

    # Test 3: Create synthetic test image
    print("\nTest 3: Create test scenario")
    test_image = np.random.randint(0, 256, (400, 400), dtype=np.uint8)

    # Load a few patch experts for testing
    from pathlib import Path
    patch_experts = {}
    for landmark_idx in [30, 36, 45]:  # Nose tip, eye corners
        patch_dir = Path(f"pyclnf/models/exported_ccnf_0.25/view_00/patch_{landmark_idx:02d}")
        if patch_dir.exists():
            try:
                patch_experts[landmark_idx] = CCNFPatchExpert(str(patch_dir))
            except:
                pass

    print(f"  Test image: {test_image.shape}")
    print(f"  Loaded {len(patch_experts)} patch experts")

    # Test 4: Test mean-shift computation
    print("\nTest 4: Compute mean-shift vector")
    landmarks_2d = pdm.params_to_landmarks_2d(initial_params)
    mean_shift = optimizer._compute_mean_shift(
        landmarks_2d, patch_experts, test_image, pdm
    )
    print(f"  Mean-shift shape: {mean_shift.shape}")
    print(f"  Mean-shift magnitude: {np.linalg.norm(mean_shift):.3f}")
    print(f"  Non-zero elements: {np.count_nonzero(mean_shift)}")

    # Test 5: Test update computation
    print("\nTest 5: Compute parameter update")
    J = pdm.compute_jacobian(initial_params)
    W = np.eye(2 * pdm.n_points)
    Lambda_inv = optimizer._compute_lambda_inv(pdm, pdm.n_params)

    delta_p = optimizer._solve_update(J, mean_shift, W, Lambda_inv, initial_params)
    print(f"  Delta params shape: {delta_p.shape}")
    print(f"  Update magnitude: {np.linalg.norm(delta_p):.6f}")
    print(f"  Max update component: {np.abs(delta_p).max():.6f}")

    # Test 6: Run full optimization
    print("\nTest 6: Run optimization loop")
    optimized_params, info = optimizer.optimize(
        pdm, initial_params, patch_experts, test_image
    )

    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final update: {info['final_update']:.6f}")
    print(f"  Parameter change: {np.linalg.norm(optimized_params - initial_params):.6f}")

    # Test 7: Verify optimized landmarks
    print("\nTest 7: Compare initial vs optimized landmarks")
    initial_landmarks = pdm.params_to_landmarks_2d(initial_params)
    optimized_landmarks = pdm.params_to_landmarks_2d(optimized_params)

    landmark_shift = np.linalg.norm(optimized_landmarks - initial_landmarks, axis=1)
    print(f"  Mean landmark shift: {landmark_shift.mean():.3f} pixels")
    print(f"  Max landmark shift: {landmark_shift.max():.3f} pixels")
    print(f"  Landmarks moved > 1px: {np.sum(landmark_shift > 1.0)}")

    print("\n" + "=" * 60)
    print("✓ NU-RLMS Optimizer Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_optimizer()
