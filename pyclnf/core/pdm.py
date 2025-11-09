"""
Point Distribution Model (PDM) - Core shape model for CLNF

Implements the PDM transform from parameters to 3D landmarks:
    xi = s · R2D · (x̄i + Φiq) + t

Where:
    - x̄i: Mean position of landmark i
    - Φi: Principal component matrix for landmark i
    - q: Non-rigid shape parameters (PCA coefficients)
    - s: Global scale
    - t: Translation [tx, ty]
    - w: Orientation [wx, wy, wz] (axis-angle)
    - R2D: First two rows of 3×3 rotation matrix from w

Parameter vector: p = [s, tx, ty, wx, wy, wz, q0, q1, ..., qm]
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class PDM:
    """Point Distribution Model for facial landmark representation."""

    def __init__(self, model_dir: str):
        """
        Load PDM from exported NumPy files.

        Args:
            model_dir: Directory containing mean_shape.npy, princ_comp.npy, eigen_values.npy
        """
        self.model_dir = Path(model_dir)

        # Load PDM components
        self.mean_shape = np.load(self.model_dir / 'mean_shape.npy')  # (3n, 1)
        self.princ_comp = np.load(self.model_dir / 'princ_comp.npy')  # (3n, m)
        self.eigen_values = np.load(self.model_dir / 'eigen_values.npy')  # (1, m)

        # Extract dimensions
        self.n_points = self.mean_shape.shape[0] // 3  # Number of landmarks (68)
        self.n_modes = self.princ_comp.shape[1]  # Number of PCA modes (34)

        # Parameter vector size: scale(1) + translation(2) + rotation(3) + shape(n_modes)
        self.n_params = 6 + self.n_modes

    def params_to_landmarks_3d(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to 3D landmark positions.

        Args:
            params: Parameter vector [s, tx, ty, wx, wy, wz, q0, ..., qm]
                   Shape: (n_params,) or (n_params, 1)

        Returns:
            landmarks_3d: 3D landmark positions, shape (n_points, 3)
        """
        params = params.flatten()

        # Extract parameters
        s = params[0]  # Scale
        tx, ty = params[1], params[2]  # Translation
        wx, wy, wz = params[3], params[4], params[5]  # Rotation (axis-angle)
        q = params[6:]  # Shape parameters

        # Apply PCA: shape = mean + principal_components @ shape_params
        # mean_shape is (3n, 1), princ_comp is (3n, m), q is (m,)
        shape_3d = self.mean_shape.flatten() + self.princ_comp @ q  # (3n,)

        # Reshape to (n, 3) for easier manipulation
        shape_3d = shape_3d.reshape(self.n_points, 3)  # (n_points, 3)

        # Compute rotation matrix from axis-angle representation
        R = self._rodrigues(np.array([wx, wy, wz]))  # (3, 3)

        # Apply similarity transform: landmarks = s * R @ shape + t
        # R is (3, 3), shape_3d is (n, 3)
        # We want to rotate each point: result[i] = s * R @ shape_3d[i] + t
        landmarks_3d = s * (shape_3d @ R.T)  # (n, 3)

        # Add translation (only to x and y, z stays as is)
        landmarks_3d[:, 0] += tx
        landmarks_3d[:, 1] += ty

        return landmarks_3d

    def params_to_landmarks_2d(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to 2D landmark positions (x, y projection).

        Args:
            params: Parameter vector [s, tx, ty, wx, wy, wz, q0, ..., qm]

        Returns:
            landmarks_2d: 2D landmark positions, shape (n_points, 2)
        """
        landmarks_3d = self.params_to_landmarks_3d(params)
        return landmarks_3d[:, :2]  # Take only x, y coordinates

    def compute_jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of 2D landmarks with respect to parameters.

        The Jacobian J has shape (2*n_points, n_params) where:
            J[2*i, j] = ∂(landmark_i.x) / ∂param_j
            J[2*i+1, j] = ∂(landmark_i.y) / ∂param_j

        This is used in the NU-RLMS optimization update step.

        Args:
            params: Parameter vector [s, tx, ty, wx, wy, wz, q0, ..., qm]

        Returns:
            jacobian: Jacobian matrix, shape (2*n_points, n_params)
        """
        params = params.flatten()

        # Extract parameters
        s = params[0]
        tx, ty = params[1], params[2]
        wx, wy, wz = params[3], params[4], params[5]
        q = params[6:]

        # Compute 3D shape before rotation
        shape_3d = self.mean_shape.flatten() + self.princ_comp @ q  # (3n,)
        shape_3d = shape_3d.reshape(self.n_points, 3)  # (n_points, 3)

        # Compute rotation matrix
        w = np.array([wx, wy, wz])
        R = self._rodrigues(w)

        # Initialize Jacobian
        J = np.zeros((2 * self.n_points, self.n_params))

        # 1. Derivative w.r.t. scale (column 0)
        # ∂(s * R @ shape) / ∂s = R @ shape
        rotated_shape = shape_3d @ R.T  # (n_points, 3)
        J[0::2, 0] = rotated_shape[:, 0]  # x components
        J[1::2, 0] = rotated_shape[:, 1]  # y components

        # 2. Derivative w.r.t. translation (columns 1-2)
        # ∂(... + tx) / ∂tx = 1, ∂(... + ty) / ∂ty = 1
        J[0::2, 1] = 1.0  # ∂x/∂tx = 1
        J[1::2, 2] = 1.0  # ∂y/∂ty = 1

        # 3. Derivative w.r.t. rotation (columns 3-5)
        # ∂(s * R(w) @ shape) / ∂w = s * (∂R/∂w @ shape)
        # This requires computing ∂R/∂wx, ∂R/∂wy, ∂R/∂wz using Rodrigues derivatives
        dR_dw = self._rodrigues_derivatives(w)  # List of 3 matrices (3x3 each)

        for k in range(3):  # For wx, wy, wz
            dR = dR_dw[k]  # (3, 3)
            d_landmarks = s * (shape_3d @ dR.T)  # (n_points, 3)
            J[0::2, 3 + k] = d_landmarks[:, 0]  # x components
            J[1::2, 3 + k] = d_landmarks[:, 1]  # y components

        # 4. Derivative w.r.t. shape parameters (columns 6:)
        # ∂(s * R @ (mean + Φ @ q)) / ∂qi = s * R @ Φ[:, i]
        # Φ is (3n, m), we need to process each mode
        for i in range(self.n_modes):
            phi_i = self.princ_comp[:, i].reshape(self.n_points, 3)  # (n_points, 3)
            d_landmarks = s * (phi_i @ R.T)  # (n_points, 3)
            J[0::2, 6 + i] = d_landmarks[:, 0]  # x components
            J[1::2, 6 + i] = d_landmarks[:, 1]  # y components

        return J

    def _rodrigues_derivatives(self, w: np.ndarray) -> list:
        """
        Compute derivatives of Rodrigues rotation matrix w.r.t. rotation vector components.

        Returns ∂R/∂wx, ∂R/∂wy, ∂R/∂wz where R = rodrigues(w).

        Args:
            w: Axis-angle rotation vector [wx, wy, wz]

        Returns:
            List of 3 matrices: [∂R/∂wx, ∂R/∂wy, ∂R/∂wz], each shape (3, 3)

        Formula (from Rodrigues derivative):
            ∂R/∂wi = ∂θ/∂wi * (cos(θ)*K + sin(θ)*K²) + (I + sin(θ)*K + (1-cos(θ))*K²) * ∂K/∂wi

        Simplified for small angles or using numerical differentiation for stability.
        """
        theta = np.linalg.norm(w)

        if theta < 1e-10:
            # Small angle: R ≈ I + K(w)
            # ∂R/∂wi ≈ ∂K/∂wi
            dR_dw = []
            for i in range(3):
                e_i = np.zeros(3)
                e_i[i] = 1.0
                dR_dw.append(self._skew(e_i))
            return dR_dw

        # For larger angles, use the analytical derivative
        # k = w / θ (unit axis)
        k = w / theta

        # ∂θ/∂wi = wi / θ
        dtheta_dw = w / theta  # (3,)

        # ∂k/∂wi = (1/θ) * I - (wi / θ²) * w^T
        # This gets complex, so we'll use a more direct formulation

        # Using the formula: ∂R/∂w = K @ R (valid for exponential map)
        # More precisely: ∂R/∂wi = [ei]_x @ R where [ei]_x is skew of unit vector ei
        R = self._rodrigues(w)

        dR_dw = []
        for i in range(3):
            # Compute ∂R/∂wi using numerical differentiation for stability
            # (can be replaced with analytical form if needed for speed)
            h = 1e-7
            w_plus = w.copy()
            w_plus[i] += h
            R_plus = self._rodrigues(w_plus)

            dR = (R_plus - R) / h
            dR_dw.append(dR)

        return dR_dw

    def _rodrigues(self, w: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle rotation vector to rotation matrix using Rodrigues formula.

        Args:
            w: Axis-angle rotation vector [wx, wy, wz], shape (3,)

        Returns:
            R: 3×3 rotation matrix

        Formula:
            θ = ||w||
            k = w / θ (unit axis)
            R = I + sin(θ) * K + (1 - cos(θ)) * K²

        where K is the skew-symmetric matrix of k:
            K = [[ 0,  -kz,  ky],
                 [ kz,  0,  -kx],
                 [-ky,  kx,  0]]
        """
        theta = np.linalg.norm(w)

        if theta < 1e-10:
            # Small angle approximation: R ≈ I + K
            return np.eye(3) + self._skew(w)

        # Normalize to get unit axis
        k = w / theta

        # Skew-symmetric matrix of k
        K = self._skew(k)

        # Rodrigues formula: R = I + sin(θ)*K + (1-cos(θ))*K²
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        return R

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from vector.

        Args:
            v: Vector [vx, vy, vz]

        Returns:
            Skew-symmetric matrix:
                [[ 0,  -vz,  vy],
                 [ vz,  0,  -vx],
                 [-vy,  vx,  0]]
        """
        return np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0]
        ])

    def init_params(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Initialize parameter vector from face bounding box or to neutral pose.

        Args:
            bbox: Optional bounding box [x, y, width, height] to estimate initial scale/translation

        Returns:
            params: Initial parameter vector [s, tx, ty, wx, wy, wz, q0, ..., qm]
        """
        params = np.zeros(self.n_params)

        if bbox is not None:
            x, y, width, height = bbox

            # Estimate scale from bbox size
            # Assume mean face has ~200 pixel width at scale=1
            mean_face_width = 200.0
            params[0] = width / mean_face_width  # scale

            # Center translation at bbox center
            params[1] = x + width / 2  # tx
            params[2] = y + height / 2  # ty
        else:
            # Neutral initialization
            params[0] = 1.0  # scale = 1
            params[1] = 0.0  # tx = 0
            params[2] = 0.0  # ty = 0

        # Rotation = 0 (frontal face)
        params[3:6] = 0.0

        # Shape parameters = 0 (mean shape)
        params[6:] = 0.0

        return params

    def clamp_shape_params(self, params: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Clamp shape parameters to valid range based on eigenvalues.

        Constrains each shape parameter qi to ±n_std standard deviations:
            -n_std * sqrt(λi) <= qi <= n_std * sqrt(λi)

        Args:
            params: Parameter vector
            n_std: Number of standard deviations (typically 3.0)

        Returns:
            params: Clamped parameter vector
        """
        params = params.copy()

        # Extract shape parameters
        q = params[6:]

        # Compute bounds from eigenvalues
        std_devs = np.sqrt(self.eigen_values.flatten())
        lower_bounds = -n_std * std_devs
        upper_bounds = n_std * std_devs

        # Clamp
        q_clamped = np.clip(q, lower_bounds, upper_bounds)

        # Update params
        params[6:] = q_clamped

        return params

    def get_info(self) -> dict:
        """Get PDM information."""
        return {
            'n_points': self.n_points,
            'n_modes': self.n_modes,
            'n_params': self.n_params,
            'mean_shape_shape': self.mean_shape.shape,
            'princ_comp_shape': self.princ_comp.shape,
            'eigen_values_shape': self.eigen_values.shape,
        }


def test_pdm():
    """Test PDM implementation."""
    print("=" * 60)
    print("Testing PDM Core Implementation")
    print("=" * 60)

    # Load PDM
    model_dir = "pyclnf/models/exported_pdm"
    pdm = PDM(model_dir)

    print("\nPDM Info:")
    for key, value in pdm.get_info().items():
        print(f"  {key}: {value}")

    # Test 1: Neutral pose (mean shape)
    print("\nTest 1: Neutral pose (mean shape)")
    params_neutral = pdm.init_params()
    print(f"  Params shape: {params_neutral.shape}")
    print(f"  Params: scale={params_neutral[0]:.3f}, tx={params_neutral[1]:.3f}, ty={params_neutral[2]:.3f}")
    print(f"  Rotation: wx={params_neutral[3]:.3f}, wy={params_neutral[4]:.3f}, wz={params_neutral[5]:.3f}")
    print(f"  Shape params (first 5): {params_neutral[6:11]}")

    landmarks_3d = pdm.params_to_landmarks_3d(params_neutral)
    landmarks_2d = pdm.params_to_landmarks_2d(params_neutral)
    print(f"  3D landmarks shape: {landmarks_3d.shape}")
    print(f"  2D landmarks shape: {landmarks_2d.shape}")
    print(f"  First 3 landmarks (2D): {landmarks_2d[:3]}")

    # Test 2: Initialize from bbox
    print("\nTest 2: Initialize from bounding box")
    bbox = (100, 100, 200, 250)  # [x, y, width, height]
    params_bbox = pdm.init_params(bbox)
    print(f"  Bbox: {bbox}")
    print(f"  Params: scale={params_bbox[0]:.3f}, tx={params_bbox[1]:.3f}, ty={params_bbox[2]:.3f}")

    landmarks_2d_bbox = pdm.params_to_landmarks_2d(params_bbox)
    print(f"  2D landmarks center: ({np.mean(landmarks_2d_bbox[:, 0]):.1f}, {np.mean(landmarks_2d_bbox[:, 1]):.1f})")
    print(f"  Expected center: ({bbox[0] + bbox[2]/2:.1f}, {bbox[1] + bbox[3]/2:.1f})")

    # Test 3: Non-zero shape parameters
    print("\nTest 3: Varying shape parameters")
    params_shape = params_neutral.copy()
    params_shape[6] = 2.0  # First PCA mode
    params_shape[7] = -1.5  # Second PCA mode

    landmarks_varied = pdm.params_to_landmarks_2d(params_shape)
    diff = np.linalg.norm(landmarks_varied - landmarks_2d)
    print(f"  Modified first 2 shape params: {params_shape[6:8]}")
    print(f"  Difference from neutral: {diff:.3f} pixels")

    # Test 4: Rotation
    print("\nTest 4: Rotation")
    params_rot = params_neutral.copy()
    params_rot[4] = 0.3  # Yaw rotation (around y-axis)

    landmarks_rot = pdm.params_to_landmarks_2d(params_rot)
    print(f"  Yaw rotation: {params_rot[4]:.3f} radians ({np.degrees(params_rot[4]):.1f}°)")
    print(f"  First 3 landmarks (rotated): {landmarks_rot[:3]}")

    # Test 5: Shape parameter clamping
    print("\nTest 5: Shape parameter clamping")
    params_extreme = params_neutral.copy()
    params_extreme[6:11] = 100.0  # Extreme values
    print(f"  Before clamping (first 5): {params_extreme[6:11]}")

    params_clamped = pdm.clamp_shape_params(params_extreme)
    print(f"  After clamping (first 5): {params_clamped[6:11]}")
    print(f"  Eigenvalues (first 5): {np.sqrt(pdm.eigen_values.flatten()[:5])}")

    # Test 6: Jacobian computation
    print("\nTest 6: Jacobian computation")
    J = pdm.compute_jacobian(params_bbox)
    print(f"  Jacobian shape: {J.shape}")
    print(f"  Expected shape: ({2 * pdm.n_points}, {pdm.n_params}) = (136, 40)")

    # Verify Jacobian accuracy using numerical differentiation
    # Test a few parameters
    h = 1e-6
    errors = []

    for param_idx in [0, 1, 2, 6, 10]:  # Test scale, tx, ty, and two shape params
        # Compute numerical derivative
        params_plus = params_bbox.copy()
        params_plus[param_idx] += h
        landmarks_plus = pdm.params_to_landmarks_2d(params_plus)

        params_minus = params_bbox.copy()
        params_minus[param_idx] -= h
        landmarks_minus = pdm.params_to_landmarks_2d(params_minus)

        numerical_deriv = (landmarks_plus - landmarks_minus) / (2 * h)
        numerical_deriv_flat = numerical_deriv.flatten()  # (136,)

        # Get analytical derivative from Jacobian
        analytical_deriv = J[:, param_idx]  # (136,)

        # Compute error
        error = np.linalg.norm(numerical_deriv_flat - analytical_deriv)
        errors.append(error)

    print(f"  Jacobian verification errors (numerical vs analytical):")
    print(f"    Param 0 (scale): {errors[0]:.2e}")
    print(f"    Param 1 (tx): {errors[1]:.2e}")
    print(f"    Param 2 (ty): {errors[2]:.2e}")
    print(f"    Param 6 (shape 0): {errors[3]:.2e}")
    print(f"    Param 10 (shape 4): {errors[4]:.2e}")
    print(f"  Max error: {max(errors):.2e} (should be < 1e-4)")

    if max(errors) < 1e-4:
        print("  ✓ Jacobian accuracy verified!")
    else:
        print("  ⚠ Jacobian may have numerical issues")

    print("\n" + "=" * 60)
    print("✓ PDM Core Implementation Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_pdm()
