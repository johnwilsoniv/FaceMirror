#!/usr/bin/env python3
"""
CalcParams using cv2.solvePnP (MediaPipe-style approach)

Simplified alternative to iterative Gauss-Newton optimization.
Decouples pose and shape estimation for better performance and debuggability.
"""

import numpy as np
import cv2
from scipy import linalg

class CalcParamsPnP:
    """
    Estimate 3D pose and shape parameters using cv2.solvePnP

    This is a simpler, faster alternative to the full Gauss-Newton optimization.
    Inspired by MediaPipe's approach.
    """

    def __init__(self, pdm_parser):
        """
        Initialize with PDM model

        Args:
            pdm_parser: PDMParser instance
        """
        self.mean_shape = pdm_parser.mean_shape  # (204, 1)
        self.princ_comp = pdm_parser.princ_comp  # (204, 34)
        self.eigen_values = pdm_parser.eigen_values  # (34,)

        # Reshape mean shape to 3D points (68 x 3)
        self.mean_shape_3d = self.mean_shape.reshape(68, 3)

        # Simple camera model (weak perspective approximation)
        # Focal length = image width (rough heuristic)
        self.focal_length = 1000.0
        self.camera_center = (512.0, 512.0)  # Assume 1024x1024 images

        self.camera_matrix = np.array([
            [self.focal_length, 0, self.camera_center[0]],
            [0, self.focal_length, self.camera_center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros(4, dtype=np.float32)  # No distortion

    def calc_params(self, landmarks_2d, max_shape_iterations=10):
        """
        Estimate pose and shape parameters

        Strategy:
        1. Use mean shape to get initial pose via solvePnP
        2. Refine shape parameters given pose
        3. (Optional) Iterate 1-2 times for better accuracy

        Args:
            landmarks_2d: (136,) array [x0...x67, y0...y67]
            max_shape_iterations: Number of refinement iterations

        Returns:
            params_global: (6,) [scale, rx, ry, rz, tx, ty]
            params_local: (34,) shape parameters
        """
        # Reshape landmarks to (68, 2)
        landmarks_2d = landmarks_2d.astype(np.float32)
        landmarks_2d_pts = np.zeros((68, 2), dtype=np.float32)
        landmarks_2d_pts[:, 0] = landmarks_2d[:68]   # X coordinates
        landmarks_2d_pts[:, 1] = landmarks_2d[68:]   # Y coordinates

        # Initialize shape parameters to zero (mean shape)
        params_local = np.zeros(34, dtype=np.float32)

        # Iterative refinement
        for iteration in range(max_shape_iterations):
            # 1. Compute current 3D shape
            if iteration == 0:
                # First iteration: use mean shape
                shape_3d = self.mean_shape_3d.copy()
            else:
                # Subsequent iterations: use current shape estimate
                shape_3d = self.calc_shape_3d(params_local)

            # 2. Solve for pose using PnP
            success, rvec, tvec = cv2.solvePnP(
                shape_3d,           # 3D model points
                landmarks_2d_pts,   # 2D observations
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE  # Iterative for accuracy
            )

            if not success:
                # Fallback to EPNP if iterative fails
                success, rvec, tvec = cv2.solvePnP(
                    shape_3d, landmarks_2d_pts,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP
                )

            if not success:
                # Return default parameters if PnP fails
                return self._get_default_params()

            # 3. Convert rvec to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # 4. Project 3D shape to 2D using estimated pose
            projected_2d, _ = cv2.projectPoints(
                shape_3d, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            projected_2d = projected_2d.reshape(68, 2)

            # 5. Compute residual
            residual = landmarks_2d_pts - projected_2d
            residual_norm = np.linalg.norm(residual)

            # 6. Refine shape parameters
            # Solve: landmarks_2d ≈ Project(mean_shape + princ_comp @ params_local)
            # This is a linear least squares problem given fixed pose
            params_local = self._refine_shape_params(
                landmarks_2d_pts, R, tvec, params_local
            )

            # Early stopping if residual is small
            if residual_norm < 1.0:  # < 1 pixel error
                break

        # Convert to OpenFace parameter format
        params_global = self._pnp_to_openface_params(rvec, tvec, landmarks_2d_pts)

        return params_global, params_local

    def calc_shape_3d(self, params_local):
        """Compute 3D shape from parameters"""
        # shape = mean + princ_comp @ params
        shape_deform = (self.princ_comp @ params_local.reshape(-1, 1)).flatten()
        shape_3d = (self.mean_shape.flatten() + shape_deform).reshape(68, 3)
        return shape_3d.astype(np.float32)

    def _refine_shape_params(self, landmarks_2d_pts, R, tvec, params_local_init):
        """
        Refine shape parameters given fixed pose

        Solve: landmarks_2d ≈ Project(mean_shape + princ_comp @ params_local)

        This is much simpler than joint optimization!
        """
        # Compute Jacobian of shape deformation w.r.t. local parameters
        # J[i,j] = ∂(projection of landmark i) / ∂(local param j)

        n_landmarks = 68
        n_params = 34

        # For simplicity, use a linear approximation
        # (Could be improved with proper Jacobian, but this is fast)

        # Project mean shape
        mean_3d = self.mean_shape_3d
        mean_projected, _ = cv2.projectPoints(
            mean_3d,
            cv2.Rodrigues(R)[0],  # Convert R to rvec
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        mean_projected = mean_projected.reshape(68, 2)

        # Residual: difference between observed and mean projection
        residual = landmarks_2d_pts - mean_projected
        residual_vec = residual.flatten()  # (136,)

        # Approximate Jacobian using finite differences
        # (In production, compute analytically for speed)
        J = np.zeros((136, n_params), dtype=np.float32)
        epsilon = 0.01

        for j in range(n_params):
            # Perturb parameter j
            params_perturbed = params_local_init.copy()
            params_perturbed[j] += epsilon

            # Compute perturbed shape
            shape_perturbed = self.calc_shape_3d(params_perturbed)

            # Project perturbed shape
            projected_perturbed, _ = cv2.projectPoints(
                shape_perturbed,
                cv2.Rodrigues(R)[0],
                tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            projected_perturbed = projected_perturbed.reshape(68, 2)

            # Finite difference: (f(x+ε) - f(x)) / ε
            diff = (projected_perturbed - mean_projected) / epsilon
            J[:, j] = diff.flatten()

        # Solve least squares: minimize ||J @ params_local - residual_vec||^2
        # Add regularization (inverse eigenvalues)
        regularization = np.diag(1.0 / self.eigen_values)

        # Regularized least squares
        JtJ = J.T @ J + regularization
        Jtr = J.T @ residual_vec

        try:
            params_local_new = linalg.solve(JtJ, Jtr, assume_a='pos')
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            params_local_new = np.linalg.lstsq(JtJ, Jtr, rcond=1e-6)[0]

        return params_local_new.astype(np.float32)

    def _pnp_to_openface_params(self, rvec, tvec, landmarks_2d_pts):
        """
        Convert cv2.solvePnP output to OpenFace parameter format

        OpenFace params: [scale, rx, ry, rz, tx, ty]
        PnP output: rvec (3,), tvec (3,)

        Need to estimate scale from landmarks
        """
        # Rotation: rvec is already in axis-angle, convert to Euler
        R, _ = cv2.Rodrigues(rvec)
        euler = self._rotation_matrix_to_euler(R)

        # Translation: use X and Y from tvec
        tx = tvec[0, 0]
        ty = tvec[1, 0]

        # Scale: estimate from bounding box
        # (Rough heuristic: compare 2D bbox size to mean 3D bbox size)
        bbox_2d = np.array([
            landmarks_2d_pts[:, 0].min(), landmarks_2d_pts[:, 1].min(),
            landmarks_2d_pts[:, 0].max(), landmarks_2d_pts[:, 1].max()
        ])
        bbox_width = bbox_2d[2] - bbox_2d[0]
        bbox_height = bbox_2d[3] - bbox_2d[1]
        bbox_size = np.sqrt(bbox_width**2 + bbox_height**2)

        # Mean shape bbox size (rough estimate)
        mean_bbox_size = 200.0  # Typical face size in 3D model
        scale = bbox_size / mean_bbox_size

        params_global = np.array([
            scale, euler[0], euler[1], euler[2], tx, ty
        ], dtype=np.float32)

        return params_global

    def _rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles (same as CalcParams)"""
        # Use quaternion intermediate for stability
        q0 = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2.0
        q1 = (R[2,1] - R[1,2]) / (4.0 * q0)
        q2 = (R[0,2] - R[2,0]) / (4.0 * q0)
        q3 = (R[1,0] - R[0,1]) / (4.0 * q0)

        # Quaternion to Euler
        t1 = 2.0 * (q0*q2 + q1*q3)
        t1 = np.clip(t1, -1.0, 1.0)

        yaw = np.arcsin(t1)
        pitch = np.arctan2(2.0 * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3)
        roll = np.arctan2(2.0 * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3)

        return np.array([pitch, yaw, roll], dtype=np.float32)

    def _get_default_params(self):
        """Return default parameters if optimization fails"""
        params_global = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        params_local = np.zeros(34, dtype=np.float32)
        return params_global, params_local


# Example usage
if __name__ == '__main__':
    from pdm_parser import PDMParser
    import pandas as pd

    print("Testing CalcParamsPnP vs CalcParams...")

    # Load PDM
    pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")

    # Load test data
    df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

    # Get landmarks for frame 0
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x_landmarks = df.loc[0, x_cols].values.astype(np.float32)
    y_landmarks = df.loc[0, y_cols].values.astype(np.float32)
    landmarks_2d = np.concatenate([x_landmarks, y_landmarks])

    # Test PnP version
    import time
    calc_pnp = CalcParamsPnP(pdm)

    start = time.perf_counter()
    params_global_pnp, params_local_pnp = calc_pnp.calc_params(landmarks_2d)
    elapsed_pnp = time.perf_counter() - start

    print(f"\nPnP version:")
    print(f"  Time: {elapsed_pnp*1000:.2f} ms")
    print(f"  Pose params: {params_global_pnp}")
    print(f"  Shape params (first 5): {params_local_pnp[:5]}")

    # Compare with original
    from calc_params import CalcParams
    calc_orig = CalcParams(pdm)

    start = time.perf_counter()
    params_global_orig, params_local_orig = calc_orig.calc_params(landmarks_2d)
    elapsed_orig = time.perf_counter() - start

    print(f"\nOriginal version:")
    print(f"  Time: {elapsed_orig*1000:.2f} ms")
    print(f"  Pose params: {params_global_orig}")
    print(f"  Shape params (first 5): {params_local_orig[:5]}")

    print(f"\nSpeedup: {elapsed_orig/elapsed_pnp:.2f}x")
