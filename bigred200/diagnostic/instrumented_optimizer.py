"""
Instrumented CLNF Optimizer for Diagnostic Data Collection

This subclass of NURLMSOptimizer captures detailed per-iteration, per-landmark
diagnostic data for root cause analysis of landmark detection errors.

Key data captured:
- Response maps for each landmark at each iteration
- Mean-shift vectors in both reference and image coordinates
- Boundary clamping events
- Response map quality metrics (sharpness, peak offset)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pyclnf'))

from pyclnf.core.optimizer import NURLMSOptimizer
from .data_structures import IterationDiagnostic, LandmarkDiagnostic


class InstrumentedNURLMSOptimizer(NURLMSOptimizer):
    """
    Optimizer with full diagnostic instrumentation.

    Captures per-iteration, per-landmark data for root cause analysis.
    """

    def __init__(self,
                 capture_response_maps: bool = True,
                 target_landmarks: Optional[List[int]] = None,
                 cpp_reference: Optional[np.ndarray] = None,
                 **kwargs):
        """
        Initialize instrumented optimizer.

        Args:
            capture_response_maps: Whether to save full response maps (expensive)
            target_landmarks: Specific landmarks to track (None = all 68)
            cpp_reference: C++ reference landmarks for this frame (68, 2)
            **kwargs: Arguments passed to NURLMSOptimizer
        """
        super().__init__(**kwargs)

        self.capture_response_maps = capture_response_maps
        # Default: track representative landmarks from each region
        self.target_landmarks = target_landmarks or list(range(68))  # All landmarks
        self.cpp_reference = cpp_reference

        # Storage for diagnostics
        self.frame_idx = 0
        self.iteration_diagnostics: List[IterationDiagnostic] = []
        self._current_iter_diag: Optional[IterationDiagnostic] = None

    def set_frame_context(self, frame_idx: int, cpp_reference: Optional[np.ndarray] = None):
        """Set context for current frame being processed."""
        self.frame_idx = frame_idx
        self.cpp_reference = cpp_reference
        self.iteration_diagnostics = []

    def get_diagnostics(self) -> List[IterationDiagnostic]:
        """Get collected diagnostics for current frame."""
        return self.iteration_diagnostics

    def _start_iteration(self, iteration: int, phase: str, window_size: int):
        """Start collecting data for a new iteration."""
        self._current_iter_diag = IterationDiagnostic(
            frame_idx=self.frame_idx,
            iteration=iteration,
            phase=phase,
            window_size=window_size,
        )

    def _end_iteration(self, update_magnitude: float, mean_shift_norm: float,
                       jacobian_norm: float, hessian_cond: Optional[float] = None,
                       reg_ratio: Optional[float] = None):
        """Finish current iteration and store diagnostics."""
        if self._current_iter_diag is not None:
            self._current_iter_diag.update_magnitude = update_magnitude
            self._current_iter_diag.mean_shift_norm = mean_shift_norm
            self._current_iter_diag.jacobian_norm = jacobian_norm
            self._current_iter_diag.hessian_cond = hessian_cond
            self._current_iter_diag.reg_ratio = reg_ratio
            self.iteration_diagnostics.append(self._current_iter_diag)
            self._current_iter_diag = None

    def _record_landmark_diagnostic(self,
                                    landmark_idx: int,
                                    current_pos: Tuple[float, float],
                                    base_pos: Tuple[float, float],
                                    response_map: Optional[np.ndarray],
                                    offset_ref: Tuple[float, float],
                                    dx_dy: Tuple[float, float],
                                    clamped: bool,
                                    ms_ref: Tuple[float, float],
                                    ms_img: Tuple[float, float],
                                    kde_total_weight: float = 0.0):
        """Record diagnostic data for one landmark."""
        if self._current_iter_diag is None:
            return

        if landmark_idx not in self.target_landmarks:
            return

        # Get C++ reference position if available
        cpp_ref_pos = None
        if self.cpp_reference is not None and landmark_idx < len(self.cpp_reference):
            cpp_ref_pos = tuple(self.cpp_reference[landmark_idx])

        # Compute response map statistics
        response_stats = None
        if response_map is not None:
            peak_idx = np.unravel_index(response_map.argmax(), response_map.shape)
            center = response_map.shape[0] // 2
            response_stats = {
                'min': float(response_map.min()),
                'max': float(response_map.max()),
                'mean': float(response_map.mean()),
                'std': float(response_map.std()),
                'peak_idx': list(peak_idx),
                'peak_offset': [int(peak_idx[1] - center), int(peak_idx[0] - center)],
                'sharpness': float(response_map.max() / (response_map.mean() + 1e-8)),
            }

        lm_diag = LandmarkDiagnostic(
            landmark_idx=landmark_idx,
            current_pos=current_pos,
            base_pos=base_pos,
            cpp_reference_pos=cpp_ref_pos,
            response_map=response_map.copy() if (response_map is not None and self.capture_response_maps) else None,
            response_map_stats=response_stats,
            offset_ref=offset_ref,
            dx_dy=dx_dy,
            clamped=clamped,
            ms_ref=ms_ref,
            ms_img=ms_img,
            kde_total_weight=kde_total_weight,
        )

        self._current_iter_diag.landmarks[landmark_idx] = lm_diag

    def _compute_mean_shift(self,
                            current_landmarks: np.ndarray,
                            base_landmarks: np.ndarray,
                            response_maps: Dict,
                            patch_experts: Dict,
                            window_size: int,
                            sim_img_to_ref: np.ndarray,
                            sim_ref_to_img: np.ndarray,
                            iteration: int = 0) -> np.ndarray:
        """
        Instrumented version of mean-shift computation.

        Captures all intermediate values for diagnostic analysis.
        """
        n_landmarks = len(current_landmarks)
        mean_shift = np.zeros(2 * n_landmarks)

        # Extract transform components
        R_ref = sim_img_to_ref[:2, :2]
        R_img = sim_ref_to_img[:2, :2]

        # Precompute KDE parameter
        a = -0.5 / (self.sigma ** 2)

        for lm_idx in range(n_landmarks):
            if lm_idx not in patch_experts:
                continue

            response_map = response_maps.get(lm_idx)
            if response_map is None:
                continue

            # Current and base positions
            current_pos = current_landmarks[lm_idx]
            base_pos = base_landmarks[lm_idx]

            # Transform offset to reference coordinates
            offset_img = current_pos - base_pos
            offset_ref = R_ref @ offset_img

            # Convert to response map coordinates
            resp_size = response_map.shape[0]
            half_ws = (resp_size - 1) / 2.0

            dx = half_ws + offset_ref[0]
            dy = half_ws + offset_ref[1]

            # Check if clamping is needed
            clamped = False
            dx_original, dy_original = dx, dy
            if dx < 0:
                dx = 0.0
                clamped = True
            elif dx > resp_size - 0.1:
                dx = resp_size - 0.1
                clamped = True
            if dy < 0:
                dy = 0.0
                clamped = True
            elif dy > resp_size - 0.1:
                dy = resp_size - 0.1
                clamped = True

            # Compute KDE mean-shift
            ms_x_ref, ms_y_ref, kde_total = self._kde_mean_shift_instrumented(
                response_map, dx, dy, a
            )

            # Transform back to image coordinates
            ms_ref = np.array([ms_x_ref, ms_y_ref])
            ms_img = R_img @ ms_ref

            mean_shift[2 * lm_idx] = ms_img[0]
            mean_shift[2 * lm_idx + 1] = ms_img[1]

            # Record diagnostic
            self._record_landmark_diagnostic(
                landmark_idx=lm_idx,
                current_pos=tuple(current_pos),
                base_pos=tuple(base_pos),
                response_map=response_map,
                offset_ref=tuple(offset_ref),
                dx_dy=(dx_original, dy_original),
                clamped=clamped,
                ms_ref=(ms_x_ref, ms_y_ref),
                ms_img=tuple(ms_img),
                kde_total_weight=kde_total,
            )

        return mean_shift

    def _kde_mean_shift_instrumented(self,
                                     response_map: np.ndarray,
                                     dx: float,
                                     dy: float,
                                     a: float) -> Tuple[float, float, float]:
        """
        KDE mean-shift with total weight tracking.

        Returns (ms_x, ms_y, total_weight) for diagnostic purposes.
        """
        resp_size = response_map.shape[0]

        mx = 0.0
        my = 0.0
        total_weight = 0.0

        for ii in range(resp_size):
            for jj in range(resp_size):
                dist_sq = (dy - ii) ** 2 + (dx - jj) ** 2
                kde_weight = np.exp(a * dist_sq)
                weight = kde_weight * response_map[ii, jj]

                total_weight += weight
                mx += weight * jj
                my += weight * ii

        if total_weight > 1e-10:
            ms_x = (mx / total_weight) - dx
            ms_y = (my / total_weight) - dy
        else:
            ms_x = 0.0
            ms_y = 0.0

        return ms_x, ms_y, total_weight

    def optimize(self,
                 pdm,
                 initial_params: np.ndarray,
                 patch_experts: dict,
                 image: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 window_size: int = 11,
                 patch_scaling: float = 0.25,
                 sigma_components: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Instrumented optimization with diagnostic collection.

        This wraps the parent optimize() method and adds diagnostic hooks.
        """
        # Clear diagnostics for this optimization run
        self.iteration_diagnostics = []

        params = initial_params.copy()
        n_params = len(params)
        n_landmarks = pdm.n_points

        # Initialize weights
        if weights is None:
            weights = np.ones(n_landmarks)

        # Weight matrix
        if self.weight_multiplier > 0:
            W = self.weight_multiplier * np.diag(np.repeat(weights, 2))
        else:
            W = np.eye(n_landmarks * 2)

        # Regularization matrix
        Lambda_inv = self._compute_lambda_inv(pdm, n_params)

        # Get initial landmarks and reference shape
        landmarks_2d_initial = pdm.params_to_landmarks_2d(params)
        reference_shape = pdm.get_reference_shape(patch_scaling, params[6:])

        # Compute similarity transforms
        from pyclnf.core.utils import align_shapes_with_scale, invert_similarity_transform
        sim_img_to_ref = align_shapes_with_scale(landmarks_2d_initial, reference_shape)
        sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

        # Precompute response maps
        response_maps = self._precompute_response_maps(
            landmarks_2d_initial, patch_experts, image, window_size,
            sim_img_to_ref, sim_ref_to_img, sigma_components, iteration=0
        )

        # =================================================================
        # PHASE 1: RIGID optimization
        # =================================================================
        rigid_params = params.copy()
        base_landmarks_rigid = landmarks_2d_initial.copy()
        previous_landmarks = None
        global_iter = 0

        for rigid_iter in range(self.max_iterations):
            self._start_iteration(global_iter, 'rigid', window_size)

            current_landmarks = pdm.params_to_landmarks_2d(rigid_params)

            # Early stopping check
            if previous_landmarks is not None and self.convergence_threshold > 0:
                landmark_change = np.linalg.norm(current_landmarks - previous_landmarks, axis=1).mean()
                if landmark_change < self.convergence_threshold:
                    self._end_iteration(0, 0, 0)
                    break
            previous_landmarks = current_landmarks.copy()

            # Compute instrumented mean-shift
            mean_shift = self._compute_mean_shift(
                current_landmarks, base_landmarks_rigid, response_maps, patch_experts,
                window_size, sim_img_to_ref, sim_ref_to_img, iteration=rigid_iter
            )

            # Compute Jacobian
            J_rigid = pdm.compute_jacobian_rigid(rigid_params)

            # Solve for update
            delta_p_rigid = self._solve_rigid_update(J_rigid, mean_shift, W, rigid_iter, window_size)

            # Update params
            delta_p_full = np.zeros(len(rigid_params))
            delta_p_full[:6] = delta_p_rigid
            rigid_params = pdm.update_params(rigid_params, delta_p_full)
            rigid_params = pdm.clamp_params(rigid_params)

            self._end_iteration(
                update_magnitude=float(np.linalg.norm(delta_p_rigid)),
                mean_shift_norm=float(np.linalg.norm(mean_shift)),
                jacobian_norm=float(np.linalg.norm(J_rigid)),
            )
            global_iter += 1

        params[:6] = rigid_params[:6]

        # =================================================================
        # PHASE 2: NON-RIGID optimization
        # =================================================================
        base_landmarks_nonrigid = pdm.params_to_landmarks_2d(params)
        previous_landmarks = None

        for nonrigid_iter in range(self.max_iterations):
            self._start_iteration(global_iter, 'nonrigid', window_size)

            current_landmarks = pdm.params_to_landmarks_2d(params)

            # Early stopping
            if previous_landmarks is not None and self.convergence_threshold > 0:
                landmark_change = np.linalg.norm(current_landmarks - previous_landmarks, axis=1).mean()
                if landmark_change < self.convergence_threshold:
                    self._end_iteration(0, 0, 0)
                    break
            previous_landmarks = current_landmarks.copy()

            # Compute instrumented mean-shift
            mean_shift = self._compute_mean_shift(
                current_landmarks, base_landmarks_nonrigid, response_maps, patch_experts,
                window_size, sim_img_to_ref, sim_ref_to_img, iteration=nonrigid_iter
            )

            # Compute full Jacobian
            J = pdm.compute_jacobian(params)

            # Solve with regularization
            JtW = J.T @ W
            JtWJ = JtW @ J
            JtWv = JtW @ mean_shift

            reg_term = self.regularization * Lambda_inv @ params
            A = JtWJ + self.regularization * Lambda_inv
            b = JtWv - reg_term

            # Compute condition number for diagnostics
            try:
                cond = np.linalg.cond(A)
            except:
                cond = None

            delta_p = np.linalg.solve(A, b)

            # Update params
            params = pdm.update_params(params, delta_p)
            params = pdm.clamp_params(params)

            self._end_iteration(
                update_magnitude=float(np.linalg.norm(delta_p)),
                mean_shift_norm=float(np.linalg.norm(mean_shift)),
                jacobian_norm=float(np.linalg.norm(J)),
                hessian_cond=cond,
                reg_ratio=float(np.linalg.norm(reg_term) / (np.linalg.norm(JtWv) + 1e-10)),
            )
            global_iter += 1

        # Return results
        final_landmarks = pdm.params_to_landmarks_2d(params)

        # Compute final update magnitude (matching base optimizer interface)
        final_update = 0.0
        if self.iteration_diagnostics:
            final_update = self.iteration_diagnostics[-1].update_magnitude

        info = {
            'iterations': global_iter,
            'converged': True,
            'final_update': final_update,  # Required by CLNF.fit()
            'window_size': window_size,
            'diagnostics': self.iteration_diagnostics,
            'iteration_history': [{'update_magnitude': d.update_magnitude} for d in self.iteration_diagnostics],
        }

        return params, info
