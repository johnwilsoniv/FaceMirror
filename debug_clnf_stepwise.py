#!/usr/bin/env python3
"""
Step-by-step CLNF debug instrumentation for comparison with C++ OpenFace.

This script adds debug output at every critical stage of CLNF optimization
to enable direct comparison with C++ OpenFace debug output.

Debug output format: [PY][STAGE] description: values
"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from pyclnf.core.pdm import PDM
from pyclnf.core.patch_expert import CCNFModel
from pyclnf.core.optimizer import NURLMSOptimizer

PROJECT_ROOT = Path(__file__).parent
TEST_IMAGE = PROJECT_ROOT / "calibration_frames" / "patient1_frame1.jpg"
PYTHON_RESULT = PROJECT_ROOT / "validation_output" / "python_baseline" / "patient1_frame1_result.json"

# Landmarks to track in detail
TRACKED_LANDMARKS = [36, 48, 30, 8]  # Left eye corner, mouth center, nose tip, jaw point

def debug_print(stage, message):
    """Print debug message with consistent formatting."""
    print(f"[PY][{stage}] {message}")

def main():
    print("="*80)
    print("CLNF Step-by-Step Debug - Python Implementation")
    print("="*80)

    # Load test image
    img = cv2.imread(str(TEST_IMAGE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get bbox from known good result
    with open(PYTHON_RESULT, 'r') as f:
        data = json.load(f)
    bbox_xyxy = data['debug_info']['face_detection']['bbox']
    bbox = (bbox_xyxy[0], bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1])

    debug_print("INPUT", f"Image shape: {gray.shape}")
    debug_print("INPUT", f"BBox (x,y,w,h): {bbox}")

    # Initialize PDM
    pdm_dir = PROJECT_ROOT / "pyclnf" / "models" / "exported_pdm"
    pdm = PDM(str(pdm_dir))

    debug_print("PDM", f"Mean shape: {pdm.mean_shape.shape}")
    debug_print("PDM", f"Num principal components: {pdm.princ_comp.shape[1]}")
    debug_print("PDM", f"Num modes: {pdm.n_modes}")

    # Initialize parameters from bbox
    params_init = pdm.init_params(bbox)

    debug_print("INIT", f"Params shape: {params_init.shape}")
    debug_print("INIT", f"Params_local (first 5): {params_init[:5]}")
    debug_print("INIT", f"Params_global (scale): {params_init[-6]:.6f}")
    debug_print("INIT", f"Params_global (rotation): [{params_init[-5]:.6f}, {params_init[-4]:.6f}, {params_init[-3]:.6f}]")
    debug_print("INIT", f"Params_global (translation): [{params_init[-2]:.6f}, {params_init[-1]:.6f}]")

    # Get initial landmarks
    landmarks_init = pdm.params_to_landmarks_2d(params_init)

    debug_print("INIT", f"Landmarks shape: {landmarks_init.shape}")
    for lm_idx in TRACKED_LANDMARKS:
        lm_x, lm_y = landmarks_init[lm_idx]
        debug_print("INIT", f"Landmark_{lm_idx}: ({lm_x:.4f}, {lm_y:.4f})")

    # Load CCNF model
    models_dir = PROJECT_ROOT / "pyclnf" / "models"
    ccnf = CCNFModel(str(models_dir), scales=[0.25])

    debug_print("CCNF", f"Loaded sigma components for window sizes: {list(ccnf.sigma_components.keys())}")

    # Get patch experts for scale 0.25, view 0
    scale_model = ccnf.scale_models.get(0.25)
    view_data = scale_model['views'].get(0)
    patch_experts = view_data['patches']

    debug_print("CCNF", f"Num patch experts: {len(patch_experts)}")

    # Create instrumented optimizer
    optimizer = InstrumentedNURLMSOptimizer(
        regularization=35,
        max_iterations=10,
        convergence_threshold=0.01,
        sigma=1.5
    )

    # Run optimization with detailed logging
    final_params, converged = optimizer.optimize(
        image=gray,
        initial_params=params_init,
        pdm=pdm,
        patch_experts=patch_experts,
        sigma_components=ccnf.sigma_components
    )

    # Final results
    final_landmarks = pdm.params_to_landmarks_2d(final_params)

    print("\n" + "="*80)
    debug_print("FINAL", f"Converged: {converged}")
    debug_print("FINAL", f"Final params_local (first 5): {final_params[:5]}")
    debug_print("FINAL", f"Final params_global (scale): {final_params[-6]:.6f}")
    for lm_idx in TRACKED_LANDMARKS:
        lm_x, lm_y = final_landmarks[lm_idx]
        debug_print("FINAL", f"Landmark_{lm_idx}: ({lm_x:.4f}, {lm_y:.4f})")
    print("="*80)


class InstrumentedNURLMSOptimizer(NURLMSOptimizer):
    """
    NURLMSOptimizer with detailed debug output at every step.
    """

    def optimize(self, image, initial_params, pdm, patch_experts, sigma_components=None):
        """Override optimize to add detailed logging."""

        # Store for tracking
        self.iteration_count = 0
        self.tracked_landmarks = TRACKED_LANDMARKS

        # Call parent with correct parameter order (pdm, initial_params, patch_experts, image, ...)
        result = super().optimize(pdm, initial_params, patch_experts, image, sigma_components=sigma_components)

        # Handle both old and new return formats
        if isinstance(result, tuple) and len(result) == 2:
            return result  # (params, info_dict) or (params, converged)
        else:
            return result, True  # Assume converged if only params returned

    def _compute_mean_shift(self, landmarks_2d, patch_experts, image, pdm,
                           window_size=11, sim_img_to_ref=None, sim_ref_to_img=None,
                           sigma_components=None, iteration=None):
        """Override to add response map logging."""

        debug_print(f"ITER{iteration}_WS{window_size}", "Computing response maps...")

        # For tracked landmarks, log detailed response map info
        mean_shift = np.zeros((len(landmarks_2d), 2))

        for i, (lm_x, lm_y) in enumerate(landmarks_2d):
            patch_expert = patch_experts.get(i)
            if patch_expert is None:
                continue

            # Compute response map with detailed logging for tracked landmarks
            response_map = self._compute_response_map(
                image, lm_x, lm_y, patch_expert, window_size,
                sim_img_to_ref, sim_ref_to_img, sigma_components,
                landmark_idx=i, iteration=iteration
            )

            if response_map is None:
                continue

            # Compute mean-shift
            ms_x, ms_y = self._compute_mean_shift_from_response(
                response_map, window_size, landmark_idx=i, iteration=iteration
            )
            mean_shift[i] = [ms_x, ms_y]

        return mean_shift

    def _compute_response_map(self, image, center_x, center_y, patch_expert,
                             window_size, sim_img_to_ref=None, sim_ref_to_img=None,
                             sigma_components=None, landmark_idx=None, iteration=None):
        """Override to add detailed logging."""

        log_detail = landmark_idx in self.tracked_landmarks

        # Build response map
        half_window = window_size // 2
        start_x = int(center_x) - half_window
        start_y = int(center_y) - half_window

        if log_detail:
            debug_print(f"ITER{iteration}_WS{window_size}_LM{landmark_idx}",
                       f"Window bounds: x=[{start_x}, {start_x + window_size}] y=[{start_y}, {start_y + window_size}]")

        response_map = np.zeros((window_size, window_size))

        for i in range(window_size):
            for j in range(window_size):
                patch_x = start_x + j
                patch_y = start_y + i

                # Extract patch
                half_w = patch_expert.width // 2
                half_h = patch_expert.height // 2
                x1 = patch_x - half_w
                y1 = patch_y - half_h
                x2 = x1 + patch_expert.width
                y2 = y1 + patch_expert.height

                if 0 <= x1 and 0 <= y1 and x2 < image.shape[1] and y2 < image.shape[0]:
                    patch = image[y1:y2, x1:x2]
                    response_map[i, j] = patch_expert.compute_response(patch)
                else:
                    response_map[i, j] = -1e10

        # Log raw response map
        if log_detail:
            peak_row, peak_col = np.unravel_index(response_map.argmax(), response_map.shape)
            peak_val = response_map[peak_row, peak_col]
            center = window_size // 2
            offset_x = peak_col - center
            offset_y = peak_row - center

            debug_print(f"ITER{iteration}_WS{window_size}_LM{landmark_idx}",
                       f"Response_RAW: peak=({peak_row},{peak_col}) val={peak_val:.6f} offset=({offset_x},{offset_y})")

        # Apply Sigma transformation if available
        if sigma_components is not None and window_size in sigma_components:
            sigma_comps = sigma_components[window_size]
            Sigma = patch_expert.compute_sigma(sigma_comps, window_size=window_size)

            # Transform
            response_vec = response_map.reshape(-1, 1)
            response_transformed = Sigma @ response_vec
            response_map = response_transformed.reshape(window_size, window_size)

            # Log transformed response map
            if log_detail:
                peak_row, peak_col = np.unravel_index(response_map.argmax(), response_map.shape)
                peak_val = response_map[peak_row, peak_col]
                offset_x = peak_col - center
                offset_y = peak_row - center

                debug_print(f"ITER{iteration}_WS{window_size}_LM{landmark_idx}",
                           f"Response_SIGMA: peak=({peak_row},{peak_col}) val={peak_val:.6f} offset=({offset_x},{offset_y})")

        return response_map

    def _compute_mean_shift_from_response(self, response_map, window_size,
                                         landmark_idx=None, iteration=None):
        """Compute mean-shift with logging."""

        log_detail = landmark_idx in self.tracked_landmarks

        # Mean-shift calculation
        sigma = self.sigma
        a = -0.5 / (sigma * sigma)

        sum_g = 0.0
        ms_x = 0.0
        ms_y = 0.0

        resp_size = response_map.shape[0]
        center = (resp_size - 1) / 2.0

        for i in range(resp_size):
            for j in range(resp_size):
                dx = j - center
                dy = i - center

                r_sq = dx*dx + dy*dy
                g = np.exp(a * r_sq)

                resp = response_map[i, j]
                weight = g * resp

                ms_x += weight * dx
                ms_y += weight * dy
                sum_g += weight

        if sum_g > 1e-8:
            ms_x /= sum_g
            ms_y /= sum_g

        if log_detail:
            debug_print(f"ITER{iteration}_WS{window_size}_LM{landmark_idx}",
                       f"Mean-shift: ({ms_x:.6f}, {ms_y:.6f}) sum_weights={sum_g:.6f}")

        return ms_x, ms_y


if __name__ == "__main__":
    main()
