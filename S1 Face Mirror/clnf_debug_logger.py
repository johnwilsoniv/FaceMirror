#!/usr/bin/env python3
"""
Enhanced CLNF detector with detailed debug logging for comparison analysis.

This wraps the standard CLNF detector to capture iteration-by-iteration diagnostics:
- Landmark positions
- PDM parameters
- Response map statistics
- Mean shift targets
- Jacobian condition numbers
- Parameter updates
- Convergence metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import numpy as np
import cv2
from pyfaceau.clnf.clnf_detector import CLNFDetector
from pyfaceau.clnf.nu_rlms import NURLMSOptimizer
from scipy.ndimage import gaussian_filter


class DebugNURLMSOptimizer(NURLMSOptimizer):
    """
    NU-RLMS optimizer with detailed debug logging.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_history = []

    def optimize(self, image, initial_landmarks, scale_idx=2, regularization=0.5):
        """
        Optimize with detailed logging of each iteration.
        """
        # Reset debug history
        self.debug_history = []

        # Initialize with current landmarks
        landmarks = initial_landmarks.copy()

        # Estimate initial PDM parameters, scale, and translation from 2D landmarks
        params, scale, translation = self.pdm.landmarks_to_params_2d(landmarks)

        # Clamp parameters to plausible range
        params = self.pdm.clamp_params(params, n_std=3.0)

        # Log initial state
        self._log_iteration(
            iteration=-1,  # -1 = initialization
            landmarks=landmarks,
            params=params,
            scale=scale,
            translation=translation,
            responses=None,
            target_landmarks=None,
            residual=None,
            jacobian=None,
            delta_params=None,
            avg_movement=0.0,
            converged=False,
        )

        # Optimization loop
        converged = False
        for iteration in range(self.max_iterations):
            # 1. Compute CEN response maps for current landmarks
            responses, extraction_bounds = self.patch_experts.response(image, landmarks, scale_idx)

            # 2. Find target positions using mean-shift on response maps
            target_landmarks = self._mean_shift_targets(landmarks, responses, extraction_bounds, scale_idx)

            # 3. Compute residual: difference between target and current positions
            residual = (target_landmarks - landmarks).flatten()

            # 4. Check convergence
            avg_movement = np.sqrt(np.mean(residual ** 2))
            if avg_movement < self.convergence_threshold:
                converged = True

            # 5. Compute Jacobian of 2D landmarks w.r.t. PDM parameters
            jacobian = self._compute_jacobian(params, scale, translation)

            # 6. Solve regularized least squares for parameter update
            delta_params = self._solve_regularized_ls(jacobian, residual, regularization)

            # Log iteration state BEFORE update
            self._log_iteration(
                iteration=iteration,
                landmarks=landmarks.copy(),
                params=params.copy(),
                scale=scale,
                translation=translation.copy(),
                responses=responses,
                target_landmarks=target_landmarks.copy(),
                residual=residual.copy(),
                jacobian=jacobian,
                delta_params=delta_params.copy(),
                avg_movement=avg_movement,
                converged=converged,
            )

            if converged:
                break

            # 7. Update parameters
            params = params + delta_params

            # 8. Clamp parameters to plausible range
            params = self.pdm.clamp_params(params, n_std=3.0)

            # 9. Convert parameters back to 2D landmarks
            landmarks = self.pdm.params_to_landmarks_2d(params, scale, translation)

        # Log final state
        self._log_iteration(
            iteration=iteration + 1,  # Final iteration
            landmarks=landmarks,
            params=params,
            scale=scale,
            translation=translation,
            responses=None,
            target_landmarks=None,
            residual=None,
            jacobian=None,
            delta_params=None,
            avg_movement=avg_movement,
            converged=converged,
        )

        return landmarks, converged, iteration + 1

    def _log_iteration(self, iteration, landmarks, params, scale, translation,
                       responses, target_landmarks, residual, jacobian, delta_params,
                       avg_movement, converged):
        """
        Log detailed state for this iteration.
        """
        log_entry = {
            'iteration': iteration,
            'landmarks': landmarks.copy(),
            'params': params.copy(),
            'scale': scale,
            'translation': translation.copy(),
            'avg_movement': avg_movement,
            'converged': converged,
        }

        # Response map statistics (only for actual iterations, not init/final)
        if responses is not None:
            response_stats = []
            for r in responses:
                if r.size > 0:
                    response_stats.append({
                        'min': float(r.min()),
                        'max': float(r.max()),
                        'mean': float(r.mean()),
                        'std': float(r.std()),
                        'shape': r.shape,
                    })
                else:
                    response_stats.append({
                        'min': 0.0,
                        'max': 0.0,
                        'mean': 0.0,
                        'std': 0.0,
                        'shape': (0, 0),
                    })
            log_entry['response_stats'] = response_stats

        # Mean shift targets
        if target_landmarks is not None:
            log_entry['target_landmarks'] = target_landmarks.copy()
            log_entry['landmark_shifts'] = target_landmarks - landmarks

        # Residual
        if residual is not None:
            log_entry['residual'] = residual.copy()
            log_entry['residual_norm'] = float(np.linalg.norm(residual))

        # Jacobian
        if jacobian is not None:
            log_entry['jacobian_condition'] = float(np.linalg.cond(jacobian))
            log_entry['jacobian_shape'] = jacobian.shape

        # Parameter update
        if delta_params is not None:
            log_entry['delta_params'] = delta_params.copy()
            log_entry['delta_params_norm'] = float(np.linalg.norm(delta_params))

        self.debug_history.append(log_entry)


class DebugCLNFDetector(CLNFDetector):
    """
    CLNF detector with debug logging enabled.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent
        super().__init__(*args, **kwargs)

        # Replace optimizer with debug version
        self.optimizer = DebugNURLMSOptimizer(
            self.pdm,
            self.patch_experts,
            max_iterations=self.optimizer.max_iterations,
            convergence_threshold=self.optimizer.convergence_threshold
        )

    def get_debug_history(self):
        """
        Get debug history from last optimization.

        Returns:
            List of iteration log entries
        """
        return self.optimizer.debug_history


def print_iteration_summary(history, iteration_idx):
    """
    Print summary of a single iteration.

    Args:
        history: Debug history list
        iteration_idx: Index into history (0 = initialization)
    """
    if iteration_idx >= len(history):
        print(f"Iteration {iteration_idx} not available")
        return

    log = history[iteration_idx]

    print(f"\n{'='*80}")
    if log['iteration'] == -1:
        print(f"INITIALIZATION (Iteration -1)")
    else:
        print(f"ITERATION {log['iteration']}")
    print(f"{'='*80}")

    print(f"  Average movement: {log['avg_movement']:.4f} pixels")
    print(f"  Converged: {log['converged']}")

    # PDM parameters
    print(f"\n  PDM Parameters:")
    print(f"    Scale: {log['scale']:.4f}")
    print(f"    Translation: [{log['translation'][0]:.2f}, {log['translation'][1]:.2f}]")
    print(f"    Params shape: {log['params'].shape}")
    print(f"    Params range: [{log['params'].min():.4f}, {log['params'].max():.4f}]")
    print(f"    Params mean: {log['params'].mean():.4f}, std: {log['params'].std():.4f}")

    # Landmarks
    print(f"\n  Landmarks:")
    landmarks = log['landmarks']
    print(f"    X range: [{landmarks[:, 0].min():.2f}, {landmarks[:, 0].max():.2f}]")
    print(f"    Y range: [{landmarks[:, 1].min():.2f}, {landmarks[:, 1].max():.2f}]")
    print(f"    Centroid: [{landmarks[:, 0].mean():.2f}, {landmarks[:, 1].mean():.2f}]")

    # Response statistics (if available)
    if 'response_stats' in log:
        print(f"\n  Response Map Statistics:")
        stats = log['response_stats']
        all_means = [s['mean'] for s in stats if s['mean'] > 0]
        all_maxs = [s['max'] for s in stats if s['max'] > 0]
        if all_means:
            print(f"    Mean response: {np.mean(all_means):.6f} (±{np.std(all_means):.6f})")
            print(f"    Max response: {np.mean(all_maxs):.6f} (±{np.std(all_maxs):.6f})")
        else:
            print(f"    No valid responses!")

    # Mean shift targets (if available)
    if 'target_landmarks' in log:
        shifts = log['landmark_shifts']
        shift_distances = np.sqrt(np.sum(shifts**2, axis=1))
        print(f"\n  Mean Shift Targets:")
        print(f"    Average shift: {shift_distances.mean():.4f} pixels")
        print(f"    Max shift: {shift_distances.max():.4f} pixels")
        print(f"    Min shift: {shift_distances.min():.4f} pixels")

    # Jacobian (if available)
    if 'jacobian_condition' in log:
        print(f"\n  Jacobian:")
        print(f"    Shape: {log['jacobian_shape']}")
        print(f"    Condition number: {log['jacobian_condition']:.2e}")

    # Parameter update (if available)
    if 'delta_params' in log:
        print(f"\n  Parameter Update:")
        print(f"    Delta params norm: {log['delta_params_norm']:.6f}")
        print(f"    Delta params range: [{log['delta_params'].min():.6f}, {log['delta_params'].max():.6f}]")

    # Residual (if available)
    if 'residual' in log:
        print(f"\n  Residual:")
        print(f"    Norm: {log['residual_norm']:.6f}")
        print(f"    Mean: {log['residual'].mean():.6f}")


def compare_iterations(history1, history2, iteration_idx, labels=("Python", "C++")):
    """
    Compare the same iteration from two different runs.

    Args:
        history1: Debug history from first run
        history2: Debug history from second run (or None)
        iteration_idx: Which iteration to compare
        labels: Tuple of (label1, label2) for display
    """
    if iteration_idx >= len(history1):
        print(f"Iteration {iteration_idx} not available in {labels[0]}")
        return

    log1 = history1[iteration_idx]

    print(f"\n{'='*80}")
    if log1['iteration'] == -1:
        print(f"COMPARING INITIALIZATION")
    else:
        print(f"COMPARING ITERATION {log1['iteration']}")
    print(f"{'='*80}")

    if history2 is None or iteration_idx >= len(history2):
        print(f"{labels[1]} data not available for comparison")
        return

    log2 = history2[iteration_idx]

    # Compare landmarks
    landmarks1 = log1['landmarks']
    landmarks2 = log2['landmarks']
    landmark_diff = np.sqrt(np.sum((landmarks1 - landmarks2)**2, axis=1))

    print(f"\n  Landmark Differences ({labels[0]} vs {labels[1]}):")
    print(f"    Mean error: {landmark_diff.mean():.4f} pixels")
    print(f"    Max error: {landmark_diff.max():.4f} pixels (landmark {landmark_diff.argmax()})")
    print(f"    Median error: {np.median(landmark_diff):.4f} pixels")

    # Compare PDM parameters
    params_diff = log1['params'] - log2['params']
    print(f"\n  PDM Parameter Differences:")
    print(f"    Mean diff: {params_diff.mean():.6f}")
    print(f"    Max diff: {params_diff.max():.6f}")
    print(f"    RMS diff: {np.sqrt(np.mean(params_diff**2)):.6f}")

    # Compare scale and translation
    print(f"\n  Scale/Translation Differences:")
    print(f"    Scale: {labels[0]}={log1['scale']:.4f}, {labels[1]}={log2['scale']:.4f}, diff={log1['scale']-log2['scale']:.6f}")
    trans_diff = log1['translation'] - log2['translation']
    print(f"    Translation diff: [{trans_diff[0]:.4f}, {trans_diff[1]:.4f}]")

    # Compare movements
    print(f"\n  Convergence Metrics:")
    print(f"    {labels[0]} avg movement: {log1['avg_movement']:.4f} pixels")
    print(f"    {labels[1]} avg movement: {log2['avg_movement']:.4f} pixels")
    print(f"    Difference: {abs(log1['avg_movement'] - log2['avg_movement']):.4f} pixels")


if __name__ == '__main__':
    print("CLNF Debug Logger")
    print("="*80)
    print("This module provides debug logging for CLNF optimization.")
    print("\nUsage:")
    print("  from clnf_debug_logger import DebugCLNFDetector")
    print("  detector = DebugCLNFDetector(model_dir, ...)")
    print("  refined_lms, converged, iters = detector.refine_landmarks(image, initial_lms)")
    print("  history = detector.get_debug_history()")
    print("  print_iteration_summary(history, iteration_idx)")
