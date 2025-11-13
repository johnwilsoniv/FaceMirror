#!/usr/bin/env python3
"""
Investigation: Why do response map peaks appear at window EDGES instead of CENTER?

Hypotheses to test:
1. Patch experts are poorly trained / incompatible
2. Image preprocessing mismatch (color space, normalization)
3. Coordinate transformation bug in response map extraction
4. Response map computation bug

This script compares OpenFace C++ vs PyCLNF response maps to identify the issue.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'pyclnf')
from pyclnf import CLNF
from pyclnf.core.optimizer import NURLMSOptimizer

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50
FACE_BBOX = (241, 555, 532, 532)


class InvestigativeOptimizer(NURLMSOptimizer):
    """Optimizer that extracts and analyzes response maps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_map_data = []

    def _compute_mean_shift(self, landmarks_2d, patch_experts, image, pdm,
                           window_size=11, sim_img_to_ref=None,
                           sim_ref_to_img=None, sigma_components=None):
        """Extract response maps for analysis."""

        n_points = landmarks_2d.shape[0]
        half_window = window_size // 2

        # Sample a few landmarks for detailed analysis
        sample_landmarks = [0, 1, 2, 17, 18, 27]  # Different facial regions

        for landmark_idx in sample_landmarks:
            if landmark_idx not in patch_experts or landmark_idx >= n_points:
                continue

            patch_expert = patch_experts[landmark_idx]
            lm_x, lm_y = landmarks_2d[landmark_idx]

            # Compute response map
            response_map = self._compute_response_map(
                image, lm_x, lm_y, patch_expert, window_size,
                sim_img_to_ref, sim_ref_to_img, sigma_components
            )

            if response_map is None:
                continue

            # Find peak
            peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
            peak_y, peak_x = peak_idx
            peak_value = response_map[peak_y, peak_x]

            # Analyze response map
            center = (window_size - 1) / 2.0
            offset_x = peak_x - center
            offset_y = peak_y - center
            offset_dist = np.sqrt(offset_x**2 + offset_y**2)

            # Compute statistics
            mean_val = np.mean(response_map)
            std_val = np.std(response_map)
            max_val = np.max(response_map)
            min_val = np.min(response_map)

            # Check if response map is "flat" (no clear peak)
            snr = (max_val - mean_val) / std_val if std_val > 0 else 0

            # Count how many pixels are "close" to the max value
            threshold = max_val * 0.9
            high_response_pixels = np.sum(response_map >= threshold)

            self.response_map_data.append({
                'landmark_idx': landmark_idx,
                'window_size': window_size,
                'lm_pos': (lm_x, lm_y),
                'peak_pos': (peak_x, peak_y),
                'peak_offset': (offset_x, offset_y),
                'peak_offset_dist': offset_dist,
                'peak_value': peak_value,
                'mean_value': mean_val,
                'std_value': std_val,
                'snr': snr,
                'high_response_pixels': high_response_pixels,
                'response_map': response_map.copy()
            })

        # Call original implementation
        return super()._compute_mean_shift(
            landmarks_2d, patch_experts, image, pdm, window_size,
            sim_img_to_ref, sim_ref_to_img, sigma_components
        )


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def main():
    print("="*80)
    print("INVESTIGATING WHY RESPONSE MAP PEAKS ARE AT WINDOW EDGES")
    print("="*80)
    print()

    # Load frame
    print(f"Loading frame {FRAME_NUM} from {VIDEO_PATH}...")
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Frame shape: {frame.shape}")
    print()

    # Create instrumented CLNF
    print("Initializing instrumented CLNF...")
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=1, window_sizes=[11])

    # Replace optimizer with diagnostic version
    clnf.optimizer = InvestigativeOptimizer(
        regularization=clnf.optimizer.regularization,
        max_iterations=1,
        convergence_threshold=clnf.optimizer.convergence_threshold,
        sigma=clnf.optimizer.sigma,
        weight_multiplier=clnf.optimizer.weight_multiplier
    )

    print("Running CLNF to extract response maps...")
    landmarks, info = clnf.fit(gray, FACE_BBOX, return_params=True)
    print()

    # Analyze results
    print("="*80)
    print("RESPONSE MAP ANALYSIS")
    print("="*80)
    print()

    data = clnf.optimizer.response_map_data

    if not data:
        print("ERROR: No response map data collected!")
        return

    print(f"Collected {len(data)} response maps from landmarks: {[d['landmark_idx'] for d in data]}")
    print()

    # Summary statistics
    print(f"{'Landmark':<10} {'Peak Offset':<15} {'SNR':<8} {'High Resp Px':<15} {'Diagnosis'}")
    print("-" * 80)

    for d in data:
        lm_idx = d['landmark_idx']
        offset_dist = d['peak_offset_dist']
        offset_x, offset_y = d['peak_offset']
        snr = d['snr']
        high_px = d['high_response_pixels']

        # Diagnose the issue
        diagnosis = []
        if offset_dist > 4.0:
            diagnosis.append("EDGE PEAK")
        if snr < 2.0:
            diagnosis.append("LOW SNR")
        if high_px > 5:
            diagnosis.append("FLAT/MULTI-PEAK")

        diagnosis_str = ", ".join(diagnosis) if diagnosis else "OK"

        print(f"{lm_idx:<10} ({offset_x:+.1f}, {offset_y:+.1f}) {offset_dist:<6.1f}px  {snr:<8.2f} {high_px:<15} {diagnosis_str}")

    print()
    print("="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    print()

    # Find worst case (largest offset)
    worst = max(data, key=lambda d: d['peak_offset_dist'])

    print(f"WORST CASE: Landmark {worst['landmark_idx']}")
    print(f"  Landmark position: ({worst['lm_pos'][0]:.2f}, {worst['lm_pos'][1]:.2f})")
    print(f"  Peak position in response map: ({worst['peak_pos'][0]}, {worst['peak_pos'][1]})")
    print(f"  Peak offset from center: ({worst['peak_offset'][0]:+.1f}, {worst['peak_offset'][1]:+.1f}) = {worst['peak_offset_dist']:.1f}px")
    print(f"  Peak value: {worst['peak_value']:.6f}")
    print(f"  Mean value: {worst['mean_value']:.6f}")
    print(f"  SNR: {worst['snr']:.2f}")
    print()

    # Print response map
    print("Response map (11×11):")
    resp = worst['response_map']
    for i in range(11):
        row_str = ""
        for j in range(11):
            val = resp[i, j]
            # Mark peak with asterisk
            if i == worst['peak_pos'][1] and j == worst['peak_pos'][0]:
                row_str += f"{val:.3f}*  "
            else:
                row_str += f"{val:.3f}   "
        print(f"  {row_str}")
    print()

    # Check if all peaks are at edges
    edge_peak_count = sum(1 for d in data if d['peak_offset_dist'] > 4.0)
    print(f"SUMMARY: {edge_peak_count}/{len(data)} response maps have peaks at edges (offset > 4px)")

    if edge_peak_count == len(data):
        print()
        print("CONCLUSION: ALL response maps have edge peaks → SYSTEMATIC ISSUE")
        print()
        print("Possible causes:")
        print("  1. Patch experts are poorly trained or incompatible")
        print("  2. Image preprocessing mismatch (normalization, color space)")
        print("  3. Coordinate transformation bug in response map extraction")
        print("  4. Response map is being computed incorrectly")
        print()
        print("NEXT STEP: Compare PyCLNF response maps with OpenFace C++ reference")
    elif edge_peak_count > len(data) // 2:
        print()
        print("CONCLUSION: MOST response maps have edge peaks → LIKELY SYSTEMATIC ISSUE")
    else:
        print()
        print("CONCLUSION: Mixed results → May be landmark-specific or image-dependent")

    print("="*80)


if __name__ == "__main__":
    main()
