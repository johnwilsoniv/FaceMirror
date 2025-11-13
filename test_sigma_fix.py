#!/usr/bin/env python3
"""
Test different sigma values to fix convergence.

ROOT CAUSE: Gaussian kernel too narrow (sigma=1.75 for ws=11)
- Peak at 5px from center: weight = exp(-0.5 × 5² / 1.75²) = 0.017 (1.7%!)
- The Gaussian ignores the peak and averages over center instead

HYPOTHESIS: sigma should scale with window_size
- Reasonable range: window_size / 3 to window_size / 2
- For ws=11: sigma ≈ 3.7 to 5.5
- For ws=9:  sigma ≈ 3.0 to 4.5
- For ws=7:  sigma ≈ 2.3 to 3.5
"""

import cv2
import numpy as np
import sys

sys.path.insert(0, 'pyclnf')
from pyclnf import CLNF

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50
FACE_BBOX = (241, 555, 532, 532)


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
    print("TESTING DIFFERENT SIGMA VALUES TO FIX CONVERGENCE")
    print("="*80)
    print()

    # Load frame
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Current broken value
    print("CURRENT (BROKEN) SIGMA:")
    print(f"  sigma = 1.75")
    print(f"  For ws=11, peak at 5px: weight = exp(-0.5 × 5² / 1.75²) = {np.exp(-0.5 * 25 / (1.75**2)):.4f} (1.7%)")
    print()

    # Test different sigma values
    sigma_tests = [
        (1.75, "Current (broken)"),
        (2.5, "Slightly larger"),
        (3.0, "ws/3.7 (narrow)"),
        (3.5, "ws/3.1 (moderate)"),
        (4.0, "ws/2.75 (wider)"),
        (4.5, "ws/2.44 (OpenFace default?)"),
        (5.0, "ws/2.2 (very wide)"),
        (5.5, "ws/2 (max reasonable)"),
    ]

    print(f"{'Sigma':<8} {'Description':<25} {'Weight@5px':<13} {'Converged':<12} {'Final Update':<15} {'Ratio':<10}")
    print("-" * 90)

    results = []

    for sigma, description in sigma_tests:
        # Test with this sigma
        clnf = CLNF(model_dir='pyclnf/models', max_iterations=20, window_sizes=[11])
        clnf.optimizer.sigma = sigma

        landmarks, info = clnf.fit(gray, FACE_BBOX, return_params=True)

        # Compute Gaussian weight at peak (5px from center)
        weight_at_peak = np.exp(-0.5 * 25 / (sigma**2))

        # Display results
        ratio = info['final_update'] / 0.005
        converged_str = "YES" if info['converged'] else "NO"

        print(f"{sigma:<8.2f} {description:<25} {weight_at_peak:<13.4f} {converged_str:<12} {info['final_update']:<15.6f} {ratio:<10.1f}x")

        results.append({
            'sigma': sigma,
            'description': description,
            'weight_at_peak': weight_at_peak,
            'converged': info['converged'],
            'final_update': info['final_update'],
            'ratio': ratio
        })

    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    # Find best sigma
    best = min(results, key=lambda x: x['final_update'])
    print(f"BEST SIGMA: {best['sigma']}")
    print(f"  Final update: {best['final_update']:.6f} (target: 0.005)")
    print(f"  Ratio: {best['ratio']:.1f}x the target")
    print(f"  Converged: {best['converged']}")
    print(f"  Weight at 5px peak: {best['weight_at_peak']:.4f} ({best['weight_at_peak']*100:.1f}%)")
    print()

    # Show improvement
    current = results[0]
    improvement = (current['final_update'] - best['final_update']) / current['final_update'] * 100
    print(f"IMPROVEMENT:")
    print(f"  Before (sigma=1.75): {current['final_update']:.6f}")
    print(f"  After (sigma={best['sigma']}): {best['final_update']:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print()

    # Recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    # Check if sigma should scale with window_size
    optimal_ratio = best['sigma'] / 11
    print(f"Optimal sigma for ws=11: {best['sigma']}")
    print(f"Ratio: sigma / window_size = {optimal_ratio:.3f}")
    print()
    print("Suggested fix:")
    print(f"  - Change default sigma from 1.5 to {best['sigma']:.1f}")
    print(f"  - OR make sigma dynamic: sigma = window_size / {11/best['sigma']:.2f}")
    print()
    print("Example implementation:")
    print("```python")
    print(f"# In optimizer __init__ or _compute_mean_shift:")
    print(f"sigma = window_size / {11/best['sigma']:.2f}  # Auto-scale with window size")
    print(f"a = -0.5 / (sigma ** 2)")
    print("```")
    print()


if __name__ == "__main__":
    main()
