#!/usr/bin/env python3
"""
Derive pyMTCNN-specific correction coefficients.

Instead of using OpenFace's correction (tuned for C++ MTCNN), we'll derive
coefficients that transform raw pyMTCNN bboxes directly to the C++ target.

This accounts for the systematic differences between pyMTCNN and C++ MTCNN.
"""

import numpy as np
import cv2
import sys
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, 'pyfaceau')
from pyclnf import CLNF

# Pre-computed data from previous analysis
RAW_PYMTCNN_BBOXES = {
    ('IMG_0433', 50): (262, 581, 431, 555),
    ('IMG_0433', 150): (274, 587, 445, 564),
    ('IMG_0433', 250): (241, 645, 422, 536),
    ('IMG_0434', 50): (268, 687, 398, 518),
    ('IMG_0434', 150): (321, 664, 393, 485),
    ('IMG_0434', 250): (328, 644, 361, 450),
    ('IMG_0435', 50): (328, 539, 409, 544),
    ('IMG_0435', 150): (305, 566, 383, 472),
    ('IMG_0435', 250): (329, 507, 427, 532),
}

CPP_CORRECTED_BBOXES = {
    ('IMG_0433', 50): (287, 688, 424, 408),
    ('IMG_0433', 150): (307, 726, 409, 394),
    ('IMG_0433', 250): (301, 740, 402, 383),
    ('IMG_0434', 50): (295, 767, 408, 414),
    ('IMG_0434', 150): (288, 780, 399, 401),
    ('IMG_0434', 250): (293, 794, 385, 372),
    ('IMG_0435', 50): (329, 669, 388, 393),
    ('IMG_0435', 150): (322, 651, 396, 397),
    ('IMG_0435', 250): (323, 659, 385, 372),
}

VIDEO_CONFIGS = [
    ('Patient Data/Normal Cohort/IMG_0433.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0434.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0435.MOV', [50, 150, 250]),
]


def apply_correction(bbox, coeffs):
    """Apply correction with given coefficients.

    Args:
        bbox: (x, y, w, h) tuple
        coeffs: [dx_coeff, dy_coeff, w_scale, h_scale]

    Returns:
        Corrected (x, y, w, h) tuple
    """
    x, y, w, h = bbox
    dx_coeff, dy_coeff, w_scale, h_scale = coeffs

    x_new = x + w * dx_coeff
    y_new = y + h * dy_coeff
    w_new = w * w_scale
    h_new = h * h_scale

    return (int(x_new), int(y_new), int(w_new), int(h_new))


def compute_bbox_error(bbox1, bbox2):
    """Compute L2 error between two bboxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Normalize by bbox size to avoid scale bias
    avg_size = (w1 + w2 + h1 + h2) / 4

    dx = (x1 - x2) / avg_size
    dy = (y1 - y2) / avg_size
    dw = (w1 - w2) / avg_size
    dh = (h1 - h2) / avg_size

    return np.sqrt(dx**2 + dy**2 + dw**2 + dh**2)


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def objective_function(coeffs, raw_bboxes, target_bboxes):
    """Objective function to minimize: total error between corrected and target bboxes."""
    total_error = 0.0

    for key in raw_bboxes.keys():
        raw_bbox = raw_bboxes[key]
        target_bbox = target_bboxes[key]

        corrected_bbox = apply_correction(raw_bbox, coeffs)
        error = compute_bbox_error(corrected_bbox, target_bbox)
        total_error += error

    return total_error


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def test_convergence(gray, bbox, max_iterations=20):
    """Test CLNF convergence with given bbox."""
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=max_iterations)
    landmarks, info = clnf.fit(gray, bbox, return_params=True)

    return {
        'converged': info['converged'],
        'iterations': info['iterations'],
        'final_update': info['final_update'],
        'ratio_to_target': info['final_update'] / 0.005
    }


def main():
    print("="*80)
    print("DERIVING PYMTCNN-SPECIFIC CORRECTION COEFFICIENTS")
    print("="*80)
    print()

    # OpenFace's original coefficients (for comparison)
    openface_coeffs = [-0.0075, 0.2459, 1.0323, 0.7751]
    print("OpenFace's C++ MTCNN correction coefficients:")
    print(f"  dx_coeff = {openface_coeffs[0]:.4f}")
    print(f"  dy_coeff = {openface_coeffs[1]:.4f}")
    print(f"  w_scale  = {openface_coeffs[2]:.4f}")
    print(f"  h_scale  = {openface_coeffs[3]:.4f}")
    print()

    # Optimize coefficients to minimize error
    print("Optimizing pyMTCNN correction coefficients...")
    print()

    # Start with OpenFace coefficients as initial guess
    initial_coeffs = openface_coeffs.copy()

    result = minimize(
        objective_function,
        initial_coeffs,
        args=(RAW_PYMTCNN_BBOXES, CPP_CORRECTED_BBOXES),
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6}
    )

    pymtcnn_coeffs = result.x

    print("="*80)
    print("DERIVED PYMTCNN CORRECTION COEFFICIENTS:")
    print("="*80)
    print(f"  dx_coeff = {pymtcnn_coeffs[0]:.4f}  (OpenFace: {openface_coeffs[0]:.4f}, Δ={pymtcnn_coeffs[0]-openface_coeffs[0]:+.4f})")
    print(f"  dy_coeff = {pymtcnn_coeffs[1]:.4f}  (OpenFace: {openface_coeffs[1]:.4f}, Δ={pymtcnn_coeffs[1]-openface_coeffs[1]:+.4f})")
    print(f"  w_scale  = {pymtcnn_coeffs[2]:.4f}  (OpenFace: {openface_coeffs[2]:.4f}, Δ={pymtcnn_coeffs[2]-openface_coeffs[2]:+.4f})")
    print(f"  h_scale  = {pymtcnn_coeffs[3]:.4f}  (OpenFace: {openface_coeffs[3]:.4f}, Δ={pymtcnn_coeffs[3]-openface_coeffs[3]:+.4f})")
    print()

    # Evaluate bbox accuracy for both corrections
    print("="*80)
    print("BBOX ACCURACY COMPARISON:")
    print("="*80)
    print()

    openface_ious = []
    pymtcnn_ious = []

    for key in RAW_PYMTCNN_BBOXES.keys():
        raw_bbox = RAW_PYMTCNN_BBOXES[key]
        target_bbox = CPP_CORRECTED_BBOXES[key]

        openface_corrected = apply_correction(raw_bbox, openface_coeffs)
        pymtcnn_corrected = apply_correction(raw_bbox, pymtcnn_coeffs)

        openface_iou = compute_iou(openface_corrected, target_bbox)
        pymtcnn_iou = compute_iou(pymtcnn_corrected, target_bbox)

        openface_ious.append(openface_iou)
        pymtcnn_ious.append(pymtcnn_iou)

        video, frame = key
        print(f"{video} f{frame}:")
        print(f"  OpenFace correction IoU:  {openface_iou:.3f}")
        print(f"  pyMTCNN correction IoU:   {pymtcnn_iou:.3f}")
        print(f"  Improvement:              {(pymtcnn_iou - openface_iou)*100:+.1f}%")
        print()

    print("Summary:")
    print(f"  OpenFace correction: mean IoU = {np.mean(openface_ious):.3f}")
    print(f"  pyMTCNN correction:  mean IoU = {np.mean(pymtcnn_ious):.3f}")
    print(f"  Improvement:         {(np.mean(pymtcnn_ious) - np.mean(openface_ious))*100:+.1f}%")
    print()

    # Test convergence with both corrections
    print("="*80)
    print("CONVERGENCE COMPARISON:")
    print("="*80)
    print()

    all_results = []

    for video_path, frame_nums in VIDEO_CONFIGS:
        video_name = Path(video_path).stem

        for frame_num in frame_nums:
            key = (video_name, frame_num)

            if key not in RAW_PYMTCNN_BBOXES:
                continue

            print(f"Testing {video_name} frame {frame_num}...")

            # Load frame
            frame = extract_frame(video_path, frame_num)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw_bbox = RAW_PYMTCNN_BBOXES[key]
            openface_bbox = apply_correction(raw_bbox, openface_coeffs)
            pymtcnn_bbox = apply_correction(raw_bbox, pymtcnn_coeffs)

            # Test convergence
            raw_conv = test_convergence(gray, raw_bbox)
            openface_conv = test_convergence(gray, openface_bbox)
            pymtcnn_conv = test_convergence(gray, pymtcnn_bbox)

            openface_improvement = ((raw_conv['final_update'] - openface_conv['final_update']) /
                                   raw_conv['final_update'] * 100)
            pymtcnn_improvement = ((raw_conv['final_update'] - pymtcnn_conv['final_update']) /
                                  raw_conv['final_update'] * 100)

            all_results.append({
                'video': video_name,
                'frame': frame_num,
                'raw_final_update': raw_conv['final_update'],
                'openface_final_update': openface_conv['final_update'],
                'pymtcnn_final_update': pymtcnn_conv['final_update'],
                'openface_improvement_pct': openface_improvement,
                'pymtcnn_improvement_pct': pymtcnn_improvement
            })

            print(f"  Raw:      {raw_conv['final_update']:.6f} ({raw_conv['ratio_to_target']:.1f}x)")
            print(f"  OpenFace: {openface_conv['final_update']:.6f} ({openface_conv['ratio_to_target']:.1f}x) [{openface_improvement:+.1f}%]")
            print(f"  pyMTCNN:  {pymtcnn_conv['final_update']:.6f} ({pymtcnn_conv['ratio_to_target']:.1f}x) [{pymtcnn_improvement:+.1f}%]")
            print()

    # Summary statistics
    print("="*80)
    print("CONVERGENCE SUMMARY:")
    print("="*80)
    print()

    openface_improvements = [r['openface_improvement_pct'] for r in all_results]
    pymtcnn_improvements = [r['pymtcnn_improvement_pct'] for r in all_results]

    print("OpenFace correction (original):")
    print(f"  Mean improvement: {np.mean(openface_improvements):+.1f}%")
    print(f"  Median:           {np.median(openface_improvements):+.1f}%")
    print(f"  Std:              {np.std(openface_improvements):.1f}%")
    print(f"  Range:            [{min(openface_improvements):+.1f}%, {max(openface_improvements):+.1f}%]")
    print(f"  Frames improved:  {sum(1 for x in openface_improvements if x > 0)}/{len(openface_improvements)}")
    print()

    print("pyMTCNN correction (derived):")
    print(f"  Mean improvement: {np.mean(pymtcnn_improvements):+.1f}%")
    print(f"  Median:           {np.median(pymtcnn_improvements):+.1f}%")
    print(f"  Std:              {np.std(pymtcnn_improvements):.1f}%")
    print(f"  Range:            [{min(pymtcnn_improvements):+.1f}%, {max(pymtcnn_improvements):+.1f}%]")
    print(f"  Frames improved:  {sum(1 for x in pymtcnn_improvements if x > 0)}/{len(pymtcnn_improvements)}")
    print()

    delta = np.mean(pymtcnn_improvements) - np.mean(openface_improvements)
    print(f"Additional improvement from pyMTCNN-specific correction: {delta:+.1f}%")
    print()

    # Show best and worst cases
    print("="*80)
    print("DETAILED BREAKDOWN:")
    print("="*80)
    print()
    print(f"{'Frame':<20} {'Raw':<10} {'OpenFace':<10} {'pyMTCNN':<10} {'OF Δ%':<10} {'pyM Δ%':<10}")
    print("-"*80)

    for r in all_results:
        frame_name = f"{r['video']} f{r['frame']}"
        print(f"{frame_name:<20} "
              f"{r['raw_final_update']:>9.6f} "
              f"{r['openface_final_update']:>9.6f} "
              f"{r['pymtcnn_final_update']:>9.6f} "
              f"{r['openface_improvement_pct']:>9.1f} "
              f"{r['pymtcnn_improvement_pct']:>9.1f}")

    print()
    print("="*80)
    print("RECOMMENDATION:")
    print("="*80)

    if delta > 5:
        print(f"✓ pyMTCNN-specific correction is SIGNIFICANTLY better (+{delta:.1f}%)")
        print("  → Use derived coefficients for pyMTCNN bboxes")
    elif delta > 0:
        print(f"~ pyMTCNN-specific correction is slightly better (+{delta:.1f}%)")
        print("  → Consider using derived coefficients")
    else:
        print(f"✗ OpenFace correction is sufficient (pyMTCNN: {delta:+.1f}%)")
        print("  → Keep using OpenFace coefficients")

    print()
    print("Code snippet for implementation:")
    print()
    print("```python")
    print("# pyMTCNN-specific correction (derived for pyMTCNN detector)")
    print(f"PYMTCNN_DX_COEFF = {pymtcnn_coeffs[0]:.4f}")
    print(f"PYMTCNN_DY_COEFF = {pymtcnn_coeffs[1]:.4f}")
    print(f"PYMTCNN_W_SCALE = {pymtcnn_coeffs[2]:.4f}")
    print(f"PYMTCNN_H_SCALE = {pymtcnn_coeffs[3]:.4f}")
    print()
    print("def apply_pymtcnn_correction(bbox):")
    print("    x, y, w, h = bbox")
    print(f"    x_new = int(x + w * {pymtcnn_coeffs[0]:.4f})")
    print(f"    y_new = int(y + h * {pymtcnn_coeffs[1]:.4f})")
    print(f"    w_new = int(w * {pymtcnn_coeffs[2]:.4f})")
    print(f"    h_new = int(h * {pymtcnn_coeffs[3]:.4f})")
    print("    return (x_new, y_new, w_new, h_new)")
    print("```")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
