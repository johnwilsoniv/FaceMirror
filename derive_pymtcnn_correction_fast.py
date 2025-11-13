#!/usr/bin/env python3
"""
Derive pyMTCNN-specific correction coefficients (fast version - no convergence testing).

We optimize coefficients to transform raw pyMTCNN bboxes to match C++ corrected bboxes.
"""

import numpy as np
from scipy.optimize import minimize

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
    """Compute L2 error between two bboxes (normalized by size)."""
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
    """Objective: minimize total error between corrected and target bboxes."""
    total_error = 0.0

    for key in raw_bboxes.keys():
        raw_bbox = raw_bboxes[key]
        target_bbox = target_bboxes[key]

        corrected_bbox = apply_correction(raw_bbox, coeffs)
        error = compute_bbox_error(corrected_bbox, target_bbox)
        total_error += error

    return total_error


def main():
    print("="*80)
    print("DERIVING PYMTCNN-SPECIFIC CORRECTION COEFFICIENTS (FAST)")
    print("="*80)
    print()

    # OpenFace's original coefficients (for C++ MTCNN)
    openface_coeffs = np.array([-0.0075, 0.2459, 1.0323, 0.7751])
    print("OpenFace's C++ MTCNN correction:")
    print(f"  dx_coeff = {openface_coeffs[0]:.4f}  [shift X by width * coeff]")
    print(f"  dy_coeff = {openface_coeffs[1]:.4f}  [shift Y by height * coeff]")
    print(f"  w_scale  = {openface_coeffs[2]:.4f}  [scale width]")
    print(f"  h_scale  = {openface_coeffs[3]:.4f}  [scale height]")
    print()

    # Compute average bbox characteristics for each detector
    print("Analyzing raw bbox characteristics...")
    print()

    pymtcnn_widths = [bbox[2] for bbox in RAW_PYMTCNN_BBOXES.values()]
    pymtcnn_heights = [bbox[3] for bbox in RAW_PYMTCNN_BBOXES.values()]

    # Invert OpenFace correction from C++ corrected bboxes to get raw C++ MTCNN
    def invert_openface_correction(corrected_bbox):
        x_corr, y_corr, w_corr, h_corr = corrected_bbox
        w_raw = w_corr / 1.0323
        h_raw = h_corr / 0.7751
        x_raw = x_corr - w_raw * (-0.0075)
        y_raw = y_corr - h_raw * 0.2459
        return (int(x_raw), int(y_raw), int(w_raw), int(h_raw))

    raw_cpp_bboxes = {k: invert_openface_correction(v) for k, v in CPP_CORRECTED_BBOXES.items()}
    cpp_widths = [bbox[2] for bbox in raw_cpp_bboxes.values()]
    cpp_heights = [bbox[3] for bbox in raw_cpp_bboxes.values()]

    print(f"Raw pyMTCNN:  width={np.mean(pymtcnn_widths):.1f}±{np.std(pymtcnn_widths):.1f}  height={np.mean(pymtcnn_heights):.1f}±{np.std(pymtcnn_heights):.1f}")
    print(f"Raw C++ MTCNN: width={np.mean(cpp_widths):.1f}±{np.std(cpp_widths):.1f}  height={np.mean(cpp_heights):.1f}±{np.std(cpp_heights):.1f}")
    print()

    width_ratio = np.mean(cpp_widths) / np.mean(pymtcnn_widths)
    height_ratio = np.mean(cpp_heights) / np.mean(pymtcnn_heights)
    print(f"Average size differences:")
    print(f"  pyMTCNN is {(1 - width_ratio) * 100:.1f}% wider than C++ MTCNN")
    print(f"  pyMTCNN is {(1 - height_ratio) * 100:.1f}% taller than C++ MTCNN")
    print()

    # Optimize coefficients
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
    print()
    print(f"  dx_coeff = {pymtcnn_coeffs[0]:.4f}  (OpenFace: {openface_coeffs[0]:.4f}, Δ={pymtcnn_coeffs[0]-openface_coeffs[0]:+.4f})")
    print(f"  dy_coeff = {pymtcnn_coeffs[1]:.4f}  (OpenFace: {openface_coeffs[1]:.4f}, Δ={pymtcnn_coeffs[1]-openface_coeffs[1]:+.4f})")
    print(f"  w_scale  = {pymtcnn_coeffs[2]:.4f}  (OpenFace: {openface_coeffs[2]:.4f}, Δ={pymtcnn_coeffs[2]-openface_coeffs[2]:+.4f})")
    print(f"  h_scale  = {pymtcnn_coeffs[3]:.4f}  (OpenFace: {openface_coeffs[3]:.4f}, Δ={pymtcnn_coeffs[3]-openface_coeffs[3]:+.4f})")
    print()

    # Evaluate bbox accuracy
    print("="*80)
    print("BBOX ACCURACY COMPARISON:")
    print("="*80)
    print()

    openface_ious = []
    pymtcnn_ious = []
    openface_errors = []
    pymtcnn_errors = []

    print(f"{'Frame':<20} {'OpenFace IoU':<15} {'pyMTCNN IoU':<15} {'Improvement'}")
    print("-"*70)

    for key in RAW_PYMTCNN_BBOXES.keys():
        raw_bbox = RAW_PYMTCNN_BBOXES[key]
        target_bbox = CPP_CORRECTED_BBOXES[key]

        openface_corrected = apply_correction(raw_bbox, openface_coeffs)
        pymtcnn_corrected = apply_correction(raw_bbox, pymtcnn_coeffs)

        openface_iou = compute_iou(openface_corrected, target_bbox)
        pymtcnn_iou = compute_iou(pymtcnn_corrected, target_bbox)

        openface_error = compute_bbox_error(openface_corrected, target_bbox)
        pymtcnn_error = compute_bbox_error(pymtcnn_corrected, target_bbox)

        openface_ious.append(openface_iou)
        pymtcnn_ious.append(pymtcnn_iou)
        openface_errors.append(openface_error)
        pymtcnn_errors.append(pymtcnn_error)

        video, frame = key
        improvement = (pymtcnn_iou - openface_iou) * 100
        print(f"{video} f{frame:<6} {openface_iou:>14.3f} {pymtcnn_iou:>14.3f} {improvement:>13.1f}%")

    print()
    print("="*80)
    print("SUMMARY:")
    print("="*80)
    print()
    print(f"IoU with C++ target:")
    print(f"  OpenFace correction: mean={np.mean(openface_ious):.3f}  std={np.std(openface_ious):.3f}")
    print(f"  pyMTCNN correction:  mean={np.mean(pymtcnn_ious):.3f}  std={np.std(pymtcnn_ious):.3f}")
    print(f"  Improvement:         {(np.mean(pymtcnn_ious) - np.mean(openface_ious))*100:+.1f}%")
    print()
    print(f"Normalized L2 error:")
    print(f"  OpenFace correction: mean={np.mean(openface_errors):.4f}")
    print(f"  pyMTCNN correction:  mean={np.mean(pymtcnn_errors):.4f}")
    print(f"  Improvement:         {(1 - np.mean(pymtcnn_errors)/np.mean(openface_errors))*100:+.1f}%")
    print()

    # Interpretation
    print("="*80)
    print("INTERPRETATION:")
    print("="*80)
    print()

    iou_improvement = (np.mean(pymtcnn_ious) - np.mean(openface_ious)) * 100
    error_improvement = (1 - np.mean(pymtcnn_errors) / np.mean(openface_errors)) * 100

    if iou_improvement > 2:
        print(f"✓ pyMTCNN-specific correction SIGNIFICANTLY better (+{iou_improvement:.1f}% IoU)")
        print("  → Derived coefficients account for pyMTCNN's different bbox characteristics")
    elif iou_improvement > 0.5:
        print(f"~ pyMTCNN-specific correction slightly better (+{iou_improvement:.1f}% IoU)")
        print("  → Small but measurable improvement")
    else:
        print(f"✗ OpenFace correction sufficient ({iou_improvement:+.1f}% IoU difference)")
        print("  → pyMTCNN and C++ MTCNN close enough that same correction works")

    print()
    print("="*80)
    print("IMPLEMENTATION CODE:")
    print("="*80)
    print()
    print("```python")
    print("# pyMTCNN-specific bbox correction")
    print("# Optimized to transform pyMTCNN bboxes → OpenFace target")
    print(f"PYMTCNN_DX_COEFF = {pymtcnn_coeffs[0]:.4f}")
    print(f"PYMTCNN_DY_COEFF = {pymtcnn_coeffs[1]:.4f}")
    print(f"PYMTCNN_W_SCALE = {pymtcnn_coeffs[2]:.4f}")
    print(f"PYMTCNN_H_SCALE = {pymtcnn_coeffs[3]:.4f}")
    print()
    print("def apply_pymtcnn_correction(bbox):")
    print('    """Apply pyMTCNN-specific bbox correction."""')
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
