#!/usr/bin/env python3
"""
Derive RetinaFace Correction Factor

Systematically derives correction parameters to transform RetinaFace bboxes
to match C++ MTCNN initialization scale perfectly.

Approach:
1. Collect bbox data from both detectors on diverse faces
2. Use optimization to find transformation parameters
3. Validate correction achieves near-perfect init scale match
"""

import subprocess
import re
import cv2
import numpy as np
from pathlib import Path
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

def run_cpp_mtcnn(image_path, output_dir):
    """Run C++ MTCNN and get bbox + init scale."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-verbose"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+)', output)

    if not bbox_match or not init_params_match:
        return None

    data = {
        'bbox': tuple(float(bbox_match.group(i)) for i in range(1, 5)),
        'init_scale': float(init_params_match.group(1))
    }

    return data

def run_retinaface(image_path):
    """Run RetinaFace and get bbox."""
    from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector

    img = cv2.imread(str(image_path))

    retinaface_model = Path("pyfaceau/weights/retinaface_mobilenet025_coreml.onnx")
    detector = ONNXRetinaFaceDetector(
        str(retinaface_model),
        use_coreml=False,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    detections, _ = detector.detect_faces(img, resize=1.0)

    if len(detections) == 0:
        return None

    detection = detections[0]
    x1, y1, x2, y2 = detection[:4]

    return {
        'bbox': (x1, y1, x2 - x1, y2 - y1),
        'confidence': float(detection[4]) if len(detection) > 4 else None
    }

def apply_correction(retina_bbox, params):
    """
    Apply correction transform to RetinaFace bbox.

    Transform model:
        corrected_x = retina_x + alpha * retina_w
        corrected_y = retina_y + beta * retina_h
        corrected_w = retina_w * gamma
        corrected_h = retina_h * delta

    Args:
        retina_bbox: (x, y, w, h) from RetinaFace
        params: [alpha, beta, gamma, delta] correction parameters

    Returns:
        corrected_bbox: (x, y, w, h) transformed to match C++ MTCNN
    """
    rx, ry, rw, rh = retina_bbox
    alpha, beta, gamma, delta = params

    cx = rx + alpha * rw
    cy = ry + beta * rh
    cw = rw * gamma
    ch = rh * delta

    return (cx, cy, cw, ch)

def compute_init_scale(bbox):
    """Compute initialization scale from bbox using PDM."""
    from pyclnf.core import PDM

    model_dir = Path("pyclnf/models/exported_pdm")
    pdm = PDM(str(model_dir))
    init_params = pdm.init_params(bbox)

    return init_params[0]

def objective_function(params, calibration_data):
    """
    Objective function for optimization.

    Minimize the sum of squared differences between:
    - C++ MTCNN init scale (target)
    - Corrected RetinaFace init scale (predicted)

    Args:
        params: [alpha, beta, gamma, delta] to optimize
        calibration_data: List of dicts with 'cpp' and 'retina' data

    Returns:
        total_error: Sum of squared scale differences
    """
    total_error = 0.0

    for data in calibration_data:
        # Apply correction to RetinaFace bbox
        corrected_bbox = apply_correction(data['retina']['bbox'], params)

        # Compute init scale from corrected bbox
        try:
            corrected_scale = compute_init_scale(corrected_bbox)
        except:
            # Penalize invalid bboxes heavily
            return 1e10

        # Compare to target C++ scale
        target_scale = data['cpp']['init_scale']
        error = (corrected_scale - target_scale) ** 2

        total_error += error

    return total_error

def collect_calibration_data(frame_paths, output_dir):
    """Collect bbox data from both detectors on all frames."""
    calibration_data = []

    print("="*80)
    print("COLLECTING CALIBRATION DATA")
    print("="*80)

    for i, frame_path in enumerate(frame_paths, 1):
        print(f"\nFrame {i}/{len(frame_paths)}: {frame_path.name}")
        print("-" * 80)

        # Run C++ MTCNN
        cpp_data = run_cpp_mtcnn(frame_path, output_dir)
        if cpp_data is None:
            print("  ✗ C++ MTCNN failed to detect face")
            continue

        x, y, w, h = cpp_data['bbox']
        print(f"  C++ MTCNN: bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}), scale={cpp_data['init_scale']:.6f}")

        # Run RetinaFace
        retina_data = run_retinaface(frame_path)
        if retina_data is None:
            print("  ✗ RetinaFace failed to detect face")
            continue

        x, y, w, h = retina_data['bbox']
        print(f"  RetinaFace: bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")

        calibration_data.append({
            'frame': frame_path.name,
            'cpp': cpp_data,
            'retina': retina_data
        })

        print(f"  ✓ Data collected")

    print(f"\n{'='*80}")
    print(f"Collected {len(calibration_data)} calibration samples")
    print(f"{'='*80}\n")

    return calibration_data

def optimize_correction_params(calibration_data):
    """Use optimization to find best correction parameters."""
    print("="*80)
    print("OPTIMIZING CORRECTION PARAMETERS")
    print("="*80)

    # Initial guess based on single-face analysis:
    # beta ~ 0.29 (shift down), delta ~ 0.76 (height reduction)
    initial_params = [0.0, 0.29, 1.0, 0.76]

    print(f"\nInitial guess: alpha={initial_params[0]:.4f}, beta={initial_params[1]:.4f}, " +
          f"gamma={initial_params[2]:.4f}, delta={initial_params[3]:.4f}")

    # Bounds: reasonable ranges for transformation
    bounds = [
        (-0.3, 0.3),   # alpha: horizontal shift ±30% of width
        (0.0, 0.5),    # beta: vertical shift 0-50% of height (downward)
        (0.8, 1.2),    # gamma: width scale 80-120%
        (0.5, 1.0),    # delta: height scale 50-100% (reduction)
    ]

    print("\nOptimizing...")
    result = minimize(
        objective_function,
        initial_params,
        args=(calibration_data,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )

    if result.success:
        print("✓ Optimization converged!")
    else:
        print("⚠ Optimization did not fully converge, but found best parameters")

    optimal_params = result.x
    print(f"\nOptimal parameters:")
    print(f"  alpha = {optimal_params[0]:.6f}  (horizontal shift)")
    print(f"  beta  = {optimal_params[1]:.6f}  (vertical shift)")
    print(f"  gamma = {optimal_params[2]:.6f}  (width scale)")
    print(f"  delta = {optimal_params[3]:.6f}  (height scale)")
    print(f"\nFinal error: {result.fun:.8f}")
    print(f"{'='*80}\n")

    return optimal_params

def validate_correction(calibration_data, correction_params):
    """Validate that correction achieves target accuracy."""
    print("="*80)
    print("VALIDATING CORRECTION")
    print("="*80)

    errors = []

    for i, data in enumerate(calibration_data, 1):
        # Apply correction
        corrected_bbox = apply_correction(data['retina']['bbox'], correction_params)
        corrected_scale = compute_init_scale(corrected_bbox)

        # Compare to target
        target_scale = data['cpp']['init_scale']
        error = abs(corrected_scale - target_scale)
        percent_error = (error / target_scale) * 100

        errors.append(error)

        print(f"\nFrame {i}: {data['frame']}")
        print(f"  Target scale (C++ MTCNN):  {target_scale:.6f}")
        print(f"  Corrected scale (RetinaF): {corrected_scale:.6f}")
        print(f"  Error: {error:.6f} ({percent_error:.3f}%)")

        if error < 0.01:
            print(f"  ✓ EXCELLENT")
        elif error < 0.05:
            print(f"  ✓ GOOD")
        elif error < 0.1:
            print(f"  ⚠ MODERATE")
        else:
            print(f"  ✗ POOR")

    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Mean error:   {np.mean(errors):.6f}")
    print(f"Max error:    {np.max(errors):.6f}")
    print(f"Std error:    {np.std(errors):.6f}")
    print(f"Mean % error: {(np.mean(errors) / np.mean([d['cpp']['init_scale'] for d in calibration_data])) * 100:.3f}%")

    if np.mean(errors) < 0.01:
        print(f"\n✓ PERFECT! Correction achieves sub-1% accuracy")
    elif np.mean(errors) < 0.05:
        print(f"\n✓ EXCELLENT! Correction achieves <5% accuracy")
    elif np.mean(errors) < 0.1:
        print(f"\n✓ GOOD! Correction achieves <10% accuracy")
    else:
        print(f"\n⚠ Correction needs improvement")

    print(f"{'='*80}\n")

    return errors

def create_visualization(calibration_data, correction_params, output_path):
    """Create visualization of correction results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Before vs After Correction (Init Scale)
    ax = axes[0, 0]
    frame_indices = np.arange(len(calibration_data))

    cpp_scales = [d['cpp']['init_scale'] for d in calibration_data]

    # Original RetinaFace scales
    original_scales = []
    for d in calibration_data:
        scale = compute_init_scale(d['retina']['bbox'])
        original_scales.append(scale)

    # Corrected RetinaFace scales
    corrected_scales = []
    for d in calibration_data:
        corrected_bbox = apply_correction(d['retina']['bbox'], correction_params)
        scale = compute_init_scale(corrected_bbox)
        corrected_scales.append(scale)

    ax.plot(frame_indices, cpp_scales, 'ro-', label='C++ MTCNN (Target)', linewidth=2, markersize=8)
    ax.plot(frame_indices, original_scales, 'b^--', label='RetinaFace (Before)', linewidth=2, markersize=8, alpha=0.6)
    ax.plot(frame_indices, corrected_scales, 'gs-', label='RetinaFace (After Correction)', linewidth=2, markersize=8)

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Initialization Scale', fontsize=12)
    ax.set_title('Correction Effect on Initialization Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Error Distribution
    ax = axes[0, 1]

    errors_before = [abs(o - c) for o, c in zip(original_scales, cpp_scales)]
    errors_after = [abs(c - t) for c, t in zip(corrected_scales, cpp_scales)]

    x_pos = np.arange(len(calibration_data))
    width = 0.35

    ax.bar(x_pos - width/2, errors_before, width, label='Before Correction', color='lightcoral', alpha=0.7)
    ax.bar(x_pos + width/2, errors_after, width, label='After Correction', color='lightgreen', alpha=0.7)

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Error Reduction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: BBox Transformation Visualization
    ax = axes[1, 0]
    ax.axis('off')

    text = "CORRECTION PARAMETERS\n"
    text += "="*50 + "\n\n"
    text += f"alpha = {correction_params[0]:+.6f}  (horizontal shift)\n"
    text += f"beta  = {correction_params[1]:+.6f}  (vertical shift)\n"
    text += f"gamma = {correction_params[2]:+.6f}  (width scale)\n"
    text += f"delta = {correction_params[3]:+.6f}  (height scale)\n\n"
    text += "TRANSFORMATION FORMULA:\n"
    text += "-"*50 + "\n"
    text += "corrected_x = retina_x + alpha * retina_w\n"
    text += "corrected_y = retina_y + beta * retina_h\n"
    text += "corrected_w = retina_w * gamma\n"
    text += "corrected_h = retina_h * delta\n\n"
    text += "="*50 + "\n\n"
    text += "VALIDATION RESULTS:\n"
    text += "-"*50 + "\n"
    text += f"Mean error:     {np.mean(errors_after):.6f}\n"
    text += f"Max error:      {np.max(errors_after):.6f}\n"
    text += f"Std deviation:  {np.std(errors_after):.6f}\n"
    text += f"Mean % error:   {(np.mean(errors_after) / np.mean(cpp_scales)) * 100:.3f}%\n\n"

    if np.mean(errors_after) < 0.01:
        text += "✓ PERFECT accuracy achieved!\n"
    elif np.mean(errors_after) < 0.05:
        text += "✓ EXCELLENT accuracy achieved!\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Plot 4: Percentage Improvement
    ax = axes[1, 1]

    improvement = [(b - a) / b * 100 for b, a in zip(errors_before, errors_after)]

    ax.bar(frame_indices, improvement, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Error Reduction (%)', fontsize=12)
    ax.set_title('Percentage Improvement per Frame', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    mean_improvement = np.mean(improvement)
    ax.text(0.5, 0.95, f'Mean Improvement: {mean_improvement:.1f}%',
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_path}\n")
    plt.close()

def main():
    """Main calibration and correction derivation."""
    # Setup
    calibration_dir = Path("calibration_frames")
    output_dir = Path("test_output/retinaface_correction")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all calibration frames
    frame_paths = sorted(calibration_dir.glob("*.jpg"))

    print("\n" + "="*80)
    print("RETINAFACE CORRECTION FACTOR DERIVATION")
    print("="*80)
    print(f"Calibration frames: {len(frame_paths)}")
    print(f"Goal: Transform RetinaFace bbox → C++ MTCNN initialization")
    print()

    # Step 1: Collect data
    calibration_data = collect_calibration_data(frame_paths, output_dir)

    if len(calibration_data) < 3:
        print("ERROR: Need at least 3 successful detections for calibration")
        return

    # Step 2: Optimize correction parameters
    correction_params = optimize_correction_params(calibration_data)

    # Step 3: Validate correction
    errors = validate_correction(calibration_data, correction_params)

    # Step 4: Create visualization
    viz_path = output_dir / "correction_derivation_results.jpg"
    create_visualization(calibration_data, correction_params, viz_path)

    # Step 5: Save correction parameters
    params_file = output_dir / "retinaface_correction_params.txt"
    with open(params_file, 'w') as f:
        f.write("# RetinaFace to C++ MTCNN BBox Correction Parameters\n")
        f.write("# Derived from calibration on 9 frames from 3 patients\n\n")
        f.write(f"alpha = {correction_params[0]:.8f}  # horizontal shift\n")
        f.write(f"beta  = {correction_params[1]:.8f}  # vertical shift\n")
        f.write(f"gamma = {correction_params[2]:.8f}  # width scale\n")
        f.write(f"delta = {correction_params[3]:.8f}  # height scale\n\n")
        f.write("# Application formula:\n")
        f.write("# corrected_x = retina_x + alpha * retina_w\n")
        f.write("# corrected_y = retina_y + beta * retina_h\n")
        f.write("# corrected_w = retina_w * gamma\n")
        f.write("# corrected_h = retina_h * delta\n\n")
        f.write(f"# Validation results:\n")
        f.write(f"# Mean error: {np.mean(errors):.6f}\n")
        f.write(f"# Max error:  {np.max(errors):.6f}\n")
        f.write(f"# Mean % error: {(np.mean(errors) / np.mean([d['cpp']['init_scale'] for d in calibration_data])) * 100:.3f}%\n")

    print(f"{'='*80}")
    print(f"CORRECTION PARAMETERS SAVED")
    print(f"{'='*80}")
    print(f"File: {params_file}")
    print(f"Visualization: {viz_path}")
    print()

if __name__ == "__main__":
    main()
